import streamlit as st
import numpy as np
import pandas as pd
import faiss
import lightgbm as lgb
import pickle
import json

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

@st.cache_resource
def load_artifacts():
    als_data = np.load('artifacts/als_model.npz')
    user_factors = als_data['user_factors']
    item_factors = als_data['item_factors']
    faiss_index = faiss.read_index('artifacts/faiss_index.bin')
    ranker = lgb.Booster(model_file='artifacts/lightgbm_ranker.txt')
    user_features = pd.read_parquet('artifacts/user_features.parquet')
    movie_features = pd.read_parquet('artifacts/movie_features.parquet')
    user_genre_affinity = pd.read_parquet('artifacts/user_genre_affinity.parquet')
    movies_metadata = pd.read_parquet('artifacts/movies_metadata.parquet')

    with open('artifacts/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    with open('artifacts/top_popular_movies.json', 'r') as f:
        top_popular_movies = json.load(f)

    user_features_dict = user_features.set_index('userId').to_dict('index')
    movie_features_dict = movie_features.set_index('movieId').to_dict('index')
    genre_affinity_dict = user_genre_affinity.groupby('userId').apply(
        lambda x: dict(zip(x['genre'], x['genre_affinity']))
    ).to_dict()
    movie_titles_dict = movies_metadata.set_index('movieId')['title'].to_dict()

    return {
        'user_factors': user_factors, 'item_factors': item_factors,
        'faiss_index': faiss_index, 'ranker': ranker,
        'user_to_idx': mappings['user_to_idx'], 'movie_to_idx': mappings['movie_to_idx'],
        'idx_to_movie': mappings['idx_to_movie'], 'feature_cols': mappings['feature_cols'],
        'top_popular_movies': top_popular_movies,
        'user_features_dict': user_features_dict, 'movie_features_dict': movie_features_dict,
        'genre_affinity_dict': genre_affinity_dict, 'movie_titles_dict': movie_titles_dict,
    }

art = load_artifacts()

def get_genre_match(user_id, genres):
    if not genres or user_id not in art['genre_affinity_dict']:
        return 0.0
    scores = [art['genre_affinity_dict'][user_id][g] for g in genres.split('|') if g in art['genre_affinity_dict'][user_id]]
    return float(np.mean(scores)) if scores else 0.0

def recommend(user_id, k=10):
    if user_id not in art['user_to_idx']:
        ids = art['top_popular_movies'][:k]
        return True, [{"title": art['movie_titles_dict'].get(m, "Unknown"), "score": None} for m in ids]

    u_idx = art['user_to_idx'][user_id]
    user_vector = art['user_factors'][u_idx].astype('float32').reshape(1, -1)
    scores, indices = art['faiss_index'].search(user_vector, min(100, len(art['idx_to_movie'])))
    candidates = [int(art['idx_to_movie'][i]) for i in indices[0]]

    u = art['user_features_dict'].get(user_id, {})
    u_count = u.get('user_rating_count', 0)
    u_avg = u.get('user_avg_rating', 3.5)
    u_std = u.get('user_rating_std', 0)

    rows = []
    for movie_id in candidates:
        m = art['movie_features_dict'].get(movie_id)
        if not m or movie_id not in art['movie_to_idx']:
            continue

        user_vec = art['user_factors'][u_idx]
        movie_vec = art['item_factors'][art['movie_to_idx'][movie_id]]
        als_score = float(np.dot(user_vec, movie_vec))
        genre_match = get_genre_match(user_id, m['genres'])

        rows.append({
            'movieId': movie_id, 'als_score': als_score,
            'user_rating_count': u_count, 'user_avg_rating': u_avg, 'user_rating_std': u_std,
            'movie_rating_count': m['movie_rating_count'], 'movie_avg_rating': m['movie_avg_rating'],
            'movie_rating_std': m['movie_rating_std'], 'movie_rating_count_log': m['movie_rating_count_log'],
            'movie_popularity_percentile': m['movie_popularity_percentile'],
            'days_since_last_rating': m['days_since_last_rating'],
            'recent_rating_count_90d_log': m['recent_rating_count_90d_log'],
            'genre_match': genre_match,
        })

    feature_df = pd.DataFrame(rows)
    if len(feature_df) == 0:
        return True, [{"title": art['movie_titles_dict'].get(m, "Unknown"), "score": None} for m in art['top_popular_movies'][:k]]

    feature_df['score'] = art['ranker'].predict(feature_df[art['feature_cols']])
    top_k = feature_df.sort_values('score', ascending=False).head(k)

    return False, [
        {"title": art['movie_titles_dict'].get(int(row['movieId']), "Unknown"), "score": float(row['score'])}
        for _, row in top_k.iterrows()
    ]

st.title("🎬 Movie Recommendation System")
st.write("A two-stage retrieval (ALS + FAISS) + ranking (LightGBM) recommender, built on MovieLens 25M.")

user_id = st.number_input("Enter a User ID", min_value=1, value=1, step=1)
k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        cold_start, recs = recommend(int(user_id), k)
        if cold_start:
            st.info("New/unseen user — showing popular movie fallback.")
        else:
            st.success(f"Personalized recommendations for User {user_id}")
        for i, r in enumerate(recs, 1):
            score_text = f" (score: {r['score']:.3f})" if r['score'] is not None else ""
            st.write(f"**{i}. {r['title']}**{score_text}")