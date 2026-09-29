from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import faiss
import lightgbm as lgb
import pickle
import json

app = FastAPI(title="Movie Recommendation API")

# ===== Load all artifacts once, at startup =====
print("Loading artifacts...")

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
user_to_idx = mappings['user_to_idx']
movie_to_idx = mappings['movie_to_idx']
idx_to_movie = mappings['idx_to_movie']
feature_cols = mappings['feature_cols']

with open('artifacts/top_popular_movies.json', 'r') as f:
    top_popular_movies = json.load(f)

# Fast dictionary lookups (same optimization as in Colab)
user_features_dict = user_features.set_index('userId').to_dict('index')
movie_features_dict = movie_features.set_index('movieId').to_dict('index')
genre_affinity_dict = user_genre_affinity.groupby('userId').apply(
    lambda x: dict(zip(x['genre'], x['genre_affinity']))
).to_dict()
movie_titles_dict = movies_metadata.set_index('movieId')['title'].to_dict()

print("Artifacts loaded successfully.")


@app.get("/")
def root():
    return {"message": "Movie Recommendation API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "movies_loaded": len(movie_features_dict), "users_loaded": len(user_features_dict)}
@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    """
    Returns top-k movie recommendations for a given user_id,
    using the full retrieval + ranking pipeline.
    """
    # ===== Retrieval stage =====
    if user_id not in user_to_idx:
        # Cold-start fallback: popularity list
        recommended_ids = top_popular_movies[:k]
        return {
            "user_id": user_id,
            "cold_start": True,
            "recommendations": [
                {"movieId": int(m), "title": movie_titles_dict.get(m, "Unknown")}
                for m in recommended_ids
            ]
        }

    already_rated = set()  # Note: we don't have raw ratings.csv in production; skipping this check here

    u_idx = user_to_idx[user_id]
    user_vector = user_factors[u_idx].astype('float32').reshape(1, -1)
    scores, indices = faiss_index.search(user_vector, min(100, len(idx_to_movie)))
    candidates = [int(idx_to_movie[i]) for i in indices[0]]

    # ===== Feature building for ranking =====
    rows = []
    if user_id in user_features_dict:
        u = user_features_dict[user_id]
        u_count, u_avg, u_std = u['user_rating_count'], u['user_avg_rating'], u['user_rating_std']
    else:
        u_count, u_avg, u_std = 0, 3.5, 0

    for movie_id in candidates:
        if movie_id in movie_features_dict:
            m = movie_features_dict[movie_id]
            m_count = m['movie_rating_count']
            m_avg = m['movie_avg_rating']
            m_std = m['movie_rating_std']
            m_count_log = m['movie_rating_count_log']
            m_percentile = m['movie_popularity_percentile']
            m_recency = m['days_since_last_rating']
            m_recent_pop = m['recent_rating_count_90d_log']
            genres = m['genres']
        else:
            continue

        user_vec = user_factors[u_idx]
        movie_vec = item_factors[movie_to_idx[movie_id]]
        als_score = float(np.dot(user_vec, movie_vec))

        genre_match = 0.0
        if user_id in genre_affinity_dict and genres:
            scores_list = [genre_affinity_dict[user_id][g] for g in genres.split('|') if g in genre_affinity_dict[user_id]]
            genre_match = float(np.mean(scores_list)) if scores_list else 0.0

        rows.append({
            'movieId': movie_id, 'als_score': als_score,
            'user_rating_count': u_count, 'user_avg_rating': u_avg, 'user_rating_std': u_std,
            'movie_rating_count': m_count, 'movie_avg_rating': m_avg, 'movie_rating_std': m_std,
            'movie_rating_count_log': m_count_log, 'movie_popularity_percentile': m_percentile,
            'days_since_last_rating': m_recency, 'recent_rating_count_90d_log': m_recent_pop,
            'genre_match': genre_match,
        })

    feature_df = pd.DataFrame(rows)
    if len(feature_df) == 0:
        raise HTTPException(status_code=404, detail="No candidates found for this user")

    feature_df['score'] = ranker.predict(feature_df[feature_cols])
    top_k = feature_df.sort_values('score', ascending=False).head(k)

    return {
        "user_id": user_id,
        "cold_start": False,
        "recommendations": [
            {"movieId": int(row['movieId']), "title": movie_titles_dict.get(int(row['movieId']), "Unknown"), "score": float(row['score'])}
            for _, row in top_k.iterrows()
        ]
    }