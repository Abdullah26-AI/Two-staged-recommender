# Two-Stage Movie Recommendation System

A retrieval + ranking recommender built on MovieLens 25M, architected the way production systems (Netflix, YouTube) narrow a large catalog down to a candidate pool, then precisely rank that pool per user.

**Live demo:** https://two-staged-recommender-opem9q4rggwqd8d9vbvibn.streamlit.app/
**Training notebook:** `Two_staged_recommendation_system.ipynb`

## Results

| Configuration | Precision@10 | NDCG@10 |
|---|---|---|
| Popularity Baseline | 25.80% | 49.33% |
| ALS Only (no ranker) | 6.58% | 20.05% |
| ALS + Ranker | 10.08% | 26.06% |
| Popularity Candidates + Ranker | **34.09%** | **55.53%** |

Catalog coverage: the system recommends across **8x more of the catalog** than popularity (1.29% vs 0.16% — 804 vs 100 unique movies across 1,000 sampled users), showing genuine personalization rather than the same fixed list for everyone.

**What this table shows:**
- Ranking helps — ALS+Ranker beats ALS alone by ~53% relative
- The ranker is sound — given popularity's own candidates, it beats the popularity baseline itself
- The bottleneck is retrieval's candidate quality, not ranking logic — isolated via the last row

## Architecture

- **Retrieval:** Alternating Least Squares (50-dim latent factors) + FAISS similarity search, trained on a temporal split (oldest 80% train / 10% val / 10% test) to avoid leaking future data
- **Ranking:** LightGBM (lambdarank), trained on ALS score, user/movie statistics, popularity (log-scaled, percentile, 90-day recency), and a genre-affinity interaction feature
- **Cold-start:** popularity fallback for new users; genre-based TF-IDF content retriever for new items
- **Serving:** FastAPI backend + Streamlit frontend, containerized with Docker; a self-contained `streamlit_app.py` powers the public demo

## Key findings from debugging

1. **Fixed two real ALS bugs** — feeding raw ratings instead of confidence-weighted implicit feedback, and conflating "what counts as signal" with "which movies can exist in the model" (which silently deleted movies from the system).
2. **Diagnosed popularity bias** in retrieval and **Missing-Not-At-Random (MNAR) bias** in the ranking evaluation — both well-documented, researched limitations of implicit-feedback recommenders (Steck, 2010; Schnabel et al., 2016), confirmed across 8+ independent mitigation attempts rather than assumed.
3. **Found and fixed a major bug via review of the evaluation pipeline**: already-rated movies were never excluded from candidates, wasting 57% of final recommendation slots on movies users had already seen. Fixing this nearly tripled every metric.
4. **Isolated the remaining bottleneck** to retrieval candidate quality by feeding the ranker popularity's own candidates — it beat the baseline, proving the ranking model itself is sound.

## Personalization in action

Two real users, same system, same moment — different picks:

**User 1** (arthouse/drama leaning): Fight Club, Amelie, Eternal Sunshine of the Spotless Mind, Andrei Rublev, Children of Paradise, Shawshank Redemption
**User 3** (sci-fi/action leaning): The Martian, Ex Machina, Inception, Edge of Tomorrow, The Dark Knight, Star Wars: Episode IV

## Known limitations

- Precision@10 trails the popularity baseline (main ALS+Ranker config) — traced to MNAR bias in validation labels, not a fixable bug; 8+ mitigations tested in the notebook
- ~38% of test-period likes never existed in training (new-item cold-start) — structural limit of periodic retraining
- No CI/CD, live monitoring, auto-retraining, or Inverse Propensity Scoring — out of scope, noted as future work

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload        # backend, http://127.0.0.1:8000
streamlit run app.py             # frontend, http://localhost:8501
```

Or with Docker:
```bash
docker build -t recsys-app .
docker run -p 8000:8000 recsys-app
```
