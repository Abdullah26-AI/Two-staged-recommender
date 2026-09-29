import streamlit as st
import requests

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("A two-stage retrieval + ranking recommender, built on MovieLens 25M.")

user_id = st.number_input("Enter a User ID", min_value=1, value=1, step=1)
k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        try:
            response = requests.get(f"http://127.0.0.1:8000/recommend/{user_id}?k={k}")
            data = response.json()

            if data.get("cold_start"):
                st.info("This is a new/unseen user — showing popular movie fallback recommendations.")
            else:
                st.success(f"Personalized recommendations for User {user_id}")

            for i, rec in enumerate(data["recommendations"], 1):
                title = rec["title"]
                score_text = f" (score: {rec['score']:.3f})" if "score" in rec else ""
                st.write(f"**{i}. {title}**{score_text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running (uvicorn main:app --reload).")