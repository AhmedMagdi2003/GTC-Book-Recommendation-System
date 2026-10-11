import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# -------------------------------
# Load pre-trained model and data (Cached)
# -------------------------------
@st.cache_resource
def load_knn_model():
    possible_paths = [
        BASE_DIR / "models" / "knn_model.pkl",
        ROOT_DIR / "App" / "models" / "knn_model.pkl",
        Path("App/models/knn_model.pkl"),
    ]
    for p in possible_paths:
        if p.exists():
            with open(p, "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError("knn_model.pkl not found!")


@st.cache_data
def load_ratings_data():
    possible_paths = [
        ROOT_DIR / "data" / "cleaned_book_ratings_plus.csv",
        BASE_DIR / "data" / "cleaned_book_ratings_plus.csv",
        Path("data/cleaned_book_ratings_plus.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            df = pd.read_csv(p)
            item_user_matrix = df.pivot_table(
                index="title",
                columns="user_id",
                values="book_rating",
                fill_value=0,
            )
            return df, item_user_matrix
    raise FileNotFoundError("cleaned_book_ratings_plus.csv not found!")


trained_model = load_knn_model()
df, item_user_matrix = load_ratings_data()


# Helper function to show recommendations with images in a grid
def show_recommendations(recommendations, df):
    cols = st.columns(min(len(recommendations), 5))
    for idx, book in enumerate(recommendations):
        col = cols[idx % len(cols)]
        with col:
            img_url = df.loc[df["title"] == book, "img_url"].values
            if len(img_url) > 0 and pd.notna(img_url[0]):
                st.image(img_url[0], use_container_width=True)
            else:
                st.info("📖 No image")
            st.caption(f"**{book}**")


# -------------------------------
# Streamlit Layout
# -------------------------------
st.sidebar.title("📚 Book Recommender")
page = st.sidebar.radio(
    "Choose a page:", ["Pre-trained Model", "Upload & Train"]
)

# -------------------------------
# Page 1: Pre-trained Model (Default)
# -------------------------------
if page == "Pre-trained Model":
    st.title("📚 Amazon Book Recommendations")
    st.markdown(
        "Explore collaborative filtering recommendations trained on reader ratings using **K-Nearest Neighbors (KNN)**."
    )

    # Recommendation function using pre-trained model
    def recommend_pretrained(book_name, k=5):
        if book_name not in item_user_matrix.index:
            return []
        book_id = np.where(item_user_matrix.index == book_name)[0][0]
        distances, suggestions = trained_model.kneighbors(
            item_user_matrix.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=k + 1,
        )
        suggested_books = item_user_matrix.index[suggestions[0][1:]]
        return list(suggested_books)

    # Selectbox for book names (from Amazon data)
    book_name = st.selectbox(
        "Choose a book to get recommendations:",
        options=sorted(item_user_matrix.index.tolist()),
    )

    if st.button("Get Recommendations (Pre-trained) 🚀"):
        recommendations = recommend_pretrained(book_name, k=5)
        if recommendations:
            st.write("### 📖 Recommended Books:")
            show_recommendations(recommendations, df)
        else:
            st.warning("No recommendations found. Try another book.")

# -------------------------------
# Page 2: Upload & Train
# -------------------------------
elif page == "Upload & Train":
    st.title("📂 Upload Your Ratings CSV")
    st.markdown("Upload custom user-item ratings to train a fresh KNN model dynamically.")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with book ratings (user_id, title, book_rating)",
        type="csv",
    )

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.success("✅ User data loaded successfully!")

        try:
            user_item_matrix = user_df.pivot_table(
                index="title",
                columns="user_id",
                values="book_rating",
                fill_value=0,
            )
            sparse_user_item = csr_matrix(user_item_matrix.values)

            # Train a fresh KNN model
            knn = NearestNeighbors(metric="cosine", algorithm="brute")
            knn.fit(sparse_user_item)

            st.success("KNN model trained on uploaded data!")

            def recommend_book(b_name, k=5):
                if b_name not in user_item_matrix.index:
                    return []
                b_id = np.where(user_item_matrix.index == b_name)[0][0]
                distances, suggestions = knn.kneighbors(
                    user_item_matrix.iloc[b_id, :].values.reshape(1, -1),
                    n_neighbors=k + 1,
                )
                suggested_books = user_item_matrix.index[suggestions[0][1:]]
                return list(suggested_books)

            book_name = st.selectbox(
                "Choose a book from your data:",
                options=sorted(user_item_matrix.index.tolist()),
            )

            if st.button("Get Custom Recommendations"):
                recommendations = recommend_book(book_name)
                if recommendations:
                    st.write("### Recommended Books:")
                    show_recommendations(recommendations, user_df)
                else:
                    st.warning("No recommendations found. Try another book.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
