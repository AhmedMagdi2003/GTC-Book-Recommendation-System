import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import os

# -------------------------------
# Load pre-trained model and data
# -------------------------------
model_filename = os.path.join("App", "models", "knn_model.pkl")
with open(model_filename, 'rb') as file:
    trained_model = pickle.load(file)

data_filename = 'data/cleaned_book_ratings_plus.csv'
df = pd.read_csv(data_filename)

# Build item-user matrix for pre-trained model
item_user_matrix = df.pivot_table(
    index='title', columns='user_id', values='book_rating', fill_value=0
)
sparse_item_user = csr_matrix(item_user_matrix.values)

# Helper function to show recommendations with images
def show_recommendations(recommendations, df):
    for book in recommendations:
        st.write(f"**{book}**")
        img_url = df.loc[df['title'] == book, 'img_url'].values
        if len(img_url) > 0 and pd.notna(img_url[0]):
            st.image(img_url[0], width=120)
        else:
            st.write("No image available.")

# -------------------------------
# Streamlit Layout
# -------------------------------
st.sidebar.title("Book Recommender")
page = st.sidebar.radio("Choose a page:", ["Upload & Train", "Pre-trained Model"])

# -------------------------------
# Page 1: Upload & Train
# -------------------------------
if page == "Upload & Train":
    st.title("📂 Upload Your Ratings CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with book ratings", type="csv")

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("✅ User data loaded successfully!")

        # Expect columns: user_id, title, book_rating, (optional) img_url
        try:
            user_item_matrix = user_df.pivot_table(
                index='title', columns='user_id', values='book_rating', fill_value=0
            )
            sparse_user_item = csr_matrix(user_item_matrix.values)

            # Train a fresh KNN model
            knn = NearestNeighbors(metric='cosine', algorithm='brute')
            knn.fit(sparse_user_item)

            st.success("KNN model trained on uploaded data!")

            # Book recommendation function
            def recommend_book(book_name, k=5):
                if book_name not in user_item_matrix.index:
                    return []
                book_id = np.where(user_item_matrix.index == book_name)[0][0]
                distances, suggestions = knn.kneighbors(
                    user_item_matrix.iloc[book_id, :].values.reshape(1, -1),
                    n_neighbors=k + 1
                )
                suggested_books = user_item_matrix.index[suggestions[0][1:]]
                return list(suggested_books)

            # Selectbox for book names
            book_name = st.selectbox(
                "Choose a book to get recommendations:",
                options=sorted(user_item_matrix.index.tolist())
            )

            if st.button("Get Recommendations"):
                recommendations = recommend_book(book_name)
                if recommendations:
                    st.write("### Recommended Books:")
                    show_recommendations(recommendations, user_df)
                else:
                    st.warning("No recommendations found. Try another book.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# -------------------------------
# Page 2: Pre-trained Model
# -------------------------------
elif page == "Pre-trained Model":
    st.title("📚 Amazon Book Recommendations")
    st.write("Get recommendations from our trained model (books with ≥10 ratings).")

    # Recommendation function using pre-trained model
    def recommend_pretrained(book_name, k=5):
        if book_name not in item_user_matrix.index:
            return []
        book_id = np.where(item_user_matrix.index == book_name)[0][0]
        distances, suggestions = trained_model.kneighbors(
            item_user_matrix.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=k + 1
        )
        suggested_books = item_user_matrix.index[suggestions[0][1:]]
        return list(suggested_books)

    # Selectbox for book names (from Amazon data)
    book_name = st.selectbox(
        "Choose a book to get recommendations:",
        options=sorted(item_user_matrix.index.tolist())
    )

    if st.button("Get Recommendations (Pre-trained)"):
        recommendations = recommend_pretrained(book_name, k=5)
        if recommendations:
            st.write("### Recommended Books:")
            show_recommendations(recommendations, df)
        else:
            st.warning("No recommendations found. Try another book.")
