import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import os

# Load the trained KNN model

model_filename = os.path.join("App", "models", "knn_model.pkl")
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the cleaned book ratings data
data_filename = 'data/cleaned_book_ratings_plus.csv'
df = pd.read_csv(data_filename)

# Create the item-user matrix
item_user_matrix = df.pivot_table(index='title', columns='user_id', values='book_rating', fill_value=0)
sparse_item_user = csr_matrix(item_user_matrix.values)

# Streamlit app layout
st.title("Book Recommendation System")
st.write("Upload a CSV file containing book ratings to get recommendations.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write("User data loaded successfully!")

    # Function to recommend books
    def recommend_book(book_name, k=5):
        if book_name not in item_user_matrix.index:
            return []

        book_id = np.where(item_user_matrix.index == book_name)[0][0]
        distances, suggestions = model.kneighbors(
            item_user_matrix.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=k + 1
        )
        suggested_books = item_user_matrix.index[suggestions[0][1:]]
        return list(suggested_books)

    # Input for book name
    book_name = st.text_input("Enter a book name to get recommendations:")
    
    if st.button("Get Recommendations"):
        recommendations = recommend_book(book_name)
        if recommendations:
            st.subheader("Recommended Books:")

            for book in recommendations:
                st.write(f"**{book}**")

                # Try to get image URL from df
                if "img_url" in df.columns:
                    img_row = df[df["title"] == book].head(1)  # take first match
                    if not img_row.empty and pd.notna(img_row["img_url"].values[0]):
                        st.image(img_row["img_url"].values[0], width=120)
        else:
            st.warning("No recommendations found.")
