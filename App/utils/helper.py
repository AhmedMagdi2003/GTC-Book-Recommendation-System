import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def create_item_user_matrix(data):
    item_user_matrix = data.pivot_table(
        index="title",
        columns="user_id",
        values="book_rating",
        fill_value=0
    )
    return csr_matrix(item_user_matrix.values), item_user_matrix.index

def recommend_books(model, item_user_matrix, book_name, k=5):
    if book_name not in item_user_matrix:
        return []

    book_id = np.where(item_user_matrix == book_name)[0][0]
    distances, suggestions = model.kneighbors(
        item_user_matrix.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=k + 1
    )
    
    suggested_books = item_user_matrix.index[suggestions[0][1:]]
    return list(suggested_books)