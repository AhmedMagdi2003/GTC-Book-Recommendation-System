# 📚 GTC Book Recommendation System

## 🎯 Project Objectives
The purpose of this project is to build a **Book Recommendation System** using the [Book-Crossing Dataset].  
The workflow includes:  
- Collecting and combining raw datasets (**Books**, **Users**, **Ratings**).  
- Validating and cleaning the data to ensure consistency.  
- Filtering noisy entries (e.g., users or books with very few ratings).  
- Preparing a fine-tuned dataset suitable for training recommendation models.  
- Laying the foundation for machine learning approaches such as collaborative filtering, content-based, and hybrid methods.  

---
# Project Structure
```
GTC-Book-Recommendation-System/

📦App
 ┣ 📂data
 ┃ ┗ 📜cleaned_book_ratings_plus.csv
 ┣ 📂models
 ┃ ┗ 📜knn_model.pkl
 ┣ 📂utils
 ┃ ┗ 📜helper.py
 ┣ 📜app.py
 ┗ 📜requirements.txt
📦data
 ┣ 📜BX-Book-Ratings.csv
 ┣ 📜BX-Books.csv
 ┣ 📜BX-Users.csv
 ┣ 📜cleaned_book_ratings.csv
 ┗ 📜cleaned_book_ratings_plus.csv
📦Notebooks
 ┣ 📂histograms
 ┃ ┣ 📜book_rating_histogram.png
 ┃ ┣ 📜num_of_rating_histogram.png
 ┃ ┣ 📜user_age_histogram.png
 ┃ ┣ 📜user_id_histogram.png
 ┃ ┗ 📜year_histogram.png
 ┣ 📜01_data_preparation_and_validation.ipynb
 ┣ 📜02_EDA.ipynb
 ┣ 📜03_feature_engineering.ipynb
 ┣ 📜04_model_training_and_validation.ipynb
 ┗ 📜05_knn_model.ipynb
```
# Wep App 
 - https://gtc-book-recommendation-system.streamlit.app/
```
## ⚙️ Installation & Requirements
1. Clone this repository:
   ```bash
   git clone https://github.com/AhmedMagdi2003/GTC-Book-Recommendation-System.git
   cd GTC-Book-Recommendation-System
2. Install requirements.txt
```bash
pip install -r requirements.txt
```
