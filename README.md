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
│
├── data/                                 
│   ├── BX-Books.csv                      # Raw book metadata
│   ├── BX-Users.csv                      # Raw user metadata
│   ├── BX-Ratings.csv                    # Raw user-book ratings
│   ├── cleaned_book_rating.csv           # Filtered and cleaned dataset
|   ├── cleaned_book_rating_plus.csv      # Filtered and cleaned dataset with new features 
├── Notebooks/
│   ├── 01_data_preparation_and_validation.ipynb
|   ├── 02_EDA.ipynb
|   ├── 03_feature_engineering.ipynb
|   ├── 04_model_training_and_validation.ipynb
|   ├── 05_knn_model.ipynb
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation

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
