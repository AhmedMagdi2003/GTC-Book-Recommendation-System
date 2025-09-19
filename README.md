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
├── data_collecting_prepare/              # Data preparation & validation
│   ├── BX-Books.csv                      # Raw book metadata
│   ├── BX-Users.csv                      # Raw user metadata
│   ├── BX-Ratings.csv                    # Raw user-book ratings
│   ├── cleaned_book_rating.csv           # Filtered and cleaned dataset
│   └── 01_data_preparation_and_validation.ipynb  # Notebook for merging, cleaning & validation
│
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation

```
## ⚙️ Installation & Requirements
1. Clone this repository:
   ```bash
   git clone https://github.com/AhmedMagid2003/GTC-Book-Recommendation-System.git
   cd GTC-Book-Recommendation-System
2. Install requirements.txt
```bash
pip install -r requirements.txt
```
