# Game Recommendation Engine using NLP
This project analyzes Steam user reviews to classify sentiment and recommend similar games using natural language processing and machine learning.

## Project Highlights

- **Applied NLP techniques** to train a LinearSVC classifier using **TF-IDF** on 1.5M+ Steam reviews with ~84.5% accuracy.
- Built a **content-based recommendation system** using **cosine similarity** over TF-IDF vectors from positive reviews.
- Processed and balanced a **6.4M+ review dataset** via cleaning, text normalization, and class distribution handling.
- Tested multiple **ML models**; selected and saved best-performing SVM using **evaluation metrics** and **cross-validation**.

## Technologies Used
- Python
- Scikit-learn
- Pandas / NumPy
- Jupyter Notebook
- TF-IDF Vectorization
- Cosine Similarity

## Files in This Repository
- `game_recommender.ipynb` — Complete project notebook
- `linear_svc_model.pkl` — Saved sentiment classification model
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer
- `requirements_game-recommendation-engine.txt` — Required libraries (custom environment)

## How to Run
1. Clone the repo  
2. Install dependencies from `requirements_game-recommendation-engine.txt`  
3. Open `game_recommender.ipynb` in Jupyter or VS Code  
4. Run cells to classify sentiment and get game recommendations

## Dataset Info

- Source: Steam Reviews (Kaggle)
- Raw Size: 6.4M+ rows
- Used: Filtered 1.5M reviews for training (Balanced dataset)
