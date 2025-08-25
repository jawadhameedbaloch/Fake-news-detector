# 📰 Fake News Detector

A machine learning project that detects whether a news article is **Fake** or **Real** using NLP and Logistic Regression.

## 🚀 Features
- Preprocessing with stopword removal, lemmatization
- Feature extraction with TF-IDF
- Logistic Regression classifier
- Evaluation with accuracy, precision, recall, F1-score
- Streamlit app for user interaction

## 📂 Files in Repo
- `app.py` → Streamlit app
- `requirements.txt` → dependencies
- `fake_news_model.pkl` → trained ML model
- `tfidf_vectorizer.pkl` → TF-IDF vectorizer
- `README.md` → project info

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
