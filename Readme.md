# 📰 Fake News Detector

A machine learning web app built with **Streamlit** that classifies news articles as **Fake** or **Real** using **NLP preprocessing, TF-IDF features, and Logistic Regression**.  
Users can paste any news article text and instantly get a prediction with confidence score.

---

## 🚀 Features
- Preprocessing: stopword removal, lemmatization, text cleaning  
- Feature extraction: **TF-IDF (5000 features)**  
- Classification: **Logistic Regression** (99% accuracy on test data)  
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- Visualizations: Word clouds, model performance graph  
- **Streamlit app** for interactive testing  

---

## 📂 Files in Repo
- `app.py` → Streamlit web app  
- `requirements.txt` → dependencies list  
- `fake_news_model.pkl` → trained ML model  
- `tfidf_vectorizer.pkl` → TF-IDF vectorizer  
- `README.md` → project documentation  

---

## ▶️ Run Locally
Clone the repo and install requirements:
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
