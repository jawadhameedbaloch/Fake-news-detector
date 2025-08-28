
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load saved model & vectorizer
import joblib
model = joblib.load("model_compressed.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake News Detector")
st.write("Paste a news article below:")

user_input = st.text_area("News Article")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news article.")
    else:
        clean_text = " ".join([w for w in user_input.lower().split() if len(w) > 2])
        vect = vectorizer.transform([clean_text])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect).max()
        st.success(f"Prediction: {pred.upper()}")
        st.info(f"Confidence: {prob:.2f}")
