import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("model_compressed.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")
st.write("Enter a news headline or article text below, and the model will predict if it is **FAKE** or **REAL**.")

# Input box
user_input = st.text_area("Paste news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Transform text using TF-IDF vectorizer
        X = vectorizer.transform([user_input])

        # Predict
        pred = model.predict(X)[0]           # 0 or 1
        prob = model.predict_proba(X)[0]     # probability for both classes

        # Map prediction
        # IMPORTANT: When you trained, LabelEncoder made: 0 = fake, 1 = real
        if pred == 0:
            st.error(f"🟥 Prediction: **FAKE**\n\nConfidence: {prob[0]:.2f}")
        else:
            st.success(f"🟩 Prediction: **REAL**\n\nConfidence: {prob[1]:.2f}")
