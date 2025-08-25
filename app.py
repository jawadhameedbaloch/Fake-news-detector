import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Paste a news article below to check if it's **Fake** or **Real**.")

# User input
user_input = st.text_area("Enter news article text here:")

if st.button("Check"):
    if user_input.strip():
        # Transform input
        vectorized = tfidf.transform([user_input])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0].max()

        # Display result
        if prediction == "real":
            st.success(f"✅ Real News (Confidence: {prob*100:.2f}%)")
        else:
            st.error(f"❌ Fake News (Confidence: {prob*100:.2f}%)")
    else:
        st.warning("Please enter some text to analyze.")
