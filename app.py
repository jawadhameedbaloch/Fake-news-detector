import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model_compressed.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Enter a news headline or article below to check if it is **Real** or **Fake**.")

# Input box
user_input = st.text_area("Paste news text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before checking.")
    else:
        # Transform input
        input_vec = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(input_vec)[0]

        # Some models output 0/1, others output "FAKE"/"REAL"
        # Normalize prediction
        if prediction in [0, "FAKE", "Fake", "fake"]:
            label = "Fake"
            color = "red"
        else:
            label = "Real"
            color = "green"

        st.markdown(f"### ✅ Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
