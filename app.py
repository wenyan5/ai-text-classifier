import streamlit as st
import pickle

# 加载模型
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 AI Spam Classifier")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    vec = vectorizer.transform([user_input])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    if result == 1:
        st.error(f"Spam detected (confidence: {prob:.2f})")
    else:
        st.success(f"Not spam (confidence: {1-prob:.2f})")
