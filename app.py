"""
Streamlit UI for the Financial Sentiment Pipeline.

This is a standalone app: it loads the SVM and BiLSTM models directly
in-process, so it does NOT need the FastAPI backend (main.py) running.
This is what makes it deployable as a free-tier live demo on Streamlit
Community Cloud, and it's also what the `streamlit` service in
docker-compose.yaml runs (with models/ mounted as a volume).

If you want a request/response microservice architecture instead (UI ->
FastAPI -> model), main.py + svm_api.py + bilstm_api.py already provide
that as separate, callable endpoints -- this file just doesn't use them.
"""

import re

import joblib
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

MAX_LENGTH = 100


@st.cache_resource
def load_models():
    svm_model = joblib.load("models/svm_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    bilstm_model = load_model("models/sentiment_model.keras")
    tokenizer = joblib.load("models/tokenizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return svm_model, tfidf, bilstm_model, tokenizer, label_encoder


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def predict_svm(text, svm_model, tfidf):
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    pred = svm_model.predict(vec)
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[pred[0]]


def predict_bilstm(text, bilstm_model, tokenizer, label_encoder):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post", truncating="post")
    probs = bilstm_model.predict(padded, verbose=0)
    label_idx = int(np.argmax(probs, axis=1)[0])
    label = label_encoder.inverse_transform([label_idx])[0]
    confidence = float(np.max(probs))
    return label, confidence


st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="📈", layout="centered")
st.title("📈 Financial News Sentiment Analysis")
st.write("Analyze financial news using SVM and BiLSTM models.")

svm_model, tfidf, bilstm_model, tokenizer, label_encoder = load_models()

model_choice = st.selectbox("Choose Model", ["SVM", "BiLSTM", "Both"])
news_input = st.text_area(
    "Enter Financial News",
    height=150,
    placeholder="Example: Tesla stock surges after strong quarterly earnings",
)

if st.button("Predict Sentiment"):
    if news_input.strip() == "":
        st.warning("Please enter financial news.")
    else:
        results = {}
        if model_choice in ("SVM", "Both"):
            results["SVM"] = {"sentiment": predict_svm(news_input, svm_model, tfidf)}
        if model_choice in ("BiLSTM", "Both"):
            label, conf = predict_bilstm(news_input, bilstm_model, tokenizer, label_encoder)
            results["BiLSTM"] = {"sentiment": label, "confidence": conf}

        for model_name, result in results.items():
            sentiment = result["sentiment"]
            if sentiment.lower() == "positive":
                st.success(f"{model_name}: 📈 {sentiment}")
            elif sentiment.lower() == "negative":
                st.error(f"{model_name}: 📉 {sentiment}")
            else:
                st.info(f"{model_name}: {sentiment}")
            if "confidence" in result:
                st.write(f"{model_name} confidence: {result['confidence']*100:.2f}%")

st.markdown("---")
st.markdown("Built with Streamlit, scikit-learn (SVM) and TensorFlow/Keras (BiLSTM).")
