import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Page configuration
st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Financial News Sentiment Analysis")

st.write(
    "Enter a financial news headline and get sentiment prediction using NLP + Linear SVM."
)

# Input box
news_input = st.text_area(
    "Enter Financial News",
    height=150,
    placeholder="Example: Tesla stock surges after strong quarterly earnings"
)

# Predict button
if st.button("Predict Sentiment"):

    if news_input.strip() == "":
        st.warning("Please enter financial news.")

    else:

        # Send request to FastAPI
        response = requests.post(
            API_URL,
            json={"news": news_input}
        )

        # Handle response
        if response.status_code == 200:

            result = response.json()

            sentiment = result["sentiment"]

            # Display based on sentiment
            if sentiment == "Positive":
                st.success(f"📈 Sentiment: {sentiment}")

            elif sentiment == "Negative":
                st.error(f"📉 Sentiment: {sentiment}")

            else:
                st.info(f"😐 Sentiment: {sentiment}")

        else:
            st.error("Error connecting to FastAPI backend.")

# Footer
st.markdown("---")
st.markdown("Built using Streamlit + FastAPI + NLP + Linear SVM")
