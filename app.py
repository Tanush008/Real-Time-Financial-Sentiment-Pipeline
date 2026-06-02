import streamlit as st
import requests

# Page Config
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="centered"
)

# Title
st.title("📈 Financial News Sentiment Analysis")

st.write(
    "Analyze financial news using SVM and BiLSTM models."
)

# Model Selection
model_choice = st.selectbox(
    "Choose Model",
    ["SVM", "BiLSTM", "Both"]
)

# News Input
news_input = st.text_area(
    "Enter Financial News",
    height=150,
    placeholder="Example: Tesla stock surges after strong quarterly earnings"
)

# Predict Button
if st.button("Predict Sentiment"):

    if news_input.strip() == "":
        st.warning("Please enter financial news.")

    else:

        # Select API endpoint
        if model_choice == "SVM":
            api_url = "http://127.0.0.1:8000/predict/svm"

        elif model_choice == "BiLSTM":
            api_url = "http://127.0.0.1:8000/predict/bilstm"

        else:
            api_url = "http://127.0.0.1:8000/predict/compare"

        try:

            response = requests.post(
                api_url,
                json={"news": news_input}
            )

            if response.status_code == 200:

                result = response.json()

                if model_choice == "Both":
                    st.subheader("Comparison Results")
                    for model_name, model_result in result.items():
                        sentiment = model_result["sentiment"]
                        confidence = model_result.get("confidence")

                        if sentiment.lower() == "positive":
                            st.success(f"{model_name}: 📈 {sentiment}")
                        elif sentiment.lower() == "negative":
                            st.error(f"{model_name}: 📉 {sentiment}")
                        else:
                            st.info(f"{model_name}: 😐 {sentiment}")

                        if confidence is not None:
                            st.write(f"{model_name} confidence: {confidence*100:.2f}%")
                else:
                    sentiment = result["sentiment"]

                    if sentiment.lower() == "positive":
                        st.success(f"📈 Sentiment: {sentiment}")

                    elif sentiment.lower() == "negative":
                        st.error(f"📉 Sentiment: {sentiment}")

                    else:
                        st.info(f"😐 Sentiment: {sentiment}")

                    # Show confidence if available (BiLSTM)
                    if "confidence" in result:
                        st.write(
                            f"Confidence: {result['confidence']*100:.2f}%"
                        )

            else:
                st.error(
                    f"Backend Error: {response.status_code}"
                )

        except Exception as e:
            st.error(f"Connection Error: {e}")

st.markdown("---")
st.markdown("Built with Streamlit + FastAPI + SVM + BiLSTM")
