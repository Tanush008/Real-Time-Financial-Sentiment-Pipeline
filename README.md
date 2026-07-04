# Finance Sentimental

Financial news sentiment analysis platform that compares two machine learning models, SVM and BiLSTM, to classify news headlines as Positive, Neutral, or Negative. The project includes:

- A FastAPI backend that exposes prediction endpoints
- A Streamlit frontend for interactive sentiment checks
- A Kafka producer and consumer pipeline for streaming headline analysis
- PostgreSQL storage for prediction history
- Grafana for dashboarding the stored results

## Overview

The application is built around a simple workflow:

1. A financial headline is entered in the Streamlit UI or sent directly to the API.
2. The FastAPI service preprocesses the text and runs it through two models:
   - SVM using TF-IDF features
   - BiLSTM using tokenized sequences
3. The API returns the prediction, and the Streamlit app renders the result.
4. In the streaming flow, the Kafka producer publishes headlines to the `headlines` topic.
5. The Kafka consumer reads those messages, runs both models, and stores the output in PostgreSQL.
6. Grafana can be connected to PostgreSQL to visualize trends in predictions over time.

## Project Structure

```text
.
├── app.py                  # Streamlit frontend
├── main.py                 # FastAPI app that exposes prediction endpoints
├── svm_api.py              # SVM model loading, preprocessing, and /predict/svm
├── bilstm_api.py           # BiLSTM model loading, preprocessing, and /predict/bilstm
├── kafka_producer.py       # Publishes sample headlines to Kafka
├── kafka_consumer.py       # Consumes headlines, predicts sentiment, stores results in PostgreSQL
├── docker-compose.yaml     # Full stack orchestration for app, Kafka, PostgreSQL, Streamlit, Grafana
├── Dockerfile.api          # Container image for the FastAPI backend and consumer
├── Dockerfile.streamlit    # Container image for the Streamlit UI
├── requirements.txt        # Python dependencies
├── dataset/
│   └── all-data.csv        # Training dataset used for model preparation
├── models/
│   ├── sentiment_model.h5   # Trained BiLSTM model
│   ├── svm_model.pkl        # Trained SVM model
│   ├── tfidf_vectorizer.pkl # TF-IDF vectorizer for SVM
│   ├── tokenizer.pkl       # Tokenizer for BiLSTM
│   └── label_encoder.pkl   # Label encoder used by BiLSTM
├── grafana/
│   └── dashboard.json       # Grafana dashboard definition
└── notebooks/
    └── senti.ipynb          # Notebook used for experimentation/training
```

## Components

### FastAPI backend

The backend lives in [main.py](main.py) and exposes these routes:

- `GET /` - health check and endpoint list
- `POST /predict/svm` - sentiment prediction using the SVM model
- `POST /predict/bilstm` - sentiment prediction using the BiLSTM model
- `POST /predict/compare` - runs both models and returns both results

Each prediction request expects JSON like:

```json
{
  "news": "Tesla stock rises after stronger-than-expected earnings"
}
```

### Streamlit UI

The UI in [app.py](app.py) provides a simple interface where users can:

- choose SVM, BiLSTM, or Both
- enter a financial headline
- view sentiment output and confidence when available

### Kafka pipeline

- [kafka_producer.py](kafka_producer.py) publishes news headlines to the `headlines` topic.
- [kafka_consumer.py](kafka_consumer.py) consumes those messages, predicts sentiment with both models, and writes the results to PostgreSQL.

The consumer also creates the `predictions` table automatically if it does not already exist.

### PostgreSQL

The database stores a history of predictions with:

- headline
- stock symbol
- SVM prediction
- BiLSTM prediction
- confidence
- agreement flag
- timestamp

### Grafana

Grafana is included in the compose stack so you can build dashboards on top of the PostgreSQL prediction table.

## Requirements

- Python 3.10+ recommended
- Docker and Docker Compose
- Optional for local-only runs: a running Kafka broker and PostgreSQL instance

## Recommended Way to Run

This project is designed to run with Docker Compose because the API, consumer, Kafka, and PostgreSQL are wired together using container service names.

### 1. Build and start the full stack

```bash
docker compose up --build
```

This starts:

- Zookeeper on `localhost:2181`
- Kafka on `localhost:9092`
- PostgreSQL on `localhost:5432`
- FastAPI on `localhost:8000`
- Kafka consumer
- Streamlit on `localhost:8501`
- Grafana on `localhost:3000`

### 2. Open the services

- FastAPI health check: `http://127.0.0.1:8000/`
- Streamlit app: `http://127.0.0.1:8501/`
- Grafana: `http://127.0.0.1:3000/`

Grafana default login in this compose setup:

- username: `admin`
- password: `admin`

## Running Individual Parts

### FastAPI backend only

If you want to run only the API locally:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Streamlit UI only

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Kafka producer test

Run this to publish sample headlines to Kafka:

```bash
python kafka_producer.py
```

### Kafka consumer

The consumer expects:

- Kafka at `kafka:9092`
- PostgreSQL at `postgres:5432`

This is why the compose stack is the best way to run it. If you run it outside Docker, you will need to update the connection settings in [kafka_consumer.py](kafka_consumer.py) first.

## API Examples

### SVM prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/svm \
  -H "Content-Type: application/json" \
  -d '{"news":"Markets rally after strong earnings"}'
```

### BiLSTM prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/bilstm \
  -H "Content-Type: application/json" \
  -d '{"news":"Company shares fall after weak guidance"}'
```

### Compare both models

```bash
curl -X POST http://127.0.0.1:8000/predict/compare \
  -H "Content-Type: application/json" \
  -d '{"news":"Tesla stock rises on record deliveries"}'
```

## Model Notes

- The SVM model uses text cleaning, stopword removal, lemmatization, and TF-IDF features.
- The BiLSTM model uses the same preprocessing, then tokenizes and pads the sequence to a maximum length of 100.
- NLTK resources `stopwords` and `wordnet` are downloaded automatically in the model services and consumer.

## Troubleshooting

### 1. Model file not found

Make sure the `models/` directory contains all required artifacts:

- `svm_model.pkl`
- `tfidf_vectorizer.pkl`
- `sentiment_model.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`

### 2. Kafka connection errors

Kafka must be running before starting the producer or consumer.

### 3. PostgreSQL connection errors

The consumer expects PostgreSQL to be available at the container hostname `postgres` when using Docker Compose.

### 4. Streamlit cannot reach the API

Confirm the FastAPI service is running on `http://127.0.0.1:8000`.

## Tech Stack

- FastAPI
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- NLTK
- Kafka
- PostgreSQL
- Grafana
