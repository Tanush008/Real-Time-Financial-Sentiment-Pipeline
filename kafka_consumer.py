from kafka import KafkaConsumer
import json
import joblib
import re
import numpy as np
import psycopg2
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Preprocessing ────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)


# ── Load SVM ─────────────────────────────────────────────────
svm_model = joblib.load('models/svm_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
svm_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ── Load BiLSTM ──────────────────────────────────────────────
lstm_model = load_model('models/sentiment_model.h5')

tokenizer = joblib.load('models/tokenizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
MAX_LENGTH = 100

# ── PostgreSQL connection ─────────────────────────────────────
conn = psycopg2.connect(
    host="postgres",
    database="sentiment_db",
    user="postgres",
    password="postgres"
)
cursor = conn.cursor()

# create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id          SERIAL PRIMARY KEY,
        headline    TEXT,
        stock       VARCHAR(20),
        svm_pred    VARCHAR(20),
        bilstm_pred VARCHAR(20),
        confidence  FLOAT,
        agree       BOOLEAN,
        event_time  TIMESTAMP,   -- when the headline was actually published (if known)
        timestamp   TIMESTAMP    -- when this pipeline processed it
    )
""")
conn.commit()

# ── Predict functions ─────────────────────────────────────────


def predict_svm(text):
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    pred = svm_model.predict(vec)
    return svm_labels[pred[0]]


def predict_bilstm(text):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH,
                           padding='post', truncating='post')
    probs = lstm_model.predict(padded, verbose=0)
    label_idx = int(np.argmax(probs, axis=1)[0])
    label = label_encoder.inverse_transform([label_idx])[0]
    confidence = float(np.max(probs))
    return label, confidence


# ── Kafka Consumer ────────────────────────────────────────────
consumer = KafkaConsumer(
    'headlines',
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='sentiment-group'
)

print("Consumer started — waiting for messages...")

for message in consumer:
    data = message.value
    headline = data.get('headline', '')
    stock = data.get('stock', 'UNKNOWN')

    # Historical replay messages carry the headline's real publish date;
    # live/sample messages won't have this, so fall back to now().
    event_time = None
    raw_event_time = data.get('published_at')
    if raw_event_time:
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                event_time = datetime.strptime(raw_event_time, fmt)
                break
            except ValueError:
                continue
    if event_time is None:
        event_time = datetime.now()

    svm_pred = predict_svm(headline)
    bilstm_pred, conf = predict_bilstm(headline)
    agree = svm_pred.lower() == bilstm_pred.lower()

    # save to PostgreSQL
    cursor.execute("""
        INSERT INTO predictions
            (headline, stock, svm_pred, bilstm_pred, confidence, agree, event_time, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (headline, stock, svm_pred, bilstm_pred, conf, agree, event_time, datetime.now()))
    conn.commit()

    print(f"[{stock}] SVM: {svm_pred} | BiLSTM: {bilstm_pred} ({conf:.2%}) | Agree: {agree}")
