import joblib
from pydantic import BaseModel
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import APIRouter

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

router = APIRouter()


model = load_model('models/sentiment_model.h5')


tokenizer = joblib.load('models/tokenizer.pkl')

label_encoder = joblib.load('models/label_encoder.pkl')

MAX_LENGTH = 100


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)


class News(BaseModel):
    news: str


@router.post("/predict/bilstm")
def predict_sentiment(data: News):
    cleaned = preprocess_text(data.news)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH,
                           padding='post', truncating='post')
    probs = model.predict(padded)
    label_idx = int(np.argmax(probs, axis=1)[0])
    label = label_encoder.inverse_transform([label_idx])[0]
    confidence = float(np.max(probs))

    return {
        "sentiment":  label,
        "confidence": round(confidence, 4)
    }
