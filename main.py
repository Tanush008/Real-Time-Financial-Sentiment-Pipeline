from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
app = FastAPI()

model = joblib.load('svm_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class News(BaseModel):
    news: str


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word)
             for word in words if word not in stop_words]
    return ' '.join(words)


@app.post("/predict")
def predict_sentiment(data: News):
    cleaned_text = preprocess_text(data.news)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction = model.predict(text_tfidf)
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return {"sentiment": label_map[prediction[0]]}

