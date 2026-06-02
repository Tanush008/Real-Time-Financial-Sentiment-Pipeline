from fastapi import FastAPI
from pydantic import BaseModel
from svm_api import (
    router as svm_router,
    preprocess_text as svm_preprocess_text,
    model as svm_model,
    tfidf as svm_tfidf,
)
from bilstm_api import (
    router as bilstm_router,
    preprocess_text as bilstm_preprocess_text,
    model as bilstm_model,
    tokenizer,
    label_encoder,
    MAX_LENGTH,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


app = FastAPI()

app.include_router(svm_router)      # mounts /predict/svm
app.include_router(bilstm_router)   # mounts /predict/bilstm


class News(BaseModel):
    news: str


@app.post("/predict/compare")
def compare_predictions(data: News):
    svm_cleaned = svm_preprocess_text(data.news)
    svm_vector = svm_tfidf.transform([svm_cleaned])
    svm_prediction = svm_model.predict(svm_vector)

    svm_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    svm_result = {
        "sentiment": svm_label_map[svm_prediction[0]],
    }

    bilstm_cleaned = bilstm_preprocess_text(data.news)
    bilstm_sequence = tokenizer.texts_to_sequences([bilstm_cleaned])
    bilstm_padded = pad_sequences(
        bilstm_sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    bilstm_probs = bilstm_model.predict(bilstm_padded, verbose=0)
    bilstm_label_idx = int(np.argmax(bilstm_probs, axis=1)[0])
    bilstm_label = label_encoder.inverse_transform([bilstm_label_idx])[0]

    bilstm_result = {
        "sentiment": bilstm_label,
        "confidence": float(np.max(bilstm_probs)),
    }

    return {
        "svm": svm_result,
        "bilstm": bilstm_result,
    }


@app.get("/")
def health():
    return {"status": "ok", "endpoints": [
        "/predict/svm",
        "/predict/bilstm",
        "/predict/compare"
    ]}
