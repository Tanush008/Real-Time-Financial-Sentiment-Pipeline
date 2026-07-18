"""
API tests for the FastAPI backend.

These mock out the actual model/tokenizer/tfidf objects so the suite runs
fast and doesn't require the (gitignored) trained model artifacts to be
present -- useful for CI. Run with:

    pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
    """
    Builds a FastAPI TestClient with svm_api / bilstm_api's model-loading
    calls patched out, so importing main.py doesn't try to read
    models/*.pkl or models/*.h5 from disk.
    """
    fake_svm_model = MagicMock()
    fake_svm_model.predict.return_value = np.array([2])  # "Positive"

    fake_tfidf = MagicMock()
    fake_tfidf.transform.return_value = "fake_vector"

    fake_bilstm_model = MagicMock()
    fake_bilstm_model.predict.return_value = np.array([[0.05, 0.10, 0.85]])  # Positive

    fake_tokenizer = MagicMock()
    fake_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]

    fake_label_encoder = MagicMock()
    fake_label_encoder.inverse_transform.side_effect = lambda idx: np.array(["Positive"])

    with patch("joblib.load") as mock_joblib_load, \
         patch("tensorflow.keras.models.load_model", return_value=fake_bilstm_model):

        def joblib_side_effect(path):
            if "svm_model" in path:
                return fake_svm_model
            if "tfidf_vectorizer" in path:
                return fake_tfidf
            if "tokenizer" in path:
                return fake_tokenizer
            if "label_encoder" in path:
                return fake_label_encoder
            return MagicMock()

        mock_joblib_load.side_effect = joblib_side_effect

        # Force a fresh import so the patches above take effect
        for mod in ["main", "svm_api", "bilstm_api"]:
            sys.modules.pop(mod, None)

        from fastapi.testclient import TestClient
        import main

        yield TestClient(main.app)


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "/predict/svm" in body["endpoints"]
    assert "/predict/bilstm" in body["endpoints"]
    assert "/predict/compare" in body["endpoints"]


def test_predict_svm_valid_input(client):
    response = client.post("/predict/svm", json={"news": "Stocks rally on strong earnings"})
    assert response.status_code == 200
    assert response.json()["sentiment"] in {"Negative", "Neutral", "Positive"}


def test_predict_svm_missing_field(client):
    response = client.post("/predict/svm", json={})
    assert response.status_code == 422  # FastAPI validation error


def test_predict_bilstm_valid_input(client):
    response = client.post("/predict/bilstm", json={"news": "Company profits fall sharply"})
    assert response.status_code == 200
    body = response.json()
    assert "sentiment" in body
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_compare_returns_both_models(client):
    response = client.post("/predict/compare", json={"news": "Markets steady ahead of earnings"})
    assert response.status_code == 200
    body = response.json()
    assert "svm" in body
    assert "bilstm" in body
    assert "sentiment" in body["svm"]
    assert "sentiment" in body["bilstm"]


def test_predict_empty_string(client):
    # Empty text should still return a valid response, not error out
    response = client.post("/predict/svm", json={"news": ""})
    assert response.status_code == 200
