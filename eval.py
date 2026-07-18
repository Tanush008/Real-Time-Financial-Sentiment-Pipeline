"""
Evaluation script for the Financial Sentiment Pipeline.

Loads the same dataset used for training, rebuilds a single stratified
train/test split (fixing the earlier bug where SVM and BiLSTM were each
evaluated on a *different* split), reloads the saved models, and reports:

  - Accuracy, macro-F1, weighted-F1 for both models
  - Full sklearn classification_report (precision/recall/F1 per class)
  - Confusion matrices (saved as PNGs)
  - A markdown results table you can paste straight into the README / resume

Run from the project root, after the models/ and dataset/ folders are
populated (see README "Reproducing the models" section):

    python eval.py

Outputs go to results/:
    results/eval_report.md
    results/confusion_matrix_svm.png
    results/confusion_matrix_bilstm.png
"""

import json
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_LENGTH = 100  # must match bilstm_api.py

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def load_data():
    data = pd.read_csv("dataset/all-data.csv", encoding="latin-1", header=None)
    data.columns = ["sentimental", "news"]
    data.drop_duplicates(inplace=True)
    data["cleaned_news"] = data["news"].apply(preprocess_text)

    label_encoder = LabelEncoder()
    data["label"] = label_encoder.fit_transform(data["sentimental"])

    X = data["cleaned_news"]
    y = data["label"]

    # Single stratified split shared by BOTH models, so the comparison is fair.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test, label_encoder


def evaluate_svm(X_test, y_test, label_names):
    model = joblib.load("models/svm_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    start = time.time()
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    latency_ms = (time.time() - start) / len(X_test) * 1000

    return finish_eval("SVM", y_test, y_pred, label_names, latency_ms)


def evaluate_bilstm(X_test, y_test, shared_label_encoder):
    model = load_model("models/sentiment_model.keras")
    tokenizer = joblib.load("models/tokenizer.pkl")
    # The saved BiLSTM label encoder may index classes in a different order
    # than the shared one used for the test split, so convert everything
    # to string labels first, then re-encode with the shared encoder.
    bilstm_label_encoder = joblib.load("models/label_encoder.pkl")

    seqs = tokenizer.texts_to_sequences(X_test)
    padded = pad_sequences(seqs, maxlen=MAX_LENGTH, padding="post", truncating="post")

    start = time.time()
    probs = model.predict(padded, verbose=0)
    latency_ms = (time.time() - start) / len(X_test) * 1000

    pred_idx = np.argmax(probs, axis=1)
    y_pred_str = bilstm_label_encoder.inverse_transform(pred_idx)
    y_pred = shared_label_encoder.transform(y_pred_str)

    return finish_eval("BiLSTM", y_test, y_pred, shared_label_encoder, latency_ms)


def finish_eval(name, y_test, y_pred, label_encoder, latency_ms):
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, digits=3
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print(f"Avg latency:  {latency_ms:.2f} ms/sample")
    print(report)

    plot_confusion_matrix(cm, label_encoder.classes_, name)

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "latency_ms": round(latency_ms, 2),
        "report": report,
    }


def plot_confusion_matrix(cm, class_names, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = RESULTS_DIR / f"confusion_matrix_{name.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def write_markdown_report(results):
    lines = [
        "# Evaluation Results\n",
        f"Test set: {TEST_SIZE:.0%} stratified split, random_state={RANDOM_STATE} "
        "(identical split used for both models).\n",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Avg Latency (ms/sample) |",
        "|-------|----------|----------|-------------|--------------------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} "
            f"| {r['weighted_f1']:.3f} | {r['latency_ms']:.2f} |"
        )
    lines.append("\n## Per-class report\n")
    for r in results:
        lines.append(f"### {r['model']}\n```\n{r['report']}\n```\n")

    out_path = RESULTS_DIR / "eval_report.md"
    out_path.write_text("\n".join(lines))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    X_test, y_test, label_encoder = load_data()
    results = [
        evaluate_svm(X_test, y_test, label_encoder),
        evaluate_bilstm(X_test, y_test, label_encoder),
    ]
    write_markdown_report(results)
    (RESULTS_DIR / "eval_results.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "report"} for r in results], indent=2)
    )
