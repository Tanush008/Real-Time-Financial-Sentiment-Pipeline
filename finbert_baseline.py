import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval import RESULTS_DIR, load_data, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

MODEL_NAME = "ProsusAI/finbert"

# FinBERT's label order is fixed by the model config, not our dataset.
FINBERT_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}


def run_finbert(X_test, y_test, label_encoder, batch_size=32):
    print(f"Loading {MODEL_NAME} (first run downloads ~400MB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    texts = X_test.tolist()
    preds = []

    start = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128)
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=1).tolist()
            preds.extend(batch_preds)
    latency_ms = (time.time() - start) / len(texts) * 1000

    # Map FinBERT's own label ids -> string labels -> our shared encoder's ids
    pred_str = [FINBERT_ID2LABEL[p].capitalize() for p in preds]
    # Our dataset uses lowercase class names ("positive"/"neutral"/"negative");
    # align capitalization with label_encoder.classes_
    class_lookup = {c.lower(): c for c in label_encoder.classes_}
    pred_str_aligned = [class_lookup[s.lower()] for s in pred_str]
    y_pred = label_encoder.transform(pred_str_aligned)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== FinBERT (zero-shot) ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Avg latency: {latency_ms:.2f} ms/sample")
    print(report)

    plot_confusion_matrix(cm, label_encoder.classes_, "FinBERT")

    return {
        "model": "FinBERT (zero-shot)",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "latency_ms": round(latency_ms, 2),
        "report": report,
    }


def append_to_report(result):
    report_path = RESULTS_DIR / "eval_report.md"
    json_path = RESULTS_DIR / "eval_results.json"

    existing = json.loads(json_path.read_text()) if json_path.exists() else []
    existing.append({k: v for k, v in result.items() if k != "report"})
    json_path.write_text(json.dumps(existing, indent=2))

    table_row = (
        f"| {result['model']} | {result['accuracy']:.3f} | {result['macro_f1']:.3f} "
        f"| {result['weighted_f1']:.3f} | {result['latency_ms']:.2f} |\n"
    )
    section = f"\n### {result['model']}\n```\n{result['report']}\n```\n"

    if report_path.exists():
        text = report_path.read_text()
        # insert the new row right after the table header, append section at the end
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("|-------"):
                lines.insert(i + 1, table_row.rstrip("\n"))
                break
        text = "\n".join(lines) + section
        report_path.write_text(text)
    else:
        report_path.write_text(f"# Evaluation Results\n\n{table_row}{section}")

    print(f"Updated {report_path} and {json_path}")


if __name__ == "__main__":
    X_test, y_test, label_encoder = load_data()
    result = run_finbert(X_test, y_test, label_encoder)
    append_to_report(result)
