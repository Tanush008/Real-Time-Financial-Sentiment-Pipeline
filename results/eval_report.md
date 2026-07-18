# Evaluation Results

Test set: 20% stratified split, random_state=42 (identical split used for both models).

| Model | Accuracy | Macro F1 | Weighted F1 | Avg Latency (ms/sample) |
|-------|----------|----------|-------------|--------------------------|
| FinBERT (zero-shot) | 0.771 | 0.718 | 0.763 | 86.26 |
| SVM | 0.945 | 0.936 | 0.945 | 0.44 |
| BiLSTM | 0.714 | 0.659 | 0.709 | 27.49 |

## Per-class report

### SVM
```
              precision    recall  f1-score   support

    negative      0.964     0.893     0.927       121
     neutral      0.941     0.979     0.960       575
    positive      0.946     0.897     0.921       272

    accuracy                          0.945       968
   macro avg      0.950     0.923     0.936       968
weighted avg      0.946     0.945     0.945       968

```

### BiLSTM
```
              precision    recall  f1-score   support

    negative      0.580     0.661     0.618       121
     neutral      0.772     0.823     0.796       575
    positive      0.636     0.507     0.564       272

    accuracy                          0.714       968
   macro avg      0.662     0.664     0.659       968
weighted avg      0.710     0.714     0.709       968

```

### FinBERT (zero-shot)
```
              precision    recall  f1-score   support

    negative      0.664     0.686     0.675       121
     neutral      0.789     0.896     0.839       575
    positive      0.779     0.544     0.641       272

    accuracy                          0.771       968
   macro avg      0.744     0.709     0.718       968
weighted avg      0.770     0.771     0.763       968

```
