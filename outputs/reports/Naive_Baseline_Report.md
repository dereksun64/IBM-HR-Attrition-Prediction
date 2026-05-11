
# Naive Baseline Classifier Report

## Objective

Train a naive classifier that always predicts employees will stay at the company.

This serves as the lower-bound baseline for the HR attrition prediction task.

---

## Baseline Strategy

Prediction rule:

- Always predict:
  - Attrition = No
  - Employee stays

The model never predicts employee resignation.

---

## Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.8401 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 Score | 0.0000 |

---

## Confusion Matrix

| | Predicted Stay | Predicted Leave |
|---|---|---|
| Actual Stay | 247 | 0 |
| Actual Leave | 47 | 0 |

---

## Interpretation

The classifier achieves relatively high accuracy because most employees stay.

However:

- Precision = 0 because the model never predicts attrition
- Recall = 0 because every employee who leaves is missed
- F1 Score = 0 because there are no positive predictions

This demonstrates why accuracy alone is misleading for imbalanced datasets.

---

## Conclusion

This naive baseline establishes the minimum acceptable performance level.

All future machine learning models should outperform this baseline, especially on Recall and F1 Score.

---

## Included Files

- naive_baseline.py
- naive_baseline_metrics.csv
- confusion_matrix.csv
- classification_report.txt
