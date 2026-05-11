
# Exploratory Data Analysis (EDA) Report

## Dataset Overview

- Rows: 1470
- Columns: 35

---

## Missing Values

Total missing values:

0

Result:

- No missing values detected
- No imputation required

---

## Correlated Features

Threshold used:

|correlation| >= 0.70

Number of highly correlated feature pairs:

7

---

## Class Imbalance

Class distribution:

Class  Count  Percentage
   No   1233   83.877551
  Yes    237   16.122449

The dataset is imbalanced because significantly more employees stay than leave.

Recommended handling strategy:

1. Stratified train/test split
2. SMOTE only on training data
3. Evaluate using:
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC

Accuracy alone is insufficient.

---

## Files Included

- missing_values_report.csv
- correlation_matrix.csv
- high_correlation_pairs.csv
- class_imbalance_report.csv
- summary_statistics.csv
- figures/correlation_heatmap.png
- figures/class_imbalance_plot.png
