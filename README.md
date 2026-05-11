# IBM HR Attrition Prediction

**Predicting Employee Attrition with Machine Learning: Clustering, Classification, and Explainability**

A complete machine-learning pipeline applied to the IBM HR Analytics dataset. This project spans EDA, preprocessing, K-Means clustering, multi-round supervised classification, hyperparameter tuning, stacking, and SHAP explainability — achieving **F1 = 0.5248** and **ROC-AUC = 0.8132**, with OverTime, YearsWithCurrManager, and EnvironmentSatisfaction identified as the top attrition drivers.

**Authors:** Derek Sun · Wenhai Ma · Jolin Chen  
**Course:** CS439  
**Paper:** [paper/main.pdf](paper/main.pdf)  
**Repository:** https://github.com/dereksun64/IBM-HR-Attrition-Prediction

---

## Dataset

| Property | Value |
|---|---|
| Source | [IBM HR Analytics Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Rows | 1,470 employees |
| Features | 35 |
| Target | `Attrition` (Yes / No) |
| Attrition Rate | 16.12% (class imbalance) |

> **Note:** If the CSV exceeds GitHub's file-size limit, download it from Kaggle and place it at `data/WA_FnUseC_HREmployeeAttrition.csv`.

---

## Project Structure

```
hr_attrition_project/
│
├── data/
│   └── WA_FnUseC_HREmployeeAttrition.csv      ← Raw IBM HR dataset (1,470 × 35)
│
├── preprocessing/
│   ├── 01_check_missing_values.py              ← Validates dataset completeness
│   └── 02_eda_analysis.py                      ← Correlation, class imbalance, summary stats
│
├── modeling/
│   ├── 03_hr_preprocessing_pipeline.py         ← Encode, scale, SMOTE, train/test split
│   ├── 04_kmeans_clustering_pipeline.py        ← K-Means segmentation (k=2)
│   ├── 05_naive_baseline.py                    ← Always-predict-majority baseline
│   ├── 06_classification_round1.py             ← LR, RF, XGB with SMOTE
│   ├── 07_classification_round2.py             ← LR, RF, XGB with class weights
│   ├── 08_hyperparameter_tuning.py             ← GridSearchCV on RF and XGB
│   ├── 09_stacking_ensemble.py                 ← Meta-learner stacking + final summary
│   ├── 10_error_analysis.py                    ← False-negative profiling
│   └── 11_explainability.py                    ← SHAP values and waterfall plots
│
├── outputs/
│   ├── HR_Processed_Dataset_All_In_One.xlsx    ← Preprocessed dataset (all splits)
│   ├── figures/                                ← All generated plots (16 PNGs)
│   │   ├── correlation_heatmap.png
│   │   ├── class_imbalance_plot.png
│   │   ├── elbow_plot.png
│   │   ├── silhouette_plot.png
│   │   ├── pca_cluster_plot.png
│   │   ├── confusion_matrices_round1.png
│   │   ├── roc_curves_round1.png
│   │   ├── confusion_matrices_round2.png
│   │   ├── roc_curves_round2.png
│   │   ├── tuning_comparison.png
│   │   ├── final_comparison.png
│   │   ├── false_negative_profile.png
│   │   ├── shap_summary_plot.png
│   │   ├── shap_feature_importance.png
│   │   ├── shap_waterfall_true_positive.png
│   │   └── shap_waterfall_false_negative.png
│   └── reports/                                ← All CSV/TXT result files
│       ├── missing_values_report.csv
│       ├── missing_value_summary.csv
│       ├── correlation_matrix.csv
│       ├── high_correlation_pairs.csv
│       ├── class_imbalance_report.csv
│       ├── summary_statistics.csv
│       ├── kmeans_model_selection_results.csv
│       ├── cluster_attrition_summary.csv
│       ├── employee_clusters.csv
│       ├── pca_cluster_coordinates.csv
│       ├── naive_baseline_metrics.csv
│       ├── confusion_matrix.csv
│       ├── classification_results_round1.csv
│       ├── classification_results_round2.csv
│       ├── tuning_results.csv
│       ├── final_results_summary.csv
│       ├── shap_values_test.csv
│       ├── classification_report.txt
│       ├── error_analysis_findings.txt
│       ├── explainability_findings.txt
│       ├── EDA_Report.md
│       ├── KMeans_Clustering_Report.md
│       └── Naive_Baseline_Report.md
│
├── paper/
│   ├── main.tex                                ← NeurIPS-style LaTeX source
│   ├── references.bib                          ← BibTeX bibliography
│   ├── main.pdf                                ← Compiled paper
│   └── figures/                                ← Paper figure copies
│
├── README.md
├── requirements.txt
├── LICENSE
└── SUBMISSION_INSTRUCTIONS.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Optional — use a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## How to Run

Run scripts in order from the project root. Each script reads from `data/` or a prior script's output and writes to `outputs/`.

**1. `preprocessing/01_check_missing_values.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/missing_value_summary.csv`
- Validates that the dataset contains zero missing values before any modeling begins.

**2. `preprocessing/02_eda_analysis.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/` (correlation_matrix, high_correlation_pairs, class_imbalance_report, summary_statistics, missing_values_report); `outputs/figures/` (correlation_heatmap, class_imbalance_plot)
- Explores feature distributions, correlations (threshold |r| ≥ 0.70), and class imbalance.

**3. `modeling/03_hr_preprocessing_pipeline.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/HR_Processed_Dataset_All_In_One.xlsx`
- Drops four non-informative columns, encodes categoricals with LabelEncoder, scales numerics with StandardScaler, and applies SMOTE to the training split only.

**4. `modeling/04_kmeans_clustering_pipeline.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/` (kmeans_model_selection_results, cluster_attrition_summary, employee_clusters, pca_cluster_coordinates); `outputs/figures/` (elbow_plot, silhouette_plot, pca_cluster_plot)
- Fits K-Means for k = 2–10, selects k = 2 via elbow + silhouette, and visualizes clusters in PCA space.

**5. `modeling/05_naive_baseline.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/naive_baseline_metrics.csv`, `outputs/reports/confusion_matrix.csv`
- Establishes a majority-class baseline (always predicts "No Attrition") that all supervised models must beat.

**6. `modeling/06_classification_round1.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/classification_results_round1.csv`; `outputs/figures/` (confusion_matrices_round1, roc_curves_round1)
- Trains Logistic Regression, Random Forest, and XGBoost with SMOTE oversampling.

**7. `modeling/07_classification_round2.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/classification_results_round2.csv`; `outputs/figures/` (confusion_matrices_round2, roc_curves_round2)
- Retrains the same three models using class-weight balancing as an alternative to SMOTE.

**8. `modeling/08_hyperparameter_tuning.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/tuning_results.csv`; `outputs/figures/tuning_comparison.png`
- Runs GridSearchCV on Random Forest and XGBoost; selects best hyperparameters by F1.

**9. `modeling/09_stacking_ensemble.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/final_results_summary.csv`; `outputs/figures/final_comparison.png`
- Builds a stacking ensemble (LR + RF + XGB base learners, LR meta-learner) and compiles results across all models.

**10. `modeling/10_error_analysis.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/error_analysis_findings.txt`; `outputs/figures/false_negative_profile.png`
- Profiles false negatives — employees who left but were predicted to stay — highlighting diverging features.

**11. `modeling/11_explainability.py`**
- Reads: `data/WA_FnUseC_HREmployeeAttrition.csv`
- Writes: `outputs/reports/` (shap_values_test, explainability_findings); `outputs/figures/` (shap_summary_plot, shap_feature_importance, shap_waterfall_true_positive, shap_waterfall_false_negative)
- Applies SHAP LinearExplainer to the best Logistic Regression model and identifies top attrition drivers.

---

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Baseline | 0.8401 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| LR Round 1 (SMOTE) | 0.7721 | 0.3936 | **0.7872** | **0.5248** | 0.8062 |
| RF Round 1 (SMOTE) | 0.8537 | 0.5909 | 0.2766 | 0.3768 | 0.7979 |
| XGB Round 1 (SMOTE) | 0.8639 | **0.6296** | 0.3617 | 0.4595 | 0.7928 |
| RF Tuned | 0.8469 | 0.5455 | 0.2553 | 0.3478 | 0.7836 |
| XGB Tuned | 0.8265 | 0.4643 | 0.5532 | 0.5049 | **0.8132** |
| Stacking | **0.8639** | 0.4643 | 0.2766 | 0.3467 | 0.8030 |

**Top SHAP drivers:** OverTime · YearsWithCurrManager · EnvironmentSatisfaction · NumCompaniesWorked · TotalWorkingYears

---

## Paper

The full write-up is at [`paper/main.pdf`](paper/main.pdf).

Full codebase: https://github.com/dereksun64/IBM-HR-Attrition-Prediction

---

## Reproducibility

- All random seeds fixed at **42** throughout (`random_state=42`)
- SMOTE applied **only** to training data — never to the test set
- Running scripts 01–11 in order fully regenerates all outputs from the raw CSV

---

## License

MIT — see [LICENSE](LICENSE).
