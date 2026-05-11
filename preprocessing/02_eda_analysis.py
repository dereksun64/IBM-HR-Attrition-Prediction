
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"


df = pd.read_csv(str(DATA_DIR / str(DATA_DIR / "WA_FnUseC_HREmployeeAttrition.csv")))

missing_values = df.isnull().sum()

missing_df = pd.DataFrame({
    "column": missing_values.index,
    "missing_values": missing_values.values
})

missing_df.to_csv(str(REPORTS_DIR / "missing_values_report.csv"), index=False)

numeric_df = df.select_dtypes(include=[np.number])

corr_matrix = numeric_df.corr()

corr_matrix.to_csv(str(REPORTS_DIR / "correlation_matrix.csv"))

threshold = 0.7

high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]

        if abs(value) >= threshold:
            high_corr_pairs.append({
                "Feature_1": corr_matrix.columns[i],
                "Feature_2": corr_matrix.columns[j],
                "Correlation": value
            })

high_corr_df = pd.DataFrame(high_corr_pairs)

high_corr_df.to_csv(
    str(REPORTS_DIR / "high_correlation_pairs.csv"),
    index=False
)

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar()
plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=90
)
plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns
)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "correlation_heatmap.png"), dpi=200)
plt.close()

attrition_counts = df["Attrition"].value_counts()

imbalance_df = pd.DataFrame({
    "Class": attrition_counts.index,
    "Count": attrition_counts.values,
    "Percentage": (
        attrition_counts.values / len(df)
    ) * 100
})

imbalance_df.to_csv(
    str(REPORTS_DIR / "class_imbalance_report.csv"),
    index=False
)

plt.figure(figsize=(6, 5))
plt.bar(
    attrition_counts.index,
    attrition_counts.values
)

plt.xlabel("Attrition")
plt.ylabel("Employee Count")
plt.title("Class Imbalance in Attrition Dataset")
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "class_imbalance_plot.png"), dpi=200)
plt.close()

summary_stats = df.describe(include="all")
summary_stats.to_csv(str(REPORTS_DIR / "summary_statistics.csv"))

print("EDA completed successfully.")
