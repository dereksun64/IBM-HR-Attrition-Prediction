"""
Missing Value Check for IBM HR Employee Attrition Dataset
Author: Student
Purpose: Check whether the dataset contains missing values before EDA/modeling.
"""

import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"


# Change this filename if your CSV has a different name.
CSV_FILE = str(DATA_DIR / str(DATA_DIR / "WA_FnUseC_HREmployeeAttrition.csv"))


def main():
    # 1. Load dataset
    df = pd.read_csv(CSV_FILE)

    # 2. Basic dataset information
    print("Dataset shape:", df.shape)
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    # 3. Check missing values for each column
    missing_values = df.isnull().sum()
    missing_rate = (missing_values / len(df) * 100).round(4)

    missing_report = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values,
        "Missing Rate (%)": missing_rate.values
    })

    print("\nMissing value report:")
    print(missing_report.to_string(index=False))

    # 4. Total missing values
    total_missing = missing_values.sum()
    print("\nTotal Missing Values:", total_missing)

    # 5. Save report to CSV
    missing_report.to_csv(str(REPORTS_DIR / "missing_value_summary.csv"), index=False)
    print("\nSaved missing value summary to: missing_value_summary.csv")

    # 6. Final conclusion
    if total_missing == 0:
        print("\nConclusion: No missing values were found. No imputation is required.")
    else:
        print("\nConclusion: Missing values were found. Data cleaning or imputation is required.")


if __name__ == "__main__":
    main()
