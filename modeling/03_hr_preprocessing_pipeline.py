
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"


df = pd.read_csv(str(DATA_DIR / str(DATA_DIR / "WA_FnUseC_HREmployeeAttrition.csv")))

drop_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=drop_cols)

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)

X_train_balanced = pd.DataFrame(
    X_train_balanced,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

with pd.ExcelWriter(
    str(OUTPUT_DIR / "HR_Processed_Dataset_All_In_One.xlsx"),
    engine="openpyxl"
) as writer:

    X_train_balanced.to_excel(
        writer,
        sheet_name="X_train_balanced",
        index=False
    )

    pd.DataFrame({
        "Attrition": y_train_balanced
    }).to_excel(
        writer,
        sheet_name="y_train_balanced",
        index=False
    )

    X_test_scaled.to_excel(
        writer,
        sheet_name="X_test_scaled",
        index=False
    )

    pd.DataFrame({
        "Attrition": y_test
    }).to_excel(
        writer,
        sheet_name="y_test",
        index=False
    )

print("Pipeline Completed Successfully.")
