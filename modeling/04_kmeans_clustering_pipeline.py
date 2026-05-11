import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"


df_raw = pd.read_csv(str(DATA_DIR / str(DATA_DIR / "WA_FnUseC_HREmployeeAttrition.csv")))
df = df_raw.copy()

drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df.drop(columns=["Attrition"])

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = list(range(2, 11))
inertias = []
silhouette_scores = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=5,
        max_iter=100,
        algorithm="lloyd"
    )
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

results_df = pd.DataFrame({
    "k": k_values,
    "inertia": inertias,
    "silhouette_score": silhouette_scores
})

best_k = int(results_df.loc[results_df["silhouette_score"].idxmax(), "k"])

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
    max_iter=200,
    algorithm="lloyd"
)

clusters = final_model.fit_predict(X_scaled)

clustered_df = df_raw.copy()
clustered_df["Cluster"] = clusters

cluster_summary = (
    clustered_df
    .groupby("Cluster")
    .agg(
        employee_count=("Attrition", "count"),
        attrition_yes_count=("Attrition", lambda s: (s == "Yes").sum()),
        attrition_no_count=("Attrition", lambda s: (s == "No").sum()),
        attrition_rate=("Attrition", lambda s: (s == "Yes").mean())
    )
    .reset_index()
)

cluster_summary["attrition_rate_percent"] = cluster_summary["attrition_rate"] * 100

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": clusters,
    "Attrition": df_raw["Attrition"]
})

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["inertia"], marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia / Within-Cluster Sum of Squares")
plt.title("Elbow Plot for K-Means")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "elbow_plot.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(results_df["k"], results_df["silhouette_score"], marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for K-Means")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "silhouette_plot.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 6))
for cluster_id in sorted(pca_df["Cluster"].unique()):
    subset = pca_df[pca_df["Cluster"] == cluster_id]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster_id}",
        alpha=0.75,
        s=25
    )

plt.xlabel(f"PC1 ({explained_variance[0] * 100:.2f}% variance)")
plt.ylabel(f"PC2 ({explained_variance[1] * 100:.2f}% variance)")
plt.title(f"PCA Visualization of K-Means Clusters (k={best_k})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "pca_cluster_plot.png"), dpi=200)
plt.close()

results_df.to_csv(str(REPORTS_DIR / "kmeans_model_selection_results.csv"), index=False)
cluster_summary.to_csv(str(REPORTS_DIR / "cluster_attrition_summary.csv"), index=False)
clustered_df.to_csv(str(REPORTS_DIR / "employee_clusters.csv"), index=False)
pca_df.to_csv(str(REPORTS_DIR / "pca_cluster_coordinates.csv"), index=False)

print("Best k:", best_k)
print(results_df)
print(cluster_summary)