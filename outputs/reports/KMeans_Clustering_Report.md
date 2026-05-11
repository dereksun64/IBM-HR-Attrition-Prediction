# K-Means Employee Clustering Report

## 1. Objective

The goal is to group employees using K-Means clustering and then report the attrition rate within each cluster.

This is an unsupervised learning task because the clustering model does not use the `Attrition` label while fitting the clusters. Attrition is used only after clustering to interpret each group.

## 2. Why K-Means?

K-Means is a good baseline method here because:

1. The goal is employee segmentation, not direct supervised prediction.
2. After encoding categorical variables and scaling numeric variables, every employee becomes a numeric feature vector.
3. K-Means is efficient and interpretable.
4. The method creates compact groups based on feature similarity.
5. Cluster-level attrition rates can reveal which employee segments have higher turnover risk.

Because K-Means is distance-based, all features were standardized before clustering.

## 3. Silhouette Score Definition

Silhouette Score measures how well each point fits into its assigned cluster compared with the nearest alternative cluster.

For one data point:

s = (b - a) / max(a, b)

where:

- a = average distance to points in the same cluster
- b = average distance to points in the nearest different cluster

Interpretation:

- Close to 1: strong cluster assignment
- Close to 0: overlapping clusters
- Below 0: likely wrong cluster assignment

## 4. Best k Selection

K-Means was tested for k = 2 through k = 10.

The best k by Silhouette Score is:

**k = 2**

Best Silhouette Score:

**0.1202**

The Elbow plot was used to inspect inertia reduction, while the Silhouette plot was used to measure cluster separation quality.

## 5. PCA Visualization

PCA was used only for visualization. The K-Means model was trained on the full scaled feature set.

Explained variance:

- PC1: 15.62%
- PC2: 6.32%

## 6. Attrition Rate Per Cluster

Overall attrition rate: **16.12%**

 Cluster  employee_count  attrition_yes_count  attrition_no_count  attrition_rate  attrition_rate_percent
       0            1005                  193                 812        0.192040               19.203980
       1             465                   44                 421        0.094624                9.462366

## 7. Interpretation

The cluster with the highest attrition rate should be treated as the highest-risk employee segment. This cluster can be analyzed further by comparing feature averages such as overtime, job satisfaction, monthly income, job level, age, years at company, and work-life balance.

## 8. Included Files

- `kmeans_clustering_pipeline.py`
- `kmeans_model_selection_results.csv`
- `cluster_attrition_summary.csv`
- `employee_clusters.csv`
- `pca_cluster_coordinates.csv`
- `figures/elbow_plot.png`
- `figures/silhouette_plot.png`
- `figures/pca_cluster_plot.png`