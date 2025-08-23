# Clustering Report
## Setup
- Input: `Mall_Customers.csv`
- Method: `kmeans` | K: `5` (mode: fixed)
- Features: Age, Annual Income (k$), Spending Score (1-100), Gender

## Diagnostics
- See `elbow.png` (fit vs K), `silhouette.png` (quality vs K), and `pca_scatter.png` (2D view).

## Results
- Silhouette score (chosen K): n/a (fixed K)
- Cluster counts: {0: 39, 1: 58, 2: 22, 3: 34, 4: 47}
