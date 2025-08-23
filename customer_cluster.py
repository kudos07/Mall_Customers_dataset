#!/usr/bin/env python3
"""
Customer Clustering / Segmentation
----------------------------------
- Loads a CSV
- Uses selected features (or infers numeric features if not provided)
- Preprocess: One-Hot Encode categoricals, scale numerics
- Clustering methods: KMeans (default) or GaussianMixture (GMM)
- K selection: fixed integer or 'auto' using silhouette across a range
- Diagnostics: elbow (inertia/BIC), silhouette vs K, PCA 2D scatter
- Outputs:
    - cluster_assignments.csv
    - cluster_profiles.csv
    - elbow.png
    - silhouette.png
    - pca_scatter.png
    - clustering_report.md

Examples (PowerShell one-liners):
  python customer_clustering.py --input Mall_Customers.csv --features "Age,Annual Income (k$),Spending Score (1-100),Gender" --method kmeans --k auto --outdir clustering_outputs
  python customer_clustering.py --input Mall_Customers.csv --features "Age,Annual Income (k$),Spending Score (1-100)" --method gmm --k 5 --outdir clustering_outputs
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def parse_args():
    ap = argparse.ArgumentParser(description="Customer clustering / segmentation.")
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--features", default="", help="Comma-separated feature columns. If empty, infer numeric features.")
    ap.add_argument("--id_col", default="", help="Optional ID column to include in assignments.")
    ap.add_argument("--method", default="kmeans", choices=["kmeans","gmm"], help="Clustering method.")
    ap.add_argument("--k", default="auto", help="Number of clusters (int) or 'auto'.")
    ap.add_argument("--k_min", type=int, default=2, help="Min K when k='auto'.")
    ap.add_argument("--k_max", type=int, default=10, help="Max K when k='auto'.")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed.")
    ap.add_argument("--outdir", default="clustering_outputs", help="Directory to save outputs.")
    return ap.parse_args()

def infer_features(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    # choose numeric columns with some variation
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    keep = []
    for c in num_cols:
        if df[c].nunique(dropna=True) > 5:
            keep.append(c)
    # If nothing, fallback to all numeric
    if not keep:
        keep = num_cols
    return keep

def build_preprocessor(df: pd.DataFrame, features):
    cat_cols = [c for c in features if df[c].dtype == "O"]
    num_cols = [c for c in features if c not in cat_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    return pre, num_cols, cat_cols

def kmeans_fit_predict(X, k, random_state):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    return labels, inertia, km

def gmm_fit_predict(X, k, random_state):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
    labels = gmm.fit_predict(X)
    bic = gmm.bic(X)
    return labels, bic, gmm

def auto_select_k(X, method, k_min, k_max, random_state):
    elbow_vals = []
    sil_scores = []
    best_k, best_sil = None, -1.0
    for k in range(k_min, k_max+1):
        if method == "kmeans":
            labels, inertia, _ = kmeans_fit_predict(X, k, random_state)
            elbow_vals.append((k, inertia))
        else:
            labels, bic, _ = gmm_fit_predict(X, k, random_state)
            elbow_vals.append((k, bic))
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = -1.0
        sil_scores.append((k, sil))
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k, best_sil, elbow_vals, sil_scores

def plot_series(pairs, xlabel, ylabel, title, out_path):
    xs = [a for a,_ in pairs]
    ys = [b for _,b in pairs]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_pca_scatter(X, labels, out_path):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=labels, s=22)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters (PCA 2D)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def make_profiles(original_df, labels, features):
    df = original_df[features].copy()
    df["_cluster"] = labels
    rows = []
    for k, sub in df.groupby("_cluster"):
        row = {"cluster": int(k), "n": int(len(sub))}
        for col in features:
            if sub[col].dtype == "O":
                top = sub[col].value_counts(normalize=True).head(3)
                row[col] = "; ".join([f"{idx}:{pct:.2f}" for idx,pct in top.items()])
            else:
                row[col+"_mean"] = float(np.nanmean(sub[col]))
                row[col+"_median"] = float(np.nanmedian(sub[col]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")

def write_report(path: Path, meta: dict):
    lines = []
    lines.append("# Clustering Report\n")
    lines.append("## Setup\n")
    lines.append(f"- Input: `{meta['input']}`\n")
    lines.append(f"- Method: `{meta['method']}` | K: `{meta['k']}` (mode: {meta['k_mode']})\n")
    lines.append(f"- Features: {', '.join(meta['features'])}\n")
    lines.append("\n## Diagnostics\n")
    lines.append("- See `elbow.png` (fit vs K), `silhouette.png` (quality vs K), and `pca_scatter.png` (2D view).\n")
    lines.append("\n## Results\n")
    if meta.get('silhouette') is not None:
        lines.append(f"- Silhouette score (chosen K): {meta['silhouette']:.4f}\n")
    else:
        lines.append("- Silhouette score (chosen K): n/a (fixed K)\n")
    lines.append(f"- Cluster counts: {meta.get('counts')}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # Build feature list
    if args.features:
        features = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        features = infer_features(df, exclude=[args.id_col] if args.id_col else [])
    for c in features:
        if c not in df.columns:
            print(f"[ERROR] Feature `{c}` not found.", file=sys.stderr)
            sys.exit(1)

    pre, num_cols, cat_cols = build_preprocessor(df, features)
    X = pre.fit_transform(df[features])

    # K selection
    if args.k == "auto":
        best_k, best_sil, elbow_vals, sil_scores = auto_select_k(X, args.method, args.k_min, args.k_max, args.random_state)
        k = best_k or 3
        # plots
        ylabel = "Inertia (lower is better)" if args.method=="kmeans" else "BIC (lower is better)"
        plot_series(elbow_vals, "K", ylabel, f"{args.method.upper()} model fit", outdir / "elbow.png")
        plot_series(sil_scores, "K", "Silhouette (higher is better)", "Silhouette across K", outdir / "silhouette.png")
        chosen_sil = float(best_sil)
    else:
        k = int(args.k)
        chosen_sil = None
        # still provide a minimal elbow curve around chosen K
        elbow_vals = []
        for k2 in range(max(2, k-2), k+3):
            if args.method == "kmeans":
                _, inertia, _ = kmeans_fit_predict(X, k2, args.random_state)
                elbow_vals.append((k2, inertia))
            else:
                _, bic, _ = gmm_fit_predict(X, k2, args.random_state)
                elbow_vals.append((k2, bic))
        ylabel = "Inertia (lower is better)" if args.method=="kmeans" else "BIC (lower is better)"
        plot_series(elbow_vals, "K", ylabel, f"{args.method.upper()} model fit (local)", outdir / "elbow.png")

    # Fit final model
    if args.method == "kmeans":
        labels, _, model = kmeans_fit_predict(X, k, args.random_state)
    else:
        labels, _, model = gmm_fit_predict(X, k, args.random_state)

    # PCA scatter
    plot_pca_scatter(X, labels, outdir / "pca_scatter.png")

    # Assignments
    assign = pd.DataFrame({"_cluster": labels})
    if args.id_col and args.id_col in df.columns:
        assign[args.id_col] = df[args.id_col].values
    assign.to_csv(outdir / "cluster_assignments.csv", index=False)

    # Profiles
    profiles = make_profiles(df, labels, features)
    profiles.to_csv(outdir / "cluster_profiles.csv", index=False)

    # Report
    counts = assign["_cluster"].value_counts().sort_index().to_dict()
    meta = {
        "input": args.input,
        "method": args.method,
        "k": int(k),
        "k_mode": "auto" if args.k == "auto" else "fixed",
        "features": features,
        "silhouette": chosen_sil,
        "counts": counts
    }
    write_report(outdir / "clustering_report.md", meta)
    print("[DONE] Wrote outputs to:", outdir.resolve())

if __name__ == "__main__":
    main()
