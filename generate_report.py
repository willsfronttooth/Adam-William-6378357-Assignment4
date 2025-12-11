# generate_report.py
"""
Main end-to-end runner:
- Loads data
- Preprocesses (imputation, outlier detect+cap, scaling)
- Runs KMeans (k=2..8) using custom implementation (kmeans++)
- Saves elbow plot and cluster scatter (price vs units_sold)
- Computes cluster statistics and writes REPORT.md with computed values
- Trains regression models (Linear, Polynomial degree=2) to predict profit;
  computes MSE, MAE, R2; saves Actual vs Predicted and residual plots.
- Optionally converts REPORT.md to REPORT.pdf if pandoc is installed.
"""
import os
import pandas as pd
import numpy as np
from src.preprocessing import load_data, missing_value_summary, drop_or_impute, detect_outliers_iqr, cap_outliers, scale_features
from src.kmeans_custom import KMeansCustom
from src.regression_models import train_linear, train_polynomial, evaluate
from src.visualize import save_elbow, save_cluster_scatter, save_actual_vs_pred, save_residuals

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = 'results'
DATA_PATH = 'data/product_sales.csv'

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # Load
    df = load_data(DATA_PATH)
    orig_shape = df.shape

    # Missing values summary
    missing = missing_value_summary(df)

    # Impute/drop
    df = drop_or_impute(df, numeric_strategy='median', categorical_strategy='mode')
    after_impute_shape = df.shape

    # Numeric features for pipeline
    numeric_features = ['price','cost','units_sold','promotion_frequency','shelf_level']

    # Detect outliers
    outlier_mask = detect_outliers_iqr(df, numeric_features, k=1.5)
    n_outliers = outlier_mask.sum()

    # Cap outliers
    df = cap_outliers(df, numeric_features, k=1.5)

    # Scale features for K-means (z-score)
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Elbow method
    k_list = list(range(2,9))
    wcss = []
    models = {}
    X_k = df_scaled[numeric_features].values
    for k in k_list:
        km = KMeansCustom(n_clusters=k, init='kmeans++', random_state=42)
        km.fit(X_k)
        wcss.append(km.inertia_)
        models[k] = km
    elbow_path = os.path.join(RESULTS_DIR, 'elbow.png')
    save_elbow(k_list, wcss, elbow_path)

    # Choose k via elbow heuristics automatically: find the "k" where elbow elbow is strongest.
    # Here we pick the k with largest second derivative approximation or fallback to 3
    diffs = np.diff(wcss)
    second_diffs = np.diff(diffs)
    if len(second_diffs) > 0:
        chosen_k = k_list[np.argmin(second_diffs) + 1]  # heuristic
    else:
        chosen_k = 3
    # clamp
    if chosen_k not in k_list:
        chosen_k = 3

    km_final = models[chosen_k]
    labels = km_final.labels_
    df['cluster'] = labels

    # Cluster statistics
    stats = df.groupby('cluster').agg(
        n_products=('product_id','count'),
        avg_price=('price','mean'),
        avg_units_sold=('units_sold','mean'),
        avg_profit=('profit','mean'),
        avg_promotion_frequency=('promotion_frequency','mean')
    ).reset_index()

    # Save a cluster scatter plot (use raw price and units_sold for interpretability)
    # For centroids, map scaled centroids back to original space:
    centers_scaled = km_final.cluster_centers_
    # inverse transform scaled centers
    centers_unscaled = centers_scaled.copy()
    centers_unscaled[:, :len(numeric_features)] = scaler.inverse_transform(centers_scaled)
    # But kmeans was run on numeric_features only; centers_unscaled shape is (k, len(numeric_features))
    cluster_scatter_path = os.path.join(RESULTS_DIR, 'cluster_price_units.png')
    # For plotting centroids, build 2D array of price vs units_sold centers:
    centers_price_units = np.column_stack([centers_unscaled[:, numeric_features.index('price')], centers_unscaled[:, numeric_features.index('units_sold')]])
    save_cluster_scatter(df, 'price', 'units_sold', labels, centers_price_units, cluster_scatter_path)

    # Regression: predict profit
    features = ['price','cost','units_sold','promotion_frequency','shelf_level']
    X = df[features].values
    y = df['profit'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Linear
    lin_model = train_linear(X_train, y_train)
    lin_eval = evaluate(lin_model, X_test, y_test)

    # Polynomial degree=2
    poly_model, poly_tf = train_polynomial(X_train, y_train, degree=2)
    poly_eval = evaluate(poly_model, X_test, y_test, poly=poly_tf)

    # Save regression plots for best model based on MSE
    if poly_eval['mse'] < lin_eval['mse']:
        best_model_name = 'Polynomial (deg=2)'
        y_pred = poly_eval['y_pred']
    else:
        best_model_name = 'Linear'
        y_pred = lin_eval['y_pred']

    actual_vs_pred_path = os.path.join(RESULTS_DIR, 'actual_vs_predicted.png')
    save_actual_vs_pred(y_test, y_pred, actual_vs_pred_path, title=f'Actual vs Predicted ({best_model_name})')
    residuals_path = os.path.join(RESULTS_DIR, 'residuals.png')
    save_residuals(y_test, y_pred, residuals_path)

    # Write REPORT.md with computed values and embed plot filenames
    report_path = 'REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Product Performance Analysis — REPORT\n\n")
        f.write(f"Author: [Your Name]  \nDate: [2025-12-10]\n\n")
        f.write("## 1. Introduction\n\n")
        f.write("This report analyzes supermarket product sales to: (1) discover product clusters using K-means and (2) build regression models to predict product profit. The dataset provided contains 200 product records.\n\n")

        f.write("## 2. Data Overview\n\n")
        f.write(f"- Original dataset shape: {orig_shape[0]} rows, {orig_shape[1]} columns\n\n")
        f.write("### Missing values (per column)\n\n")
        f.write("```\n")
        f.write(str(missing.to_dict()))
        f.write("\n```\n\n")
        f.write(f"- After dropping rows missing product_id or product_name and imputing numeric/categorical values, dataset shape: {after_impute_shape[0]} rows, {after_impute_shape[1]} columns\n\n")
        f.write(f"- Number of rows flagged as having an outlier (IQR k=1.5): {int(n_outliers)}\n\n")

        f.write("## 3. Data Preprocessing\n\n")
        f.write("- Missing values: product_id/product_name rows were dropped. Numeric columns imputed with median; categorical with mode.\n")
        f.write("- Outliers: detected using IQR (k=1.5) and capped at IQR fences (winsorization) to preserve sample size.\n")
        f.write("- Scaling: Z-score standardization applied to numeric features for K-means.\n\n")

        f.write("## 4. K-means Clustering Analysis\n\n")
        f.write("### 4.1 Elbow Method\n\n")
        f.write(f"- WCSS for k=2..8: {dict(zip(k_list, [float(x) for x in wcss]))}\n\n")
        f.write(f"![Elbow plot]({elbow_path})\n\n")
        f.write(f"Recommended k (heuristic): {chosen_k}\n\n")

        f.write("### 4.2 Cluster statistics (for chosen k)\n\n")
        f.write(stats.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"![Cluster scatter]({cluster_scatter_path})\n\n")

        f.write("### 4.3 Interpretation & Business Insights\n\n")
        for _, row in stats.iterrows():
            cid = int(row['cluster'])
            f.write(f"- Cluster {cid} (n={int(row['n_products'])}): avg_price=${row['avg_price']:.2f}, avg_units_sold={row['avg_units_sold']:.1f}, avg_profit=${row['avg_profit']:.2f}, avg_promotion_frequency={row['avg_promotion_frequency']:.2f}\n")
            f.write("  - Suggested label & insight: [write a business insight here based on cluster characteristics]\n\n")

        f.write("## 5. Regression Analysis\n\n")
        f.write("Target: profit\n\n")
        f.write("### Models & Metrics (test set)\n\n")
        f.write("| Model | MSE | MAE | R^2 |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| Linear | {lin_eval['mse']:.3f} | {lin_eval['mae']:.3f} | {lin_eval['r2']:.3f} |\n")
        f.write(f"| Polynomial (deg=2) | {poly_eval['mse']:.3f} | {poly_eval['mae']:.3f} | {poly_eval['r2']:.3f} |\n\n")

        f.write(f"Best model by MSE: {best_model_name}\n\n")
        f.write(f"![Actual vs Predicted]({actual_vs_pred_path})\n\n")
        f.write(f"![Residuals]({residuals_path})\n\n")

        f.write("## 6. Conclusion\n\n")
        f.write("- Summarize key findings and recommended next steps.\n\n")

        f.write("## 7. Reproducibility\n\n")
        f.write("Run `python generate_report.py` to reproduce this REPORT.md and all plots. Requirements are in requirements.txt.\n")

    print("REPORT.md generated. Results saved to 'results/' folder.")
    # Optional: try to convert to PDF with pandoc
    try:
        import subprocess
        subprocess.run(['pandoc', 'REPORT.md', '-o', 'REPORT.pdf'], check=True)
        print("REPORT.pdf created (pandoc must be installed for this).")
    except Exception:
        print("Pandoc not available or conversion failed — REPORT.md remains as Markdown.")

if __name__ == '__main__':
    main()