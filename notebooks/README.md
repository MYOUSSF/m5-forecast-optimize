# Notebooks

Run in order. Each notebook writes outputs consumed by the next.

---

### `02_feature_engineering.ipynb`
Builds the feature matrix from the raw M5 CSVs one store at a time, keeping peak RAM under 400 MB. Writes one `features_<store>.parquet` file per store to `CACHE_DIR`. Covers lag features (7, 14, 28, 35 days), rolling mean/std, price momentum, calendar and SNAP indicators, and label-encoded IDs. Includes plots of feature distributions, rolling means vs actuals, and the sales target distribution.

### `03_forecasting.ipynb`
Trains q10, q50, and q90 XGBoost quantile models using 3-fold walk-forward cross-validation over a 28-day horizon. Streams directly from the per-store parquets — the full 57M-row dataset is never loaded into RAM at once. Logs parameters and metrics to MLflow and produces a SHAP beeswarm plot for the q50 model. Saves `forecasts.parquet` with columns `q10`, `q50`, `q90`, `q_star`, and `inventory` for use in notebook 04.

### `04_optimization.ipynb`
Runs three inventory optimization modules against the saved forecasts and prints a combined dollar-impact summary. No model training — executes in seconds.

| Module | Method | What it decides |
|---|---|---|
| **Newsvendor** | Critical ratio (CR = 0.9989) | Optimal order quantity Q* per item |
| **Budget allocation** | Greedy marginal revenue per dollar | How to split a $500K budget across stores |
| **Markdown scheduling** | Price-elasticity model (ε = −2.5) | Which items to discount and by how much |