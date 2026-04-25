"""
LightGBM Quantile Forecasting
Trains three models (q=0.1, 0.5, 0.9) using walk-forward cross-validation.
Logs all runs to MLflow and computes WRMSSE + dollar-value error cost.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import shap
import logging
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.data.features import (
    FEATURE_COLS, TARGET_COL, DATE_COL, ID_COL,
    AVG_UNIT_COST, AVG_SELL_PRICE,
    HOLDING_COST_RATE, STOCKOUT_PENALTY_RATE,
)

logger = logging.getLogger(__name__)

QUANTILES = [0.10, 0.50, 0.90]
HORIZON   = 28   # M5 forecast horizon in days

# ── LightGBM base params ─────────────────────────────────────────────────────
BASE_PARAMS = dict(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    num_leaves        = 127,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)


# ── Walk-forward split ────────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame,
                         n_splits: int = 3,
                         horizon: int = HORIZON,
                         gap: int = 0):
    """
    Yield (train_idx, val_idx) for expanding-window walk-forward CV.
    Each validation window is *horizon* days; gap avoids leakage.
    """
    dates = df[DATE_COL].sort_values().unique()
    total = len(dates)
    val_size = horizon

    for i in range(n_splits, 0, -1):
        val_end   = total - (i - 1) * val_size
        val_start = val_end - val_size
        train_end = val_start - gap

        train_dates = dates[:train_end]
        val_dates   = dates[val_start:val_end]

        train_idx = df[df[DATE_COL].isin(train_dates)].index
        val_idx   = df[df[DATE_COL].isin(val_dates)].index

        yield train_idx, val_idx


# ── WRMSSE (simplified) ───────────────────────────────────────────────────────

def wrmsse(y_true: np.ndarray, y_pred: np.ndarray,
           scale: Optional[np.ndarray] = None) -> float:
    """
    Weighted Root Mean Squared Scaled Error (simplified, equal weights).
    scale defaults to naive (lag-28) RMSE if not supplied.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if scale is None:
        scale = np.sqrt(np.mean(y_true ** 2)) + 1e-8
    return float(rmse / scale)


# ── Dollar-value error cost ────────────────────────────────────────────────────

def dollar_error_cost(y_true: np.ndarray,
                       y_pred: np.ndarray) -> dict:
    """
    Translate forecast errors into $ cost using newsvendor logic:
      - Over-forecast (pred > actual) → holding cost on excess inventory
      - Under-forecast (pred < actual) → stockout penalty on unmet demand
    Returns dict with total and decomposed costs.
    """
    over_mask  = y_pred > y_true
    under_mask = y_pred < y_true

    holding_cost_per_unit = AVG_UNIT_COST * HOLDING_COST_RATE / 365
    stockout_cost_per_unit = AVG_SELL_PRICE * STOCKOUT_PENALTY_RATE

    overstock_units  = np.sum(np.maximum(y_pred - y_true, 0)[over_mask])
    understock_units = np.sum(np.maximum(y_true - y_pred, 0)[under_mask])

    holding_cost  = overstock_units  * holding_cost_per_unit
    stockout_cost = understock_units * stockout_cost_per_unit

    return {
        "total_cost_usd"    : holding_cost + stockout_cost,
        "holding_cost_usd"  : holding_cost,
        "stockout_cost_usd" : stockout_cost,
        "overstock_units"   : overstock_units,
        "understock_units"  : understock_units,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_quantile_models(df: pd.DataFrame,
                           output_dir: str | Path = "models",
                           experiment_name: str = "m5_lgbm_quantile",
                           n_cv_splits: int = 3) -> dict[float, lgb.LGBMRegressor]:
    """
    Train one LightGBM model per quantile with walk-forward CV.
    Logs metrics, params, and models to MLflow.

    Returns dict: {quantile → fitted model}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)
    models = {}

    for q in QUANTILES:
        logger.info("Training quantile q=%.2f", q)

        params = {**BASE_PARAMS, "objective": "quantile", "alpha": q}
        cv_wrmsse, cv_dollar = [], []

        # Walk-forward CV
        for fold, (train_idx, val_idx) in enumerate(
            walk_forward_splits(df, n_splits=n_cv_splits)
        ):
            X_tr = df.loc[train_idx, FEATURE_COLS]
            y_tr = df.loc[train_idx, TARGET_COL]
            X_val = df.loc[val_idx, FEATURE_COLS]
            y_val = df.loc[val_idx, TARGET_COL].values

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )

            preds = np.maximum(model.predict(X_val), 0)  # sales ≥ 0

            fold_wrmsse = wrmsse(y_val, preds)
            fold_dollar = dollar_error_cost(y_val, preds)

            cv_wrmsse.append(fold_wrmsse)
            cv_dollar.append(fold_dollar["total_cost_usd"])
            logger.info("  Fold %d | WRMSSE=%.4f | $cost=%.0f",
                        fold, fold_wrmsse, fold_dollar["total_cost_usd"])

        # Final model on full data
        X_all = df[FEATURE_COLS]
        y_all = df[TARGET_COL]
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X_all, y_all, callbacks=[lgb.log_evaluation(-1)])

        models[q] = final_model

        # Save model in LightGBM native text format (safe, no pickle)
        model_path = output_dir / f"lgbm_q{int(q*100)}.txt"
        final_model.booster_.save_model(str(model_path))

        # MLflow logging — params/metrics + native model file as artifact
        with mlflow.start_run(run_name=f"lgbm_q{int(q*100)}"):
            mlflow.log_params({**params, "n_cv_splits": n_cv_splits,
                               "horizon": HORIZON})
            mlflow.log_metrics({
                "cv_wrmsse_mean" : float(np.mean(cv_wrmsse)),
                "cv_wrmsse_std"  : float(np.std(cv_wrmsse)),
                "cv_dollar_mean" : float(np.mean(cv_dollar)),
            })
            mlflow.log_artifact(str(model_path), artifact_path=f"model_q{int(q*100)}")

        logger.info("  ✓ q=%.2f | CV WRMSSE=%.4f ± %.4f | CV $cost=%.0f",
                    q, np.mean(cv_wrmsse), np.std(cv_wrmsse), np.mean(cv_dollar))

    return models


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_quantiles(models: dict[float, lgb.LGBMRegressor],
                       X: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference for all quantiles.
    Returns DataFrame with columns: q10, q50, q90.
    """
    preds = {}
    for q, model in models.items():
        preds[f"q{int(q*100):02d}"] = np.maximum(model.predict(X[FEATURE_COLS]), 0)
    return pd.DataFrame(preds, index=X.index)


# ── SHAP Analysis ─────────────────────────────────────────────────────────────

def compute_shap(model: lgb.LGBMRegressor,
                  X: pd.DataFrame,
                  max_display: int = 20,
                  save_path: Optional[Path] = None) -> np.ndarray:
    """
    Compute SHAP values for the median (q50) model.
    Saves beeswarm plot if save_path given.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[FEATURE_COLS])

    if save_path:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X[FEATURE_COLS],
                          max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP beeswarm saved → %s", save_path)

    return shap_values
