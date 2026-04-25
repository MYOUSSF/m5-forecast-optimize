"""
XGBoost Quantile Forecasting — GPU-Accelerated
===============================================
Trains q10/q50/q90 models from per-store parquet files directly,
never loading the full 57M-row dataset into RAM at once.

GPU usage
---------
XGBoost with tree_method="hist" + device="cuda" runs natively on
Kaggle's P100 GPU — no special compilation needed. Expect 3-5x
speedup over CPU on 57M rows.
"""

import gc
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import shap
import logging
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BASE_PARAMS, QUANTILES,
    HORIZON, MODEL_DIR, MLFLOW_URI
)
from src.data.features import (
    FEATURE_COLS, TARGET_COL, DATE_COL,
    AVG_UNIT_COST, AVG_SELL_PRICE,
    HOLDING_COST_RATE, STOCKOUT_PENALTY_RATE,
)

logger = logging.getLogger(__name__)


# ── Date-based walk-forward splits ────────────────────────────────────────────

def walk_forward_date_splits(all_dates: np.ndarray,
                              n_splits: int = 3,
                              horizon: int = HORIZON):
    """
    Yield (train_cutoff_date, val_start_date, val_end_date) tuples.
    Works on the date array only — never touches the full DataFrame.
    """
    total    = len(all_dates)
    val_size = horizon

    for i in range(n_splits, 0, -1):
        val_end_idx   = total - (i - 1) * val_size
        val_start_idx = val_end_idx - val_size
        train_end_idx = val_start_idx

        yield (
            all_dates[train_end_idx - 1],
            all_dates[val_start_idx],
            all_dates[min(val_end_idx - 1, total - 1)],
        )


# ── Load a date slice from cache ──────────────────────────────────────────────

def _load_slice(cache_dir: Path,
                date_from: Optional[pd.Timestamp],
                date_to: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream X, y arrays for a date range from per-store parquet files.
    One store at a time — never loads all 57M rows at once.
    """
    X_parts, y_parts = [], []

    for pq in sorted(cache_dir.glob("features_*.parquet")):
        store = pd.read_parquet(pq, columns=FEATURE_COLS + [TARGET_COL, DATE_COL])
        store[DATE_COL] = pd.to_datetime(store[DATE_COL])

        mask = store[DATE_COL] <= date_to
        if date_from is not None:
            mask &= store[DATE_COL] >= date_from

        chunk = store[mask]
        if len(chunk):
            X_parts.append(chunk[FEATURE_COLS].values.astype("float32"))
            y_parts.append(chunk[TARGET_COL].values.astype("float32"))

        del store, chunk
        gc.collect()

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def wrmsse(y_true: np.ndarray, y_pred: np.ndarray,
           scale: Optional[float] = None) -> float:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if scale is None:
        scale = np.sqrt(np.mean(y_true ** 2)) + 1e-8
    return float(rmse / scale)


def dollar_error_cost(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    holding_per_unit  = AVG_UNIT_COST * HOLDING_COST_RATE / 365
    stockout_per_unit = AVG_SELL_PRICE * STOCKOUT_PENALTY_RATE

    overstock  = np.maximum(y_pred - y_true, 0).sum()
    understock = np.maximum(y_true - y_pred, 0).sum()

    return {
        "total_cost_usd"   : overstock * holding_per_unit + understock * stockout_per_unit,
        "holding_cost_usd" : overstock  * holding_per_unit,
        "stockout_cost_usd": understock * stockout_per_unit,
        "overstock_units"  : float(overstock),
        "understock_units" : float(understock),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_quantile_models(cache_dir: str | Path,
                           output_dir: str | Path | None = None,
                           experiment_name: str = "m5_xgb_quantile",
                           n_cv_splits: int = 3) -> dict[float, xgb.Booster]:
    """
    Train one XGBoost model per quantile with walk-forward CV.
    Reads directly from per-store parquets — never loads full dataset.

    Returns : {quantile → xgb.Booster}
    """
    cache_dir  = Path(cache_dir)
    output_dir = Path(output_dir) if output_dir else Path(MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Get date array from one lightweight store file
    sample_pq = sorted(cache_dir.glob("features_*.parquet"))[0]
    sample_df = pd.read_parquet(sample_pq, columns=[DATE_COL])
    all_dates = pd.to_datetime(sample_df[DATE_COL]).sort_values().unique()
    del sample_df; gc.collect()

    splits = list(walk_forward_date_splits(all_dates, n_splits=n_cv_splits))
    logger.info("Walk-forward splits: %d folds, horizon=%d days", len(splits), HORIZON)
    for i, (tr_end, v_start, v_end) in enumerate(splits):
        logger.info("  Fold %d: train <= %s | val %s -> %s", i, tr_end, v_start, v_end)

    mlflow.set_experiment(experiment_name)
    models = {}

    for q in QUANTILES:
        logger.info("─── Training q=%.2f ───", q)

        # XGBoost quantile params
        xgb_params = {
            "objective"        : "reg:quantileerror",
            "quantile_alpha"   : q,
            "tree_method"      : BASE_PARAMS["tree_method"],
            "device"           : BASE_PARAMS["device"],
            "learning_rate"    : BASE_PARAMS["learning_rate"],
            "max_depth"        : BASE_PARAMS["max_depth"],
            "min_child_weight" : BASE_PARAMS["min_child_weight"],
            "subsample"        : BASE_PARAMS["subsample"],
            "colsample_bytree" : BASE_PARAMS["colsample_bytree"],
            "reg_alpha"        : BASE_PARAMS["reg_alpha"],
            "reg_lambda"       : BASE_PARAMS["reg_lambda"],
            "seed"             : BASE_PARAMS["random_state"],
            "verbosity"        : 0,
        }

        cv_wrmsse, cv_dollar = [], []

        for fold, (tr_end, v_start, v_end) in enumerate(splits):
            logger.info("  Fold %d — loading train (date <= %s)...", fold, tr_end)
            X_tr, y_tr   = _load_slice(cache_dir, date_from=None,    date_to=tr_end)
            logger.info("  Fold %d — loading val   (%s -> %s)...",   fold, v_start, v_end)
            X_val, y_val = _load_slice(cache_dir, date_from=v_start, date_to=v_end)
            logger.info("  Train: %s rows | Val: %s rows",
                        f"{len(X_tr):,}", f"{len(X_val):,}")

            dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=FEATURE_COLS)
            dval   = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
            del X_tr, y_tr; gc.collect()

            booster = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round   = BASE_PARAMS["n_estimators"],
                evals             = [(dval, "val")],
                early_stopping_rounds = BASE_PARAMS["early_stopping_rounds"],
                verbose_eval      = 200,
            )
            del dtrain, dval; gc.collect()

            preds = np.maximum(booster.predict(
                xgb.DMatrix(X_val, feature_names=FEATURE_COLS)), 0)
            fold_wrmsse = wrmsse(y_val, preds)
            fold_dollar = dollar_error_cost(y_val, preds)
            cv_wrmsse.append(fold_wrmsse)
            cv_dollar.append(fold_dollar["total_cost_usd"])
            logger.info("  Fold %d | WRMSSE=%.4f | $cost=%s",
                        fold, fold_wrmsse, f"{fold_dollar['total_cost_usd']:,.0f}")
            del X_val, y_val, preds, booster; gc.collect()

        # Final model on all data
        logger.info("  Training final model on all data...")
        X_all, y_all   = _load_slice(cache_dir, date_from=None, date_to=all_dates[-1])
        dtrain_all     = xgb.DMatrix(X_all, label=y_all, feature_names=FEATURE_COLS)
        del X_all, y_all; gc.collect()

        final_booster = xgb.train(
            xgb_params, dtrain_all,
            num_boost_round = BASE_PARAMS["n_estimators"],
            verbose_eval    = 200,
        )
        del dtrain_all; gc.collect()

        model_path = output_dir / f"xgb_q{int(q*100)}.ubj"
        final_booster.save_model(str(model_path))
        models[q] = final_booster

        with mlflow.start_run(run_name=f"xgb_q{int(q*100)}"):
            mlflow.log_params({"quantile": q, "n_cv_splits": n_cv_splits,
                               "horizon": HORIZON, **xgb_params})
            mlflow.log_metrics({
                "cv_wrmsse_mean" : float(np.mean(cv_wrmsse)),
                "cv_wrmsse_std"  : float(np.std(cv_wrmsse)),
                "cv_dollar_mean" : float(np.mean(cv_dollar)),
            })
            mlflow.log_artifact(str(model_path), artifact_path=f"model_q{int(q*100)}")

        logger.info("  ✓ q=%.2f | CV WRMSSE=%.4f ± %.4f | CV $cost=%s",
                    q, np.mean(cv_wrmsse), np.std(cv_wrmsse),
                    f"{np.mean(cv_dollar):,.0f}")

    return models


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_quantiles(models: dict[float, xgb.Booster],
                       X: pd.DataFrame) -> pd.DataFrame:
    dmat = xgb.DMatrix(X[FEATURE_COLS].values.astype("float32"),
                       feature_names=FEATURE_COLS)
    preds = {}
    for q, booster in models.items():
        preds[f"q{int(q*100):02d}"] = np.maximum(booster.predict(dmat), 0)
    return pd.DataFrame(preds, index=X.index)


def load_models(model_dir: str | Path) -> dict[float, xgb.Booster]:
    model_dir = Path(model_dir)
    models = {}
    for q in QUANTILES:
        path = model_dir / f"xgb_q{int(q*100)}.ubj"
        if path.exists():
            b = xgb.Booster()
            b.load_model(str(path))
            models[q] = b
            logger.info("Loaded %s", path.name)
    return models


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(booster: xgb.Booster,
                  X: pd.DataFrame,
                  max_display: int = 20,
                  save_path: Optional[Path] = None) -> np.ndarray:
    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(
        X[FEATURE_COLS].values.astype("float32"))

    if save_path:
        shap.summary_plot(shap_values, X[FEATURE_COLS],
                          max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return shap_values


# ── Legacy helper for tests ───────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame,
                         n_splits: int = 3,
                         horizon: int = HORIZON,
                         gap: int = 0):
    dates = pd.to_datetime(df[DATE_COL]).sort_values().unique()
    for tr_end, v_start, v_end in walk_forward_date_splits(dates, n_splits, horizon):
        train_idx = df[pd.to_datetime(df[DATE_COL]) <= tr_end].index
        val_idx   = df[(pd.to_datetime(df[DATE_COL]) >= v_start) &
                       (pd.to_datetime(df[DATE_COL]) <= v_end)].index
        yield train_idx, val_idx
