"""
Main Pipeline
=============
End-to-end run: feature engineering → LightGBM training → 3 optimization modules → summary.

Usage
-----
    python run_pipeline.py --data_dir data/ --n_items 500 --output_dir outputs/

Arguments
---------
--data_dir   : path to folder with M5 CSVs (default: data/)
--n_items    : number of items to subsample (None = full; default: 500 for dev)
--output_dir : where to save artefacts (models, plots, reports)
--full       : flag to run on full dataset (overrides n_items)
"""

import argparse
import logging
import json
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.data.features import build_features, FEATURE_COLS
from src.forecasting.xgb_quantile import (
    train_quantile_models, predict_quantiles, compute_shap
)
from src.optimization.newsvendor import inventory_dollar_impact, optimal_order_quantity, CostParams
from src.optimization.budget_alloc_improved import budget_dollar_impact, BudgetParams
from src.optimization.markdown import markdown_dollar_impact, MarkdownParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/")
    p.add_argument("--n_items",    type=int, default=500)
    p.add_argument("--output_dir", default="outputs/")
    p.add_argument("--full",       action="store_true")
    p.add_argument("--n_cv",       type=int, default=3,
                   help="Walk-forward CV folds")
    return p.parse_args()


def make_forecast_df(cache_dir: Path, models: dict, id_col="id") -> pd.DataFrame:
    """Run inference and attach forecasts to the last 28 days of the feature df."""
    # Get last date from one store
    sample_pq = next(cache_dir.glob("features_*.parquet"))
    sample_df = pd.read_parquet(sample_pq, columns=["date"])
    last_date = pd.to_datetime(sample_df["date"]).max()
    cutoff    = last_date - pd.Timedelta(days=27)

    out_parts = []
    for pq in cache_dir.glob("features_*.parquet"):
        df = pd.read_parquet(pq)
        df["date"] = pd.to_datetime(df["date"])
        test_df = df[df["date"] >= cutoff].copy()

        preds = predict_quantiles(models, test_df)
        test_df = test_df.reset_index(drop=True)
        preds   = preds.reset_index(drop=True)

        out_parts.append(pd.concat([
            test_df[["id", "store_id", "item_id", "date", "sales", "sell_price"]],
            preds
        ], axis=1))

    return pd.concat(out_parts, ignore_index=True)


def simulate_inventory(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate realistic on-hand inventory for the markdown module.
    Inventory = q90 * U(0.8, 1.4) — some items are overstocked.
    """
    rng = np.random.default_rng(42)
    n   = len(forecast_df)
    inventory = forecast_df["q90"].values * rng.uniform(0.8, 1.4, n)
    forecast_df = forecast_df.copy()
    forecast_df["inventory"] = np.round(inventory).astype(int)
    return forecast_df


def main():
    args = parse_args()
    n_items = None if args.full else args.n_items
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("M5 Forecast → Optimize Pipeline")
    logger.info("Items: %s | CV folds: %d", n_items or 'ALL', args.n_cv)
    logger.info("=" * 60)

    # ── 1. Feature Engineering ────────────────────────────────────────────────
    logger.info("[1/5] Building features...")
    cache_dir = build_features(args.data_dir, n_items=n_items)
    logger.info("Feature matrix cached to %s", cache_dir)

    # ── 2. Train XGBoost quantile models ────────────────────────────────────
    logger.info("[2/5] Training XGBoost quantile models...")
    models = train_quantile_models(
        cache_dir,
        output_dir=output_dir / "models",
        n_cv_splits=args.n_cv,
    )

    # ── 3. Generate forecasts ─────────────────────────────────────────────────
    logger.info("[3/5] Generating 28-day probabilistic forecasts...")
    forecast_df = make_forecast_df(cache_dir, models)
    forecast_df = simulate_inventory(forecast_df)

    # Compute q_star for markdown module
    cost_p = CostParams()
    forecast_df["q_star"] = optimal_order_quantity(
        forecast_df["q10"].values,
        forecast_df["q50"].values,
        forecast_df["q90"].values,
        cost=cost_p,
    )

    forecast_path = output_dir / "forecasts.parquet"
    forecast_df.to_parquet(forecast_path, index=False)
    logger.info("Forecasts saved → %s", forecast_path)

    # ── 4. SHAP analysis ──────────────────────────────────────────────────────
    logger.info("[4/5] Computing SHAP feature importance...")
    sample_pq = next(cache_dir.glob("features_*.parquet"))
    df_sample = pd.read_parquet(sample_pq)
    sample = df_sample.sample(min(5000, len(df_sample)), random_state=42)
    compute_shap(
        models[0.50], sample,
        save_path=output_dir / "shap_beeswarm.png"
    )

    # ── 5. Optimization modules ───────────────────────────────────────────────
    logger.info("[5/5] Running optimization modules...")

    # Module 1 — Newsvendor
    nv_impact = inventory_dollar_impact(forecast_df)
    logger.info("  [Newsvendor] 28d saving: $%s | Enterprise (ann.): $%s",
                f"{nv_impact['saving_28d_usd']:,.0f}",
                f"{nv_impact['enterprise_saving_usd']:,.0f}")

    # Module 2 — Budget Allocation
    ba_params = BudgetParams(budget_usd=500_000)
    ba_impact = budget_dollar_impact(forecast_df, params=ba_params)
    logger.info("  [BudgetAlloc] Revenue uplift 28d: $%s | Enterprise (ann.): $%s | Shadow price: $%.4f/$ | Items: %s",
                f"{ba_impact['revenue_uplift_28d_usd']:,.0f}",
                f"{ba_impact['enterprise_uplift_usd']:,.0f}",
                ba_impact["shadow_price_per_dollar"] or 0,
                f"{ba_impact['n_items']:,}")
    if ba_impact["actual_revenue_28d_usd"] is not None:
        logger.info("  [BudgetAlloc] Actual revenue 28d: $%s | Uplift vs actual: $%s (%.1f%%)",
                    f"{ba_impact['actual_revenue_28d_usd']:,.0f}",
                    f"{ba_impact['revenue_uplift_vs_actual_28d_usd']:,.0f}",
                    ba_impact["revenue_uplift_vs_actual_pct"])

    # Module 3 — Markdown
    md_impact = markdown_dollar_impact(forecast_df)
    logger.info("  [Markdown] Revenue gain 28d: $%s | Enterprise (ann.): $%s",
                f"{md_impact['revenue_gain_28d_usd']:,.0f}",
                f"{md_impact['enterprise_gain_usd']:,.0f}")

    # ── Summary Report ────────────────────────────────────────────────────────
    store_rollup_df = ba_impact.pop("store_rollup_df")
    item_detail_df  = ba_impact.pop("item_detail_df")
    md_item_df      = md_impact.pop("item_detail_df")

    summary = {
        "run_timestamp"     : datetime.now().isoformat(),
        "n_items_sampled"   : n_items or "ALL",
        "n_forecast_rows"   : len(forecast_df),
        "newsvendor"        : nv_impact,
        "budget_allocation" : ba_impact,
        "markdown"          : {k: v for k, v in md_impact.items()},
        "combined_enterprise_annual_usd": (
            nv_impact["enterprise_saving_usd"]
            + ba_impact["enterprise_uplift_usd"]
            + md_impact["enterprise_gain_usd"]
        ),
    }

    report_path = output_dir / "optimization_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    store_rollup_df.to_csv(output_dir / "store_allocations.csv",  index=False)
    item_detail_df.to_csv( output_dir / "item_allocations.csv",   index=False)
    md_item_df.to_csv(     output_dir / "markdown_items.csv",     index=False)

    # Log summary to MLflow
    with mlflow.start_run(run_name="optimization_summary"):
        mlflow.log_metrics({
            "nv_saving_28d"                  : nv_impact["saving_28d_usd"],
            "nv_enterprise_annual"           : nv_impact["enterprise_saving_usd"],
            "ba_uplift_28d"                  : ba_impact["revenue_uplift_28d_usd"],
            "ba_enterprise_annual"           : ba_impact["enterprise_uplift_usd"],
            "ba_shadow_price"                : ba_impact["shadow_price_per_dollar"] or 0,
            "ba_actual_revenue_28d"          : ba_impact["actual_revenue_28d_usd"] or 0,
            "ba_uplift_vs_actual_28d"        : ba_impact["revenue_uplift_vs_actual_28d_usd"] or 0,
            "ba_enterprise_uplift_vs_actual" : ba_impact["enterprise_uplift_vs_actual_usd"] or 0,
            "md_gain_28d"                    : md_impact["revenue_gain_28d_usd"],
            "md_enterprise_annual"           : md_impact["enterprise_gain_usd"],
            "combined_enterprise_ann"        : summary["combined_enterprise_annual_usd"],
        })

    logger.info("=" * 60)
    logger.info("COMBINED ENTERPRISE ANNUAL IMPACT: $%s",
                f"{summary['combined_enterprise_annual_usd']:,.0f}")
    logger.info("Report saved → %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()