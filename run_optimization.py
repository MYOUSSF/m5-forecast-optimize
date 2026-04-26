"""
Optimization Pipeline (standalone)
====================================
Runs the three optimization modules against an existing forecasts.parquet
produced by run_pipeline.py. No feature engineering or model training required.

Usage
-----
    python run_optimization.py --output_dir outputs/

    # Custom budget or cost params
    python run_optimization.py --output_dir outputs/ --budget 1000000

    # Point at a non-default forecast file
    python run_optimization.py --forecast_path outputs/forecasts.parquet

Arguments
---------
--output_dir    : folder that contains forecasts.parquet and receives new outputs
                  (default: outputs/)
--forecast_path : explicit path to forecasts.parquet, overrides --output_dir
--budget        : replenishment budget in USD (default: 500,000)
--c_u           : stockout penalty per unit in USD (default: from features.py)
--c_o           : holding cost per unit per day in USD (default: from features.py)
"""

import argparse
import json
import logging
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.optimization.newsvendor import (
    inventory_dollar_impact, optimal_order_quantity, CostParams,
)
from src.optimization.budget_alloc_improved import budget_dollar_impact, BudgetParams
from src.optimization.markdown import markdown_dollar_impact, MarkdownParams
from src.data.features import AVG_UNIT_COST, AVG_SELL_PRICE, HOLDING_COST_RATE, STOCKOUT_PENALTY_RATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimize")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    default="outputs/")
    p.add_argument("--forecast_path", default=None,
                   help="Explicit path to forecasts.parquet. "
                        "Defaults to <output_dir>/forecasts.parquet.")
    p.add_argument("--budget", type=float, default=500_000,
                   help="Replenishment budget in USD (default: 500,000)")
    p.add_argument("--c_u",   type=float, default=None,
                   help="Stockout penalty per unit $ (default: AVG_SELL_PRICE × STOCKOUT_PENALTY_RATE)")
    p.add_argument("--c_o",   type=float, default=None,
                   help="Holding cost per unit per day $ (default: AVG_UNIT_COST × HOLDING_COST_RATE / 365)")
    return p.parse_args()


def load_forecast(forecast_path: Path) -> pd.DataFrame:
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"forecasts.parquet not found at {forecast_path}.\n"
            "Run run_pipeline.py first to generate it."
        )
    df = pd.read_parquet(forecast_path)
    logger.info("Loaded forecasts: %s rows | columns: %s", f"{len(df):,}", list(df.columns))

    required = {"q10", "q50", "q90", "sell_price"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"forecasts.parquet is missing required columns: {missing}")

    return df


def ensure_q_star(df: pd.DataFrame, cost: CostParams) -> pd.DataFrame:
    """Recompute q_star if absent or if cost params have changed."""
    df = df.copy()
    df["q_star"] = optimal_order_quantity(
        df["q10"].values, df["q50"].values, df["q90"].values, cost=cost
    )
    return df


def ensure_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Use existing inventory column or simulate if absent."""
    if "inventory" in df.columns:
        logger.info("Using existing inventory column from forecasts.parquet")
        return df
    logger.warning("No inventory column found — simulating from q90 (set seed=42)")
    rng = np.random.default_rng(42)
    df  = df.copy()
    df["inventory"] = np.round(
        df["q90"].values * rng.uniform(0.8, 1.4, len(df))
    ).astype(int)
    return df


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = Path(args.forecast_path) if args.forecast_path else output_dir / "forecasts.parquet"

    logger.info("=" * 60)
    logger.info("M5 Optimization Pipeline (standalone)")
    logger.info("Forecast : %s", forecast_path)
    logger.info("Budget   : $%s", f"{args.budget:,.0f}")
    logger.info("=" * 60)

    # ── Load forecasts ────────────────────────────────────────────────────────
    forecast_df = load_forecast(forecast_path)

    # ── Cost params ───────────────────────────────────────────────────────────
    c_u = args.c_u or AVG_SELL_PRICE * STOCKOUT_PENALTY_RATE
    c_o = args.c_o or AVG_UNIT_COST  * HOLDING_COST_RATE / 365
    cost_p = CostParams(c_u=c_u, c_o=c_o)
    logger.info("Cost params: c_u=$%.4f  c_o=$%.4f  CR=%.4f",
                cost_p.c_u, cost_p.c_o, cost_p.critical_ratio)

    # ── Ensure q_star and inventory columns are present ───────────────────────
    forecast_df = ensure_q_star(forecast_df, cost_p)
    forecast_df = ensure_inventory(forecast_df)

    # ── Module 1 — Newsvendor ─────────────────────────────────────────────────
    logger.info("[1/3] Newsvendor inventory optimization...")
    nv_impact = inventory_dollar_impact(forecast_df, cost=cost_p)
    logger.info("  28d saving       : $%s", f"{nv_impact['saving_28d_usd']:,.0f}")
    logger.info("  Enterprise annual: $%s", f"{nv_impact['enterprise_saving_usd']:,.0f}")

    # ── Module 2 — Budget Allocation (LP) ─────────────────────────────────────
    logger.info("[2/3] LP budget allocation...")
    ba_params = BudgetParams(budget_usd=args.budget)
    ba_impact = budget_dollar_impact(forecast_df, params=ba_params)
    logger.info("  Uplift vs naive  : $%s (%.1f%%)",
                f"{ba_impact['revenue_uplift_28d_usd']:,.0f}",
                ba_impact["revenue_uplift_pct"])
    if ba_impact["actual_revenue_28d_usd"] is not None:
        logger.info("  Actual revenue   : $%s",
                    f"{ba_impact['actual_revenue_28d_usd']:,.0f}")
        logger.info("  Uplift vs actual : $%s (%.1f%%)",
                    f"{ba_impact['revenue_uplift_vs_actual_28d_usd']:,.0f}",
                    ba_impact["revenue_uplift_vs_actual_pct"])
    logger.info("  Shadow price     : $%.4f per $1 of budget",
                ba_impact["shadow_price_per_dollar"] or 0)
    logger.info("  Enterprise annual: $%s", f"{ba_impact['enterprise_uplift_usd']:,.0f}")

    # ── Module 3 — Markdown Scheduling ───────────────────────────────────────
    logger.info("[3/3] Markdown scheduling...")
    md_impact = markdown_dollar_impact(forecast_df)
    logger.info("  Items marked down: %s (%.1f%%)",
                f"{md_impact['n_items_marked_down']:,}",
                md_impact["pct_items_marked_down"])
    logger.info("  Avg markdown depth: %.1f%%", md_impact["avg_markdown_depth_pct"])
    logger.info("  Revenue gain 28d : $%s", f"{md_impact['revenue_gain_28d_usd']:,.0f}")
    logger.info("  Enterprise annual: $%s", f"{md_impact['enterprise_gain_usd']:,.0f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    store_rollup_df = ba_impact.pop("store_rollup_df")
    item_detail_df  = ba_impact.pop("item_detail_df")
    md_item_df      = md_impact.pop("item_detail_df")

    combined_enterprise = (
        nv_impact["enterprise_saving_usd"]
        + ba_impact["enterprise_uplift_usd"]
        + md_impact["enterprise_gain_usd"]
    )

    summary = {
        "run_timestamp"                  : datetime.now().isoformat(),
        "forecast_path"                  : str(forecast_path),
        "n_forecast_rows"                : len(forecast_df),
        "budget_usd"                     : args.budget,
        "cost_params"                    : {"c_u": c_u, "c_o": c_o,
                                            "critical_ratio": cost_p.critical_ratio},
        "newsvendor"                     : nv_impact,
        "budget_allocation"              : ba_impact,
        "markdown"                       : md_impact,
        "combined_enterprise_annual_usd" : combined_enterprise,
    }

    report_path = output_dir / "optimization_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    store_rollup_df.to_csv(output_dir / "store_allocations.csv",  index=False)
    item_detail_df.to_csv( output_dir / "item_allocations.csv",   index=False)
    md_item_df.to_csv(     output_dir / "markdown_items.csv",     index=False)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{output_dir / 'mlflow.db'}")
    mlflow.set_experiment("m5_optimization")
    with mlflow.start_run(run_name="optimization_standalone"):
        mlflow.log_params({"budget_usd": args.budget, "c_u": c_u, "c_o": c_o})
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
            "combined_enterprise_ann"        : combined_enterprise,
        })
        mlflow.log_artifact(str(report_path))

    logger.info("=" * 60)
    logger.info("COMBINED ENTERPRISE ANNUAL: $%s", f"{combined_enterprise:,.0f}")
    logger.info("Report  → %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()