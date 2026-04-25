"""
Module 2 — Store-Level Budget Allocation
=========================================
Given a fixed replenishment budget B and probabilistic demand forecasts,
allocate inventory spend across stores to maximise expected revenue:

    max  Σ_s  sell_price × E[min(Q_s, D_s)]
    s.t. Σ_s  unit_cost × Q_s  ≤  B

This is separable and concave → greedy marginal revenue per dollar is optimal.

Expected revenue model
----------------------
E[min(Q, D)] is approximated by integrating over the piecewise-linear CDF
implied by (q10, q50, q90):

  Q ≤ q10  →  E[min] ≈ Q                          (sell everything, ~certain)
  q10<Q≤q50 →  E[min] ≈ q10 + (Q-q10)*(1-0.10/0.50)  (prob of selling tapers)
  q50<Q≤q90 →  E[min] ≈ q50 + (Q-q50)*(1-0.50/0.90)  (prob tapers further)
  Q > q90   →  E[min] ≈ q90                        (hard cap: demand exhausted)

This gives a monotone concave function, which is the correct shape for the
greedy marginal-revenue algorithm to work properly.
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.data.features import AVG_UNIT_COST, AVG_SELL_PRICE


@dataclass
class BudgetParams:
    budget_usd: float  = 500_000.0
    unit_cost:  float  = AVG_UNIT_COST
    sell_price: float  = AVG_SELL_PRICE


# ── Store-level aggregation ───────────────────────────────────────────────────

def aggregate_by_store(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Sum item-level forecasts to store-level totals."""
    return (forecast_df
            .groupby("store_id")[["q10", "q50", "q90"]]
            .sum()
            .reset_index())


# ── Expected revenue (correct piecewise-linear CDF) ──────────────────────────

def expected_revenue(order_qty: np.ndarray,
                      q10: np.ndarray,
                      q50: np.ndarray,
                      q90: np.ndarray,
                      sell_price: float = AVG_SELL_PRICE) -> np.ndarray:
    """
    E[min(Q, D)] via piecewise-linear CDF interpolation.

    Slopes are derived from quantile probabilities:
      - Below q10  : slope = 1.0  (selling every unit with near-certainty)
      - q10 → q50  : slope = (0.50 - 0.10) / (q50 - q10) per unit of Q,
                     integrated gives (1 - 0.10) at q10 tapering to (1 - 0.50) at q50
      - q50 → q90  : same logic, tapering to (1 - 0.90)
      - Above q90  : slope = 0    (additional stock almost never sells)
    """
    Q   = np.asarray(order_qty, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)

    # Avoid division by zero for degenerate quantiles
    span_low  = np.maximum(q50 - q10, 1e-6)
    span_high = np.maximum(q90 - q50, 1e-6)

    # Slope of E[min] in each region (marginal sell-through probability)
    slope_low  = (0.50 - 0.10) / span_low   * span_low   # = 0.40 per unit of span
    slope_high = (0.90 - 0.50) / span_high  * span_high  # = 0.40 per unit of span

    # Actually we want the *average* probability of selling a marginal unit:
    #   in [q10, q50]: prob ranges from 0.90 down to 0.50  → avg = 0.70
    #   in [q50, q90]: prob ranges from 0.50 down to 0.10  → avg = 0.30
    # So E[min] increments are:
    #   each unit in [q10,q50] contributes ~0.70 expected sales
    #   each unit in [q50,q90] contributes ~0.30 expected sales

    e_sales = np.where(
        Q <= q10,
        Q,                                                      # region 1: sell all
        np.where(
            Q <= q50,
            q10 + (Q - q10) * 0.70,                            # region 2
            np.where(
                Q <= q90,
                q10 + span_low * 0.70 + (Q - q50) * 0.30,     # region 3
                q10 + span_low * 0.70 + span_high * 0.30       # region 4: capped
            )
        )
    )
    return sell_price * e_sales


# ── Greedy allocation ─────────────────────────────────────────────────────────

def greedy_budget_allocation(store_df: pd.DataFrame,
                              params: Optional[BudgetParams] = None,
                              n_steps: int = 10_000) -> pd.DataFrame:
    """
    Greedy marginal-revenue-per-dollar allocation.

    At each step assigns a unit bundle to the store with the highest
    incremental revenue per dollar spent. Runs until budget exhausted.

    n_steps controls resolution: higher = more accurate but slower.
    Default 10,000 gives good accuracy on store-level M5 aggregates.
    """
    if params is None:
        params = BudgetParams()

    stores = store_df["store_id"].values
    q10    = store_df["q10"].values.astype(float)
    q50    = store_df["q50"].values.astype(float)
    q90    = store_df["q90"].values.astype(float)
    n      = len(stores)

    # Step size: spread budget across n_steps increments
    total_units = params.budget_usd / params.unit_cost
    step_qty    = max(1.0, total_units / n_steps)

    # Naive baseline: allocate proportional to q50
    total_q50   = q50.sum()
    naive_alloc = (q50 / total_q50) * total_units

    # Greedy allocation
    alloc  = np.zeros(n)
    budget = params.budget_usd

    while budget >= params.unit_cost * step_qty:
        cur_rev  = expected_revenue(alloc,              q10, q50, q90, params.sell_price)
        next_rev = expected_revenue(alloc + step_qty,   q10, q50, q90, params.sell_price)
        mr_per_dollar = (next_rev - cur_rev) / (params.unit_cost * step_qty + 1e-9)

        best = np.argmax(mr_per_dollar)

        # Stop if no store benefits from more inventory (all past q90)
        if mr_per_dollar[best] <= 0:
            break

        alloc[best] += step_qty
        budget       -= params.unit_cost * step_qty

    rev_naive   = expected_revenue(naive_alloc, q10, q50, q90, params.sell_price)
    rev_optimal = expected_revenue(alloc,       q10, q50, q90, params.sell_price)

    return pd.DataFrame({
        "store_id"            : stores,
        "demand_q50"          : q50,
        "alloc_naive_units"   : naive_alloc,
        "alloc_optimal_units" : alloc,
        "revenue_naive_usd"   : rev_naive,
        "revenue_optimal_usd" : rev_optimal,
        "cost_naive_usd"      : naive_alloc * params.unit_cost,
        "cost_optimal_usd"    : alloc       * params.unit_cost,
    })


# ── Dollar impact summary ─────────────────────────────────────────────────────

def budget_dollar_impact(forecast_df: pd.DataFrame,
                          params: Optional[BudgetParams] = None) -> dict:
    if params is None:
        params = BudgetParams()

    store_df = aggregate_by_store(forecast_df)
    result   = greedy_budget_allocation(store_df, params)

    total_rev_naive   = result["revenue_naive_usd"].sum()
    total_rev_optimal = result["revenue_optimal_usd"].sum()
    rev_uplift        = total_rev_optimal - total_rev_naive
    uplift_pct        = rev_uplift / (total_rev_naive + 1e-8) * 100

    result["realloc_delta"] = result["alloc_optimal_units"] - result["alloc_naive_units"]
    top_gainer = result.loc[result["realloc_delta"].idxmax(), "store_id"]
    top_loser  = result.loc[result["realloc_delta"].idxmin(), "store_id"]

    annualised_uplift = rev_uplift * (365 / 28)
    enterprise_uplift = annualised_uplift * 470

    return {
        "policy"                         : "Greedy Marginal Revenue Allocation",
        "budget_usd"                     : params.budget_usd,
        "total_revenue_naive_usd"        : round(total_rev_naive,   2),
        "total_revenue_optimal_usd"      : round(total_rev_optimal, 2),
        "revenue_uplift_28d_usd"         : round(rev_uplift,        2),
        "revenue_uplift_pct"             : round(uplift_pct,        2),
        "annualised_uplift_10stores_usd" : round(annualised_uplift,  0),
        "enterprise_uplift_usd"          : round(enterprise_uplift,  0),
        "top_gaining_store"              : top_gainer,
        "top_losing_store"               : top_loser,
        "store_allocation_df"            : result,
    }