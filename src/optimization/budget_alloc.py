"""
Module 2 — Store-Level Budget Allocation
=========================================
Given a fixed replenishment budget B and probabilistic demand forecasts,
allocate inventory spend across stores to maximise expected revenue (or
equivalently, minimise expected stockout losses subject to a budget constraint).

Formulation
-----------
    max  Σ_s  revenue(q_s)
    s.t. Σ_s  cost(q_s) ≤ B
         q_s ≥ 0  ∀s

where  revenue(q_s) = sell_price × E[min(q_s, D_s)]
and    cost(q_s)    = unit_cost × q_s

This is separable → solved greedily via marginal-revenue-per-dollar ranking.

Dollar impact: baseline = allocate B proportional to q50 across stores.
               optimal  = greedy MR/$ allocation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Optional
from dataclasses import dataclass

from src.data.features import AVG_UNIT_COST, AVG_SELL_PRICE


@dataclass
class BudgetParams:
    budget_usd: float = 500_000.0      # replenishment budget per cycle ($)
    unit_cost: float = AVG_UNIT_COST   # $ COGS per unit
    sell_price: float = AVG_SELL_PRICE # $ retail price per unit


# ── Store-level forecast aggregation ──────────────────────────────────────────

def aggregate_by_store(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse item-level forecasts to store-level totals.
    forecast_df must have columns: store_id, q10, q50, q90.
    """
    agg = (forecast_df
           .groupby("store_id")[["q10", "q50", "q90"]]
           .sum()
           .reset_index())
    return agg


# ── Revenue function (expected sales = E[min(Q, D)]) ─────────────────────────

def expected_revenue(order_qty: np.ndarray,
                      q10: np.ndarray,
                      q50: np.ndarray,
                      q90: np.ndarray,
                      sell_price: float = AVG_SELL_PRICE) -> np.ndarray:
    """
    Approximate E[min(Q, D)] using linear interpolation on the forecast CDF.
    Revenue = sell_price × E[min(Q, D)].
    """
    order_qty = np.asarray(order_qty, dtype=float)
    q10, q50, q90 = map(np.asarray, (q10, q50, q90))

    # E[min(Q,D)] ≈ Q  when Q ≤ q10 (almost certain to sell all)
    #             ≈ q50  when Q ≥ q90 (demand capped)
    # Linearly interpolate between quantile points
    e_sales = np.where(
        order_qty <= q10, order_qty,
        np.where(
            order_qty <= q50,
            q10 + (order_qty - q10) * 0.8,
            np.where(
                order_qty <= q90,
                q50 + (order_qty - q50) * 0.5,
                q90 * 0.95  # diminishing returns beyond q90
            )
        )
    )
    return sell_price * e_sales


# ── Greedy marginal allocation ────────────────────────────────────────────────

def greedy_budget_allocation(store_df: pd.DataFrame,
                              params: Optional[BudgetParams] = None,
                              n_steps: int = 1000) -> pd.DataFrame:
    """
    Greedy algorithm: repeatedly assign one unit to the store with the
    highest marginal revenue per dollar until budget is exhausted.

    Parameters
    ----------
    store_df : DataFrame with columns [store_id, q10, q50, q90]
    params   : BudgetParams
    n_steps  : resolution of the greedy search

    Returns
    -------
    DataFrame with columns [store_id, q50_naive, q_optimal, revenue_naive,
                             revenue_optimal, cost_naive, cost_optimal]
    """
    if params is None:
        params = BudgetParams()

    stores = store_df["store_id"].values
    q10    = store_df["q10"].values
    q50    = store_df["q50"].values
    q90    = store_df["q90"].values
    n      = len(stores)

    unit_budget = params.unit_cost  # cost per unit

    # Baseline: proportional allocation (order proportional to q50)
    total_q50   = q50.sum()
    naive_alloc = (q50 / total_q50) * (params.budget_usd / params.unit_cost)

    # Greedy allocation
    alloc    = np.zeros(n)
    budget   = params.budget_usd
    step_qty = max(1.0, total_q50 / n_steps)  # qty per step

    while budget >= unit_budget * step_qty:
        # Compute marginal revenue for each store at current allocation
        current_rev = expected_revenue(alloc,       q10, q50, q90, params.sell_price)
        next_rev    = expected_revenue(alloc + step_qty, q10, q50, q90, params.sell_price)
        marginal_mr = (next_rev - current_rev) / (unit_budget * step_qty + 1e-9)

        # Allocate to store with highest MR/$
        best = np.argmax(marginal_mr)
        alloc[best] += step_qty
        budget       -= unit_budget * step_qty

    # Build results
    rev_naive   = expected_revenue(naive_alloc, q10, q50, q90, params.sell_price)
    rev_optimal = expected_revenue(alloc,       q10, q50, q90, params.sell_price)

    result = pd.DataFrame({
        "store_id"        : stores,
        "demand_q50"      : q50,
        "alloc_naive_units"   : naive_alloc,
        "alloc_optimal_units" : alloc,
        "revenue_naive_usd"   : rev_naive,
        "revenue_optimal_usd" : rev_optimal,
        "cost_naive_usd"      : naive_alloc * params.unit_cost,
        "cost_optimal_usd"    : alloc       * params.unit_cost,
    })
    return result


# ── Dollar impact summary ─────────────────────────────────────────────────────

def budget_dollar_impact(forecast_df: pd.DataFrame,
                          params: Optional[BudgetParams] = None) -> dict:
    """
    Full pipeline: aggregate → allocate → compute dollar impact.
    Returns summary dict suitable for MLflow logging / dashboard display.
    """
    if params is None:
        params = BudgetParams()

    store_df = aggregate_by_store(forecast_df)
    result   = greedy_budget_allocation(store_df, params)

    total_rev_naive   = result["revenue_naive_usd"].sum()
    total_rev_optimal = result["revenue_optimal_usd"].sum()
    rev_uplift        = total_rev_optimal - total_rev_naive
    uplift_pct        = rev_uplift / (total_rev_naive + 1e-8) * 100

    # Store with biggest reallocation
    result["realloc_delta"] = (result["alloc_optimal_units"]
                               - result["alloc_naive_units"])
    top_gainer = result.loc[result["realloc_delta"].idxmax(), "store_id"]
    top_loser  = result.loc[result["realloc_delta"].idxmin(), "store_id"]

    annualised_uplift  = rev_uplift * (365 / 28)
    enterprise_uplift  = annualised_uplift * 470   # scale to 4700 stores

    return {
        "policy"                        : "Greedy Marginal Revenue Allocation",
        "budget_usd"                    : params.budget_usd,
        "total_revenue_naive_usd"       : round(total_rev_naive,   2),
        "total_revenue_optimal_usd"     : round(total_rev_optimal, 2),
        "revenue_uplift_28d_usd"        : round(rev_uplift,        2),
        "revenue_uplift_pct"            : round(uplift_pct,        2),
        "annualised_uplift_10stores_usd": round(annualised_uplift,  0),
        "enterprise_uplift_usd"         : round(enterprise_uplift,  0),
        "top_gaining_store"             : top_gainer,
        "top_losing_store"              : top_loser,
        "store_allocation_df"           : result,
    }
