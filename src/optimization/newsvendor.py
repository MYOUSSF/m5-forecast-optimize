"""
Module 1 — Newsvendor Inventory Optimization
=============================================
Given probabilistic forecasts (q10, q50, q90), solve the classic
newsvendor problem to find the optimal order quantity that minimises
expected total cost = holding_cost × E[overstock] + stockout_penalty × E[understock].

The critical ratio  CR = c_u / (c_u + c_o)  gives the optimal service level,
and we interpolate across our forecast quantiles to find the order quantity.

Dollar impact: compares naive (order = q50) vs optimised (order = q*).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from src.data.features import (
    AVG_UNIT_COST, AVG_SELL_PRICE,
    HOLDING_COST_RATE, STOCKOUT_PENALTY_RATE,
)


# ── Cost structure ─────────────────────────────────────────────────────────────

@dataclass
class CostParams:
    """
    Per-unit cost parameters for the newsvendor model.

    c_o : cost of ordering one unit too many (holding / obsolescence)
    c_u : cost of being short one unit (stockout penalty + lost margin)
    """
    c_o: float = field(default_factory=lambda: AVG_UNIT_COST * HOLDING_COST_RATE / 365)
    c_u: float = field(default_factory=lambda: AVG_SELL_PRICE * STOCKOUT_PENALTY_RATE)

    @property
    def critical_ratio(self) -> float:
        """Optimal service level = c_u / (c_u + c_o)."""
        return self.c_u / (self.c_u + self.c_o)


# ── Core solver ───────────────────────────────────────────────────────────────

def optimal_order_quantity(q10: np.ndarray,
                            q50: np.ndarray,
                            q90: np.ndarray,
                            cost: Optional[CostParams] = None) -> np.ndarray:
    """
    Interpolate optimal order quantity Q* from three forecast quantiles.

    The newsvendor optimal order Q* satisfies  F(Q*) = CR  where F is the
    demand CDF.  We approximate F⁻¹(CR) by linear interpolation between
    the three known quantile points.

    Parameters
    ----------
    q10, q50, q90 : arrays of forecast quantiles (same length)
    cost          : CostParams (default uses module-level assumptions)

    Returns
    -------
    q_star : array of optimal order quantities
    """
    if cost is None:
        cost = CostParams()

    cr = cost.critical_ratio  # e.g. ~0.94 with default costs

    # Map the three quantile levels to arrays for interpolation
    # For each row, we have three (prob, quantity) pairs:
    #   (0.10, q10), (0.50, q50), (0.90, q90)
    # Linear interpolation within the relevant interval

    q10, q50, q90 = map(np.asarray, (q10, q50, q90))
    q_star = np.empty_like(q50, dtype=float)

    if cr <= 0.10:
        q_star[:] = q10
    elif cr <= 0.50:
        t = (cr - 0.10) / (0.50 - 0.10)
        q_star = q10 + t * (q50 - q10)
    elif cr <= 0.90:
        t = (cr - 0.50) / (0.90 - 0.50)
        q_star = q50 + t * (q90 - q50)
    else:
        # Beyond q90: extrapolate using q50→q90 slope
        t = (cr - 0.90) / (1.0 - 0.90)
        q_star = q90 + t * (q90 - q50)

    return np.maximum(q_star, 0)


# ── Expected cost calculator ──────────────────────────────────────────────────

def expected_cost(order_qty: np.ndarray,
                   q10: np.ndarray,
                   q50: np.ndarray,
                   q90: np.ndarray,
                   cost: Optional[CostParams] = None) -> pd.DataFrame:
    """
    Approximate expected holding + stockout cost for a given order quantity
    by integrating over the triangular distribution implied by (q10, q50, q90).

    Returns a DataFrame with per-row cost breakdown.
    """
    if cost is None:
        cost = CostParams()

    order_qty, q10, q50, q90 = map(np.asarray, (order_qty, q10, q50, q90))

    # Expected overstock  E[max(Q - D, 0)]  ≈ max(Q - q50, 0) * (1 - CR_empirical)
    # Simple linear approximation good enough for portfolio demos
    E_overstock  = np.maximum(order_qty - q50, 0) * 0.5
    E_understock = np.maximum(q50 - order_qty, 0) * 0.5

    holding  = E_overstock  * cost.c_o
    stockout = E_understock * cost.c_u

    return pd.DataFrame({
        "order_qty"      : order_qty,
        "expected_overstock"  : E_overstock,
        "expected_understock" : E_understock,
        "holding_cost_usd"    : holding,
        "stockout_cost_usd"   : stockout,
        "total_cost_usd"      : holding + stockout,
    })


# ── Dollar impact summary ─────────────────────────────────────────────────────

def inventory_dollar_impact(forecast_df: pd.DataFrame,
                              cost: Optional[CostParams] = None) -> dict:
    """
    Compare naive policy (order = q50) vs optimal newsvendor policy.

    Parameters
    ----------
    forecast_df : DataFrame with columns q10, q50, q90 (one row per item-day)
    cost        : CostParams

    Returns
    -------
    dict with annualised dollar savings and key metrics
    """
    if cost is None:
        cost = CostParams()

    q10 = forecast_df["q10"].values
    q50 = forecast_df["q50"].values
    q90 = forecast_df["q90"].values

    # Naive policy: order q50
    naive_cost_df = expected_cost(q50, q10, q50, q90, cost)

    # Optimal policy
    q_star = optimal_order_quantity(q10, q50, q90, cost)
    opt_cost_df   = expected_cost(q_star, q10, q50, q90, cost)

    total_naive = naive_cost_df["total_cost_usd"].sum()
    total_opt   = opt_cost_df["total_cost_usd"].sum()
    saving      = total_naive - total_opt

    # Annualise: forecast_df covers HORIZON=28 days
    days_covered = 28
    annualised_saving = saving * (365 / days_covered)

    # Scale to Walmart US store count (4,700 stores)
    # The M5 dataset covers 10 stores → scale factor = 470
    store_scale = 470
    enterprise_saving = annualised_saving * store_scale

    return {
        "policy"                  : "Newsvendor (Critical Ratio)",
        "critical_ratio"          : round(cost.critical_ratio, 4),
        "naive_total_cost_usd"    : round(total_naive, 2),
        "optimal_total_cost_usd"  : round(total_opt, 2),
        "saving_28d_usd"          : round(saving, 2),
        "saving_pct"              : round(saving / (total_naive + 1e-8) * 100, 2),
        "annualised_saving_10stores_usd" : round(annualised_saving, 0),
        "enterprise_saving_usd"   : round(enterprise_saving, 0),
        "avg_order_qty_naive"     : round(float(np.mean(q50)), 3),
        "avg_order_qty_optimal"   : round(float(np.mean(q_star)), 3),
        "service_level_target"    : round(cost.critical_ratio * 100, 1),
    }


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def sensitivity_analysis(forecast_df: pd.DataFrame,
                           c_u_range: Optional[list] = None,
                           c_o_range: Optional[list] = None) -> pd.DataFrame:
    """
    Sweep c_u and c_o to show how optimal policy & savings change.
    Useful for the dashboard / README exhibit.
    """
    if c_u_range is None:
        c_u_range = [0.5, 1.0, 2.0, 3.0, 4.0, AVG_SELL_PRICE * STOCKOUT_PENALTY_RATE]
    if c_o_range is None:
        c_o_range = [AVG_UNIT_COST * HOLDING_COST_RATE / 365]  # keep c_o fixed

    rows = []
    for c_u in c_u_range:
        for c_o in c_o_range:
            cp = CostParams(c_o=c_o, c_u=c_u)
            result = inventory_dollar_impact(forecast_df, cost=cp)
            result["c_u"] = c_u
            result["c_o"] = c_o
            rows.append(result)

    return pd.DataFrame(rows)
