"""
Module 3 — Promotion / Markdown Scheduling
============================================
Given inventory levels and demand forecasts, decide WHICH items to mark down,
HOW DEEP the markdown should be, and WHEN to apply it to clear excess
inventory before end-of-season while maximising net revenue.

Model
-----
For each item with current_inventory > optimal_stock:
  excess = inventory - q_star  (where q_star comes from newsvendor module)

  Markdown depth d ∈ {0%, 10%, 20%, 30%, 40%, 50%} drives demand lift via
  a price-elasticity model:
      demand_lift_factor(d) = 1 + elasticity × d

  Net revenue per unit after markdown = sell_price × (1 - d)
  Revenue with markdown  = min(inventory, forecasted_demand × lift) × net_price
  Revenue without markdown = min(inventory, q50) × sell_price

  We choose the markdown depth that maximises net revenue while clearing
  at least 80% of the excess (clearance constraint).

Dollar impact: total revenue gain from markdown vs. no-action policy
               (no-action = sell what you can at full price + hold/dispose rest).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.data.features import AVG_SELL_PRICE, AVG_UNIT_COST


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class MarkdownParams:
    sell_price: float  = AVG_SELL_PRICE
    unit_cost:  float  = AVG_UNIT_COST
    elasticity: float  = -2.5          # price elasticity of demand (typical grocery: -2 to -3)
    clearance_target: float = 0.80     # must clear ≥ 80% of excess
    disposal_cost_pct: float = 0.15    # cost to dispose unsold unit (% of COGS)
    holding_cost_per_day: float = 0.05 # $ per unit per day holding cost

MARKDOWN_DEPTHS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])


# ── Price-elasticity demand lift ──────────────────────────────────────────────

def demand_lift(markdown_depth: float, elasticity: float = -2.5) -> float:
    """
    Multiplicative demand lift from a price reduction.
      lift = 1 + |elasticity| × markdown_depth
    E.g., 20% markdown with elasticity -2.5 → lift = 1.50 (+50% demand)
    """
    return 1.0 + abs(elasticity) * markdown_depth


# ── Item-level markdown optimiser ────────────────────────────────────────────

def optimal_markdown(inventory:   np.ndarray,
                      q_star:      np.ndarray,
                      q50:         np.ndarray,
                      params: Optional[MarkdownParams] = None,
                      horizon_days: int = 28) -> pd.DataFrame:
    """
    For each item, find the markdown depth that maximises net revenue
    subject to clearing ≥ clearance_target of excess inventory.

    Parameters
    ----------
    inventory   : current on-hand inventory per item
    q_star      : newsvendor optimal order qty (items with inventory > q_star are excess)
    q50         : median forecast demand over horizon
    params      : MarkdownParams
    horizon_days: planning horizon

    Returns
    -------
    DataFrame with optimal markdown depth and revenue comparison per item
    """
    if params is None:
        params = MarkdownParams()

    inventory, q_star, q50 = map(np.asarray, (inventory, q_star, q50))
    n = len(inventory)

    excess = np.maximum(inventory - q_star, 0)
    needs_markdown = excess > 0

    results = []

    for i in range(n):
        inv = inventory[i]
        exc = excess[i]
        d50 = q50[i]

        # No-action baseline: sell q50 at full price, hold/dispose rest
        units_sold_no_action = min(inv, d50)
        unsold_no_action     = max(inv - d50, 0)
        disposal_cost        = unsold_no_action * params.unit_cost * params.disposal_cost_pct
        holding_cost         = unsold_no_action * params.holding_cost_per_day * horizon_days
        rev_no_action = (units_sold_no_action * params.sell_price
                         - disposal_cost - holding_cost)

        if not needs_markdown[i]:
            results.append({
                "excess_units"         : 0,
                "markdown_depth_pct"   : 0.0,
                "units_sold_markdown"  : units_sold_no_action,
                "units_sold_no_action" : units_sold_no_action,
                "rev_markdown_usd"     : rev_no_action,
                "rev_no_action_usd"    : rev_no_action,
                "rev_gain_usd"         : 0.0,
                "clearance_rate"       : 0.0,
                "markdown_applied"     : False,
            })
            continue

        best_depth = 0.0
        best_rev   = rev_no_action
        best_row   = None

        for depth in MARKDOWN_DEPTHS:
            lift         = demand_lift(depth, params.elasticity)
            lifted_demand = d50 * lift
            net_price     = params.sell_price * (1 - depth)

            units_sold    = min(inv, lifted_demand)
            unsold        = max(inv - lifted_demand, 0)
            clearance_rate = 1 - (unsold / (exc + 1e-8))

            # Must clear ≥ clearance_target of excess
            if clearance_rate < params.clearance_target and depth < MARKDOWN_DEPTHS[-1]:
                continue

            rev = units_sold * net_price - unsold * params.unit_cost * params.disposal_cost_pct

            if rev > best_rev:
                best_rev   = rev
                best_depth = depth
                best_row   = {
                    "excess_units"         : exc,
                    "markdown_depth_pct"   : best_depth * 100,
                    "units_sold_markdown"  : units_sold,
                    "units_sold_no_action" : units_sold_no_action,
                    "rev_markdown_usd"     : best_rev,
                    "rev_no_action_usd"    : rev_no_action,
                    "rev_gain_usd"         : best_rev - rev_no_action,
                    "clearance_rate"       : clearance_rate,
                    "markdown_applied"     : True,
                }

        if best_row is None:
            # Fallback: deepest markdown if nothing clears enough
            depth  = MARKDOWN_DEPTHS[-1]
            lift   = demand_lift(depth, params.elasticity)
            units_sold  = min(inv, d50 * lift)
            net_price   = params.sell_price * (1 - depth)
            rev    = units_sold * net_price
            best_row = {
                "excess_units"         : exc,
                "markdown_depth_pct"   : depth * 100,
                "units_sold_markdown"  : units_sold,
                "units_sold_no_action" : units_sold_no_action,
                "rev_markdown_usd"     : rev,
                "rev_no_action_usd"    : rev_no_action,
                "rev_gain_usd"         : rev - rev_no_action,
                "clearance_rate"       : 1 - max(inv - d50 * lift, 0) / (exc + 1e-8),
                "markdown_applied"     : True,
            }

        results.append(best_row)

    return pd.DataFrame(results)


# ── Dollar impact summary ─────────────────────────────────────────────────────

def markdown_dollar_impact(forecast_df: pd.DataFrame,
                            inventory_col: str = "inventory",
                            q_star_col: str    = "q_star",
                            params: Optional[MarkdownParams] = None) -> dict:
    """
    Run markdown optimisation across all items and return dollar impact summary.

    forecast_df must contain: inventory, q_star, q10, q50, q90
    """
    if params is None:
        params = MarkdownParams()

    result_df = optimal_markdown(
        inventory  = forecast_df[inventory_col].values,
        q_star     = forecast_df[q_star_col].values,
        q50        = forecast_df["q50"].values,
        params     = params,
    )

    n_items_marked = result_df["markdown_applied"].sum()
    total_rev_gain  = result_df["rev_gain_usd"].sum()
    total_excess    = result_df["excess_units"].sum()
    avg_depth       = result_df.loc[result_df["markdown_applied"], "markdown_depth_pct"].mean()
    avg_clearance   = result_df.loc[result_df["markdown_applied"], "clearance_rate"].mean()

    annualised_gain = total_rev_gain * (365 / 28)
    enterprise_gain = annualised_gain * 470

    return {
        "policy"                       : "Price-Elasticity Markdown Optimisation",
        "n_items_marked_down"          : int(n_items_marked),
        "pct_items_marked_down"        : round(n_items_marked / len(result_df) * 100, 1),
        "avg_markdown_depth_pct"       : round(float(avg_depth), 1) if not np.isnan(avg_depth) else 0.0,
        "avg_clearance_rate_pct"       : round(float(avg_clearance * 100), 1) if not np.isnan(avg_clearance) else 0.0,
        "total_excess_units"           : round(float(total_excess), 0),
        "revenue_gain_28d_usd"         : round(total_rev_gain, 2),
        "annualised_gain_10stores_usd" : round(annualised_gain, 0),
        "enterprise_gain_usd"          : round(enterprise_gain, 0),
        "item_detail_df"               : result_df,
    }
