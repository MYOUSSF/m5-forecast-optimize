"""
Module 2 — Item-Store Level Budget Allocation (LP)
===================================================
Given a fixed replenishment budget B and probabilistic demand forecasts,
allocate inventory spend across all item-store pairs to maximise expected
revenue subject to the budget constraint.

Improvements over v1
--------------------
1. Item-store level    : operates on (item_id, store_id) pairs instead of
                         aggregating to store totals — preserves price
                         variation across items.

2. Item-specific prices: revenue and budget constraint both use each item's
                         actual sell_price from the forecast DataFrame.

3. Exact LP solver     : the concave piecewise-linear objective is reformulated
                         as a linear program and solved exactly via
                         scipy.optimize.linprog (HiGHS backend), replacing the
                         discretisation-limited greedy algorithm.

LP Formulation
--------------
Introduce auxiliary variable e_i = E[min(Q_i, D_i)] for each item.
Maximising a concave piecewise-linear function is equivalent to:

    max  Σ_i  sell_price_i × e_i

    s.t. — piecewise-linear upper bounds on e_i (one per segment):
           e_i ≤ Q_i                                        [slope 1.00, Q ≤ q10]
           e_i ≤ q10_i + (Q_i − q10_i) × 0.80             [slope 0.80, q10–q50]
           e_i ≤ e50_i + (Q_i − q50_i) × 0.35             [slope 0.35, q50–q90]
           e_i ≤ e90_i + (Q_i − q90_i) × 0.10             [slope 0.10, Q > q90]

         — budget constraint:
           Σ_i  unit_cost_i × Q_i  ≤  B

         — non-negativity:
           Q_i ≥ 0,  e_i ≥ 0   ∀ i

Since we are maximising a concave objective, the solver automatically picks
the tightest binding segment for each e_i — no integer variables needed.

The dual variable on the budget constraint gives the shadow price: how much
an extra dollar of budget is worth in revenue.

Dollar impact: baseline = allocate B proportional to q50 × sell_price
               optimal  = LP solution
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from typing import Optional
from dataclasses import dataclass

from src.data.features import HOLDING_COST_RATE


@dataclass
class BudgetParams:
    budget_usd:  float = 500_000.0              # total replenishment budget ($)
    margin_rate: float = 1.0 - HOLDING_COST_RATE  # COGS as fraction of sell_price


# ── Expected revenue: E[min(Q, D)] × sell_price ───────────────────────────────

def expected_revenue_item(order_qty:  np.ndarray,
                           q10:        np.ndarray,
                           q50:        np.ndarray,
                           q90:        np.ndarray,
                           sell_price: np.ndarray) -> np.ndarray:
    """
    Approximate sell_price × E[min(Q, D)] for each item using piecewise-linear
    interpolation across the three known CDF points (0.10, 0.50, 0.90).

    The approximation is constructed so that:
      - It is non-decreasing in Q (more stock → weakly more revenue)
      - It is concave in Q (diminishing marginal returns)
      - It never hard-caps: marginal revenue stays positive beyond q90,
        decaying gradually toward zero as Q → ∞

    Piecewise regions
    -----------------
    Q ≤ q10 : sell almost certainly — E[min] ≈ Q
    q10 < Q ≤ q50 : interpolate with slope 0.80 (80% chance of selling marginal unit)
    q50 < Q ≤ q90 : interpolate with slope 0.35 (35% chance)
    Q > q90        : slope decays as 0.10 × q90 / Q  (ensures concavity, no hard cap)

    The slopes (0.80, 0.35, 0.10) are derived from the implied CDF:
      - Between q10 and q50 the CDF rises from 0.10 to 0.50  → avg(1-F) ≈ 0.80
      - Between q50 and q90 the CDF rises from 0.50 to 0.90  → avg(1-F) ≈ 0.35
      - Beyond q90 the survival function 1-F decays toward 0
    """
    order_qty  = np.asarray(order_qty,  dtype=float)
    q10, q50, q90 = map(np.asarray, (q10, q50, q90))
    sell_price = np.asarray(sell_price, dtype=float)

    # Anchor values of E[min(Q,D)] at each quantile boundary
    e_at_q10 = q10                              # slope=1 up to q10
    e_at_q50 = e_at_q10 + (q50 - q10) * 0.80
    e_at_q90 = e_at_q50 + (q90 - q50) * 0.35

    # Beyond q90: marginal E[min] = 0.10 × q90 / Q  (integrates to log, always > 0)
    # E[min(Q,D)] ≈ e_at_q90 + 0.10 × q90 × ln(Q / q90)  for Q > q90
    safe_q90 = np.maximum(q90, 1e-6)
    safe_qty = np.maximum(order_qty, 1e-6)

    e_sales = np.where(
        order_qty <= q10,
        order_qty,
        np.where(
            order_qty <= q50,
            e_at_q10 + (order_qty - q10) * 0.80,
            np.where(
                order_qty <= q90,
                e_at_q50 + (order_qty - q50) * 0.35,
                e_at_q90 + 0.10 * safe_q90 * np.log(safe_qty / safe_q90)
            )
        )
    )
    # E[min(Q,D)] is bounded above by q90 + small log term — clip at inventory ceiling
    e_sales = np.maximum(e_sales, 0.0)

    return sell_price * e_sales


# ── Naive baseline: allocate proportional to expected revenue ─────────────────

def _naive_allocation(q50:        np.ndarray,
                       sell_price: np.ndarray,
                       unit_cost:  np.ndarray,
                       budget:     float) -> np.ndarray:
    """
    Baseline: allocate budget proportional to q50 × sell_price (expected revenue
    at median demand). Uses item-specific prices throughout.
    """
    weights = q50 * sell_price
    total   = weights.sum()
    if total < 1e-9:
        return np.zeros_like(q50)
    budget_per_item = (weights / total) * budget
    return budget_per_item / np.maximum(unit_cost, 1e-9)


# ── LP budget allocation ──────────────────────────────────────────────────────

def lp_budget_allocation(forecast_df: pd.DataFrame,
                          params: Optional[BudgetParams] = None) -> pd.DataFrame:
    """
    Solve the budget allocation problem as a linear program using
    scipy.optimize.linprog (HiGHS backend).

    Decision variables  (length 2n, ordered [Q_0..Q_{n-1}, e_0..e_{n-1}]):
      Q_i  : units allocated to item i
      e_i  : auxiliary variable = E[min(Q_i, D_i)]  (expected sales)

    Objective (minimise negative revenue, since linprog minimises):
      min  −Σ_i  sell_price_i × e_i

    Constraints:
      4n inequality constraints — one per piecewise segment per item:
        e_i − Q_i              ≤  0               [segment 1: slope 1.00]
        e_i − 0.80 × Q_i      ≤  e10_i − 0.80 × q10_i   [segment 2: slope 0.80]
        e_i − 0.35 × Q_i      ≤  e50_i − 0.35 × q50_i   [segment 3: slope 0.35]
        e_i − 0.10 × Q_i      ≤  e90_i − 0.10 × q90_i   [segment 4: slope 0.10]

      1 budget constraint:
        Σ_i  unit_cost_i × Q_i  ≤  B

    Bounds:
      Q_i ≥ 0,  e_i ≥ 0

    The dual variable on the budget constraint is the shadow price:
    the marginal revenue of an additional dollar of budget.

    Parameters
    ----------
    forecast_df : DataFrame with columns
                  [id, item_id, store_id, q10, q50, q90, sell_price]
    params      : BudgetParams

    Returns
    -------
    DataFrame with one row per item-store pair.
    """
    if params is None:
        params = BudgetParams()

    df = forecast_df.copy().reset_index(drop=True)
    n  = len(df)

    sell_price = df["sell_price"].values.astype(float)
    unit_cost  = np.maximum(sell_price * (1.0 - params.margin_rate), 0.01)
    q10 = df["q10"].values.astype(float)
    q50 = df["q50"].values.astype(float)
    q90 = df["q90"].values.astype(float)

    # ── Anchor values of E[min(Q,D)] at each quantile boundary ───────────────
    e_at_q10 = q10
    e_at_q50 = e_at_q10 + (q50 - q10) * 0.80
    e_at_q90 = e_at_q50 + (q90 - q50) * 0.35

    # ── Objective: minimise −sell_price × e  (linprog minimises) ─────────────
    # Variables: [Q_0..Q_{n-1}, e_0..e_{n-1}]
    c = np.concatenate([np.zeros(n), -sell_price])

    # ── Inequality constraints A_ub @ x ≤ b_ub ───────────────────────────────
    # 4 segment constraints per item + 1 budget constraint
    # Build as sparse for efficiency with 30K items

    row_idx, col_idx, data = [], [], []
    b_ub = []

    for seg, (slope_Q, slope_e, rhs_offset) in enumerate([
        # e_i - slope_Q * Q_i ≤ rhs_offset_i
        # rearranged from:  e_i ≤ anchor + slope * (Q_i - breakpoint)
        (1.00, 1.0,  0.0),                          # seg 1: e ≤ Q
        (0.80, 1.0,  e_at_q10 - 0.80 * q10),        # seg 2: e ≤ e10 + 0.80(Q-q10)
        (0.35, 1.0,  e_at_q50 - 0.35 * q50),        # seg 3: e ≤ e50 + 0.35(Q-q50)
        (0.10, 1.0,  e_at_q90 - 0.10 * q90),        # seg 4: e ≤ e90 + 0.10(Q-q90)
    ]):
        base_row = seg * n
        for i in range(n):
            r = base_row + i
            # coefficient on Q_i
            row_idx.append(r);  col_idx.append(i);   data.append(-slope_Q)
            # coefficient on e_i
            row_idx.append(r);  col_idx.append(n+i); data.append(slope_e)
        rhs = rhs_offset if np.ndim(rhs_offset) > 0 else np.full(n, rhs_offset)
        b_ub.extend(rhs.tolist())

    # Budget constraint: Σ unit_cost_i × Q_i ≤ B
    budget_row = 4 * n
    for i in range(n):
        row_idx.append(budget_row); col_idx.append(i); data.append(unit_cost[i])
    b_ub.append(params.budget_usd)

    A_ub = csr_matrix((data, (row_idx, col_idx)), shape=(4 * n + 1, 2 * n))
    b_ub = np.array(b_ub)

    # ── Bounds: Q_i ≥ 0, e_i ≥ 0 ─────────────────────────────────────────────
    bounds = [(0, None)] * (2 * n)

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={"disp": False},
    )

    if result.status != 0:
        raise RuntimeError(f"LP did not converge: {result.message}")

    alloc = result.x[:n]

    # Shadow price on budget constraint (dual variable, last row of y)
    # linprog returns ineq_marginals for HiGHS
    shadow_price = float(result.ineqlin.marginals[-1]) if hasattr(result, "ineqlin") else None

    # ── Naive baseline ────────────────────────────────────────────────────────
    naive_alloc = _naive_allocation(q50, sell_price, unit_cost, params.budget_usd)

    # ── Build results ─────────────────────────────────────────────────────────
    rev_naive   = expected_revenue_item(naive_alloc, q10, q50, q90, sell_price)
    rev_optimal = expected_revenue_item(alloc,       q10, q50, q90, sell_price)

    out = df[["id", "item_id", "store_id", "sell_price"]].copy()
    out["unit_cost"]             = unit_cost
    out["demand_q50"]            = q50
    out["alloc_naive_units"]     = naive_alloc
    out["alloc_optimal_units"]   = alloc
    out["revenue_naive_usd"]     = rev_naive
    out["revenue_optimal_usd"]   = rev_optimal
    out["cost_naive_usd"]        = naive_alloc * unit_cost
    out["cost_optimal_usd"]      = alloc       * unit_cost
    out["realloc_delta_units"]   = alloc - naive_alloc
    out["shadow_price_per_dollar"] = shadow_price  # same for all rows

    return out


# ── Store-level rollup ────────────────────────────────────────────────────────

def rollup_by_store(item_result: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level allocation results to store level for reporting."""
    return (item_result
            .groupby("store_id")
            .agg(
                alloc_naive_units   = ("alloc_naive_units",   "sum"),
                alloc_optimal_units = ("alloc_optimal_units", "sum"),
                revenue_naive_usd   = ("revenue_naive_usd",   "sum"),
                revenue_optimal_usd = ("revenue_optimal_usd", "sum"),
                cost_naive_usd      = ("cost_naive_usd",      "sum"),
                cost_optimal_usd    = ("cost_optimal_usd",    "sum"),
                n_items             = ("item_id",             "count"),
            )
            .reset_index())


# ── Dollar impact summary ─────────────────────────────────────────────────────

def budget_dollar_impact(forecast_df: pd.DataFrame,
                          params: Optional[BudgetParams] = None) -> dict:
    """
    Full pipeline: LP allocation → dollar impact summary.
    forecast_df must contain: id, item_id, store_id, q10, q50, q90, sell_price.

    Returns summary dict suitable for MLflow logging / dashboard display,
    plus item_detail_df and store_rollup_df for deeper inspection.

    Key extra output vs v1
    ----------------------
    shadow_price_per_dollar : marginal revenue of an extra $1 of budget.
                              E.g. 0.85 means an extra $1,000 budget yields ~$850
                              in additional expected revenue.
    """
    if params is None:
        params = BudgetParams()

    item_result  = lp_budget_allocation(forecast_df, params)
    store_result = rollup_by_store(item_result)

    total_rev_naive   = item_result["revenue_naive_usd"].sum()
    total_rev_optimal = item_result["revenue_optimal_usd"].sum()
    rev_uplift        = total_rev_optimal - total_rev_naive
    uplift_pct        = rev_uplift / (total_rev_naive + 1e-8) * 100

    top_gainer = (item_result
                  .groupby("store_id")["realloc_delta_units"]
                  .sum().idxmax())
    top_loser  = (item_result
                  .groupby("store_id")["realloc_delta_units"]
                  .sum().idxmin())

    n_items_reallocated = (item_result["realloc_delta_units"].abs() > 0.5).sum()
    shadow_price = item_result["shadow_price_per_dollar"].iloc[0]

    annualised_uplift = rev_uplift * (365 / 28)
    enterprise_uplift = annualised_uplift * 470

    return {
        "policy"                        : "LP — Item-Store Level (HiGHS)",
        "budget_usd"                    : params.budget_usd,
        "n_items"                       : len(item_result),
        "n_items_reallocated"           : int(n_items_reallocated),
        "total_revenue_naive_usd"       : round(total_rev_naive,   2),
        "total_revenue_optimal_usd"     : round(total_rev_optimal, 2),
        "revenue_uplift_28d_usd"        : round(rev_uplift,        2),
        "revenue_uplift_pct"            : round(uplift_pct,        2),
        "annualised_uplift_10stores_usd": round(annualised_uplift,  0),
        "enterprise_uplift_usd"         : round(enterprise_uplift,  0),
        "shadow_price_per_dollar"       : round(shadow_price, 4) if shadow_price else None,
        "top_gaining_store"             : top_gainer,
        "top_losing_store"              : top_loser,
        "item_detail_df"                : item_result,
        "store_rollup_df"               : store_result,
    }
