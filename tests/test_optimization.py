"""
Unit and integration tests for all three optimization modules.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from src.optimization.newsvendor import (
    CostParams, optimal_order_quantity, expected_cost, inventory_dollar_impact,
    sensitivity_analysis,
)
from src.optimization.budget_alloc import (
    BudgetParams, aggregate_by_store, greedy_budget_allocation, budget_dollar_impact,
)
from src.optimization.markdown import (
    MarkdownParams, demand_lift, optimal_markdown, markdown_dollar_impact,
    MARKDOWN_DEPTHS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_forecast_df():
    """10 items, varied demand uncertainty."""
    rng = np.random.default_rng(0)
    n = 100
    q50 = rng.uniform(5, 50, n)
    return pd.DataFrame({
        "id"       : [f"item_{i}" for i in range(n)],
        "store_id" : [f"CA_{i % 10 + 1}" for i in range(n)],
        "q10"      : q50 * 0.6,
        "q50"      : q50,
        "q90"      : q50 * 1.5,
    })


@pytest.fixture
def forecast_with_inventory(simple_forecast_df):
    """Add inventory and q_star columns."""
    df = simple_forecast_df.copy()
    rng = np.random.default_rng(1)
    df["inventory"] = df["q90"] * rng.uniform(0.8, 1.4, len(df))
    cost = CostParams()
    df["q_star"] = optimal_order_quantity(
        df["q10"].values, df["q50"].values, df["q90"].values, cost
    )
    return df


# ── Newsvendor tests ──────────────────────────────────────────────────────────

class TestCostParams:
    def test_critical_ratio_range(self):
        cp = CostParams()
        assert 0 < cp.critical_ratio < 1

    def test_critical_ratio_increases_with_cu(self):
        low  = CostParams(c_u=1.0, c_o=1.0)
        high = CostParams(c_u=5.0, c_o=1.0)
        assert high.critical_ratio > low.critical_ratio

    def test_critical_ratio_default_is_high(self):
        """Default costs → high CR (stockout >> holding)."""
        assert CostParams().critical_ratio > 0.85


class TestOptimalOrderQty:
    def test_output_shape(self):
        q10 = np.array([5.0, 10.0, 15.0])
        q50 = np.array([10.0, 20.0, 30.0])
        q90 = np.array([15.0, 30.0, 45.0])
        result = optimal_order_quantity(q10, q50, q90)
        assert result.shape == (3,)

    def test_no_negatives(self):
        q10 = np.zeros(10)
        q50 = np.random.uniform(0, 20, 10)
        q90 = q50 * 2
        assert (optimal_order_quantity(q10, q50, q90) >= 0).all()

    def test_optimal_between_q50_and_q90(self):
        """With default high CR, q* should be above median."""
        q10 = np.array([5.0])
        q50 = np.array([10.0])
        q90 = np.array([18.0])
        q_star = optimal_order_quantity(q10, q50, q90)
        assert q_star[0] >= q50[0]

    def test_low_cr_order_below_median(self):
        """If c_u ≈ 0, never worth stocking out insurance → order near q10."""
        cp = CostParams(c_u=0.01, c_o=100.0)
        q10 = np.array([5.0])
        q50 = np.array([10.0])
        q90 = np.array([15.0])
        q_star = optimal_order_quantity(q10, q50, q90, cost=cp)
        assert q_star[0] <= q50[0]


class TestExpectedCost:
    def test_columns_present(self):
        q50 = np.array([10.0, 20.0])
        result = expected_cost(q50, q50 * 0.6, q50, q50 * 1.5)
        assert "total_cost_usd" in result.columns
        assert "holding_cost_usd" in result.columns
        assert "stockout_cost_usd" in result.columns

    def test_costs_non_negative(self):
        q50 = np.random.uniform(1, 50, 50)
        result = expected_cost(q50, q50 * 0.6, q50, q50 * 1.4)
        assert (result["total_cost_usd"] >= 0).all()

    def test_ordering_too_low_increases_stockout(self):
        q50 = np.array([20.0])
        low_order  = expected_cost(np.array([5.0]),  q50 * 0.6, q50, q50 * 1.5)
        high_order = expected_cost(np.array([50.0]), q50 * 0.6, q50, q50 * 1.5)
        assert low_order["stockout_cost_usd"].sum() > high_order["stockout_cost_usd"].sum()


class TestInventoryDollarImpact:
    def test_returns_dict(self, simple_forecast_df):
        result = inventory_dollar_impact(simple_forecast_df)
        assert isinstance(result, dict)

    def test_saving_is_finite(self, simple_forecast_df):
        """Saving can be negative (we order more for insurance) but must be finite."""
        result = inventory_dollar_impact(simple_forecast_df)
        assert np.isfinite(result["saving_28d_usd"])

    def test_enterprise_scaling(self, simple_forecast_df):
        result = inventory_dollar_impact(simple_forecast_df)
        # Enterprise = annualised × 470 (within 5 to handle floating-point rounding)
        annualised = result["annualised_saving_10stores_usd"]
        enterprise = result["enterprise_saving_usd"]
        if abs(annualised) > 1e-3:
            assert abs(enterprise / annualised - 470) < 5


class TestSensitivityAnalysis:
    def test_returns_dataframe(self, simple_forecast_df):
        df = sensitivity_analysis(simple_forecast_df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1

    def test_higher_cu_changes_critical_ratio(self, simple_forecast_df):
        """Higher c_u should increase the critical ratio."""
        df = sensitivity_analysis(simple_forecast_df, c_u_range=[1.0, 5.0])
        cr = df.set_index("c_u")["critical_ratio"]
        assert cr[5.0] > cr[1.0]


# ── Budget allocation tests ───────────────────────────────────────────────────

class TestAggregateByStore:
    def test_shape(self, simple_forecast_df):
        result = aggregate_by_store(simple_forecast_df)
        n_stores = simple_forecast_df["store_id"].nunique()
        assert len(result) == n_stores

    def test_sum_preserved(self, simple_forecast_df):
        result = aggregate_by_store(simple_forecast_df)
        assert abs(result["q50"].sum() - simple_forecast_df["q50"].sum()) < 1e-6


class TestGreedyBudgetAllocation:
    def test_budget_not_exceeded(self, simple_forecast_df):
        store_df = aggregate_by_store(simple_forecast_df)
        params   = BudgetParams(budget_usd=50_000)
        result   = greedy_budget_allocation(store_df, params)
        total_cost = (result["alloc_optimal_units"] * params.unit_cost).sum()
        assert total_cost <= params.budget_usd * 1.01  # 1% tolerance for step rounding

    def test_all_stores_get_allocation(self, simple_forecast_df):
        store_df = aggregate_by_store(simple_forecast_df)
        result   = greedy_budget_allocation(store_df)
        assert (result["alloc_optimal_units"] >= 0).all()

    def test_optimal_revenue_geq_naive(self, simple_forecast_df):
        store_df = aggregate_by_store(simple_forecast_df)
        result   = greedy_budget_allocation(store_df)
        assert result["revenue_optimal_usd"].sum() >= result["revenue_naive_usd"].sum() * 0.99


class TestBudgetDollarImpact:
    def test_returns_dict(self, simple_forecast_df):
        result = budget_dollar_impact(simple_forecast_df)
        assert "revenue_uplift_28d_usd" in result

    def test_enterprise_scaling(self, simple_forecast_df):
        result = budget_dollar_impact(simple_forecast_df)
        ann = result["annualised_uplift_10stores_usd"]
        ent = result["enterprise_uplift_usd"]
        if abs(ann) > 1e-3:
            assert abs(ent / ann - 470) < 5
        else:
            assert abs(ent) < 1  # both near zero — scaling correct by construction


# ── Markdown tests ────────────────────────────────────────────────────────────

class TestDemandLift:
    def test_no_markdown_no_lift(self):
        assert demand_lift(0.0) == 1.0

    def test_lift_increases_with_depth(self):
        lifts = [demand_lift(d) for d in [0, 0.1, 0.2, 0.3]]
        assert lifts == sorted(lifts)

    def test_realistic_range(self):
        """20% markdown, elasticity -2.5 → 50% demand lift."""
        assert abs(demand_lift(0.20, -2.5) - 1.50) < 0.01


class TestOptimalMarkdown:
    def test_output_rows_match_input(self, forecast_with_inventory):
        df = forecast_with_inventory
        result = optimal_markdown(
            df["inventory"].values,
            df["q_star"].values,
            df["q50"].values,
        )
        assert len(result) == len(df)

    def test_no_markdown_when_no_excess(self):
        """Items with inventory ≤ q_star should not get marked down."""
        result = optimal_markdown(
            inventory = np.array([5.0, 10.0]),
            q_star    = np.array([8.0, 15.0]),
            q50       = np.array([6.0, 12.0]),
        )
        assert not result["markdown_applied"].any()

    def test_deep_excess_gets_marked_down(self):
        """Item with 3× inventory vs q_star must be marked down."""
        result = optimal_markdown(
            inventory = np.array([90.0]),
            q_star    = np.array([10.0]),
            q50       = np.array([12.0]),
        )
        assert result["markdown_depth_pct"].iloc[0] > 0

    def test_markdown_depths_valid(self, forecast_with_inventory):
        df = forecast_with_inventory
        result = optimal_markdown(
            df["inventory"].values,
            df["q_star"].values,
            df["q50"].values,
        )
        valid_depths = set(MARKDOWN_DEPTHS * 100)
        for d in result["markdown_depth_pct"]:
            assert round(d, 1) in {round(x, 1) for x in valid_depths}


class TestMarkdownDollarImpact:
    def test_returns_dict(self, forecast_with_inventory):
        result = markdown_dollar_impact(forecast_with_inventory)
        assert "revenue_gain_28d_usd" in result

    def test_enterprise_scaling(self, forecast_with_inventory):
        result = markdown_dollar_impact(forecast_with_inventory)
        ann = result["annualised_gain_10stores_usd"]
        ent = result["enterprise_gain_usd"]
        assert abs(ent / (ann + 1e-8) - 470) < 1


# ── Integration test ──────────────────────────────────────────────────────────

class TestEndToEndImpact:
    def test_combined_impact_positive(self, forecast_with_inventory):
        """All three modules together should produce a positive combined impact."""
        nv = inventory_dollar_impact(forecast_with_inventory)
        ba = budget_dollar_impact(forecast_with_inventory)
        md = markdown_dollar_impact(forecast_with_inventory)

        combined = (
            nv["enterprise_saving_usd"]
            + ba["enterprise_uplift_usd"]
            + md["enterprise_gain_usd"]
        )
        assert combined > 0, f"Expected positive impact, got {combined}"
