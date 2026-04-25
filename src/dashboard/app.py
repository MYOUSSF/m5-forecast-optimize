"""
Streamlit Dashboard — M5 Forecast Optimization
================================================
Interactive display of forecast outputs and all three optimization modules
with dollar-value impact at item, store, and enterprise scale.

Run with:
    streamlit run src/dashboard/app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="M5 Forecast Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .metric-card {
        background: #0f1117;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #48bb78;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .section-header {
        border-left: 3px solid #48bb78;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data(output_dir: str = "outputs"):
    out = Path(output_dir)
    data = {}

    report_path = out / "optimization_report.json"
    if report_path.exists():
        with open(report_path) as f:
            data["report"] = json.load(f)

    forecast_path = out / "forecasts.parquet"
    if forecast_path.exists():
        data["forecasts"] = pd.read_parquet(forecast_path)
        data["forecasts"]["date"] = pd.to_datetime(data["forecasts"]["date"])

    store_path = out / "store_allocations.csv"
    if store_path.exists():
        data["store_alloc"] = pd.read_csv(store_path)

    md_path = out / "markdown_items.csv"
    if md_path.exists():
        data["markdown"] = pd.read_csv(md_path)

    return data


# ── Colour palette ────────────────────────────────────────────────────────────
C_GREEN  = "#48bb78"
C_BLUE   = "#63b3ed"
C_ORANGE = "#f6ad55"
C_RED    = "#fc8181"
C_GRAY   = "#4a5568"
PLOTLY_TEMPLATE = "plotly_dark"


def fmt_dollar(v):
    if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("## ⚙️ Settings")
    output_dir = st.sidebar.text_input("Output directory", value="outputs")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "End-to-end M5 demand forecasting + inventory optimisation. "
        "Three modules translate probabilistic forecasts into dollar impact."
    )
    st.sidebar.markdown("**Modules**")
    st.sidebar.markdown("- 📦 Newsvendor (order qty)")
    st.sidebar.markdown("- 💰 Budget Allocation (store-level)")
    st.sidebar.markdown("- 🏷️ Markdown Scheduling (clearance)")
    return output_dir


# ── KPI cards ─────────────────────────────────────────────────────────────────

def kpi_card(label, value, delta=None):
    delta_html = f'<div style="color:#68d391;font-size:0.8rem">▲ {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── Forecast tab ──────────────────────────────────────────────────────────────

def tab_forecast(data):
    st.markdown('<div class="section-header"><h3>📈 28-Day Demand Forecasts</h3></div>',
                unsafe_allow_html=True)

    if "forecasts" not in data:
        st.warning("Run the pipeline first: `python run_pipeline.py`")
        return

    fc = data["forecasts"]

    # Item selector
    items = fc["id"].unique()
    selected = st.selectbox("Select item-store series", items[:50])
    item_fc = fc[fc["id"] == selected].sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=item_fc["date"], y=item_fc["q90"],
        fill=None, mode="lines", line=dict(width=0),
        name="q90", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=item_fc["date"], y=item_fc["q10"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(99,179,237,0.15)", name="80% PI"
    ))
    fig.add_trace(go.Scatter(
        x=item_fc["date"], y=item_fc["q50"],
        mode="lines+markers", line=dict(color=C_BLUE, width=2),
        marker=dict(size=4), name="Median forecast"
    ))
    if "sales" in item_fc.columns:
        fig.add_trace(go.Scatter(
            x=item_fc["date"], y=item_fc["sales"],
            mode="markers", marker=dict(color=C_GREEN, size=5, symbol="x"),
            name="Actual sales"
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=350,
        title=f"Demand forecast — {selected}",
        xaxis_title="Date", yaxis_title="Units",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast distribution
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(fc, x="q50", nbins=60,
                             title="Distribution of Median Forecasts",
                             color_discrete_sequence=[C_BLUE],
                             template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fc["pi_width"] = fc["q90"] - fc["q10"]
        fig3 = px.histogram(fc, x="pi_width", nbins=60,
                             title="Forecast Uncertainty (PI Width)",
                             color_discrete_sequence=[C_ORANGE],
                             template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig3, use_container_width=True)


# ── Newsvendor tab ────────────────────────────────────────────────────────────

def tab_newsvendor(data):
    st.markdown('<div class="section-header"><h3>📦 Module 1: Newsvendor Inventory Optimization</h3></div>',
                unsafe_allow_html=True)

    if "report" not in data:
        st.warning("No report found. Run the pipeline first.")
        return

    nv = data["report"]["newsvendor"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("28-Day Cost Saving", fmt_dollar(nv["saving_28d_usd"]),
                       f"{nv['saving_pct']}% reduction")
    with c2: kpi_card("Annual (10 stores)", fmt_dollar(nv["annualised_saving_10stores_usd"]))
    with c3: kpi_card("Enterprise Annual", fmt_dollar(nv["enterprise_saving_usd"]),
                       "scaled to 4,700 stores")
    with c4: kpi_card("Service Level Target", f"{nv['service_level_target']}%",
                       f"CR = {nv['critical_ratio']}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=["Naive (q50)", "Newsvendor Optimal"],
            y=[nv["naive_total_cost_usd"], nv["optimal_total_cost_usd"]],
            marker_color=[C_RED, C_GREEN],
            text=[fmt_dollar(nv["naive_total_cost_usd"]),
                  fmt_dollar(nv["optimal_total_cost_usd"])],
            textposition="outside",
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                           title="Total Inventory Cost: Naive vs Optimal",
                           yaxis_title="Cost ($)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "forecasts" in data:
            fc = data["forecasts"]
            fig2 = px.scatter(
                fc.sample(min(2000, len(fc))),
                x="q50", y="q_star",
                color="q90",
                color_continuous_scale="Viridis",
                title="Optimal Order Qty vs Median Forecast",
                labels={"q50": "Median Demand Forecast",
                        "q_star": "Optimal Order Qty (q*)",
                        "q90": "q90"},
                template=PLOTLY_TEMPLATE, height=320,
                opacity=0.6,
            )
            fig2.add_shape(type="line", x0=0, y0=0,
                           x1=fc["q50"].max(), y1=fc["q50"].max(),
                           line=dict(color=C_GRAY, dash="dash"))
            st.plotly_chart(fig2, use_container_width=True)

    st.info(
        f"**Interpretation:** The newsvendor model uses the critical ratio "
        f"CR = {nv['critical_ratio']} (derived from holding vs stockout costs) "
        f"to set order quantities at the {nv['service_level_target']}th percentile of demand. "
        f"Points above the dashed line = items stocked above median → insurance against stockout."
    )


# ── Budget allocation tab ─────────────────────────────────────────────────────

def tab_budget(data):
    st.markdown('<div class="section-header"><h3>💰 Module 2: Store Budget Allocation</h3></div>',
                unsafe_allow_html=True)

    if "report" not in data:
        st.warning("No report found.")
        return

    ba = data["report"]["budget_allocation"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("28-Day Revenue Uplift", fmt_dollar(ba["revenue_uplift_28d_usd"]),
                       f"+{ba['revenue_uplift_pct']}%")
    with c2: kpi_card("Annual (10 stores)", fmt_dollar(ba["annualised_uplift_10stores_usd"]))
    with c3: kpi_card("Enterprise Annual", fmt_dollar(ba["enterprise_uplift_usd"]))
    with c4: kpi_card("Budget Deployed", fmt_dollar(ba["budget_usd"]))

    st.markdown("---")

    if "store_alloc" in data:
        sa = data["store_alloc"]
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Naive (proportional)",
                                  x=sa["store_id"],
                                  y=sa["alloc_naive_units"],
                                  marker_color=C_GRAY))
            fig.add_trace(go.Bar(name="Optimal (MR/$)",
                                  x=sa["store_id"],
                                  y=sa["alloc_optimal_units"],
                                  marker_color=C_GREEN))
            fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE,
                               height=320, title="Units Allocated per Store",
                               xaxis_title="Store", yaxis_title="Units")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Naive Revenue",
                                   x=sa["store_id"],
                                   y=sa["revenue_naive_usd"],
                                   marker_color=C_GRAY))
            fig2.add_trace(go.Bar(name="Optimal Revenue",
                                   x=sa["store_id"],
                                   y=sa["revenue_optimal_usd"],
                                   marker_color=C_BLUE))
            fig2.update_layout(barmode="group", template=PLOTLY_TEMPLATE,
                                height=320, title="Expected Revenue per Store",
                                xaxis_title="Store", yaxis_title="Revenue ($)")
            st.plotly_chart(fig2, use_container_width=True)

        sa["realloc_delta"] = sa["alloc_optimal_units"] - sa["alloc_naive_units"]
        fig3 = px.bar(sa, x="store_id", y="realloc_delta",
                       color="realloc_delta",
                       color_continuous_scale=["#fc8181", "#4a5568", "#48bb78"],
                       title="Reallocation Delta (Optimal − Naive) per Store",
                       template=PLOTLY_TEMPLATE, height=280)
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.info(
        f"**Interpretation:** Greedy marginal-revenue-per-dollar allocation "
        f"concentrates inventory on high-demand stores. "
        f"Top gaining store: **{ba['top_gaining_store']}** | "
        f"Top reducing store: **{ba['top_losing_store']}**."
    )


# ── Markdown tab ──────────────────────────────────────────────────────────────

def tab_markdown(data):
    st.markdown('<div class="section-header"><h3>🏷️ Module 3: Markdown / Clearance Scheduling</h3></div>',
                unsafe_allow_html=True)

    if "report" not in data:
        st.warning("No report found.")
        return

    md = data["report"]["markdown"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Items Marked Down", f"{md['n_items_marked_down']:,}",
                       f"{md['pct_items_marked_down']}% of items")
    with c2: kpi_card("Avg Markdown Depth", f"{md['avg_markdown_depth_pct']}%")
    with c3: kpi_card("28-Day Revenue Gain", fmt_dollar(md["revenue_gain_28d_usd"]))
    with c4: kpi_card("Enterprise Annual", fmt_dollar(md["enterprise_gain_usd"]))

    st.markdown("---")

    if "markdown" in data:
        mk = data["markdown"]
        col1, col2 = st.columns(2)

        with col1:
            depth_counts = (mk[mk["markdown_applied"]]
                            ["markdown_depth_pct"]
                            .value_counts()
                            .reset_index())
            depth_counts.columns = ["depth_pct", "count"]
            fig = px.bar(depth_counts.sort_values("depth_pct"),
                          x="depth_pct", y="count",
                          color="depth_pct",
                          color_continuous_scale="Oranges",
                          title="Markdown Depth Distribution",
                          labels={"depth_pct": "Markdown Depth (%)",
                                  "count": "# Items"},
                          template=PLOTLY_TEMPLATE, height=320)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sample = mk[mk["markdown_applied"]].sample(min(2000, len(mk[mk["markdown_applied"]])))
            fig2 = px.scatter(sample,
                               x="excess_units", y="rev_gain_usd",
                               color="markdown_depth_pct",
                               color_continuous_scale="RdYlGn",
                               title="Revenue Gain vs Excess Inventory",
                               labels={"excess_units": "Excess Units",
                                       "rev_gain_usd": "Revenue Gain ($)",
                                       "markdown_depth_pct": "Depth %"},
                               template=PLOTLY_TEMPLATE, height=320,
                               opacity=0.7)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(mk[mk["markdown_applied"]], x="clearance_rate",
                             nbins=40, title="Clearance Rate Distribution",
                             color_discrete_sequence=[C_ORANGE],
                             template=PLOTLY_TEMPLATE, height=250)
        st.plotly_chart(fig3, use_container_width=True)


# ── Combined impact tab ───────────────────────────────────────────────────────

def tab_combined(data):
    st.markdown('<div class="section-header"><h3>🎯 Combined Business Impact</h3></div>',
                unsafe_allow_html=True)

    if "report" not in data:
        st.warning("No report found.")
        return

    r  = data["report"]
    nv = r["newsvendor"]
    ba = r["budget_allocation"]
    md = r["markdown"]

    total = r["combined_enterprise_annual_usd"]
    kpi_card("TOTAL ENTERPRISE ANNUAL IMPACT", fmt_dollar(total),
             "All 3 modules, scaled to 4,700 stores")

    st.markdown("---")

    modules  = ["Newsvendor\n(Inventory)", "Budget\nAllocation", "Markdown\nScheduling"]
    ann_vals = [
        nv["enterprise_saving_usd"],
        ba["enterprise_uplift_usd"],
        md["enterprise_gain_usd"],
    ]

    fig = go.Figure(go.Bar(
        x=modules, y=ann_vals,
        marker_color=[C_GREEN, C_BLUE, C_ORANGE],
        text=[fmt_dollar(v) for v in ann_vals],
        textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=380,
        title="Enterprise Annual Impact by Module (scaled to 4,700 US Walmart stores)",
        yaxis_title="Annual Impact ($)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Waterfall
    fig2 = go.Figure(go.Waterfall(
        name="Impact",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Newsvendor", "Budget Allocation", "Markdown", "Total"],
        y=[nv["enterprise_saving_usd"], ba["enterprise_uplift_usd"],
           md["enterprise_gain_usd"], 0],
        connector={"line": {"color": C_GRAY}},
        increasing={"marker": {"color": C_GREEN}},
        totals={"marker": {"color": C_BLUE}},
        text=[fmt_dollar(v) for v in ann_vals + [total]],
    ))
    fig2.update_layout(template=PLOTLY_TEMPLATE, height=350,
                        title="Cumulative Impact Waterfall")
    st.plotly_chart(fig2, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown("# 📦 M5 Forecast Optimization Dashboard")
    st.markdown("*Demand forecasting → inventory optimization → dollar impact*")
    st.markdown("---")

    output_dir = sidebar()
    data = load_data(output_dir)

    if not data:
        st.error("No output data found. Run the pipeline first:\n\n"
                 "```bash\npython run_pipeline.py --n_items 500\n```")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecasts",
        "📦 Newsvendor",
        "💰 Budget Allocation",
        "🏷️ Markdown",
        "🎯 Combined Impact",
    ])

    with tab1: tab_forecast(data)
    with tab2: tab_newsvendor(data)
    with tab3: tab_budget(data)
    with tab4: tab_markdown(data)
    with tab5: tab_combined(data)


if __name__ == "__main__":
    main()
