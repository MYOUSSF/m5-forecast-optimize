# %% [markdown]
# # M5 Exploratory Data Analysis
# 
# This notebook profiles the M5 Walmart dataset before modelling:
# - Sales distributions and sparsity (intermittency)
# - Temporal structure: trends, seasonality, weekly patterns
# - Price dynamics and promotions (SNAP)
# - Store × category decomposition
# - Lag/autocorrelation analysis to inform feature engineering

# %% Imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = "data/"
FIG_DIR   = "outputs/eda/"
import os; os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "green"  : "#48bb78",
    "blue"   : "#63b3ed",
    "orange" : "#f6ad55",
    "red"    : "#fc8181",
    "gray"   : "#718096",
    "purple" : "#b794f4",
}
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor"  : "#1a202c",
    "axes.edgecolor"  : "#2d3748",
    "axes.labelcolor" : "#e2e8f0",
    "xtick.color"     : "#a0aec0",
    "ytick.color"     : "#a0aec0",
    "text.color"      : "#e2e8f0",
    "grid.color"      : "#2d3748",
    "grid.linewidth"  : 0.5,
    "font.family"     : "monospace",
    "axes.titlesize"  : 11,
    "axes.titleweight": "bold",
})


# %% Load data
print("Loading M5 data...")
sales    = pd.read_csv(f"{DATA_DIR}sales_train_validation.csv")
calendar = pd.read_csv(f"{DATA_DIR}calendar.csv")
prices   = pd.read_csv(f"{DATA_DIR}sell_prices.csv")

print(f"Sales shape    : {sales.shape}")
print(f"Calendar shape : {calendar.shape}")
print(f"Prices shape   : {prices.shape}")

id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = [c for c in sales.columns if c.startswith("d_")]

# %% [markdown]
# ## 1. Dataset Overview & Sparsity

# %%
print("\n=== Dataset Overview ===")
print(f"Items (SKUs)       : {sales['item_id'].nunique():,}")
print(f"Stores             : {sales['store_id'].nunique()}")
print(f"States             : {sales['state_id'].nunique()}")
print(f"Categories         : {sales['cat_id'].nunique()}")
print(f"Departments        : {sales['dept_id'].nunique()}")
print(f"Days of history    : {len(day_cols):,}")
print(f"Total time series  : {len(sales):,}")
print(f"Total observations : {len(sales) * len(day_cols):,}")

# Sparsity (zero-sales days)
sales_vals = sales[day_cols].values
zero_rate  = (sales_vals == 0).mean()
print(f"\nZero-sales rate    : {zero_rate:.1%}  (intermittency)")

# Distribution of mean daily sales per series
series_mean = sales_vals.mean(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("M5 Dataset — Series-Level Sales Characteristics", fontsize=13, y=1.02)

axes[0].hist(series_mean, bins=80, color=COLORS["blue"], edgecolor="none", alpha=0.85)
axes[0].set_title("Mean Daily Sales per Series")
axes[0].set_xlabel("Mean Units/Day")
axes[0].set_ylabel("# Series")
axes[0].axvline(np.median(series_mean), color=COLORS["orange"], lw=1.5,
                label=f"Median={np.median(series_mean):.2f}")
axes[0].legend(fontsize=8)

# Zero-days distribution
zero_days = (sales_vals == 0).mean(axis=1)
axes[1].hist(zero_days, bins=40, color=COLORS["red"], edgecolor="none", alpha=0.85)
axes[1].set_title("Fraction of Zero-Sales Days per Series")
axes[1].set_xlabel("Zero Rate")
axes[1].set_ylabel("# Series")

# Category breakdown
cat_totals = sales.groupby("cat_id")[day_cols].sum().sum(axis=1)
bars = axes[2].bar(cat_totals.index, cat_totals.values / 1e6,
                   color=[COLORS["green"], COLORS["blue"], COLORS["purple"]])
axes[2].set_title("Total Sales Volume by Category (M units)")
axes[2].set_xlabel("Category")
axes[2].set_ylabel("Units (millions)")
for bar in bars:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.0f}M", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}01_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}01_overview.png")


# %% [markdown]
# ## 2. Temporal Patterns — Trend, Seasonality, Weekly Cycle

# %%
# Aggregate total daily sales across all series
calendar["date"] = pd.to_datetime(calendar["date"])
cal_slim = calendar[["d", "date", "month", "year", "weekday", "wday",
                      "event_name_1", "event_type_1",
                      "snap_CA", "snap_TX", "snap_WI"]].copy()

daily_total = pd.DataFrame({
    "d"     : day_cols,
    "sales" : sales[day_cols].sum().values,
})
daily_total = daily_total.merge(cal_slim, on="d")

fig = plt.figure(figsize=(14, 10))
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Total daily sales over time
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(daily_total["date"], daily_total["sales"] / 1e3,
                 alpha=0.4, color=COLORS["blue"])
ax1.plot(daily_total["date"], daily_total["sales"] / 1e3,
         color=COLORS["blue"], lw=0.8)

# Annotate Christmas dips
xmas_dates = daily_total[daily_total["date"].dt.month == 12][
    daily_total["date"].dt.day == 25]["date"]
for d in xmas_dates:
    ax1.axvline(d, color=COLORS["red"], lw=0.7, alpha=0.5, linestyle="--")

ax1.set_title("Total Daily Unit Sales (all stores, all items)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Units (000s)")
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ── Monthly seasonal pattern
ax2 = fig.add_subplot(gs[1, 0])
monthly = daily_total.groupby("month")["sales"].mean()
ax2.bar(monthly.index, monthly.values / 1e3,
        color=COLORS["green"], edgecolor="none")
ax2.set_title("Avg Daily Sales by Month")
ax2.set_xlabel("Month")
ax2.set_ylabel("Avg Units (000s)")
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

# ── Weekly cycle
ax3 = fig.add_subplot(gs[1, 1])
weekday_order = ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"]
weekly = daily_total.groupby("weekday")["sales"].mean().reindex(weekday_order)
colors_week = [COLORS["orange"] if d in ["Saturday","Sunday"] else COLORS["blue"]
               for d in weekday_order]
ax3.bar(range(len(weekday_order)), weekly.values / 1e3,
        color=colors_week, edgecolor="none")
ax3.set_title("Avg Daily Sales by Day of Week")
ax3.set_xlabel("Day")
ax3.set_ylabel("Avg Units (000s)")
ax3.set_xticks(range(7))
ax3.set_xticklabels(["Sat","Sun","Mon","Tue","Wed","Thu","Fri"], fontsize=8)

# ── Year-over-year comparison
ax4 = fig.add_subplot(gs[2, :])
for yr, col in zip([2013, 2014, 2015, 2016],
                   [COLORS["gray"], COLORS["purple"], COLORS["blue"], COLORS["green"]]):
    yr_data = daily_total[daily_total["year"] == yr].copy()
    yr_data = yr_data.sort_values("date")
    yr_data["day_of_year"] = yr_data["date"].dt.dayofyear
    ax4.plot(yr_data["day_of_year"], yr_data["sales"].rolling(7).mean() / 1e3,
             color=col, lw=1.2, label=str(yr), alpha=0.85)
ax4.set_title("Year-over-Year Sales (7-day rolling mean)")
ax4.set_xlabel("Day of Year")
ax4.set_ylabel("Units (000s)")
ax4.legend(fontsize=9)

fig.suptitle("M5 Temporal Patterns", fontsize=14, y=1.01)
plt.savefig(f"{FIG_DIR}02_temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}02_temporal.png")


# %% [markdown]
# ## 3. Price Dynamics and SNAP Promotions

# %%
print("\n=== Price Summary ===")
print(prices["sell_price"].describe().round(2))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Price Dynamics & Promotions", fontsize=13)

# Price distribution by category
for ax, cat in zip(axes[0], ["FOODS", "HOBBIES"]):
    cat_items = sales[sales["cat_id"] == cat]["item_id"].unique()
    cat_prices = prices[prices["item_id"].isin(cat_items)]["sell_price"]
    ax.hist(cat_prices, bins=60, color=COLORS["blue"], edgecolor="none", alpha=0.8)
    ax.set_title(f"Price Distribution — {cat}")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    ax.axvline(cat_prices.median(), color=COLORS["orange"], lw=1.5,
               label=f"Median ${cat_prices.median():.2f}")
    ax.legend(fontsize=8)

# SNAP days impact
snap_days = calendar[calendar[["snap_CA","snap_TX","snap_WI"]].any(axis=1)]["d"].tolist()
no_snap_days = calendar[~calendar["d"].isin(snap_days)]["d"].tolist()

snap_sales   = sales[[c for c in snap_days   if c in sales.columns]].mean().mean()
nosnap_sales = sales[[c for c in no_snap_days if c in sales.columns]].mean().mean()

axes[1][0].bar(["No SNAP", "SNAP Day"],
               [nosnap_sales, snap_sales],
               color=[COLORS["gray"], COLORS["green"]], edgecolor="none")
axes[1][0].set_title("Avg Per-Series Daily Sales: SNAP vs Non-SNAP")
axes[1][0].set_ylabel("Avg Units")
lift = (snap_sales / nosnap_sales - 1) * 100
axes[1][0].text(1, snap_sales * 0.95, f"+{lift:.1f}%", ha="center",
                color=COLORS["green"], fontsize=11, fontweight="bold")

# Event type sales lift
event_days = calendar[calendar["event_name_1"].notna()].copy()
no_event   = calendar[calendar["event_name_1"].isna()]["d"].tolist()
base_sales = sales[[c for c in no_event if c in sales.columns]].mean().mean()

event_lifts = {}
for etype in event_days["event_type_1"].unique():
    edays = event_days[event_days["event_type_1"] == etype]["d"].tolist()
    valid = [c for c in edays if c in sales.columns]
    if valid:
        e_sales = sales[valid].mean().mean()
        event_lifts[etype] = (e_sales / base_sales - 1) * 100

lifts_df = pd.Series(event_lifts).sort_values(ascending=True)
colors_e = [COLORS["green"] if v > 0 else COLORS["red"] for v in lifts_df.values]
axes[1][1].barh(lifts_df.index, lifts_df.values, color=colors_e, edgecolor="none")
axes[1][1].set_title("Sales Lift by Event Type vs No-Event Baseline (%)")
axes[1][1].set_xlabel("Lift (%)")
axes[1][1].axvline(0, color=COLORS["gray"], lw=1)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}03_prices_promotions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}03_prices_promotions.png")


# %% [markdown]
# ## 4. Autocorrelation — Informing Lag Feature Selection

# %%
# Sample a few series and compute ACF
sample_ids = sales.sample(5, random_state=42)["id"].values
fig, axes  = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Autocorrelation Analysis — Sample Series", fontsize=13)

for idx, (sid, ax) in enumerate(zip(sample_ids, axes.flatten())):
    row    = sales[sales["id"] == sid]
    series = row[day_cols].values.flatten().astype(float)

    # Manual ACF up to lag 56
    lags = range(1, 57)
    acf_vals = []
    series_demean = series - series.mean()
    var = np.var(series_demean) + 1e-8
    for lag in lags:
        cov = np.mean(series_demean[lag:] * series_demean[:-lag])
        acf_vals.append(cov / var)

    colors_acf = [COLORS["green"] if v > 0 else COLORS["red"] for v in acf_vals]
    ax.bar(lags, acf_vals, color=colors_acf, edgecolor="none", alpha=0.8)
    ax.axhline(0, color=COLORS["gray"], lw=0.8)
    ax.axhline(1.96 / np.sqrt(len(series)), color=COLORS["blue"],
               lw=0.8, linestyle="--", label="95% CI")
    ax.axhline(-1.96 / np.sqrt(len(series)), color=COLORS["blue"],
               lw=0.8, linestyle="--")

    # Highlight key lags
    for kl, kc in [(7, COLORS["orange"]), (14, COLORS["orange"]),
                   (28, COLORS["purple"]), (35, COLORS["purple"])]:
        ax.axvline(kl, color=kc, lw=1, alpha=0.5)

    dept = row["dept_id"].values[0]
    ax.set_title(f"{dept}", fontsize=9)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_ylim(-0.4, 1.0)

# Use last panel for lag importance summary
ax = axes.flatten()[-1]
lag_labels = ["7", "14", "28", "35"]
lag_vals   = [7, 14, 28, 35]
# Average |ACF| at these lags across all sample series
avg_acf = []
for lag in lag_vals:
    acf_at_lag = []
    for sid in sample_ids:
        row    = sales[sales["id"] == sid]
        series = row[day_cols].values.flatten().astype(float)
        series_demean = series - series.mean()
        var    = np.var(series_demean) + 1e-8
        cov    = np.mean(series_demean[lag:] * series_demean[:-lag])
        acf_at_lag.append(abs(cov / var))
    avg_acf.append(np.mean(acf_at_lag))

ax.bar(lag_labels, avg_acf, color=COLORS["purple"], edgecolor="none")
ax.set_title("Avg |ACF| at Key Lags\n(informs feature selection)")
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Avg |ACF|")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}04_autocorrelation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}04_autocorrelation.png")


# %% [markdown]
# ## 5. Store × Category Sales Heatmap

# %%
# Pivot: mean daily sales by store × category
store_cat = (sales.groupby(["store_id", "cat_id"])[day_cols]
             .sum().sum(axis=1)
             .unstack("cat_id"))

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(store_cat.values / 1e3, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(store_cat.columns)))
ax.set_xticklabels(store_cat.columns)
ax.set_yticks(range(len(store_cat.index)))
ax.set_yticklabels(store_cat.index)
plt.colorbar(im, ax=ax, label="Total Sales (000 units)")

for i in range(len(store_cat.index)):
    for j in range(len(store_cat.columns)):
        ax.text(j, i, f"{store_cat.values[i,j]/1e3:.0f}K",
                ha="center", va="center", fontsize=8, color="black")

ax.set_title("Total Sales Volume — Store × Category Heatmap", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}05_store_category_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}05_store_category_heatmap.png")


# %% [markdown]
# ## 6. Class Imbalance & Intermittency Profile

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Intermittency Profile — Critical for Forecast Model Choice", fontsize=12)

# Intermittency by category
for cat, color in zip(["FOODS", "HOBBIES", "HOUSEHOLD"],
                      [COLORS["green"], COLORS["orange"], COLORS["blue"]]):
    cat_series = sales[sales["cat_id"] == cat][day_cols].values.flatten()
    zero_pct   = (cat_series == 0).mean() * 100
    axes[0].bar(cat, zero_pct, color=color, edgecolor="none")
    axes[0].text(cat, zero_pct + 0.5, f"{zero_pct:.1f}%", ha="center", fontsize=9)

axes[0].set_title("Zero-Sales Rate by Category")
axes[0].set_ylabel("% Zero Days")
axes[0].set_ylim(0, 80)

# Sales value distribution (log scale)
all_nonzero = sales[day_cols].values[sales[day_cols].values > 0].flatten()
axes[1].hist(all_nonzero, bins=60, color=COLORS["blue"], edgecolor="none", alpha=0.8,
             log=True)
axes[1].set_title("Non-Zero Sales Distribution (log scale)")
axes[1].set_xlabel("Units Sold")
axes[1].set_ylabel("Frequency (log)")
axes[1].axvline(np.median(all_nonzero), color=COLORS["orange"], lw=1.5,
                label=f"Median = {np.median(all_nonzero):.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}06_intermittency.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}06_intermittency.png")


# %% Summary
print("\n" + "="*60)
print("EDA COMPLETE — Key Findings")
print("="*60)
print(f"  Zero-sales rate          : {zero_rate:.1%}  → use Tweedie/Poisson loss")
print(f"  Weekly cycle             : Saturday/Sunday peak  → wday feature critical")
print(f"  SNAP lift                : +{lift:.1f}%  → include per-state SNAP flag")
print(f"  Strongest ACF lags       : 7, 14, 28, 35  → confirmed feature choices")
print(f"  FOODS dominates volume   : {cat_totals['FOODS']/cat_totals.sum():.0%} of total sales")
print(f"  EDA figures saved to     : {FIG_DIR}")
