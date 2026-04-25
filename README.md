# M5 Demand Forecasting + Inventory Optimization

> **End-to-end ML pipeline**: probabilistic demand forecasting on 57M+ Walmart sales records → three inventory optimization modules → **$346.7M estimated enterprise annual impact** (markdown module; newsvendor & budget allocation modules have known approximation bugs — see Results)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU-brightgreen.svg)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange.svg)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen.svg)]()

---

## Business Problem

Walmart operates 4,700+ US stores managing tens of thousands of SKUs daily. Poor demand forecasts cascade into three compounding costs:

| Problem | Symptom | Annual Cost (industry est.) |
|---|---|---|
| Over-ordering | Excess inventory → holding, obsolescence, markdowns | ~1–2% of revenue |
| Under-ordering | Stockouts → lost sales, customer dissatisfaction | ~4% of revenue |
| Suboptimal allocation | High-demand stores under-stocked, low-demand over-stocked | ~0.5% of revenue |

This project quantifies how **probabilistic forecasting** (quantile regression) enables smarter decisions than point forecasting across all three dimensions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        M5 Raw Data (59M rows)                       │
│         sales_train_validation.csv · calendar.csv · sell_prices.csv │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                ┌───────────▼────────────┐
                │   Feature Engineering   │
                │  · 4 sales lags (7–35d) │
                │  · Rolling mean/std     │
                │  · Price momentum       │
                │  · Calendar + SNAP      │
                │  · Label-encoded IDs    │
                └───────────┬────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │        XGBoost Quantile Regression       │
         │  GPU (P100) · Walk-forward CV · MLflow   │
         │                                          │
         │   q10 ──── q50 (median) ──── q90         │
         │    ↓            ↓              ↓          │
         │  Lower       Point         Upper         │
         │  bound    forecast         bound         │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────┼──────────────────────┐
         │                  │                       │
┌────────▼────────┐ ┌───────▼────────┐ ┌──────────▼───────────┐
│  Module 1       │ │  Module 2      │ │  Module 3             │
│  Newsvendor     │ │  Budget Alloc  │ │  Markdown Scheduling  │
│                 │ │                │ │                       │
│ Critical ratio  │ │ Greedy MR/$    │ │ Price-elasticity      │
│ → optimal Q*   │ │ → store alloc  │ │ → clearance depth     │
│                 │ │                │ │                       │
│ $X holding vs  │ │ $Y rev uplift  │ │ $Z revenue recovery   │
│ stockout saving │ │ per cycle      │ │ from excess stock     │
└─────────────────┘ └────────────────┘ └───────────────────────┘
                            │
               ┌────────────▼──────────────┐
               │   Streamlit Dashboard      │
               │   + MLflow Experiment UI   │
               └───────────────────────────┘
```

---

## Results

### Forecasting Performance

| Model | CV WRMSSE (mean ± std) | CV Dollar Cost (mean) |
|---|---|---|
| q10 | 0.7929 ± 0.0126 | $4,372,260 |
| q50 | 0.5720 ± 0.0116 | $2,441,506 |
| q90 | 0.8724 ± 0.0060 | $513,511 |

| Setting | Value |
|---|---|
| CV folds | 3 walk-forward |
| Forecast horizon | 28 days |
| Train rows (fold 0 → 2) | 54.9M → 55.8M → 56.6M |
| Val rows per fold | 853,720 |
| GPU training time (all 3 models) | ~2.5 hours (P100) |
| Quantile objective | `reg:quantileerror` (XGBoost) |

> **Walk-forward validation** ensures no data leakage: each fold trains on all data before the validation window, mimicking real production deployment. The q50 model is explained via SHAP TreeExplainer on a 3,000-row sample.

### Optimization Dollar Impact

Actual output from `04_optimization.ipynb` (10 M5 stores, extrapolated to 4,700):

| Module | 28-day impact | Enterprise annual (×470 stores) | Status |
|---|---|---|---|
| **Newsvendor** | −$9K | −$52.9M | ⚠️ Bug — see below |
| **Budget Allocation** | −$7K | −$45.2M | ⚠️ Bug — see below |
| **Markdown Scheduling** | +$73K | +$444.8M | ✅ Working |
| **Combined** | — | **+$346.7M** | — |

```
========================================================
ENTERPRISE ANNUAL IMPACT  (4,700 US stores)
========================================================
  Module 1 — Newsvendor (inventory cost) : $-52.9M
  Module 2 — Budget allocation (revenue) : $-45.2M
  Module 3 — Markdown scheduling (revenue): $444.8M
--------------------------------------------------------
  TOTAL                                  : $346.7M
========================================================
```

**Known issues in optimization modules:**

- **Newsvendor**: `fmt()` formatting function misclassifies float values as integers for some output fields (critical_ratio, avg_order_qty display as `$1` and `$5`). More critically, the `saving_28d_usd` is negative (−$8,639), meaning the optimal policy costs *more* than the naive q50 baseline. This likely reflects a bug in the `expected_cost()` approximation — the linear overstock/understock approximation breaks down at CR = 0.9989, where Q* is far above q90, producing large holding costs that swamp the stockout savings.
- **Budget Allocation**: Greedy MR/$ allocation underperforms proportional baseline (−$7K over 28 days). The `expected_revenue()` approximation caps revenue at `q90 × 0.95` regardless of allocation, which removes the marginal incentive to shift units to high-demand stores.

Cost parameters confirmed from output:
- `c_o = $0.0048/unit/day`, `c_u = $4.20/unit`, **CR = 0.9989** → 99.9th percentile service level target

> *Extrapolation assumes uniform store characteristics. The markdown module result ($444.8M) is directionally valid; the newsvendor and budget allocation figures require fixing the approximation bugs before use.*

---

## Module Deep-Dives

### Module 1: Newsvendor Inventory Optimization

The **newsvendor model** finds the optimal order quantity Q* that minimises:

```
Expected Cost = c_o × E[max(Q - D, 0)]  +  c_u × E[max(D - Q, 0)]
                     (holding cost)              (stockout penalty)
```

The closed-form solution is the **Critical Ratio**:

```
F(Q*) = CR = c_u / (c_u + c_o)
```

With Walmart-scale assumptions:
- `c_o = $0.0048/unit/day`  (25% holding cost rate on $7 avg COGS)
- `c_u = $4.20/unit`        (40% lost margin + goodwill on $10.50 avg price)
- **CR = 0.9989** → 99.9th percentile service level target

Since stockout costs dwarf holding costs, the model recommends stocking well above the median. However, the current `expected_cost()` implementation uses a linear approximation that breaks down at CR values this extreme — Q* is pushed far beyond q90, producing holding costs that exceed the stockout savings. This is a known bug to fix before production use.

### Module 2: Store Budget Allocation

Given a fixed replenishment budget B, allocate units across stores to maximise total expected revenue:

```
max  Σ_s  sell_price × E[min(Q_s, D_s)]
s.t. Σ_s  unit_cost × Q_s  ≤  B
```

Solved via **greedy marginal revenue per dollar** (separable objective):
- At each step, assign inventory to the store with highest `ΔRevenue / ΔCost`
- Converges to the optimal for separable concave objectives

**Key insight**: naive proportional allocation ignores demand uncertainty. High-variance stores need a larger buffer to achieve the same expected fill rate.

### Module 3: Markdown / Clearance Scheduling

For items where `inventory > Q*` (excess stock), find the markdown depth `d ∈ {0%, 5%, ..., 50%}` that maximises net revenue subject to clearing ≥80% of excess:

```
Revenue(d) = min(inventory, q50 × lift(d)) × sell_price × (1 - d)

lift(d) = 1 + |elasticity| × d       (price-elasticity model)
```

With grocery-typical elasticity of -2.5: a 20% markdown generates +50% demand lift.

**When to mark down vs hold**: the model prefers early shallow markdowns over late deep ones — consistent with academic evidence on dynamic pricing.

---

## EDA Key Findings

From `notebooks/eda.py`:

| Finding | Implication |
|---|---|
| **62% zero-sales rate** (intermittent demand) | Tweedie loss / Poisson models outperform MSE; WRMSSE handles zeros correctly |
| **Strong weekly cycle** (Sat > Fri by ~40%) | `wday` is the #1 most important feature (confirmed by SHAP) |
| **SNAP days +12–18% lift** | Per-state SNAP binary feature included |
| **Lag-7 ACF ~0.7, Lag-28 ACF ~0.5** | Confirms choice of lag features at 7, 14, 28, 35 days |
| **FOODS = 68% of total volume** | Category-level models would improve over global model |

---

## Project Structure

```
m5-forecast-optimize/
├── data/                          # M5 CSVs (not committed — see setup)
├── notebooks/
│   ├── 02_feature_engineering.ipynb  # Feature pipeline, per-store parquets
│   ├── 03_forecasting.ipynb          # XGBoost training, SHAP, forecast export
│   └── 04_optimization.ipynb         # Newsvendor, budget alloc, markdown
├── src/
│   ├── data/
│   │   └── features.py            # Full feature engineering pipeline
│   ├── forecasting/
│   │   └── xgb_quantile.py        # XGBoost q10/q50/q90 + SHAP + MLflow
│   ├── optimization/
│   │   ├── newsvendor.py          # Module 1: critical ratio inventory
│   │   ├── budget_alloc.py        # Module 2: greedy MR/$ allocation
│   │   └── markdown.py            # Module 3: price-elasticity clearance
│   └── dashboard/
│       └── app.py                 # Streamlit interactive dashboard
├── tests/
│   └── test_optimization.py       # 32 unit + integration tests
├── config.py                      # Single source of truth for paths & params
├── run_pipeline.py                # End-to-end pipeline entry point
└── requirements.txt
```

---

## Setup & Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download M5 data

```bash
# Requires Kaggle API credentials
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/
```

### 3. Run the pipeline

**Notebook workflow (recommended):**
```bash
# Run notebooks in order
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_forecasting.ipynb      # GPU recommended; ~2.5h on P100
jupyter notebook notebooks/04_optimization.ipynb
```

**Or run end-to-end from CLI:**
```bash
# Development run (500 items, fast iteration)
python run_pipeline.py --n_items 500 --output_dir outputs/

# Full dataset run (GPU recommended)
python run_pipeline.py --full --n_cv 3 --output_dir outputs/
```

### 4. View results

```bash
# Streamlit dashboard
streamlit run src/dashboard/app.py

# MLflow experiment tracker
mlflow ui --backend-store-uri mlruns/
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Design Decisions

**Why XGBoost over LightGBM?**
Both are strong gradient boosting frameworks for tabular data. XGBoost was chosen here because:
- `reg:quantileerror` is a first-class objective with native GPU support via `device="cuda"`
- The P100 Kaggle GPU yields 3–5× speedup over CPU on 57M rows
- `tree_method="hist"` keeps memory footprint low, enabling streaming from per-store parquets
- SHAP TreeExplainer integrates directly with `xgb.Booster` objects

**Why quantile regression over point forecasting?**
The three optimization modules all require demand uncertainty estimates:
- Newsvendor: CR maps to a quantile of the forecast distribution
- Budget allocation: high-variance stores need larger inventory buffers  
- Markdown: excess = inventory − Q*, which depends on the uncertainty-adjusted optimal Q*

A point forecast (q50 only) systematically underestimates the required inventory when stockout costs >> holding costs.

**Why walk-forward CV?**
Standard k-fold cross-validation violates the temporal ordering assumption, causing data leakage (future data leaks into training). Walk-forward ensures each validation window is strictly after its training window — mimicking real production deployment.

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| Feature engineering at scale | `src/data/features.py` — lag/rolling/price/calendar features on 59M rows |
| Quantile regression | `xgb_quantile.py` — three quantile models with walk-forward CV |
| Operations research | `newsvendor.py` — critical ratio derivation and interpolation |
| Constrained optimisation | `budget_alloc.py` — greedy MR/$ with budget constraint |
| Pricing models | `markdown.py` — price-elasticity demand model |
| MLOps | MLflow experiment tracking, model registry, artefact logging |
| Model interpretability | SHAP TreeExplainer, beeswarm plot |
| Software engineering | Type hints, logging, argparse CLI, `__init__.py` packaging |
| Testing | 32 pytest unit + integration tests with fixtures |
| Data visualisation | Streamlit dashboard + 6 EDA figures |

---

## References

- Makridakis et al. (2022). *M5 accuracy competition: Results, findings and conclusions*. International Journal of Forecasting.
- Cachon & Terwiesch (2009). *Matching Supply with Demand*. McGraw-Hill. (Newsvendor chapter)
- Künzel et al. (2019). *Metalearners for estimating heterogeneous treatment effects*. PNAS.
- Ke et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS.