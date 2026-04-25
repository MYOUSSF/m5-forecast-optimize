"""
M5 Feature Engineering Pipeline — Memory-Safe Version
======================================================
Processes one store at a time to keep peak RAM under 4 GB.
Full M5 (30,490 series × 1,941 days) fits comfortably in Kaggle's 30 GB.

Strategy
--------
1. Load raw CSVs once
2. For each of the 10 stores, build all features on ~3,049 series
3. Save each store's feature block to parquet
4. Concatenate at the end (or stream directly to LightGBM)
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Cost assumptions ──────────────────────────────────────────────────────────
HOLDING_COST_RATE     = 0.25
STOCKOUT_PENALTY_RATE = 0.40
AVG_UNIT_COST         = 7.0
AVG_SELL_PRICE        = 10.50


def _mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e6


# ── Load ──────────────────────────────────────────────────────────────────────

def load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    sales    = pd.read_csv(data_dir / "sales_train_validation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    prices   = pd.read_csv(data_dir / "sell_prices.csv")
    logger.info("Loaded: sales %s | calendar %s | prices %s",
                sales.shape, calendar.shape, prices.shape)
    return sales, calendar, prices


# ── Per-store feature builder ─────────────────────────────────────────────────

def _build_store_features(store_sales: pd.DataFrame,
                           calendar: pd.DataFrame,
                           prices: pd.DataFrame,
                           lags: list[int],
                           windows: list[int]) -> pd.DataFrame:
    """
    Build the complete feature matrix for a single store's items.
    store_sales : subset of sales wide DataFrame for one store
    Returns a long-format DataFrame ready for modelling.
    """
    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in store_sales.columns if c.startswith("d_")]

    # ── Melt ──────────────────────────────────────────────────────────────────
    df = store_sales[id_cols + day_cols].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    df["d_int"] = df["d"].str[2:].astype("int16")
    df["sales"] = df["sales"].astype("float32")

    # ── Calendar ──────────────────────────────────────────────────────────────
    cal = calendar[["d", "date", "wm_yr_wk", "weekday", "wday",
                     "month", "year", "event_name_1", "event_type_1",
                     "snap_CA", "snap_TX", "snap_WI"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    df = df.merge(cal, on="d", how="left")

    # ── Prices ────────────────────────────────────────────────────────────────
    store_id   = store_sales["store_id"].iloc[0]
    store_prices = prices[prices["store_id"] == store_id].copy()
    store_prices["sell_price"] = store_prices["sell_price"].astype("float32")
    df = df.merge(store_prices[["item_id", "wm_yr_wk", "sell_price"]],
                  on=["item_id", "wm_yr_wk"], how="left")

    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    # Price features
    df["price_lag_1w"]  = df.groupby("id")["sell_price"].shift(7).astype("float32")
    df["price_pct_chg"] = ((df["sell_price"] - df["price_lag_1w"])
                           / (df["price_lag_1w"] + 1e-6)).astype("float32")
    mean_price          = (df.groupby(["cat_id", "date"])["sell_price"]
                             .transform("mean").astype("float32"))
    df["price_rel_cat"] = (df["sell_price"] / (mean_price + 1e-6)).astype("float32")
    df.drop(columns=["price_lag_1w"], inplace=True)

    # ── Lag features ──────────────────────────────────────────────────────────
    for lag in lags:
        df[f"lag_{lag}"] = (df.groupby("id")["sales"]
                              .shift(lag).astype("float32"))

    # ── Rolling features (lag-28 base) ────────────────────────────────────────
    grp_sizes = df.groupby("id", sort=False).size().values
    base      = np.full(len(df), np.nan, dtype="float32")
    sales_arr = df["sales"].to_numpy(dtype="float32")

    pos = 0
    for size in grp_sizes:
        if size > 28:
            base[pos + 28 : pos + size] = sales_arr[pos : pos + size - 28]
        pos += size
    del sales_arr

    for w in windows:
        mean_arr = np.empty(len(df), dtype="float32")
        std_arr  = np.empty(len(df), dtype="float32")
        pos = 0
        for size in grp_sizes:
            s = pd.Series(base[pos : pos + size])
            mean_arr[pos : pos + size] = s.rolling(w, min_periods=1).mean().values
            std_arr[pos  : pos + size] = s.rolling(w, min_periods=1).std().values
            pos += size
        df[f"rmean_{w}"] = mean_arr.astype("float32")
        df[f"rstd_{w}"]  = std_arr.astype("float32")

    del base, mean_arr, std_arr

    # ── Calendar features ─────────────────────────────────────────────────────
    df["is_weekend"]    = df["wday"].isin([1, 2]).astype("int8")
    df["is_month_end"]  = (df["date"].dt.day >= 25).astype("int8")
    df["is_month_start"] = (df["date"].dt.day <= 5).astype("int8")
    df["day_of_year"]   = df["date"].dt.dayofyear.astype("int16")

    state = store_sales["state_id"].iloc[0]
    snap_col = f"snap_{state}"
    df["snap"]      = df[snap_col].astype("int8")
    df["has_event"] = df["event_name_1"].notna().astype("int8")

    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]:
        df[col + "_enc"] = df[col].astype("category").cat.codes.astype("int16")

    # Drop rows with NaN lags
    df = df.dropna(subset=["lag_28", "rmean_7"]).reset_index(drop=True)

    # Drop columns not needed for modelling — saves ~3 GB on full dataset
    drop_cols = ["d", "wm_yr_wk", "weekday", "dept_id", "cat_id", "state_id",
                 "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    gc.collect()
    return df


# ── Master pipeline ───────────────────────────────────────────────────────────

def build_features(data_dir: str | Path,
                   n_items: Optional[int] = None,
                   lags: list[int] = [7, 14, 28, 35],
                   windows: list[int] = [7, 14, 28],
                   cache_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Build features for all stores, processing one store at a time.
    Writes each store to a parquet file in cache_dir and returns immediately —
    never concatenates all stores into RAM.

    Downstream training and inference read directly from the per-store
    parquets via _load_slice() and the store cache, keeping peak RAM
    to one store at a time (~400 MB each).

    Parameters
    ----------
    data_dir  : folder with M5 CSVs
    n_items   : subsample N items total (None = all 30,490)
    cache_dir : folder to write per-store parquets

    Returns
    -------
    cache_dir : Path to the folder of per-store parquet files
    """
    sales, calendar, prices = load_raw(data_dir)

    if n_items:
        sales = sales.sample(n=n_items, random_state=42)

    stores    = sales["store_id"].unique()
    from config import CACHE_DIR as _DEFAULT_CACHE
    cache_dir = Path(cache_dir) if cache_dir else Path(_DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %d stores | cache -> %s", len(stores), cache_dir)

    for i, store in enumerate(stores):
        cache_file = cache_dir / f"features_{store}.parquet"

        if cache_file.exists():
            logger.info("[%d/%d] %s — cached, skipping", i+1, len(stores), store)
            continue

        store_sales = sales[sales["store_id"] == store]
        logger.info("[%d/%d] %s — building features (%d series)",
                    i+1, len(stores), store, len(store_sales))

        store_df = _build_store_features(store_sales, calendar, prices, lags, windows)
        store_df.to_parquet(cache_file, index=False)
        logger.info("  -> %.0f MB saved to %s", _mem_mb(store_df), cache_file.name)
        del store_df
        gc.collect()

    files = sorted(cache_dir.glob("features_*.parquet"))
    logger.info("Done. %d store parquets in %s", len(files), cache_dir)
    return cache_dir


# ── Convenience: melt_sales for EDA notebook ─────────────────────────────────

def melt_sales(sales: pd.DataFrame,
               n_items: Optional[int] = None) -> pd.DataFrame:
    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if n_items:
        sales = sales.sample(n=n_items, random_state=42)
    df = sales[id_cols + day_cols].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    df["d_int"] = df["d"].str[2:].astype("int16")
    df["sales"] = df["sales"].astype("float32")
    return df


def attach_calendar(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar[["d", "date", "wm_yr_wk", "weekday", "wday",
                     "month", "year", "event_name_1", "event_type_1",
                     "snap_CA", "snap_TX", "snap_WI"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    return df.merge(cal, on="d", how="left")


def attach_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["sell_price"] = prices["sell_price"].astype("float32")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    df["price_lag_1w"]  = df.groupby("id")["sell_price"].shift(7).astype("float32")
    df["price_pct_chg"] = ((df["sell_price"] - df["price_lag_1w"])
                           / (df["price_lag_1w"] + 1e-6)).astype("float32")
    mean_price = (df.groupby(["store_id", "cat_id", "date"])["sell_price"]
                    .transform("mean").astype("float32"))
    df["price_rel_cat"] = (df["sell_price"] / (mean_price + 1e-6)).astype("float32")
    df.drop(columns=["price_lag_1w"], inplace=True)
    return df


def build_lag_features(df: pd.DataFrame,
                        lags: list[int] = [7, 14, 28, 35]) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag).astype("float32")
    return df


def build_rolling_features(df: pd.DataFrame,
                             windows: list[int] = [7, 14, 28]) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    grp_sizes = df.groupby("id", sort=False).size().values
    base = np.full(len(df), np.nan, dtype="float32")
    sales_arr = df["sales"].to_numpy(dtype="float32")
    pos = 0
    for size in grp_sizes:
        if size > 28:
            base[pos + 28 : pos + size] = sales_arr[pos : pos + size - 28]
        pos += size
    del sales_arr
    for w in windows:
        mean_arr = np.empty(len(df), dtype="float32")
        std_arr  = np.empty(len(df), dtype="float32")
        pos = 0
        for size in grp_sizes:
            s = pd.Series(base[pos : pos + size])
            mean_arr[pos : pos + size] = s.rolling(w, min_periods=1).mean().values
            std_arr[pos  : pos + size] = s.rolling(w, min_periods=1).std().values
            pos += size
        df[f"rmean_{w}"] = mean_arr.astype("float32")
        df[f"rstd_{w}"]  = std_arr.astype("float32")
    del base
    gc.collect()
    return df


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"]    = df["wday"].isin([1, 2]).astype("int8")
    df["is_month_end"]  = (df["date"].dt.day >= 25).astype("int8")
    df["is_month_start"] = (df["date"].dt.day <= 5).astype("int8")
    df["day_of_year"]   = df["date"].dt.dayofyear.astype("int16")
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["snap"] = np.int8(0)
    for state, col in snap_map.items():
        df.loc[df["state_id"] == state, "snap"] = df.loc[df["state_id"] == state, col].astype("int8")
    df["has_event"] = df["event_name_1"].notna().astype("int8")
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]:
        df[col + "_enc"] = df[col].astype("category").cat.codes.astype("int16")
    gc.collect()
    return df


# ── Feature list ──────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "lag_7", "lag_14", "lag_28", "lag_35",
    "rmean_7", "rmean_14", "rmean_28",
    "rstd_7",  "rstd_14",  "rstd_28",
    "sell_price", "price_pct_chg", "price_rel_cat",
    "wday", "month", "year", "day_of_year",
    "is_weekend", "is_month_end", "is_month_start",
    "snap", "has_event",
    "item_id_enc", "dept_id_enc", "cat_id_enc",
    "store_id_enc", "state_id_enc", "weekday_enc",
]

TARGET_COL = "sales"
DATE_COL   = "date"
ID_COL     = "id"
