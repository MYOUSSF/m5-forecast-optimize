"""
Project Configuration
=====================
Single source of truth for all paths and runtime settings.
Override via environment variables or by editing this file.

Priority (highest to lowest):
  1. Environment variables
  2. This file's defaults

Usage
-----
Local:
    python run_pipeline.py

Kaggle:
    Set ENV=kaggle before running, or just let auto-detect handle it.

Other:
    export DATA_DIR=/my/data M5_CACHE_DIR=/my/cache python run_pipeline.py
"""

import os
from pathlib import Path

# ── Environment auto-detection ────────────────────────────────────────────────

def _detect_env() -> str:
    """Detect runtime environment."""
    if Path("/kaggle/input").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"

ENV = os.environ.get("ENV", _detect_env())

# ── Path defaults per environment ─────────────────────────────────────────────

_DEFAULTS = {
    "kaggle": {
        "DATA_DIR"  : "/kaggle/input/competitions/m5-forecasting-accuracy/",
        "CACHE_DIR" : "/kaggle/working/store_cache",
        "OUTPUT_DIR": "/kaggle/working/outputs",
        "MODEL_DIR" : "/kaggle/working/models",
        "MLFLOW_URI": "/kaggle/working/mlruns",
    },
    "colab": {
        "DATA_DIR"  : "/content/drive/MyDrive/m5/data/",
        "CACHE_DIR" : "/content/store_cache",
        "OUTPUT_DIR": "/content/outputs",
        "MODEL_DIR" : "/content/models",
        "MLFLOW_URI": "/content/mlruns",
    },
    "local": {
        "DATA_DIR"  : str(Path(__file__).parent / "data"),
        "CACHE_DIR" : str(Path(__file__).parent / "store_cache"),
        "OUTPUT_DIR": str(Path(__file__).parent / "outputs"),
        "MODEL_DIR" : str(Path(__file__).parent / "models"),
        "MLFLOW_URI": str(Path(__file__).parent / "mlruns"),
    },
}

def _get(key: str) -> str:
    """Return env var if set, else environment default."""
    return os.environ.get(key, _DEFAULTS[ENV][key])

# ── Exported config values ────────────────────────────────────────────────────

DATA_DIR   = _get("DATA_DIR")
CACHE_DIR  = _get("CACHE_DIR")
OUTPUT_DIR = _get("OUTPUT_DIR")
MODEL_DIR  = _get("MODEL_DIR")
MLFLOW_URI = _get("MLFLOW_URI")

# ── Training settings ─────────────────────────────────────────────────────────

N_ITEMS     = int(os.environ.get("N_ITEMS", 0)) or None  # None = full dataset
N_CV_SPLITS = int(os.environ.get("N_CV_SPLITS", 3))
HORIZON     = int(os.environ.get("HORIZON", 28))

# ── model parameters ────────────────────────────────────────────────────────────

QUANTILES = [0.10, 0.50, 0.90]

BASE_PARAMS = dict(
    n_estimators      = 2000,
    learning_rate     = 0.05,
    max_depth         = 6,
    min_child_weight  = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    tree_method       = "hist",
    device            = "cuda",
    random_state      = 42,
    verbosity         = 0,
    early_stopping_rounds = 50,
)

# ── Print summary ─────────────────────────────────────────────────────────────

def print_config():
    print(f"Environment : {ENV}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"CACHE_DIR   : {CACHE_DIR}")
    print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"MODEL_DIR   : {MODEL_DIR}")
    print(f"MLFLOW_URI  : {MLFLOW_URI}")
    print(f"N_ITEMS     : {N_ITEMS or 'ALL'}")
    print(f"N_CV_SPLITS : {N_CV_SPLITS}")

if __name__ == "__main__":
    print_config()
