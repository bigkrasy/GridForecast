# src/gridforecast/config.py
"""
GridForecast — Configuration.

Всички paths, constants и settings на едно място.

В Docker: MODELS_DIR, DATA_DIR се override-ват от env vars
(защото `pip install .` копира кода в site-packages, а моделът
е mount-нат на /app/models).
"""

import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Environment variable override (за Docker)
# Ако MODELS_DIR / DATA_DIR са зададени → използват се
# Иначе → PROJECT_ROOT / "models" (за локално)
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))

SYNTHETIC_DIR = DATA_DIR / "synthetic"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))

# ============================================================
# Create directories — само локално (не в Docker)
# ============================================================
# В Docker → appuser не може да пише в /app/data/
# Затова: skip ако сме в Docker (IN_DOCKER=1)
IS_DOCKER = os.getenv("IN_DOCKER") == "1"

if not IS_DOCKER:
    for d in [SYNTHETIC_DIR, RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            pass  # skip silently


# ============================================================
# Data
# ============================================================
TRAIN_RATIO = 0.8
FORECAST_HORIZON_HOURS = 24
RANDOM_SEED = 42

CONSUMPTION_STABLE = SYNTHETIC_DIR / "consumption_stable.parquet"
CONSUMPTION_DRIFT = SYNTHETIC_DIR / "consumption_drift.parquet"


# ============================================================
# Models
# ============================================================
PROPHET_DAILY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_YEARLY_SEASONALITY = True
PROPHET_SEASONALITY_MODE = "additive"
PROPHET_GROWTH = "flat"

BOOSTING_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

LSTM_SEQUENCE_LENGTH = 168
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 30
LSTM_PATIENCE = 7
LSTM_BATCH_SIZE = 64
LSTM_LEARNING_RATE = 1e-3

LAG_FEATURES = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]


# ============================================================
# Drift detection
# ============================================================
DRIFT_THRESHOLDS = {
    "psi_alert": 0.25,
    "psi_warn": 0.10,
    "ks_pvalue": 0.05,
    "mae_multiplier_alert": 1.5,
    "mae_multiplier_warn": 1.3,
    "rolling_window_days": 7,
}


# ============================================================
# API
# ============================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================
# Plotting (OPTIONAL — само за notebooks)
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["axes.titleweight"] = "bold"

    HAS_PLOTTING = True
except ImportError:
    plt = None
    sns = None
    HAS_PLOTTING = False


# ============================================================
# Utils
# ============================================================
def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("GridForecast — Configuration")
    print("=" * 70)
    print(f"\nPaths:")
    print(f"  PROJECT_ROOT:  {PROJECT_ROOT}")
    print(f"  DATA_DIR:      {DATA_DIR}")
    print(f"  MODELS_DIR:    {MODELS_DIR}")
    print(f"  FIGURES_DIR:   {FIGURES_DIR}")
    print(f"\nEnvironment:")
    print(f"  IS_DOCKER:     {IS_DOCKER}")
    print(f"  HAS_PLOTTING:  {HAS_PLOTTING}")
    print(f"\nAPI:")
    print(f"  API_HOST:      {API_HOST}")
    print(f"  API_PORT:      {API_PORT}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()