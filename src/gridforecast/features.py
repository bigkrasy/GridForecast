"""
GridForecast — feature engineering (single source of truth).

Тази функция се внася:
  • от notebooks (05, 07, 08) при training
  • от api/main.py при inference

Гаранция: тренировъчните features и inference features са 1:1.
Ако промениш нещо тук — трябва да retrain-неш модела.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from gridforecast.config import MODELS_DIR

# ─────────────────────────────────────────────────────────────
# КОНСТАНТИ
# ─────────────────────────────────────────────────────────────
LAG_HOURS: Final[tuple[int, ...]] = (1, 2, 3, 24, 48, 168)
ROLLING_WINDOWS: Final[tuple[int, ...]] = (24, 168)
TARGET_COL: Final[str] = "consumption_mw"
FEATURE_COLUMNS_FILE: Final[Path] = MODELS_DIR / "feature_columns.json"


# ─────────────────────────────────────────────────────────────
# CORE: create_features()
# ─────────────────────────────────────────────────────────────
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Прилага пълния feature engineering pipeline.
    
    ВАЖНО: НЕ drop-ва NaN редове. Първите 168 реда имат NaN в lag_168h.
    Викащият решава дали да drop-не (training) или да запълни (inference).
    
    Създадени колони (в допълнение към входните):
      • lag_{1,2,3,24,48,168}h        (6)
      • rolling_{mean,std,min,max}_{24,168}h  (8)
      • hour, dow, month              (3) — hour/month презаписват входните,
                                            dow е НОВА колона
      • hour_sin/cos, dow_sin/cos, month_sin/cos  (6)
    """
    _validate_input(df)

    out = df.copy()

    # 1. Lag features
    for lag in LAG_HOURS:
        out[f"lag_{lag}h"] = out[TARGET_COL].shift(lag)

    # 2. Rolling features (shift(1) ПЪРВО → без leakage)
    for window in ROLLING_WINDOWS:
        base = out[TARGET_COL].shift(1).rolling(window, min_periods=window)
        out[f"rolling_mean_{window}h"] = base.mean()
        out[f"rolling_std_{window}h"] = base.std()
        out[f"rolling_min_{window}h"] = base.min()
        out[f"rolling_max_{window}h"] = base.max()

    # 3. Calendar features (hour/month презаписват; dow е НОВА)
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month

    # 4. Cyclical encoding (използваме новите колони, гарантирано от index)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


def get_feature_columns(df_feat: pd.DataFrame) -> list[str]:
    """Връща списъка с feature колони (без target-а)."""
    return [c for c in df_feat.columns if c != TARGET_COL]


# ─────────────────────────────────────────────────────────────
# EXPORT / LOAD feature_columns.json
# ─────────────────────────────────────────────────────────────
def save_feature_columns(feature_cols: list[str], path: Path = FEATURE_COLUMNS_FILE) -> None:
    """Записва реда на features в JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"features": feature_cols}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_feature_columns(path: Path = FEATURE_COLUMNS_FILE) -> list[str]:
    """Зарежда реда на features от JSON."""
    if not path.exists():
        raise RuntimeError(f"feature_columns.json липсва в {path}.")
    data = json.loads(path.read_text(encoding="utf-8"))
    cols = data.get("features")
    if not isinstance(cols, list) or not cols:
        raise RuntimeError(f"Невалиден feature_columns.json: {data!r}")
    return cols


# ─────────────────────────────────────────────────────────────
# ВАЛИДАЦИЯ
# ─────────────────────────────────────────────────────────────
def _validate_input(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"create_features() изисква DatetimeIndex. Получи: {type(df.index).__name__}")
    if TARGET_COL not in df.columns:
        raise KeyError(f"Липсва колона '{TARGET_COL}' в DataFrame.")
    if df.index.has_duplicates:
        raise ValueError("DatetimeIndex има дублирани timestamps.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex трябва да е сортиран възходящо.")