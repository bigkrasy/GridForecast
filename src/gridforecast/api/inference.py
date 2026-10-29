"""
GridForecast — inference логика.

Използва `create_features()` (single source of truth), за да гарантира,
че inference features са 1:1 с training features.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gridforecast.config import MODELS_DIR
from gridforecast.features import (
    TARGET_COL,
    create_features,
    load_feature_columns,
)


class ModelStore:
    """
    Singleton за модела.

    Зарежда `production_model.pkl` веднъж при startup и го държи в паметта.
    """
    model = None
    feature_cols: list[str] | None = None
    metadata: dict | None = None

    @classmethod
    def load(cls, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = MODELS_DIR / "production_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Липсва модел: {model_path}\n"
                f"Очаква се от 08_retraining.ipynb."
            )

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        cls.model = data["model"]
        cls.feature_cols = data.get("feature_cols") or load_feature_columns()

        # Валидация: model.feature_cols == feature_columns.json
        n_in = getattr(cls.model, "n_features_in_", None)
        if n_in is not None and n_in != len(cls.feature_cols):
            raise ValueError(
                f"Mismatch: model.n_features_in_={n_in} "
                f"vs feature_cols={len(cls.feature_cols)}"
            )

        cls.metadata = {
            "model_name": data.get("model_name", "LightGBM"),
            "fit_mape": float(data.get("fit_mape", 2.96)),
            "baseline_mape": float(data.get("baseline_mape", 9.20)),
            "retraining_strategy": data.get(
                "retraining_strategy",
                "monthly_monitoring + quarterly_full_retrain",
            ),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "n_features": len(cls.feature_cols),
        }

    @classmethod
    def is_loaded(cls) -> bool:
        return cls.model is not None

    @classmethod
    def clear(cls) -> None:
        cls.model = None
        cls.feature_cols = None
        cls.metadata = None


def build_inference_row(
    timestamp: datetime,
    temperature_c: float,
    is_holiday: bool,
    is_weekend: bool,
    history: list[dict],
) -> pd.DataFrame:
    """
    Изгражда 1 feature row за inference.

    Стъпки:
      1. Вземи `history` (list от {timestamp, consumption_mw})
      2. Добави нов ред с `consumption_mw=NaN` (ще го предскажем)
      3. Извикай `create_features()` → DataFrame с всички 23 features
      4. Вземи последния ред (за timestamp-а)
      5. Валидирай: 23 features, no NaN

    Връща:
        pd.DataFrame с 1 ред × 23 features (в правилния ред).
    """
    if ModelStore.feature_cols is None:
        raise RuntimeError("ModelStore не е зареден. Викни ModelStore.load().")

    # 1. History → DataFrame
    hist_df = pd.DataFrame(history)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    hist_df = hist_df.set_index("timestamp").sort_index()

    if TARGET_COL not in hist_df.columns:
        raise ValueError(f"history липсва колона '{TARGET_COL}'")

    # 2. Добави нов ред (consumption_mw=NaN)
    ts = pd.Timestamp(timestamp)
    if ts <= hist_df.index[-1]:
        raise ValueError(
            f"timestamp ({ts}) трябва да е СЛЕД последната history точка "
            f"({hist_df.index[-1]})"
        )

    new_row = pd.DataFrame(
        {
            TARGET_COL: [np.nan],
            "temperature_c": [float(temperature_c)],
            "is_holiday": [int(is_holiday)],
            "is_weekend": [int(is_weekend)],
        },
        index=pd.DatetimeIndex([ts], name="timestamp"),
    )

    full_df = pd.concat([hist_df, new_row])

    # 3. create_features() — single source of truth
    df_feat = create_features(full_df)

    # 4. Вземи последния ред (за timestamp-а)
    last_row = df_feat.iloc[[-1]]

    # 5. Валидация
    missing = [c for c in ModelStore.feature_cols if c not in last_row.columns]
    if missing:
        raise RuntimeError(
            f"create_features() не е създал очаквани колони: {missing}"
        )

    X = last_row[ModelStore.feature_cols]

    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"NaN в inference features: {nan_cols}. "
            f"history трябва да е ≥ 168 реда."
        )

    return X.reset_index(drop=True)


def predict_single(
    timestamp: datetime,
    temperature_c: float,
    is_holiday: bool,
    is_weekend: bool,
    history: list[dict],
) -> float:
    """Prediction за един timestamp."""
    if not ModelStore.is_loaded():
        raise RuntimeError("ModelStore не е зареден")

    X = build_inference_row(timestamp, temperature_c, is_holiday, is_weekend, history)
    # Подаваме DataFrame (не .values) — LGBM е трениран с feature names
    y_pred = ModelStore.model.predict(X)[0]
    return float(y_pred)


def predict_batch(
    requests: list[dict],
) -> list[float]:
    """Batch prediction."""
    if not ModelStore.is_loaded():
        raise RuntimeError("ModelStore не е зареден")

    X_list = [
        build_inference_row(
            timestamp=r["timestamp"],
            temperature_c=r["temperature_c"],
            is_holiday=r["is_holiday"],
            is_weekend=r["is_weekend"],
            history=r["history"],
        )
        for r in requests
    ]
    X_all = pd.concat(X_list, ignore_index=True)
    # Подаваме DataFrame (не .values) — LGBM е трениран с feature names
    y_preds = ModelStore.model.predict(X_all)
    return [float(p) for p in y_preds]