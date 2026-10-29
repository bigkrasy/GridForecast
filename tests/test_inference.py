"""
test_inference.py — pytest тестове за inference логиката.

Пусни: pytest tests/test_inference.py -v
"""
import pandas as pd
import pytest

from gridforecast.api.inference import (
    ModelStore,
    build_inference_row,
    predict_single,
)
from gridforecast.config import SYNTHETIC_DIR, TRAIN_RATIO


@pytest.fixture(scope="module", autouse=True)
def load_model():
    """Зарежда модела веднъж за целия module."""
    ModelStore.load()
    yield
    ModelStore.clear()


@pytest.fixture(scope="module")
def real_data():
    """Реални данни от consumption_stable.parquet."""
    df = pd.read_parquet(SYNTHETIC_DIR / "consumption_stable.parquet")
    df.index = pd.DatetimeIndex(df.index)
    split_idx = int(TRAIN_RATIO * len(df))
    return df, split_idx


def test_model_loaded():
    assert ModelStore.is_loaded()
    assert ModelStore.metadata is not None
    assert ModelStore.metadata["n_features"] == 23
    assert len(ModelStore.feature_cols) == 23


def test_model_matches_json():
    from gridforecast.features import load_feature_columns

    json_cols = load_feature_columns()
    assert ModelStore.feature_cols == json_cols
    assert len(json_cols) == 23


def test_build_inference_row(real_data):
    df, split_idx = real_data
    history_end = split_idx - 1
    history = df.iloc[history_end - 168:history_end + 1]
    next_row = df.iloc[history_end + 1]

    history_list = [
        {"timestamp": ts, "consumption_mw": float(row["consumption_mw"])}
        for ts, row in history.iterrows()
    ]

    X = build_inference_row(
        timestamp=df.index[history_end + 1],
        temperature_c=float(next_row["temperature_c"]),
        is_holiday=bool(next_row["is_holiday"]),
        is_weekend=bool(next_row["is_weekend"]),
        history=history_list,
    )

    assert X.shape == (1, 23)
    assert list(X.columns) == ModelStore.feature_cols
    assert not X.isna().any().any()


def test_predict_single_accuracy(real_data):
    df, split_idx = real_data
    history_end = split_idx - 1
    history = df.iloc[history_end - 168:history_end + 1]
    next_row = df.iloc[history_end + 1]

    history_list = [
        {"timestamp": ts, "consumption_mw": float(row["consumption_mw"])}
        for ts, row in history.iterrows()
    ]

    y_pred = predict_single(
        timestamp=df.index[history_end + 1],
        temperature_c=float(next_row["temperature_c"]),
        is_holiday=bool(next_row["is_holiday"]),
        is_weekend=bool(next_row["is_weekend"]),
        history=history_list,
    )

    y_true = float(next_row["consumption_mw"])
    error_pct = abs(y_true - y_pred) / y_true * 100

    assert y_pred > 0
    assert error_pct < 5.0, f"MAPE {error_pct:.2f}% > 5%"


def test_build_inference_row_raises_on_short_history(real_data):
    df, split_idx = real_data
    history_end = split_idx - 1
    # Само 50 реда — недостатъчно за lag_168h
    history = df.iloc[history_end - 50:history_end + 1]
    next_row = df.iloc[history_end + 1]

    history_list = [
        {"timestamp": ts, "consumption_mw": float(row["consumption_mw"])}
        for ts, row in history.iterrows()
    ]

    with pytest.raises(ValueError, match="NaN"):
        build_inference_row(
            timestamp=df.index[history_end + 1],
            temperature_c=float(next_row["temperature_c"]),
            is_holiday=bool(next_row["is_holiday"]),
            is_weekend=bool(next_row["is_weekend"]),
            history=history_list,
        )


def test_build_inference_row_raises_on_timestamp_before_history(real_data):
    df, split_idx = real_data
    history_end = split_idx - 1
    history = df.iloc[history_end - 168:history_end + 1]
    next_row = df.iloc[history_end + 1]

    history_list = [
        {"timestamp": ts, "consumption_mw": float(row["consumption_mw"])}
        for ts, row in history.iterrows()
    ]

    # timestamp = последна history точка → трябва да гръмне
    with pytest.raises(ValueError, match="СЛЕД последната"):
        build_inference_row(
            timestamp=df.index[history_end],  # същият
            temperature_c=float(next_row["temperature_c"]),
            is_holiday=bool(next_row["is_holiday"]),
            is_weekend=bool(next_row["is_weekend"]),
            history=history_list,
        )