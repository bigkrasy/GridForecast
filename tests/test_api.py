"""
test_api.py — pytest тестове за GridForecast API.

Пусни: pytest tests/test_api.py -v
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridforecast.api.main import app
from gridforecast.config import SYNTHETIC_DIR, TRAIN_RATIO


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_predict_payload():
    df = pd.read_parquet(SYNTHETIC_DIR / "consumption_stable.parquet")
    df.index = pd.DatetimeIndex(df.index)
    split_idx = int(TRAIN_RATIO * len(df))
    history_end = split_idx - 1
    history = df.iloc[history_end - 168:history_end + 1]
    next_row = df.iloc[history_end + 1]

    return {
        "timestamp": df.index[history_end + 1].isoformat(),
        "temperature_c": float(next_row["temperature_c"]),
        "is_holiday": bool(next_row["is_holiday"]),
        "is_weekend": bool(next_row["is_weekend"]),
        "history": [
            {"timestamp": ts.isoformat(), "consumption_mw": float(row["consumption_mw"])}
            for ts, row in history.iterrows()
        ],
    }


def make_synthetic_history(n: int = 168) -> list[dict]:
    end_ts = datetime(2024, 12, 1, 23, 0, 0)
    return [
        {
            "timestamp": (end_ts - timedelta(hours=n - 1 - i)).isoformat(),
            "consumption_mw": 1000.0 + (i % 24) * 10.0,
        }
        for i in range(n)
    ]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data


def test_root_alias(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert data["n_features"] == 23
    assert len(data["feature_columns"]) == 23
    assert data["fit_mape"] > 0
    assert "drift_summary" in data


def test_model_info_has_drift_summary(client):
    r = client.get("/model-info")
    data = r.json()
    assert isinstance(data["drift_summary"], list)


def test_predict_with_real_data(client, real_predict_payload):
    r = client.post("/predict", json=real_predict_payload)
    assert r.status_code == 200, f"Response: {r.json()}"
    data = r.json()

    assert "predicted_mw" in data
    assert data["predicted_mw"] > 0
    assert data["model_name"] == "LightGBM"
    assert data["model_version"] == "v2"
    assert "latency_ms" in data


def test_predict_with_synthetic_history(client):
    r = client.post(
        "/predict",
        json={
            "timestamp": "2024-12-02T00:00:00",
            "temperature_c": 5.0,
            "is_holiday": False,
            "is_weekend": False,
            "history": make_synthetic_history(168),
        },
    )
    assert r.status_code == 200
    assert r.json()["predicted_mw"] > 0


def test_predict_accuracy(client, real_predict_payload):
    df = pd.read_parquet(SYNTHETIC_DIR / "consumption_stable.parquet")
    df.index = pd.DatetimeIndex(df.index)
    split_idx = int(TRAIN_RATIO * len(df))
    actual = float(df.iloc[split_idx]["consumption_mw"])

    r = client.post("/predict", json=real_predict_payload)
    data = r.json()
    error_pct = abs(actual - data["predicted_mw"]) / actual * 100

    assert error_pct < 5.0, f"MAPE {error_pct:.2f}% > 5%"


def test_predict_missing_history(client):
    r = client.post(
        "/predict",
        json={
            "timestamp": "2024-12-01T12:00:00",
            "temperature_c": 5.0,
            "is_holiday": False,
            "is_weekend": False,
        },
    )
    assert r.status_code == 422


def test_predict_short_history(client):
    r = client.post(
        "/predict",
        json={
            "timestamp": "2024-12-01T12:00:00",
            "temperature_c": 5.0,
            "is_holiday": False,
            "is_weekend": False,
            "history": make_synthetic_history(50),
        },
    )
    assert r.status_code == 422


def test_predict_timestamp_before_history(client):
    history = make_synthetic_history(168)
    r = client.post(
        "/predict",
        json={
            "timestamp": history[-1]["timestamp"],
            "temperature_c": 5.0,
            "is_holiday": False,
            "is_weekend": False,
            "history": history,
        },
    )
    assert r.status_code == 400


def test_predict_batch(client):
    payload = {
        "requests": [
            {
                "timestamp": "2024-12-02T00:00:00",
                "temperature_c": 5.0,
                "is_holiday": False,
                "is_weekend": False,
                "history": make_synthetic_history(168),
            },
            {
                "timestamp": "2024-12-02T01:00:00",
                "temperature_c": 6.0,
                "is_holiday": False,
                "is_weekend": False,
                "history": make_synthetic_history(168),
            },
        ]
    }
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["n_predictions"] == 2
    assert len(data["predictions"]) == 2


def test_predict_batch_empty(client):
    r = client.post("/predict/batch", json={"requests": []})
    assert r.status_code == 422


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "gridforecast_requests_total" in text
    assert "gridforecast_model_loaded 1" in text