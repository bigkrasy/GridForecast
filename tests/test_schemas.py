import pytest
from pydantic import ValidationError

from gridforecast.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftSummaryRow,
    HealthResponse,
    HistoryPoint,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


# ============================================================
# HistoryPoint
# ============================================================
def test_history_point_valid():
    """Валиден HistoryPoint."""
    hp = HistoryPoint(timestamp="2024-12-01T12:00:00", consumption_mw=1000.0)
    assert hp.consumption_mw == 1000.0


def test_history_point_negative_consumption():
    """consumption_mw < 0 → ValidationError (gt=0)."""
    with pytest.raises(ValidationError):
        HistoryPoint(timestamp="2024-12-01T12:00:00", consumption_mw=-100)


def test_history_point_zero_consumption():
    """consumption_mw = 0 → ValidationError (gt=0)."""
    with pytest.raises(ValidationError):
        HistoryPoint(timestamp="2024-12-01T12:00:00", consumption_mw=0)


# ============================================================
# PredictRequest
# ============================================================
def _make_history(n: int = 168) -> list[dict]:
    """Генерира n реда history."""
    from datetime import datetime, timedelta

    end = datetime(2024, 12, 1, 12, 0, 0)
    return [
        {
            "timestamp": (end - timedelta(hours=n - 1 - i)).isoformat(),
            "consumption_mw": 1000.0 + i,
        }
        for i in range(n)
    ]


def test_predict_request_valid():
    """Валиден PredictRequest с 168 history."""
    req = PredictRequest(
        timestamp="2024-12-01T13:00:00",
        temperature_c=5.5,
        is_holiday=False,
        is_weekend=True,
        history=_make_history(168),
    )
    assert req.temperature_c == 5.5
    assert len(req.history) == 168


def test_predict_request_more_history():
    """Повече от 168 history → OK."""
    req = PredictRequest(
        timestamp="2024-12-01T13:00:00",
        temperature_c=5.5,
        history=_make_history(200),
    )
    assert len(req.history) == 200


def test_predict_request_short_history():
    """history < 168 → ValidationError (min_length=168)."""
    with pytest.raises(ValidationError):
        PredictRequest(
            timestamp="2024-12-01T13:00:00",
            temperature_c=5.5,
            history=_make_history(100),
        )


def test_predict_request_empty_history():
    """history = [] → ValidationError."""
    with pytest.raises(ValidationError):
        PredictRequest(
            timestamp="2024-12-01T13:00:00",
            temperature_c=5.5,
            history=[],
        )


def test_predict_request_missing_history():
    """Липсва history → ValidationError."""
    with pytest.raises(ValidationError):
        PredictRequest(
            timestamp="2024-12-01T13:00:00",
            temperature_c=5.5,
        )


def test_predict_request_default_flags():
    """is_holiday / is_weekend → default False."""
    req = PredictRequest(
        timestamp="2024-12-01T13:00:00",
        temperature_c=5.5,
        history=_make_history(168),
    )
    assert req.is_holiday is False
    assert req.is_weekend is False


# ============================================================
# PredictResponse
# ============================================================
def test_predict_response_valid():
    """Валиден PredictResponse."""
    resp = PredictResponse(
        timestamp="2024-12-01T13:00:00",
        predicted_mw=1234.5,
        model_name="LightGBM",
        model_version="v2",
        latency_ms=12.5,
    )
    assert resp.predicted_mw == 1234.5
    assert resp.model_name == "LightGBM"
    assert resp.model_version == "v2"
    assert resp.latency_ms == 12.5


# ============================================================
# BatchPredictRequest
# ============================================================
def test_batch_predict_request_valid():
    """Валиден BatchPredictRequest с 2 requests."""
    req = BatchPredictRequest(
        requests=[
            PredictRequest(
                timestamp="2024-12-01T13:00:00",
                temperature_c=5.5,
                history=_make_history(168),
            ),
            PredictRequest(
                timestamp="2024-12-01T14:00:00",
                temperature_c=6.0,
                history=_make_history(168),
            ),
        ]
    )
    assert len(req.requests) == 2


def test_batch_predict_request_empty():
    """requests = [] → ValidationError (min_length=1)."""
    with pytest.raises(ValidationError):
        BatchPredictRequest(requests=[])


# ============================================================
# HealthResponse
# ============================================================
def test_health_response_healthy():
    """HealthResponse със status="healthy"."""
    resp = HealthResponse(
        status="healthy",
        model_loaded=True,
        timestamp="2024-12-01T12:00:00",
    )
    assert resp.status == "healthy"
    assert resp.model_loaded is True


def test_health_response_degraded():
    """HealthResponse със status="degraded"."""
    resp = HealthResponse(
        status="degraded",
        model_loaded=False,
        timestamp="2024-12-01T12:00:00",
    )
    assert resp.status == "degraded"
    assert resp.model_loaded is False


def test_health_response_invalid_status():
    """status != "healthy"/"degraded" → ValidationError."""
    with pytest.raises(ValidationError):
        HealthResponse(
            status="unknown",
            model_loaded=True,
            timestamp="2024-12-01T12:00:00",
        )


# ============================================================
# DriftSummaryRow
# ============================================================
def test_drift_summary_row_valid():
    """Валиден DriftSummaryRow."""
    row = DriftSummaryRow(
        method="PSI (consumption, full)",
        value=0.5318,
        threshold=0.25,
        status="DRIFT",
        weight="LOW",
        note="false positive (seasonal)",
    )
    assert row.method == "PSI (consumption, full)"
    assert row.status == "DRIFT"


def test_drift_summary_row_default_note():
    """note → default "" (ако не е подаден)."""
    row = DriftSummaryRow(
        method="PSI (temperature, full)",
        value=0.0883,
        threshold=0.25,
        status="OK",
        weight="LOW",
    )
    assert row.note == ""


# ============================================================
# ModelInfoResponse
# ============================================================
def test_model_info_response_valid():
    """Валиден ModelInfoResponse."""
    resp = ModelInfoResponse(
        model_name="LightGBM",
        model_version="v2",
        n_features=23,
        feature_columns=["temperature_c", "is_holiday"] + [f"lag_{i}h" for i in range(1, 22)],
        fit_mape=2.96,
        baseline_mape=9.20,
        retraining_strategy="monthly_monitoring + quarterly_full_retrain",
        loaded_at="2024-12-01T12:00:00",
        drift_summary=[],
    )
    assert resp.n_features == 23
    assert len(resp.feature_columns) == 23
    assert resp.fit_mape == 2.96


def test_model_info_response_default_drift_summary():
    """drift_summary → default [] (ако не е подаден)."""
    resp = ModelInfoResponse(
        model_name="LightGBM",
        model_version="v2",
        n_features=23,
        feature_columns=[f"f{i}" for i in range(23)],
        fit_mape=2.96,
        baseline_mape=9.20,
        retraining_strategy="monthly",
        loaded_at="2024-12-01T12:00:00",
    )
    assert resp.drift_summary == []