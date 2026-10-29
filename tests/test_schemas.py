import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from gridforecast.api.schemas import (
    ForecastContextRow,
    ForecastRequest,
    BatchForecastRequest,
)

def make_context(n=168):
    """Генерира валиден контекст от n часа."""
    end_ts = datetime(2024, 12, 1, 23, 0, 0)
    context = []
    for i in range(n):
        ts = end_ts - timedelta(hours=n-1-i)
        context.append(ForecastContextRow(
            timestamp=ts.isoformat(),
            consumption_mw=1000.0,
            temperature_c=15.0,
            is_weekend=ts.weekday() >= 5,
            is_holiday=False
        ))
    return context

def test_forecast_context_row_valid():
    row = ForecastContextRow(
        timestamp="2024-12-01T00:00:00",
        consumption_mw=1000.0,
        temperature_c=15.0,
        is_weekend=True,
        is_holiday=False
    )
    assert row.consumption_mw == 1000.0
    assert row.is_weekend is True

def test_forecast_context_row_invalid_missing_field():
    with pytest.raises(ValidationError):
        ForecastContextRow(
            timestamp="2024-12-01T00:00:00",
            temperature_c=15.0,
            is_weekend=True,
            is_holiday=False
        )

def test_forecast_request_valid():
    context = make_context(168)
    req = ForecastRequest(context=context, horizon_hours=24)
    assert len(req.context) == 168
    assert req.horizon_hours == 24

def test_forecast_request_default_horizon():
    context = make_context(168)
    req = ForecastRequest(context=context)
    assert req.horizon_hours == 24

def test_forecast_request_too_short():
    """Тества, че контекст < 168 хвърля ValidationError."""
    context = make_context(10)
    with pytest.raises(ValidationError):
        ForecastRequest(context=context)

def test_batch_forecast_request_valid():
    context = make_context(168)
    single_req = ForecastRequest(context=context)
    batch_req = BatchForecastRequest(requests=[single_req])
    assert len(batch_req.requests) == 1