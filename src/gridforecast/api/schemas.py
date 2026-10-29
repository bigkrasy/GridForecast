"""
GridForecast — Pydantic schemas за API.

Дизайн:
- `PredictRequest`: подаваш САМО raw features + history. API-то само
  извиква `create_features()` и взема последния ред.
- `PredictResponse`: predicted_mw + timestamp + model info.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# Single prediction
# ============================================================
class HistoryPoint(BaseModel):
    """Една точка от history (за lag/rolling features)."""
    timestamp: datetime
    consumption_mw: float = Field(..., gt=0, description="Консумация в MW (> 0)")


class PredictRequest(BaseModel):
    """
    Заявка за единична прогноза.

    `history` трябва да съдържа последните ≥ 168 часа (за lag_168h + rolling_168h).
    `timestamp` е моментът, за който искаме прогноза (обикновено след последната history точка).
    """
    timestamp: datetime = Field(..., description="Момент на прогнозата (ISO 8601)")
    temperature_c: float = Field(..., description="Температура в °C")
    is_holiday: bool = Field(default=False)
    is_weekend: bool = Field(default=False)
    history: list[HistoryPoint] = Field(
        ...,
        min_length=168,
        description="Минимум 168 часа история (за lag_168h и rolling_168h)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-12-01T12:00:00",
                "temperature_c": 5.5,
                "is_holiday": False,
                "is_weekend": True,
                "history": [
                    {"timestamp": "2024-11-24T12:00:00", "consumption_mw": 1100.0},
                    # ... още 167 реда
                ],
            }
        }
    }


class PredictResponse(BaseModel):
    """Отговор от /predict."""
    model_config = {"protected_namespaces": ()}

    timestamp: datetime
    predicted_mw: float
    model_name: str
    model_version: str
    latency_ms: float


# ============================================================
# Batch prediction
# ============================================================
class BatchPredictRequest(BaseModel):
    """Batch заявка — списък от единични PredictRequest."""
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Отговор от /predict/batch."""
    model_config = {"protected_namespaces": ()}

    predictions: list[PredictResponse]
    n_predictions: int
    total_latency_ms: float


# ============================================================
# Health + model info
# ============================================================
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    timestamp: str


class DriftSummaryRow(BaseModel):
    """Един ред от drift_summary.parquet."""
    method: str
    value: float
    threshold: float
    status: str
    weight: str
    note: str = ""


class ModelInfoResponse(BaseModel):
    """Model metadata + drift summary."""
    model_config = {"protected_namespaces": ()}

    model_name: str
    model_version: str
    n_features: int
    feature_columns: list[str]
    fit_mape: float
    baseline_mape: float
    retraining_strategy: str
    loaded_at: str
    drift_summary: list[DriftSummaryRow] = Field(
        default_factory=list,
        description="Последен drift check (ако има drift_summary.parquet)",
    )