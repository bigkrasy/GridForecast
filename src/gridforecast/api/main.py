"""
GridForecast API — прогнозиране на потребление на електроенергия.

Endpoints:
  GET  /              → health check
  GET  /health        → health check (alias)
  GET  /model-info    → model metadata + drift summary
  GET  /metrics       → Prometheus metrics
  POST /predict       → single prediction
  POST /predict/batch → batch prediction
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from gridforecast.api.inference import ModelStore, predict_batch, predict_single
from gridforecast.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftSummaryRow,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from gridforecast.config import MODELS_DIR

logger = logging.getLogger("gridforecast.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ============================================================
# Metrics (simple)
# ============================================================
_metrics = {
    "requests_total": 0,
    "predictions_total": 0,
    "errors_total": 0,
    "start_time": time.time(),
}


# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 GridForecast API starting...")
    try:
        ModelStore.load()
        logger.info(
            f"✅ Model loaded: {ModelStore.metadata['model_name']} "
            f"({ModelStore.metadata['n_features']} features, "
            f"MAPE={ModelStore.metadata['fit_mape']:.2f}%)"
        )
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        raise
    yield
    ModelStore.clear()
    logger.info("👋 GridForecast API stopped")


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="GridForecast API",
    description="Прогнозиране на потребление на електроенергия (MW)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    _metrics["requests_total"] += 1
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            _metrics["errors_total"] += 1
        return response
    except Exception:
        _metrics["errors_total"] += 1
        raise


# ============================================================
# Endpoints
# ============================================================
@app.get("/", response_model=HealthResponse, tags=["health"])
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    return HealthResponse(
        status="healthy" if ModelStore.is_loaded() else "degraded",
        model_loaded=ModelStore.is_loaded(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
async def model_info():
    if not ModelStore.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    # Зареди drift_summary.parquet (ако има)
    drift_summary: list[DriftSummaryRow] = []
    drift_path = MODELS_DIR / "drift_summary.parquet"
    if drift_path.exists():
        try:
            df = pd.read_parquet(drift_path)
            for row in df.to_dict(orient="records"):
                drift_summary.append(
                    DriftSummaryRow(
                        method=str(row.get("method", "")),
                        value=float(row.get("value", 0.0)),
                        threshold=float(row.get("threshold", 0.0)),
                        status=str(row.get("status", "")),
                        weight=str(row.get("weight", "")),
                        note=str(row.get("note", "")),
                    )
                )
        except Exception as e:
            logger.warning(f"Не мога да заредя drift_summary: {e}")

    m = ModelStore.metadata
    return ModelInfoResponse(
        model_name=m["model_name"],
        model_version="v2",
        n_features=m["n_features"],
        feature_columns=ModelStore.feature_cols or [],
        fit_mape=m["fit_mape"],
        baseline_mape=m["baseline_mape"],
        retraining_strategy=m["retraining_strategy"],
        loaded_at=m["loaded_at"],
        drift_summary=drift_summary,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(request: PredictRequest):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    try:
        history = [h.model_dump() for h in request.history]
        y_pred = predict_single(
            timestamp=request.timestamp,
            temperature_c=request.temperature_c,
            is_holiday=request.is_holiday,
            is_weekend=request.is_weekend,
            history=history,
        )

        _metrics["predictions_total"] += 1
        latency_ms = (time.time() - start) * 1000

        return PredictResponse(
            timestamp=request.timestamp,
            predicted_mw=y_pred,
            model_name=ModelStore.metadata["model_name"],
            model_version="v2",
            latency_ms=round(latency_ms, 2),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["prediction"],
)
async def predict_batch_endpoint(request: BatchPredictRequest):
    if not ModelStore.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    try:
        req_dicts = [
            {
                "timestamp": r.timestamp,
                "temperature_c": r.temperature_c,
                "is_holiday": r.is_holiday,
                "is_weekend": r.is_weekend,
                "history": [h.model_dump() for h in r.history],
            }
            for r in request.requests
        ]
        y_preds = predict_batch(req_dicts)

        _metrics["predictions_total"] += len(y_preds)
        total_latency_ms = (time.time() - start) * 1000

        predictions = [
            PredictResponse(
                timestamp=req.timestamp,
                predicted_mw=pred,
                model_name=ModelStore.metadata["model_name"],
                model_version="v2",
                latency_ms=0,
            )
            for req, pred in zip(request.requests, y_preds)
        ]

        return BatchPredictResponse(
            predictions=predictions,
            n_predictions=len(predictions),
            total_latency_ms=round(total_latency_ms, 2),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
async def metrics():
    uptime = time.time() - _metrics["start_time"]
    loaded = 1 if ModelStore.is_loaded() else 0
    mape = ModelStore.metadata.get("fit_mape", 0) if ModelStore.metadata else 0

    lines = [
        "# HELP gridforecast_requests_total Total HTTP requests",
        "# TYPE gridforecast_requests_total counter",
        f"gridforecast_requests_total {_metrics['requests_total']}",
        "",
        "# HELP gridforecast_predictions_total Total predictions made",
        "# TYPE gridforecast_predictions_total counter",
        f"gridforecast_predictions_total {_metrics['predictions_total']}",
        "",
        "# HELP gridforecast_errors_total Total HTTP errors",
        "# TYPE gridforecast_errors_total counter",
        f"gridforecast_errors_total {_metrics['errors_total']}",
        "",
        "# HELP gridforecast_uptime_seconds Service uptime",
        "# TYPE gridforecast_uptime_seconds gauge",
        f"gridforecast_uptime_seconds {uptime:.2f}",
        "",
        "# HELP gridforecast_model_loaded Model loaded flag",
        "# TYPE gridforecast_model_loaded gauge",
        f"gridforecast_model_loaded {loaded}",
        "",
        "# HELP gridforecast_model_mape Model MAPE (%)",
        "# TYPE gridforecast_model_mape gauge",
        f"gridforecast_model_mape {mape}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    from gridforecast.config import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)