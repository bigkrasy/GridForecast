# ============================================================
# GridForecast — Production Dockerfile (multi-stage, non-root)
# ============================================================

# ─── Stage 1: Builder ───
FROM python:3.11-slim AS builder

WORKDIR /build

# Build deps (нужни за компилиране на pandas/numpy/lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip wheel --no-cache-dir -r requirements-prod.txt -w /wheels

# ─── Stage 2: Runtime ───
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Runtime deps (libgomp1 е задължителен за LightGBM, curl за healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Project (src layout)
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .   # НЕ -e (production)

# Test config (махни .coveragerc ако липсва)
COPY pytest.ini .
# COPY .coveragerc .   # ← разкоментирай, ако имаш файла

# Runtime dirs
RUN mkdir -p /app/models /app/data/synthetic /app/logs

# Non-root user
RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "gridforecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]