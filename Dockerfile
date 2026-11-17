# Serves the FastAPI fare predictor. Port 7860 is the Hugging Face Spaces default.
FROM python:3.12-slim

# XGBoost needs the OpenMP runtime, which the slim image doesn't include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (uid 1000, as Hugging Face Spaces expects)
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first so this layer is cached when only code changes
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Only what the app needs at runtime
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/xgb_model.json models/
COPY --chown=appuser:appuser data/model_training_columns.json \
     data/taxi_zone_lookup.csv \
     data/distance_lookup.csv \
     data/duration_lookup.csv \
     data/

USER appuser
EXPOSE 7860

CMD ["uvicorn", "src.interface:app", "--host", "0.0.0.0", "--port", "7860"]