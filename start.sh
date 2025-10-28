#!/usr/bin/env bash
set -e

# Forcer l'entraînement si demandé
if [ "${FORCE_TRAIN:-false}" = "true" ]; then
  rm -rf artifacts/*
fi

# Entraîner si artefacts absents
if [ ! -f "artifacts/model.joblib" ]; then
  echo "[start.sh] No artifacts found — training a fresh model..."
  python scripts/fetch_german_credit.py || true
  python src/train/train.py
fi

: "${PORT:=8000}"
: "${THRESHOLD:=0.05}"
: "${LOG_AUDIT:=true}"

echo "[start.sh] Starting API on 0.0.0.0:${PORT} (THRESHOLD=${THRESHOLD}, LOG_AUDIT=${LOG_AUDIT})"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
