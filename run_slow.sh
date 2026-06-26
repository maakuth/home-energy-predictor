#!/bin/bash

set -e

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Use a separate lockfile so this can run alongside run_frequent.sh
(
  flock -n 200 || { echo "⚠️ A slow HEPO task is already running. Skipping."; exit 1; }

  echo "=== Starting HEPO Slow Task: $(date) ==="

  # SARIMA Prediction (Benchmark)
  # This runs the benchmark model and archives predictions for later analysis.
  # Does not affect the house plan — slow tasks are safe to skip/run infrequently.

  # Snapshot shared state files for exclusive slow-task use (cp + atomic mv)
  # so we never compete with run_frequent.sh / run_weekly.sh writes.
  cp state/processed_data.csv state/.processed_data_slow.tmp 2>/dev/null && \
    mv state/.processed_data_slow.tmp state/processed_data_slow.csv 2>/dev/null || true
  cp state/sarima_model_params.pkl state/.sarima_model_params_slow.tmp 2>/dev/null && \
    mv state/.sarima_model_params_slow.tmp state/sarima_model_params_slow.pkl 2>/dev/null || true

  echo "[1/1] Predicting Future (SARIMA Benchmark)..."
  SLOW_DATA_PATH=state/processed_data_slow.csv \
  SLOW_PARAMS_PATH=state/sarima_model_params_slow.pkl \
    python3 sarimax_predictor.py

  echo "=== Slow Task Complete: $(date) ==="

) 200>/tmp/hepo_slow.lock
