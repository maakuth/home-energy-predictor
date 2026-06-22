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
  echo "[1/1] Predicting Future (SARIMA Benchmark)..."
  python3 sarimax_predictor.py

  echo "=== Slow Task Complete: $(date) ==="

) 200>/tmp/hepo_slow.lock
