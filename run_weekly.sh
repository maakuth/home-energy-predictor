#!/bin/bash
set -e

# Change to the project directory
cd "."

# Activate virtual environment
source .venv/bin/activate

# Use a lockfile to prevent race conditions with run_frequent.sh
(
  # For weekly, we wait for the lock (don't exit immediately like frequent)
  flock 200

  echo "=== Starting Weekly HEPO Pipeline: $(date) ==="

  # 1. Extract Data (Full 2 Years for Training)
  echo "[1/6] Extracting Data (730 days for training)..."
  python extract_data.py --days 730

  # 2. Process Data
  echo "[2/6] Processing Data..."
  python process_data.py

  # 3. Retrain XGBoost Model
  echo "[3/6] Retraining XGBoost Model (full dataset)..."
  python train_model.py

  # 4. Retrain SARIMA Model (extended 30-day window for better stability)
  echo "[4/6] Retraining SARIMA Model (30-day window for weekly runs)..."
  python train_sarima.py --days 30

  # 5. Predict Future
  echo "[5/6] Predicting Future..."
  python predict_future.py

  # 6. Optimize & Push
  echo "[6/6] Optimizing & Pushing Plan..."
  python optimize_plan.py
  python push_to_ha.py

  # 7. Performance Analysis (weekly: analyze 14 days)
  echo "[7/6] Analyzing Performance (Last 14 Days)..."
  python analyze_performance.py --days 14 --backtest

  # 8. Evolution Analysis (weekly: analyze 14 days)
  echo "[8/6] Analyzing Plan Evolution (Last 14 Days)..."
  python analyze_evolution.py --days 14

  echo "=== Weekly Pipeline Complete: $(date) ==="

) 200>/tmp/hepo.lock
