#!/bin/bash
set -e

# Change to the project directory
cd "."

# Activate virtual environment
source venv/bin/activate

# Use a lockfile to prevent race conditions with run_frequent.sh
(
  # For daily, we wait for the lock (don't exit immediately like frequent)
  flock 200

  echo "=== Starting Daily HEPO Pipeline: $(date) ==="

  # 1. Extract Data (Full 2 Years for Training)
  echo "[1/5] Extracting Data..."
  python extract_data.py --days 730

  # 2. Process Data
  echo "[2/5] Processing Data..."
  python process_data.py

  # 3. Retrain Model
  echo "[3/6] Retraining XGBoost Model..."
  python train_model.py

  echo "[4/6] Retraining SARIMA Model (14-day window)..."
  python train_sarima.py

  # 5. Predict Future
  echo "[5/7] Predicting Future..."
  python predict_future.py

  # 6. Optimize & Push
  echo "[6/7] Optimizing & Pushing Plan..."
  python optimize_plan.py
  python push_to_ha.py

  # 7. Performance Analysis
  echo "[7/7] Analyzing Performance (Last 7 Days)..."
  python analyze_performance.py --days 7 --backtest

  echo "=== Pipeline Complete: $(date) ==="

) 200>/tmp/hepo.lock
