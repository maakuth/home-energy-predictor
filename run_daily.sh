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
  echo "[3/5] Retraining Model..."
  python train_model.py

  # 4. Predict Future
  echo "[4/5] Predicting Future..."
  python predict_future.py

  # 5. Optimize & Push
  echo "[5/5] Optimizing & Pushing Plan..."
  python optimize_plan.py
  python push_to_ha.py

  # 6. Performance Analysis
  echo "[6/6] Analyzing Performance (Last 7 Days)..."
  python analyze_performance.py --days 7 --backtest

  echo "=== Pipeline Complete: $(date) ==="

) 200>/tmp/hepo.lock
