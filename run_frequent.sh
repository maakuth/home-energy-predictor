#!/bin/bash

set -e

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Use a lockfile to prevent race conditions with run_daily.sh
(
  flock -n 200 || { echo "⚠️ Another HEPO process is running. Skipping frequent update."; exit 1; }

  echo "=== Starting Frequent HEPO Update: $(date) ==="
  
  # 1. Extract Latest States (only 3 days for speed)
  echo "[1/5] Extracting Data..."
  python3 extract_data.py --days 3
  
  # 2. Process
  echo "[2/5] Processing Data..."
  python3 process_data.py
  
  # 3. Predict (Rolling Horizon) - Existing XGBoost/ML prediction
  echo "[3/5] Predicting Future (Existing ML)..."
  python3 predict_future.py
  
  # 4. SARIMA Prediction (Benchmark)
  # This runs in parallel but doesn't affect the house plan
  echo "[4/5] Predicting Future (SARIMA Benchmark)..."
  python3 sarimax_predictor.py
  
  # 5. Optimize & Push (Uses XGBoost output from predict_future.py)
  echo "[5/5] Optimizing & Pushing Plan..."
  python3 optimize_plan.py
  python3 push_to_ha.py

  # Reset net metering state when crossing a 15-minute boundary
  # so each quarter starts clean. Uses epoch interval index
  # (epoch_seconds / 900) instead of clock position, so SARIMAX
  # delays don't cause us to skip the reset.
  track_file="/tmp/hepo_net_metering_interval"
  current_interval=$(($(date +%s) / 900))

  if [ ! -f "$track_file" ] || [ "$(cat "$track_file")" -lt "$current_interval" ]; then
    echo '{}' > net_metering_state.json
    echo "$current_interval" > "$track_file"
  fi

  echo "=== Update Complete: $(date) ==="

) 200>/tmp/hepo.lock
