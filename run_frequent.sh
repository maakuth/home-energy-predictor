#!/bin/bash
set -e

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Starting Frequent HEPO Update: $(date) ==="

# 1. Extract Latest States
echo "[1/4] Extracting Data..."
python extract_data.py

# 2. Process
echo "[2/4] Processing Data..."
python process_data.py

# 3. Predict (Rolling Horizon)
echo "[3/4] Predicting Future..."
python predict_future.py

# 4. Optimize & Push
echo "[4/4] Optimizing & Pushing Plan..."
python optimize_plan.py
python push_to_ha.py

# 5. Optional Accuracy Reflection (could be run less often, but hourly is fine)
echo "[5/5] Analyzing Accuracy..."
python analyze_performance.py

echo "=== Update Complete: $(date) ==="
