#!/bin/bash
set -e

# Change to the project directory
cd "."

# Activate virtual environment
source venv/bin/activate

echo "=== Starting Daily HEPO Pipeline: Tue Mar 24 13:12:24 UTC 2026 ==="

# 1. Extract Data
echo "[1/5] Extracting Data..."
python extract_data.py

# 2. Process Data
echo "[2/5] Processing Data..."
python process_data.py

# 3. Retrain Model (Optional - could be conditional)
echo "[3/5] Retraining Model..."
python train_model.py

# 4. Predict Tomorrow
echo "[4/5] Predicting Tomorrow..."
#python predict_tomorrow.py
python predict_future.py

# 5. Optimize & Push
echo "[5/5] Optimizing & Pushing Plan..."
python optimize_plan.py
python push_to_ha.py

echo "=== Pipeline Complete: Tue Mar 24 13:12:24 UTC 2026 ==="
