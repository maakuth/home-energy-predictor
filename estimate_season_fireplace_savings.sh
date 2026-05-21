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

# Use a lockfile to prevent race conditions with other scripts
(
  flock -n 200 || { echo "⚠️ Another HEPO process is running. Skipping seasonal update."; exit 1; }

  echo "=== Starting Seasonal Data Extraction: $(date) ==="
  
  # 1. Extract Data (365 days for the whole season)
  echo "[1/2] Extracting Data for 365 days..."
  python3 extract_data.py --days 365
  
  # 2. Process
  echo "[2/3] Processing Data..."
  python3 process_data.py
  
  # 3. Estimate Fireplace Savings
  echo "[3/3] Estimating Fireplace Savings..."
  python3 estimate_fireplace_savings.py
  
  echo "=== Estimation Complete: $(date) ==="
  echo "The raw_data.csv and processed_data.csv files were updated and savings estimated."

) 200>/tmp/hepo.lock
