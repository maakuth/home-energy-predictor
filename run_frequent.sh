#!/bin/bash

set -e

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Explicit archive mode is dry-run by default. The Python runner owns the same
# lock and requires --execute plus separate --approve-publish for live controls.
for arg in "$@"; do
  case "$arg" in
    --archive|--archive=*) exec python3 -m utils.archive_pipeline --mode frequent "$@" ;;
  esac
done
if [ "$#" -ne 0 ]; then
  echo "Use --archive PATH for archive mode, or no arguments for CSV rollback." >&2
  exit 2
fi

# Optional, non-actuating forecast snapshot collection. This is disabled unless
# FORECAST_SNAPSHOT_SYSTEMD_UNIT names an independently installed service or
# timer target. It is triggered before optimization work, but --no-block keeps
# it out of this optimization cgroup and it cannot delay or fail this run.
# Trigger failures are reported and ignored.
if [ -n "${FORECAST_SNAPSHOT_SYSTEMD_UNIT:-}" ]; then
  if ! systemctl --user start --no-block "$FORECAST_SNAPSHOT_SYSTEMD_UNIT"; then
    echo "⚠️ Forecast snapshot trigger failed for $FORECAST_SNAPSHOT_SYSTEMD_UNIT" >&2
  fi
fi

# Use a lockfile to prevent race conditions with run_daily.sh
(
  flock -n 200 || { echo "⚠️ Another HEPO process is running. Skipping frequent update."; exit 1; }

  echo "=== Starting Frequent HEPO Update: $(date) ==="

  # Reset net metering state when crossing a 15-minute boundary
  # so each quarter starts clean. Done up front so a later pipeline
  # failure cannot leave a stale baseline accumulating in period balance.
  # Uses epoch interval index (epoch_seconds / 900) instead of clock position.
  mkdir -p state
  track_file="state/hepo_net_metering_interval"
  current_interval=$(($(date +%s) / 900))

  if [ ! -f "$track_file" ] || [ "$(cat "$track_file")" -lt "$current_interval" ]; then
    echo '{}' > state/net_metering_state.json
    echo "$current_interval" > "$track_file"
  fi

  # 1. Extract Latest States (only 3 days for speed)
  echo "[1/4] Extracting Data..."
  python3 extract_data.py --days 3
  
  # 2. Process
  echo "[2/4] Processing Data..."
  python3 process_data.py
  
  # 3. Predict (Rolling Horizon) - Existing XGBoost/ML prediction
  echo "[3/4] Predicting Future (Existing ML)..."
  python3 predict_future.py
  
  # 4. Optimize & Push (Uses XGBoost output from predict_future.py)
  echo "[4/4] Optimizing & Pushing Plan..."
  python3 optimize_plan.py
  HEPO_PUBLISH_LOCK_FD=200 HEPO_PIPELINE_LOCK_PATH=/tmp/hepo.lock python3 push_to_ha.py

  echo "=== Update Complete: $(date) ==="

) 200>/tmp/hepo.lock
