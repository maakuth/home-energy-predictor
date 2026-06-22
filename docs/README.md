# Home Energy Predictor (HEPO) v1.7.0

An ML-powered agent that predicts household energy consumption and optimizes usage against electricity spot prices using a Model Predictive Control (MPC) approach.

> **Note:** This is a custom tool built for personal use. There are no plans to genericize it, but it may serve as inspiration. 

Built with a mixture of LLM agents using a number of models, especially Gemini 3 series and a number of OpenCode Zen models.

Licensed under AGPL-3.0.

## Features

### 1. Data Pipeline
- **Extraction (`extract_data.py`)**: Fetches historical sensor states from Home Assistant (PostgreSQL). Supports `--days` for fast (3-day) or full (365-730 day) extractions.
- **Processing (`process_data.py`)**: Denoises sensor data with a rolling median filter, infers fireplace activity, and computes derived features.
- **Feedback Loop**: All predictions and optimization plans are archived in `hepo.db` (SQLite) for performance analysis.

### 2. Machine Learning
- **XGBoost Regressor (`train_model.py`)**: Predicts total home consumption (gross load) using features like outside temp, solar forecast, accumulator temp, fireplace status, EV SOC/position, and temporal features.
- **SARIMAX Benchmark (`train_sarima.py`, `sarimax_predictor.py`)**: Fits a SARIMAX model on historical baseload as a benchmark/ensemble comparison. Trained and saved separately from XGBoost.
- **Model Versioning**: Semantic versioning stored in `VERSION` file (not tied to git). All predictions and analysis are tagged with the version for performance tracking.

### 3. Rolling Horizon Inference & Optimization
- **Prediction (`predict_future.py`)**: Generates a 24-48 hour forecast at 15-minute intervals starting from "now". Converts predicted power (kW) to energy (kWh) per interval.
- **Optimization (`optimize_plan.py`)**: MPC-based battery dispatch planning with:
  - **Dynamic Price Alignment**: Robustly aligns 15-minute Nordpool spot prices to prediction intervals.
  - **Battery Actions**: `CHARGE_SOLAR`, `CHARGE_GRID`, `DISCHARGE_LOAD`, `DISCHARGE_EXPORT`.
  - **GSHP Optimization**: Strategic stop/start based on price lookahead.
  - **EV Charging**: Price-aware charging scheduling.
  - **Fuse Limit Enforcement**: Grid import capped by main fuse rating.

### 4. Battery Planners
Pluggable architecture under `battery_planners/`:
- **Heuristic Planner**: Marginal opportunity-cost ranking with gradual ramps.
- **Nemotron-Linprog Planner**: LP-based dispatch using `scipy.optimize.linprog` (HiGHS) with configurable horizon, discount, terminal value, and degradation cost.
- **Factory pattern** (`factory.py`) selects planner via `BATTERY_PLANNER_TYPE` env var.

### 5. Performance Analysis
- **`analyze_performance.py`**: Compares predictions vs. actual consumption, computes MAE/bias in 3-hour windows, calculates battery ROI (planned savings, spread), and stores results in `performance_analysis` table.
- **`analyze_evolution.py`**: Tracks how plans evolve as target time approaches — MAE by lead time, stability (stddev of predictions, action flips), and planned cost evolution.
- **Strategic Adaptation**: Historical metrics can guide tuning (e.g., positive bias → optimizer too conservative; decreasing spread → discharge thresholds need adjustment).

## Usage

### Frequent Optimization (every 15-60 min)
```bash
./run_frequent.sh
```
Extracts 3 days, predicts from "now", optimizes battery/GSHP/EV, pushes to HA.

### Mid-Cycle Update (every 2-5 min)
```bash
./run_often.py
```
Lightweight: reads current HA state, computes battery setpoint from existing plan, pushes single power command.

### Weekly Retraining
```bash
./run_weekly.sh
```
Extracts 730 days, retrains XGBoost + SARIMA, runs analysis for last 14 days.

### Automation (Cron)
```bash
# Battery setpoint update every 5 minutes
*/5 * * * * /path/to/hepo/.venv/bin/python3 /path/to/hepo/run_often.py >> /path/to/hepo/frequent.log 2>&1

# Full re-optimization every 30 minutes
*/30 * * * * /path/to/hepo/run_frequent.sh >> /path/to/hepo/frequent.log 2>&1

# Weekly retraining every Sunday at 04:00
0 4 * * 0 /path/to/hepo/run_weekly.sh >> /path/to/hepo/weekly.log 2>&1
```

## Configuration

Copy `.env.template` to `.env` and configure:
- Database credentials (PostgreSQL)
- Home Assistant host/token
- Electricity pricing, taxes, VAT, export deductions
- Battery capacity, efficiency, SOC limits, grid-charge margin, net metering
- GSHP temperatures, COP, heat loss, strategic stop threshold
- EV target SOC, capacity, charge power
- LP planner horizon, discount, terminal value, degradation cost

See `docs/ENV_VARIABLES.md` for detailed documentation of all parameters.

## Project Structure

| File | Purpose |
|------|---------|
| `extract_data.py` | Fetch sensor data from HA PostgreSQL |
| `process_data.py` | Clean, denoise, feature-engineer raw data |
| `train_model.py` | Train XGBoost regression model |
| `predict_future.py` | Generate consumption forecast |
| `optimize_plan.py` | MPC battery/GSHP/EV optimizer |
| `train_sarima.py` | Train SARIMAX benchmark model |
| `sarimax_predictor.py` | SARIMAX forecast generation |
| `push_to_ha.py` | Push plan and metrics to HA sensors |
| `analyze_performance.py` | MAE/bias/ROI analysis |
| `analyze_evolution.py` | Plan stability & evolution tracking |
| `battery_planners/` | Pluggable battery dispatch algorithms |
| `check_battery_state.py` | Quick HA battery SOC/power diagnostic |
| `compare_models.py` | Compare XGBoost vs SARIMA performance |
| `hepo.db` | SQLite database for predictions & metrics |

## Tests

```bash
venv/bin/python3 -m pytest
# Fast subset (skip slow SARIMA tests):
venv/bin/python -m pytest -k 'not sarima'
```

Run full suite before committing. See `AGENTS.md` for development guidelines.
