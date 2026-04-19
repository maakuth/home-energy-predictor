# Home Energy Predictor (HEPO)

An ML-powered agent that predicts household energy consumption and optimizes usage against electricity spot prices using a Model Predictive Control (MPC) approach.

## Current Features

### 1. Data Pipeline
- **Extraction (`extract_data.py`)**: Fetches historical sensor states from Home Assistant (PostgreSQL).
  - Supports `--days` argument for fast (e.g., 3-day) or full (365-day) extractions.
- **Processing (`process_data.py`)**:
  - Denoises sensor data with a rolling median filter.
  - Implements **Fireplace Logic** to infer when the fireplace is active.
- **Feedback Loop Storage**: All predictions are archived in a local SQLite database (`hepo.db`) for performance analysis. This includes the full optimization plan: **battery intents, power levels, expected SOC, and assumed market prices.**

### 2. Machine Learning
- **Model (`train_model.py`)**: Uses **XGBoost Regressor** to predict total home consumption (gross load).
  - Features: Outside temp, solar forecast, accumulator temp, fireplace status, EV SOC/position, and temporal features.

### 3. Rolling Horizon Inference & Optimization
- **Prediction (`predict_future.py`)**:
  - Generates a 24-48 hour forecast at 15-minute intervals starting from "now."
  - Converts predicted Power (kW) to Energy (kWh) per interval for accurate battery planning.
- **Optimization (`optimize_plan.py`)**:
  - **Dynamic Price Alignment**: Robustly aligns 15-minute spot prices (e.g., from Nordpool) to prediction intervals.
  - **MPC Strategy**: Re-evaluates the battery dispatch plan every 15-60 minutes based on real-time SOC and consumption.
  - **Granular Battery Actions**:
    - `CHARGE_SOLAR`: Fill from PV surplus.
    - `CHARGE_GRID`: Fill from cheap grid power (arbitrage).
    - `DISCHARGE_LOAD`: Cover house consumption.
    - `DISCHARGE_EXPORT`: Sell stored energy to the grid.

### 4. Performance Reflection
- **Analysis (`analyze_performance.py`)**:
  - Compares archived predictions against actual observed consumption from the HA database.
  - Aggregates results into **3-hour windows** to provide stable Mean Absolute Error (MAE) and Bias metrics.
  - **Battery Evaluation**: Calculates the **Planned ROI** of the battery strategy (Savings in €, Avg Charge/Discharge prices, and price Spread).
  - **Persistence**: Automatically stores analysis results into a dedicated `performance_analysis` table in `hepo.db` for long-term trend tracking and strategic adaptation.
  - Pushes accuracy metrics to `sensor.hepo_accuracy`.

## Usage

### 1. Frequent Optimization (Rolling Horizon)
Run this every 15-60 minutes via Cron to keep the battery and heating plan reactive to real-time changes.
```bash
./run_frequent.sh
```
*Effect: Fast extraction (3 days), predicts from "now", optimizes, and updates HA.*

### 2. Daily Retraining
Run this once a day (e.g., at night or at 18:00) to keep the ML model updated with the latest trends.
```bash
./run_daily.sh
```
*Effect: Full extraction (365 days), retrains XGBoost model, updates forecast.*

### 3. Automation (Cron)
Example crontab for a robust setup:
```bash
# Update optimization plan every 30 minutes
*/30 * * * * /path/to/hepo/run_frequent.sh >> /path/to/hepo/frequent.log 2>&1

# Retrain the model every day at 18:05
5 18 * * * /path/to/hepo/run_daily.sh >> /path/to/hepo/daily.log 2>&1
```

## Configuration
Set these in `.env` (refer to `README.md` for full list):
- `BATTERY_CAPACITY_KWH`: Total usable capacity.
- `GRID_TRANSFER_EUR_PER_KWH`: Variable transfer costs.
- `IMPORT_VAT_MULTIPLIER`: Tax calculations.
- `DATA_RESAMPLE_INTERVAL`: Typically `15min`.

## Future Battery Integration
When the physical battery is installed, the following updates are required:

### 1. Load Calculation (`process_data.py`)
Currently, `total_home_power = grid_power + solar_actual`.
With a battery, you must subtract the battery's net discharge to get the **true house load**:
`total_home_power = grid_power + solar_actual - (battery_discharge_power - battery_charge_power)`

### 2. Closed-Loop SOC (`optimize_plan.py`)
- **Initial State:** Update `fetch_ha_state` to pull the **real-time Battery SOC** from Home Assistant at the start of each optimization run.
- **Constraints:** Fine-tune `BATTERY_CHARGE_EFFICIENCY` and `BATTERY_DISCHARGE_EFFICIENCY` based on real-world data from the inverter.

### 3. Inverter Automation
Use the `battery_action` field from the `sensor.hepo_optimization_plan` to trigger the inverter's operating modes (e.g., via Modbus or a Home Assistant integration).
