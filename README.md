# Home Energy Predictor (HEPO)

An ML-powered agent that predicts household energy consumption and optimizes usage against electricity spot prices.

## Current Features

### 1. Data Pipeline
- **Extraction (`extract_data.py`)**: Pulls the last 365 days of sensor history from Home Assistant (PostgreSQL).
- **Processing (`process_data.py`)**:
  - Denoises sensor data with a rolling median filter.
  - Implements **Fireplace Logic** to infer when the fireplace is active (based on accumulator temperature RoC and low heat pump power).
  - Feature engineering: temporal features (hour, day, month), lagged states.

### 2. Machine Learning
- **Model (`train_model.py`)**:
  - Uses **XGBoost Regressor**.
  - Features: Outside temp, solar forecast, accumulator temp, fireplace status, EV SOC/position, temporal features.
  - Current performance should be measured per prediction interval (default 15 min).

### 3. Inference
- **Prediction (`predict_tomorrow.py`)**:
  - Fetches real-time state (current temp, accumulator temp, EV SOC) and solar forecast from Home Assistant API.
  - Generates a 24-hour forecast at 15-minute intervals (96 points by default).
  - Saves predictions to `tomorrow_predictions.npy`.

### 4. Optimization & Integration
- **Optimization (`optimize_plan.py`)**:
  - Fetches market prices from Home Assistant and aligns them to 15-minute intervals (configurable), preferring `sensor.current_electricity_market_price` (fallback: Nordpool sensors).
  - Fetches solar forecast from Solcast (`sensor.solcast_pv_forecast_forecast_tomorrow`).
  - Builds asymmetric tariff prices:
    - **Import price** = market price + transfer + tax + adders (optional VAT multiplier)
    - **Export price** = market price - export deductions
  - Runs battery dispatch planning with configurable battery constraints (default capacity 40 kWh):
    - SOC window, reserve SOC, max charge/discharge power, charge/discharge efficiency
    - Solar surplus charging, expensive-interval discharge, optional export arbitrage
  - **EV Strategy**: Identifies the cheapest charging intervals corresponding to 4 hours of charging time (configurable).
  - **Heating Strategy**: Boosts GSHP setpoint if the **effective price** is below the daily 20th percentile.
    - *Effective Price* is considered 0.0 €/kWh if solar production > 0.5 kWh.
  - Generates `optimization_plan.json` with legacy EV/heat flags and battery fields (`battery_action`, `soc_kwh`, `soc_pct`, `grid_import_kwh`, `grid_export_kwh`, `estimated_hour_cost`, `estimated_hour_savings`, etc.).
- **Integration (`push_to_ha.py`)**:
  - Pushes the optimization plan to a Home Assistant sensor: `sensor.hepo_optimization_plan`.
  - The plan is stored in the sensor's attributes as a JSON list.

## Battery/Tariff Configuration

Set these in `.env` to tune battery dispatch economics and constraints:

- `BATTERY_CAPACITY_KWH` (default `40.0`)
- `BATTERY_MIN_SOC_PCT` (default `10.0`)
- `BATTERY_MAX_SOC_PCT` (default `90.0`)
- `BATTERY_RESERVE_SOC_PCT` (default `BATTERY_MIN_SOC_PCT`)
- `BATTERY_INITIAL_SOC_PCT` (default `50.0`)
- `BATTERY_MAX_CHARGE_KW` (default `10.0`)
- `BATTERY_MAX_DISCHARGE_KW` (default `10.0`)
- `BATTERY_CHARGE_EFFICIENCY` (default `0.95`)
- `BATTERY_DISCHARGE_EFFICIENCY` (default `0.95`)
- `BATTERY_ALLOW_EXPORT` (default `true`)

- `GRID_TRANSFER_EUR_PER_KWH` (default `0.0`)
- `ELECTRICITY_TAX_EUR_PER_KWH` (default `0.0`)
- `IMPORT_FIXED_ADDERS_EUR_PER_KWH` (default `0.0`)
- `IMPORT_VAT_MULTIPLIER` (default `1.0`)
- `EXPORT_DEDUCTION_EUR_PER_KWH` (default `0.0`)

- `DATA_RESAMPLE_INTERVAL` (default `15min`)
- `PREDICTION_INTERVAL_MINUTES` (default `15`)
- `PLAN_INTERVAL_MINUTES` (default `15`)
- `PLAN_INTERVAL_HOURS` (default derived from `PLAN_INTERVAL_MINUTES`)
- `EV_CHARGE_HOURS` (default `4.0`)

## Usage

### Manual Execution
1. Configure `.env` with DB and HA credentials.
2. Run the full pipeline:
   ```bash
   ./run_daily.sh
   ```

### Automation (Cron)
To run the pipeline daily at 18:00 (after spot prices are published), add this to your crontab:
```bash
0 18 * * * /path/to/home-energy-predictor/run_daily.sh >> /path/to/home-energy-predictor/hepo.log 2>&1
```
