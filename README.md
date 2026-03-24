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
  - Current Performance: **MAE ~1.01 kWh** (hourly).

### 3. Inference
- **Prediction (`predict_tomorrow.py`)**:
  - Fetches real-time state (current temp, accumulator temp, EV SOC) and solar forecast from Home Assistant API.
  - Generates a 24-hour hour-by-hour energy consumption forecast.
  - Saves predictions to `tomorrow_predictions.npy`.

### 4. Optimization & Integration
- **Optimization (`optimize_plan.py`)**:
  - Fetches tomorrow's hourly spot prices from Nordpool (`sensor.nordpool_total`).
  - **EV Strategy**: Identifies the 4 cheapest hours for charging.
  - **Heating Strategy**: Boosts GSHP setpoint if price is below the daily 20th percentile.
  - Generates `optimization_plan.json`.
- **Integration (`push_to_ha.py`)**:
  - Pushes the optimization plan to a Home Assistant sensor: `sensor.hepo_optimization_plan`.
  - The plan is stored in the sensor's attributes as a JSON list.

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
