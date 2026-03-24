# Technical Specification: Home Energy Prediction & Optimization (HEPO)

## 1. System Overview
The goal is to predict the next 24 hours of household power consumption (with a focus on heating and EV charging) to optimize usage against tomorrow's electricity spot prices. The system will run on a local PC with access to a Home Assistant (PostgreSQL) database.

## 2. Data Engineering & Cleaning

### Input Entities (from Home Assistant PostgreSQL)
- **Environment:** `sensor.outside_temperature`
- **Ground Source Heat Pump (GSHP) Power:** `sensor.mlp_teho`
- **Air-to-Air Heat Pump (AAHP) Power:**
  - `sensor.saikaan_olohuone_current_power`
  - `sensor.mokkimokin_ilp_power`
  - `sensor.mummun_energy` (**Derived Power:** Calculated as the delta of energy total between state updates, $\Delta \text{kWh} / \Delta t$).
- **Thermal Storage:** `sensor.mlp_varaajan_lampotila` (Accumulator top temperature)
- **EV State:**
  - `sensor.xpz_491_battery_level` (SOC)
  - `device_tracker.xpz_491_position` (Is Home?)
- **Total Household Power:** `sensor.sahkokauppa_nyt`
- **Solar Forecast:** `sensor.solcast_pv_forecast_forecast_tomorrow` (Solcast API integration in HA).

### Pipeline Steps
1. **Extraction:** SQL query to pull the last 365 days of state history for the above entities.
2. **Denoising:** 
   - Apply a Rolling Median Filter (window=3) to remove "insane" spikes.
   - Clip values based on physical limits (e.g., Power > 0, Temp < 50°C).
3. **Resampling:** Aggregate all data to 1-hour intervals.
   - **Continuous values (Temp/Power):** `mean()`
   - **State values (EV Home):** `max()` (If home at any point in the hour, consider home).
4. **Gap Filling:** Linear interpolation for gaps < 2 hours; drop/ignore larger gaps to avoid synthetic bias.

## 3. Feature Engineering: The Fireplace Logic
To isolate the heating system's relationship with outside temperature, the agent must infer "Fireplace Assistance."
- **Logic:** $\text{RoC} = \Delta T_{acc}(t) - T_{acc}(t-1)$.
- **Identify "Fireplace Hours":** If $\text{RoC} > 0.3^\circ\text{C/hr}$ AND $(\text{GSHP\_Power} + \text{AAHP\_Power}) < \text{Threshold}$, then `is_fireplace_active = True`.
- **Heating Demand Factor:** Create a feature `Pheat_norm` which represents the power used when the fireplace is off. This helps the model learn the true heat-loss coefficient of the building.

## 4. Model Specification
Implement a Gradient Boosted Regressor (XGBoost or LightGBM) for its ability to handle non-linear efficiencies (COP) of heat pumps.

### Model Features (X)
- **Weather:** `outside_temp`, `sensor.solcast_pv_forecast_forecast_tomorrow`.
- **Thermal State:** `accumulator_temp`, `is_fireplace_active` (lagged).
- **EV State:** `is_home`, `soc_needed` (Target SOC - Current SOC).
- **Temporal:** `hour_of_day`, `day_of_week`, `month` (to proxy ground temp for GSHP).

### Target (Y)
- `total_power_usage_kwh` (hourly).

## 5. Deployment & Execution Loop
The CLI agent will orchestrate the following script cycle at 18:00 daily.

### Step 1: Data Refresh
Sync the latest 24 hours of HA data and the 24-hour solar/weather forecast.

### Step 2: Inference
Generate a 24-hour vector $[P_1, P_2, \dots, P_{24}]$ representing predicted power usage for tomorrow.

### Step 3: Optimization Logic
Compare prediction against tomorrow’s spot prices (€/kWh):
- **EV Strategy:** Identify the $N$ cheapest hours while `ev_is_home == True` to reach target SOC.
- **Thermal Strategy:** If spot price is in the bottom 20th percentile, increase GSHP setpoint by $2^\circ\text{C}$ to charge the 500L accumulator (thermal buffering).

### Step 4: HA Integration
Push the "Optimization Plan" back to Home Assistant via REST API or MQTT to trigger automations.

## 6. Feedback & Retraining
- **Accuracy Check:** Every day at 17:55, calculate the Mean Absolute Error (MAE) of the previous day's forecast.
- **Automatic Retraining:** If the 7-day rolling MAE increases by $>20\%$, trigger a full model retrain on the most recent 90 days of data.

## Implementation Stack
- **Language:** Python 3.10+
- **DB Connector:** `psycopg2` or `SQLAlchemy`
- **ML Libraries:** `scikit-learn`, `xgboost`, `pandas`
- **API:** Home Assistant REST API
