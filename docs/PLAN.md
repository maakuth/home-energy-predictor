# Technical Specification: Home Energy Prediction & Optimization (HEPO)

## 1. System Overview
HEPO (Home Energy Prediction & Optimization) is an ML-powered system designed to predict household energy consumption and optimize the operation of a Ground Source Heat Pump (GSHP) and a home battery system. It uses a **Model Predictive Control (MPC)** approach with a 15-minute rolling horizon to minimize electricity costs against volatile spot prices (e.g., Nordpool).

## 2. Data Engineering & Pipeline

### Input Entities (from Home Assistant PostgreSQL)
- **Environment:** `sensor.outside_temperature`
- **GSHP Power:** `sensor.mlp_teho` (Watts)
- **AAHP Power:** `sensor.saikaan_olohuone_current_power`, `sensor.mokkimokin_ilp_power`
- **Thermal Storage:** `sensor.mlp_varaajan_lampotila` (500L Accumulator)
- **Sauna:** `sensor.sauna_kiuas_lampotila` (Used to infer sauna activity)
- **EV State:** `sensor.xpz_491_battery_level` (SOC), `device_tracker.xpz_491_position` (Home/Away)
- **Home Battery:** `sensor.be_soc` (State of Charge %), `sensor.be_stat_batt_power` (Power in Watts, positive=charging from grid/solar, negative=discharging to home)
- **Grid Power:** `sensor.sahkokauppa_nyt` (Bidirectional grid meter)
- **Solar Production:** `sensor.solcast_pv_forecast_actual_power` (Actual) and `sensor.solcast_pv_forecast_forecast_tomorrow` (Forecast).

### Pipeline Steps
1. **Extraction (`extract_data.py`):** Fetches historical state history. Supports fast 3-day sync or full 365-day extraction.
2. **Denoising (`process_data.py`):** Applies a 15-point Rolling Median Filter to remove sensor spikes.
3. **Resampling:** Aggregates data to 15-minute intervals.
4. **Calculations (Battery-Aware):** 
   - `total_home_power = grid_power + solar_actual - battery_power`
   - `baseload_power = total_home_power - (gshp_power / 1000) - (leaf_power / 1000)` (Target for the ML model)
   - **Critical:** Battery power MUST be subtracted to get true home load. Without this correction:
     - When battery charges from solar: home load is calculated as if solar went to house (wrong!)
     - When battery discharges: home load is underestimated (wrong!)
     - Model learns incorrect patterns from inflated/deflated load values
   - **Convention:** Battery power positive = charging (from grid/solar), negative = discharging (to home).
   - **Backward Compatibility:** If battery sensor unavailable, battery_power defaults to 0.

## 3. Machine Learning & Feature Engineering

### Model Specification (`train_model.py`)
- **Algorithm:** XGBoost Regressor.
- **Target (Y):** `baseload_power` (kW).
- **Weighting:** Data after Oct 1, 2025 (structural expansion) is weighted 3x to prioritize recent building performance.

### Features (X)
- **Weather:** `outside_temp`, `wind_speed`, `solar_forecast`.
- **Thermal State:** `accumulator_temp`, `acc_roc` (Rate of Change), `is_fireplace_lag1` (Fireplace inference).
- **EV State:** `ev_soc`, `ev_position`.
- **Anchors:** `baseload_lag_1h`, `baseload_lag_24h` (Lagged baseload for autocorrelation).
- **Temporal:** `hour`, `quarter_hour`, `day_of_week`, `month`.
- **Context:** `is_extended_complex` (Boolean flag for building expansion), `is_sauna_active`.

### Fireplace & Sauna Logic
- **Fireplace:** Inferred if `acc_roc > 0.3°C/hr` while heat pump power is low.
- **Sauna:** Inferred from sauna temperature sensor. If inactive, a **Sauna Heuristic** projects potential usage during typical weekend/evening windows.

## 4. Rolling Horizon Optimization (MPC)

The system executes a 24-48 hour optimization plan every 15-60 minutes (`predict_future.py` + `optimize_plan.py`).

### Battery Dispatch Strategy
The optimizer uses a **marginal opportunity-cost ranking** (8-hour lookahead window) to decide dispatch actions. This creates gradual power ramps rather than binary on/off blocks.

- **CHARGE_SOLAR:** Capture excess PV generation **only when storing is more valuable than exporting**. If the current export price exceeds `opportunity_cost × round_trip_efficiency`, the solar is exported to grid instead. This enables grid-arbitrage strategies (export solar at peak, charge cheap grid later).
- **CHARGE_GRID:** Charge during cheap price windows if the "round-trip" profit (future_price * efficiency > current_price) is positive.
- **DISCHARGE_LOAD:** Offset house consumption during high-price peaks. The battery only discharges the *excess* energy beyond what is reserved for strictly better future peaks. This preserves peak capacity while monetizing stranded energy.
- **DISCHARGE_EXPORT:** Sell energy back to the grid during extreme price spikes (if `BATTERY_ALLOW_EXPORT` is enabled). Like load discharge, it is limited to the excess energy not needed for better opportunities.

### GSHP (Heat Pump) Dispatch Strategy
- **Thermal Buffering:** Increases setpoint/runs the pump when prices are in the bottom 30th percentile or when solar surplus is available.
- **Strategic Stop:** Pauses heating during price peaks if the accumulator temperature allows, considering a 2-8 hour lookahead.
- **Heat Loss Modeling:** Uses a cooling coefficient (`GSHP_HEAT_LOSS_K`) and outdoor temperature to model the reservoir's thermal decay, plus a baseline demand (`GSHP_BASELINE_DEMAND_KW`, default 1.0 kW) to account for DHW, circulation, and standby losses even when no space heating is required.

## 5. Deployment & Integration

### Execution Loop
- **Often (`run_often.sh`):** Runs every minute. Lightweight update: reads current battery/grid sensor states and pushes latest battery intent to Home Assistant (no heavy computation).
- **Frequent (`run_frequent.sh`):** Runs every 30 mins. Performs fast extraction, generates fresh predictions, and updates the MPC plan.
- **Daily (`run_daily.sh`):** Runs once a day. Performs full extraction and retrains the XGBoost model.

### Home Assistant Integration
- **Optimization Plan:** Pushed to `sensor.hepo_optimization_plan` as a JSON attribute, containing 15-minute resolution actions for the inverter and GSHP.
- **Battery Control:** `number.hoymiles_remote_control_hoymiles_battery_power` (in Watts) provides real-time control setpoint for the Hoymiles battery inverter. Positive = discharge (provide power to home), Negative = charge (draw power from grid/solar). Updated every minute via `run_often.sh`.
- **Accuracy Tracking:** `sensor.hepo_accuracy` reports 3-hour windowed MAE and Bias from `analyze_performance.py`.

## 6. Battery Availability & Degradation Mode

The system gracefully handles battery unavailability for testing and maintenance periods:

### How It Works

1. **Battery Detection:**
   - On startup, checks if `sensor.be_soc` exists and is available in Home Assistant
   - If unavailable → automatically falls back to no-battery optimization
   - Can also be disabled via `HEPO_DISABLE_BATTERY=1` in `.env`

2. **With Battery Available:**
   - Full battery dispatch optimization (charge/discharge planning)
   - Real-time control setpoint pushed to Hoymiles inverter every minute
   - Battery SoC tracked and used for planning

3. **Without Battery (Degradation Mode):**
   - GSHP and baseload still optimized normally
   - Battery control skipped silently (no errors)
   - System continues to operate and learn from load patterns
   - Log shows: `⊘ Battery Unavailable: ...W (skipped)`

### Use Case

Perfect for test deployments:
- Day 1-5: Battery not installed → System learns baseload patterns
- Day 6+: Battery installed → System adds battery optimization automatically
- No code changes needed, no errors logged

### Configuration

To manually disable battery optimization during testing:
```env
HEPO_DISABLE_BATTERY=1  # Set to 1 to disable, 0 or omit to enable
```

## 7. Technical Stack
- **Core:** Python 3.10+, Pandas, XGBoost.
- **Database:** PostgreSQL (Source), SQLite (Performance Archiving), CSV (Intermediate).
- **Config:** Environment variables (.env) for hardware constraints (battery capacity, efficiencies, COP).
