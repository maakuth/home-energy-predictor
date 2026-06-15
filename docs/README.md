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

- **Performance Reflection**:
  - **Analysis (`analyze_performance.py`)**:
    - Compares archived predictions against actual observed consumption from the HA database.
    - Aggregates results into **3-hour windows** to provide stable Mean Absolute Error (MAE) and Bias metrics.
    - **Battery Evaluation**: Calculates the **Planned ROI** of the battery strategy (Savings in €, Avg Charge/Discharge prices, and price Spread).
    - **Persistence**: Automatically stores analysis results into a dedicated `performance_analysis` table in `hepo.db` for long-term trend tracking and strategic adaptation.
    - Pushes accuracy metrics to `sensor.hepo_accuracy`.
  - **Evolution Tracking (`analyze_evolution.py`)**:
    - Evaluates how plans **evolve** as the target time approaches.
    - Tracks **MAE by Lead Time** (e.g., how accurate is a 24h prediction vs. a 1h prediction?).
    - Measures **Stability**: Calculates the StdDev of predictions for the same timestamp and the frequency of **Battery Action flips** (e.g., changing from charge to idle).
    - Analyzes **Planned Cost Evolution** to see if the optimizer finds better deals as market price certainty increases.


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
*Effect: Full extraction (365 days), retrains XGBoost model, updates forecast, and analyzes plan evolution.*

### 3. Automation (Cron)
Example crontab for a robust setup:
```bash
# Update optimization plan every 30 minutes
*/30 * * * * /path/to/hepo/run_frequent.sh >> /path/to/hepo/frequent.log 2>&1

# Retrain the model every day at 18:05
5 18 * * * /path/to/hepo/run_daily.sh >> /path/to/hepo/daily.log 2>&1
```

## Configuration

### Setting up `.env`
1. Copy the template to create your local configuration:
   ```bash
   cp .env.template .env
   ```
2. Edit `.env` with your actual values:
   - Database credentials (PostgreSQL)
   - Home Assistant host and token
   - Electricity pricing and fees
   - Battery and GSHP parameters
   - EV charging preferences

**Important**: The `.env` file is **never committed to git** (it's in `.gitignore`). This protects your credentials and local configuration from being overwritten by agent development or CI/CD processes.

### Key Configuration Parameters
- `BATTERY_CAPACITY_KWH`: Total usable capacity. **Set to `0` to disable battery simulation entirely** — the optimizer will plan as if no battery exists, and `grid_import_kwh`/`grid_export_kwh`/estimated costs will reflect a battery-less home. GSHP and EV optimizations are unaffected (battery dispatch runs downstream of those).
- `MAIN_FUSE_SIZE_A`: Main fuse rating per phase (default `25`). Used to enforce maximum grid import limit (`3 × fuse × 230V`). Prevents battery grid-charging from overloading the connection when EV or other large loads are already drawing power.
- `GRID_TRANSFER_EUR_PER_KWH`: Variable transfer costs.
- `IMPORT_VAT_MULTIPLIER`: Tax calculations.
- `DATA_RESAMPLE_INTERVAL`: Typically `15min`.

### Battery Optimization Logic
The battery optimizer uses a **24-hour forward simulation** with a **marginal opportunity-cost ranking** to make dispatch decisions. Unlike a simple "charge when cheap, discharge when expensive" strategy, it computes the *marginal value* of keeping every stored kWh for future use, and only discharges/export when the current price is at least as good as that value. This creates **gradual ramps** in battery power rather than binary on/off blocks.

Key features:
1. **Partial Discharge/Export**: When the current price is profitable but not the *best* in the horizon, the battery only discharges the "excess" energy not needed for strictly better future peaks. This preserves peak capacity while monetizing stranded energy.
2. **Solar Export**: Solar surplus is **not** automatically captured. The optimizer compares `current_export_price` vs `opportunity_cost × round_trip_efficiency`. When exporting is better, PV goes to the grid instead of the battery, freeing room for later cheap grid charging.
3. **Spill-Risk Check**: Before charging from the grid, it simulates the battery forward with *only* solar charging. If the battery would hit 100% SOC while solar is still spilling, grid charging is blocked for that interval. This prevents paying for grid energy that would just displace free solar later.
4. **Profitability Check**: If no spill risk exists, grid charging is only allowed when `(best_future_price × round_trip_efficiency) > current_import_price`, respecting the full asymmetric price spread (grid fees, taxes, VAT on import; deductions on export).
5. **Fuse Limit Check**: Even if profitable, grid charging is clamped to the remaining import capacity after accounting for house load, GSHP, EV, and Leaf loads: `available = max_fuse_kw − committed_import`.

## Future Battery Integration
When the physical battery is installed, switch from simulation to real control by:

1. Set `BATTERY_CAPACITY_KWH` to the actual capacity (e.g., `40`) in `.env`.
2. Update load calculation in `process_data.py` — currently `total_home_power = grid_power + solar_actual`. With a battery, subtract the battery's net discharge to get the **true house load**: `total_home_power = grid_power + solar_actual - (battery_discharge_power - battery_charge_power)`.
3. The optimizer already reads real-time SOC from `sensor.be_soc` in Home Assistant when available, so closed-loop SOC tracking works out of the box. Fine-tune `BATTERY_CHARGE_EFFICIENCY` and `BATTERY_DISCHARGE_EFFICIENCY` based on real-world inverter data.
4. Use the `battery_action` field from `sensor.hepo_optimization_plan` to trigger the inverter's operating modes (e.g., via Modbus or a Home Assistant integration).
