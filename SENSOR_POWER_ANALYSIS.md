# HEPO Sensor Power Architecture & Load Calculation Analysis

## Executive Summary

The home energy predictor currently **does not account for battery power** in its historical data processing or predictions. The load calculations assume a battery-less system where grid power + solar production equals home consumption. When a physical battery is installed, the load calculation formula needs fundamental revision.

---

## 1. Sensor Architecture & Raw Data Collection

### Sensors Extracted (extract_data.py)

| Sensor Entity | Column Name | Type | Units | Notes |
|---|---|---|---|---|
| `sensor.sahkokauppa_nyt` | `total_power` | Power Flow | kW | **Grid meter** - positive = import, negative = export |
| `sensor.solarh_63038_real_power_kw` | `solar_actual` | Power | kW | **Solar production** from PV inverter |
| `sensor.mlp_teho` | `gshp_power` | Power | W | Ground Source Heat Pump consumption |
| `sensor.saikaan_olohuone_current_power` | `aahp_living_power` | Power | W | Air-to-Air Heat Pump (living area) |
| `sensor.mokkimokin_ilp_power` | `aahp_cabin_power` | Power | W | Air-to-Air Heat Pump (cabin) |
| `sensor.tasmota_energy_power_3` | `leaf_power` | Power | W | Nissan Leaf EV charging power |
| `sensor.mummun_energy` | `mummun_power` | Energy → Power | W | Converted from cumulative energy to power |
| `sensor.be_soc` | *(Not extracted)* | State of Charge | % | **House battery** (read during optimization only) |
| `sensor.mlp_varaajan_lampotila` | `accumulator_temp` | Temperature | °C | GSHP accumulator tank |
| Various weather sensors | `outside_temp`, `wind_speed` | Temp/Wind | °C / m/s | Weather data |

**Key Observation:** `sensor.be_soc` (house battery state) is **NOT** extracted into historical data during `extract_data.py`. It's only read in real-time during optimization in `optimize_plan.py`.

---

## 2. Current Load Calculation (BATTERY-FREE ASSUMPTION)

### Step 1: Total Home Power (process_data.py, lines 49-52)

```python
if 'total_power' in df.columns and 'solar_actual' in df.columns:
    df['total_home_power'] = df['total_power'] + df['solar_actual']
elif 'total_power' in df.columns:
    df['total_home_power'] = df['total_power']
```

**What it means:**
- `total_power`: Grid import/export from the meter (kW)
  - Positive = importing from grid
  - Negative = exporting to grid
- `solar_actual`: Solar production (kW)
  - Always positive
- `total_home_power`: **Reconstructed home consumption** by adding them together

**Formula:**
```
total_home_power = grid_power + solar_production
```

**Physics interpretation:**
```
Home Load = Grid Import + Solar Generation
           ↓
home consumption must equal what comes from grid + what's produced locally
(assuming no battery exists or battery is ignored)
```

**Example:**
- Grid meter reads: +2 kW (importing)
- Solar produces: +3 kW
- Total home power: 2 + 3 = 5 kW
- Meaning: The house is using 5 kW total

---

### Step 2: Baseload Calculation (process_data.py, lines 54-61)

```python
if 'total_home_power' in df.columns:
    gshp_kw = (df['gshp_power'] / 1000.0) if 'gshp_power' in df.columns else 0.0
    leaf_kw = (df['leaf_power'] / 1000.0) if 'leaf_power' in df.columns else 0.0
    df['baseload_power'] = df['total_home_power'] - gshp_kw - leaf_kw
    df['baseload_power'] = df['baseload_power'].clip(lower=0)
else:
    df['baseload_power'] = 0.0
```

**What it subtracts:**
- `gshp_power` (kW): Ground source heat pump
- `leaf_power` (kW): Nissan Leaf EV charging

**NOT subtracted:**
- `aahp_living_power`, `aahp_cabin_power`: Air-to-air heat pumps (not separated)
- `mummun_power`: Unknown load (kitchen equipment?)
- Battery power: **Not extracted or tracked in historical data**

**Formula:**
```
baseload_power = (grid_power + solar_production) - gshp_power - leaf_power
baseload_power = all_other_consumption (GSHP, AAHP, other appliances, losses)
```

**Clipping:** Baseload is clipped to ≥ 0 (can't be negative)

---

## 3. Current Assumptions About Power Flow

### The Current Model's View (Battery-Free World)

```
┌─────────────────────────────────────────────────┐
│  HOME CONSUMPTION (total_home_power)            │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │ BASELOAD (everything except GSHP/Leaf) │   │
│  │ = grid_power + solar_actual             │   │
│  │         - gshp_power - leaf_power       │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  + GSHP (Ground Source Heat Pump)              │
│  + Leaf (EV Charging)                          │
│  = Total measured from: grid_power + solar_act │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Key Assumption:**
```
POWER BALANCE (No Battery):
  Home Consumption = Grid Import - Grid Export + Solar Production
  
  If grid_power > 0 (import) and solar_actual > 0 (production):
    Home Load = grid_power + solar_actual
    (No battery to interfere)
```

---

## 4. Where Load is Used

### In Feature Engineering (train_model.py, lines 16-27)

```python
target = 'baseload_power'  # What we're predicting

features = [
    'outside_temp', 'wind_speed', 'solar_forecast',
    'accumulator_temp', 'gshp_pump_temp', 'is_gshp_pump_running', 
    'acc_roc', 'is_fireplace_lag1',
    'ev_soc', 'ev_position',
    'leaf_power_lag_1h', 'leaf_energy_24h',
    'baseload_lag_1h', 'baseload_lag_24h',  # ← Historical baseload as features
    'is_extended_complex',
    'hour', 'minute', 'quarter_hour', 'day_of_week', 'month',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
]
```

**Model trains on:**
- **Target:** `baseload_power` (calculated as grid + solar - gshp - leaf)
- **Features include:** Lagged baseload (1h, 24h history)

---

### In Prediction (predict_future.py, lines 194-224)

The `get_baseload_at_lag()` function reconstructs historical baseload using the **same formula**:

```python
def get_baseload_at_lag(hours_back):
    target_ts = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    
    total = get_nearest(total_df, target_ts)        # grid_power (kW)
    solar = get_nearest(solar_df, target_ts)        # solar_actual (kW)
    gshp = get_nearest(gshp_df, target_ts) / 1000   # gshp_power (W→kW)
    leaf = get_nearest(leaf_df, target_ts) / 1000   # leaf_power (W→kW)
    
    return max(0.0, total + solar - gshp - leaf)    # baseload
```

**Called at lines 248-249:**
```python
lag1h_val = get_baseload_at_lag(1)     # Baseload 1 hour ago
lag24h_val = get_baseload_at_lag(24)   # Baseload 24 hours ago
```

These become features for the model to make today's prediction.

---

## 5. Battery Impact: THE MISSING PIECE

### Current Situation (No Battery Installed)

The system measures what comes from the grid and assumes all home consumption comes from:
1. Grid import, or
2. Solar production

**If a home battery is installed**, this assumption **breaks**.

### With a Home Battery Present

```
POWER BALANCE WITH BATTERY:
┌─────────────────────────────────────────────────────┐
│ Home Load (what we care about) = ?                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Grid meter sees: grid_power (import or export)     │
│  Solar system sees: solar_actual (production)       │
│  Battery exchanges: battery_net_power               │
│                                                     │
│  True Home Load = grid_power + solar_actual         │
│                  - battery_net_power                │
│                                                     │
│  Where:                                             │
│    battery_net_power = battery_discharge - charge   │
│    Positive = battery is discharging (providing)    │
│    Negative = battery is charging (storing)         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Example Scenario with Battery

**Moment 1: Battery Charging from Solar**
- Grid meter: -2 kW (exporting solar surplus)
- Solar actual: +5 kW (producing)
- Battery: charging at +3 kW
- **Current formula would calculate:**
  - `total_home_power = -2 + 5 = +3 kW`
  - BUT actual home consumption is only 0 kW
  - The battery is storing the other 3 kW
- **Correct formula should be:**
  - `total_home_power = -2 + 5 - 3 = 0 kW` (actual load is zero)

**Moment 2: Battery Discharging to Load**
- Grid meter: +1 kW (importing)
- Solar actual: +0.5 kW
- Battery: discharging at +2 kW
- **Current formula would calculate:**
  - `total_home_power = 1 + 0.5 = 1.5 kW`
  - BUT actual home consumption is 3.5 kW
  - Battery is providing 2 kW of it
- **Correct formula should be:**
  - `total_home_power = 1 + 0.5 + 2 = 3.5 kW` (actual load)

---

## 6. The CRITICAL Issue: History & Model Training

### Problem 1: Past data cannot be corrected retrospectively

The system stores `grid_power` and `solar_actual` in raw HA PostgreSQL database. Once a battery is installed:
- Old historical data (before battery) will be correct
- New historical data (with battery) **will be missing battery_power** if not extracted
- Mixing the two leads to **systematic bias**

### Problem 2: Model retraining needs battery sensors

If a battery exists and isn't measured, the model will **overestimate baseload** because:
```
Observed: baseload = grid_power + solar_actual - gshp - leaf
  (without battery_power subtraction)

Actual: baseload = grid_power + solar_actual - battery_net_power - gshp - leaf
  (with battery discharge subtracted)

Training on observed values → model learns an inflated baseload
Prediction on new data → systematic overestimation
```

---

## 7. README Guidance: What Should Happen

From `README.md` lines 79-84:

```
When the physical battery is installed, switch from simulation to real control by:

1. Set BATTERY_CAPACITY_KWH to the actual capacity (e.g., 40) in .env.

2. Update load calculation in process_data.py — currently:
   total_home_power = grid_power + solar_actual
   
   With a battery, subtract battery's net discharge to get true house load:
   total_home_power = grid_power + solar_actual 
                    - (battery_discharge_power - battery_charge_power)

3. The optimizer already reads real-time SOC from sensor.be_soc in Home Assistant
   when available, so closed-loop SOC tracking works out of the box.
   Fine-tune BATTERY_CHARGE_EFFICIENCY and BATTERY_DISCHARGE_EFFICIENCY.

4. Use battery_action field from sensor.hepo_optimization_plan to trigger inverter.
```

**This explicitly acknowledges the battery power adjustment is needed.**

---

## 8. Where Battery Power Is Currently Handled

### In Optimization (optimize_plan.py)

Battery dispatch **is simulated** during optimization:

```python
def plan_battery_dispatch(predictions, solar_array, import_prices, export_prices):
    # Lines 107-292
    # Simulates: charge_from_solar, charge_from_grid, discharge_to_load, discharge_to_export
    # Computes: battery_power_kw (line 278)
    
    battery_plan.append({
        'battery_action': battery_action,
        'battery_power_kw': float((charge_total - discharge_total) / interval_hours),
        'charge_from_solar_kwh': float(charge_from_solar),
        'charge_from_grid_kwh': float(charge_from_grid),
        'discharge_to_load_kwh': float(discharge_to_load),
        'discharge_to_export_kwh': float(discharge_to_export),
        'soc_kwh': float(soc_kwh),
        'grid_import_kwh': float(grid_import_kwh),
        'grid_export_kwh': float(grid_export_kwh),
        ...
    })
```

**Battery is PLANNED but not MEASURED from history.**

### In Historical Analysis (analyze_performance.py)

Battery performance is evaluated retrospectively:

```python
def summarize_battery_performance(df_merged):
    # Lines 141-216
    # Compares archived PLAN (what optimize_plan.py predicted) 
    # against ACTUAL consumption (grid_power + solar_actual - gshp)
    
    # But it uses:
    # - predicted_usage (baseload calculated without battery subtraction)
    # - actual_usage (also calculated without battery subtraction if battery exists)
```

**Analysis assumes battery_free load, so results are biased if battery exists.**

---

## 9. Sensors NOT Currently Extracted

These would be needed for proper battery accounting:

| What's Needed | Current Status | Notes |
|---|---|---|
| Battery power (charge/discharge) | **Not extracted** | `sensor.be_battery_power` or inverter MPPT values |
| Battery SOC | **Read real-time only** | `sensor.be_soc` (not in historical extraction) |
| Battery temperature | **Not extracted** | Inverter provides this |
| Inverter status | **Not extracted** | Would help diagnose battery mode |

---

## 10. Summary Table: Power Components

| Component | Sensor | Extracted | Used in Baseload | Used in Training | Notes |
|---|---|---|---|---|---|
| Grid power | `total_power` | ✅ Yes | ✅ Added | ✅ Indirectly (in baseload) | Core measurement |
| Solar production | `solar_actual` | ✅ Yes | ✅ Added | ✅ Feature + in baseload | Core measurement |
| GSHP | `gshp_power` | ✅ Yes | ✅ Subtracted | ✅ Subtracted before training | Explicitly removed |
| Leaf (EV) | `leaf_power` | ✅ Yes | ✅ Subtracted | ✅ Subtracted + lagged features | Explicitly removed |
| AAHP (Heat pumps) | `aahp_*_power` | ✅ Yes | ❌ NOT separated | ❌ Included in baseload | Treated as part of load |
| Mummun | `mummun_power` | ✅ Yes | ❌ NOT separated | ❌ Included in baseload | Unknown appliance |
| **Battery power** | None | ❌ No | ❌ Ignored | ❌ Not accounted | **CRITICAL GAP** |
| Battery SOC | `be_soc` | ❌ No (real-time only) | N/A | N/A | Only in optimization |

---

## 11. Implications & Recommendations

### Current State
- ✅ System works correctly for **battery-free scenarios** (simulation mode)
- ✅ Battery optimization is **simulated** during planning
- ✅ Real-time SOC reading works
- ❌ **No historical battery power data** means past consumption is mislabeled when battery exists
- ❌ **Model retraining with battery present** will learn biased baseload values

### When Battery Is Installed

**Must do immediately:**
1. Extract `battery_power` (charge/discharge) to raw_data.csv from inverter
2. Update `process_data.py` line 50 formula:
   ```python
   df['total_home_power'] = df['total_power'] + df['solar_actual'] - df.get('battery_power', 0)
   ```
3. Retrain model (`train_model.py`) on corrected historical data
4. Verify `analyze_performance.py` metrics improve (should show baseload decrease)

**Should do:**
1. Add battery temperature to features (degrades with heat)
2. Extract full battery metrics (voltage, current, SOC history)
3. Create separate columns for `battery_charge_power` and `battery_discharge_power` (signed properly)
4. Add tests validating baseload calculation with known battery actions

---

## 12. Code Locations Reference

| Task | File | Lines | Current Behavior |
|---|---|---|---|
| Extract sensors | `extract_data.py` | 12-28 | Defines what's pulled from HA |
| Calculate total_home_power | `process_data.py` | 49-52 | `grid + solar` (ignores battery) |
| Calculate baseload | `process_data.py` | 54-61 | Subtracts only GSHP & Leaf |
| Train model | `train_model.py` | 9-106 | Uses `baseload_power` as target |
| Predict baseload lags | `predict_future.py` | 194-224 | Recalculates via `grid + solar - gshp - leaf` |
| Plan battery dispatch | `optimize_plan.py` | 107-292 | Simulates battery, doesn't use historical |
| Analyze performance | `analyze_performance.py` | 76-216 | Compares vs actual without battery adjustment |

---

## Conclusion

The HEPO system currently operates under a **battery-free power balance assumption**:
```
Home Load = Grid Import + Solar Production - Known High Loads (GSHP, Leaf)
```

When a physical battery is installed, this equation becomes invalid. The battery introduces an unmeasured power flow that must be:
1. **Extracted** from the inverter
2. **Subtracted** during load calculation
3. **Included** in model retraining
4. **Validated** in performance analysis

Currently, **the battery is only simulated in the optimizer, not measured in history**. This creates a data integrity issue that must be resolved before the physical battery can be reliably controlled.

