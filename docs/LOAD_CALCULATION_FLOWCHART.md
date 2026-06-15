# HEPO Load Calculation & Data Flow

## Data Flow Diagram: From Home Assistant to Model Training

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HOME ASSISTANT SENSORS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Grid Meter          Solar System         Heat Pump         EV Charger │
│  ─────────           ──────────          ─────────         ────────────│
│  total_power ────────solar_actual ──────gshp_power ───────leaf_power   │
│  (import/export)     (production)        (consumption)     (consumption)│
│                                                                         │
│  Accumulator Tank    GSHP Pump Temp      Heat Pump Assist              │
│  ────────────────    ──────────────      ────────────────              │
│  accumulator_temp    gshp_pump_temp      aahp_*_power                  │
│  (tank temperature)  (supply line temp)  (both living & cabin)         │
│                                                                         │
│  Battery State       Weather                                           │
│  ──────────────      ───────                                           │
│  be_soc ───── X      outside_temp, wind_speed                          │
│  (read real-time     (forecasts)                                       │
│   only, NOT                                                            │
│   extracted)                                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ extract_data.py
                              │ (PostgreSQL query)
                              ↓
        ┌──────────────────────────────────────────┐
        │   raw_data.csv (15-min or 1-min)         │
        ├──────────────────────────────────────────┤
        │ Index | total_power | solar_actual |     │
        │ ...   | gshp_power  | leaf_power   | ... │
        └──────────────────────────────────────────┘
                              │
                              │ process_data.py
                              ↓
        ┌────────────────────────────────────────────────────────────┐
        │  processed_data.csv (TRAINING DATA)                        │
        ├────────────────────────────────────────────────────────────┤
        │                                                            │
        │  STEP 1: total_home_power calculation (Line 49-52)        │
        │  ───────────────────────────────────────────────────      │
        │                                                            │
        │    IF solar_actual exists:                                │
        │      total_home_power = total_power + solar_actual        │
        │    ELSE:                                                  │
        │      total_home_power = total_power                       │
        │                                                            │
        │  Interpretation: Home consumption from grid + solar        │
        │  Assumption: NO BATTERY (or battery is ignored)           │
        │                                                            │
        │  ────────────────────────────────────────────────────    │
        │  STEP 2: baseload_power calculation (Line 54-61)          │
        │  ───────────────────────────────────────────────────      │
        │                                                            │
        │    gshp_kw = gshp_power / 1000.0                          │
        │    leaf_kw = leaf_power / 1000.0                          │
        │                                                            │
        │    baseload_power = total_home_power - gshp_kw - leaf_kw  │
        │    baseload_power = max(0, baseload_power)  # Clip        │
        │                                                            │
        │  Interpretation: All consumption except GSHP & Leaf       │
        │  Includes: AAHP, appliances, mummun, UNKNOWN LOADS        │
        │  Does NOT include: Battery power (not measured)           │
        │                                                            │
        │  ────────────────────────────────────────────────────    │
        │  STEP 3: Feature engineering (Line 92-150)                │
        │  ───────────────────────────────────────────────────      │
        │                                                            │
        │    • baseload_lag_1h = baseload_power.shift(4)            │
        │    • baseload_lag_24h = baseload_power.shift(96)          │
        │    • leaf_power_lag_1h, leaf_energy_24h                   │
        │    • Temporal features: hour, day_of_week, month          │
        │    • Cyclical encoding: sin/cos of hour, day, month       │
        │    • Environment: outside_temp, accumulator_temp, etc.    │
        │                                                            │
        │  Output columns:                                          │
        │  - baseload_power (TARGET for training)                   │
        │  - baseload_lag_1h, baseload_lag_24h (FEATURES)           │
        │  - All other features for model input                     │
        │                                                            │
        └────────────────────────────────────────────────────────────┘
                              │
                     ┌────────┴────────┐
                     │                 │
                     ↓                 ↓
           ┌──────────────────┐   ┌──────────────────┐
           │  train_model.py  │   │ sarimax_predict  │
           ├──────────────────┤   │ or.py            │
           │ XGBoost Model    │   ├──────────────────┤
           │ ─────────────    │   │ Seasonal ARIMA   │
           │ Target: base     │   │ ─────────────── │
           │         load_    │   │ Target: base     │
           │         power    │   │         load_    │
           │                  │   │         power    │
           │ Features:        │   │                  │
           │ • outside_temp   │   │ Uses hist baseload│
           │ • baseload_lag_* │   │ for seasonal      │
           │ • solar_forecast │   │ patterns          │
           │ • ev_soc         │   │                  │
           │ • temporal       │   │                  │
           │ • ... (21 total) │   │                  │
           │                  │   │                  │
           │ Outputs:         │   │ Outputs:         │
           │ • energy_model   │   │ • SARIMA params  │
           │   .json          │   │ • sarimax_       │
           │ • model_features │   │   predictions.   │
           │   .json          │   │   json           │
           └──────────────────┘   └──────────────────┘
                     │                    │
                     └────────┬───────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ↓                    ↓
        ┌──────────────────────┐  ┌───────────────────┐
        │ predict_future.py    │  │ optimize_plan.py  │
        ├──────────────────────┤  ├───────────────────┤
        │ INFERENCE STAGE      │  │ OPTIMIZATION      │
        │ ──────────────────   │  │ ────────────────  │
        │                      │  │                   │
        │ For each future      │  │ For each predicted│
        │ timestamp (15-min    │  │ timepoint:        │
        │ intervals):          │  │                   │
        │                      │  │ • Calculate net   │
        │ 1. Fetch real-time   │  │   load (baseload  │
        │    states (current   │  │   + GSHP +        │
        │    temp, wind, etc.) │  │   planned EV)     │
        │                      │  │                   │
        │ 2. Calculate lag     │  │ • Simulate        │
        │    features:         │  │   battery dispatch│
        │    lag1h_val =       │  │   - charge from   │
        │    total_power +     │  │     solar         │
        │    solar_actual -    │  │   - charge from   │
        │    gshp_power -      │  │     grid (cheap)  │
        │    leaf_power        │  │   - discharge     │
        │    (1h ago)          │  │     to load       │
        │                      │  │   - discharge     │
        │ 3. XGBoost predicts  │  │     to export     │
        │    baseload_power    │  │                   │
        │    for each interval │  │ Outputs:          │
        │                      │  │ • optimization_   │
        │ 4. Store with solar  │  │   plan.json       │
        │    forecast          │  │   (battery        │
        │                      │  │    intents)       │
        │ Output:              │  │ • Pushes to HA:   │
        │ future_predictions   │  │   sensor.hepo_opt │
        │ .json                │  │   imization_plan  │
        │                      │  │                   │
        └──────────────────────┘  └───────────────────┘
                      │                    │
                      └────────┬───────────┘
                               │
                               ↓
        ┌──────────────────────────────────────┐
        │  analyze_performance.py              │
        ├──────────────────────────────────────┤
        │ Compares predictions vs actual:      │
        │                                      │
        │ actual_usage = total_power +         │
        │                solar_actual -        │
        │                gshp_actual -         │
        │                leaf_actual           │
        │                                      │
        │ (NOTE: Still assumes no battery)    │
        │                                      │
        │ Calculates:                          │
        │ • MAE (Mean Absolute Error)          │
        │ • Bias (under/over prediction)       │
        │ • Battery ROI metrics                │
        │ • GSHP timing quality                │
        │                                      │
        │ Stores in hepo.db:                   │
        │ performance_analysis table           │
        │                                      │
        └──────────────────────────────────────┘
```

---

## Formula Reference Card

### 1. Total Home Power (What the house uses)

**Current (Battery-Free):**
```
total_home_power = grid_power + solar_actual
```

**With Battery (FUTURE):**
```
total_home_power = grid_power + solar_actual - battery_net_power

Where:
  battery_net_power = battery_discharge - battery_charge
  Positive = discharging (providing power to load)
  Negative = charging (storing excess power)
```

### 2. Baseload Power (After removing known loads)

**Current:**
```
baseload_power = (grid_power + solar_actual) 
                 - gshp_power - leaf_power
                 
            = total_home_power - gshp_power - leaf_power
```

**Clipping:** `max(0, baseload_power)` (can't be negative)

**Includes:** AAHP, appliances, fireplace, mummun, phantom loads
**Excludes:** GSHP, Nissan Leaf, (Battery - if present)

### 3. Baseload Lag Features (Historical anchors)

**For 1-hour lag:**
```
baseload_lag_1h = baseload_power[t-1h]
                = grid_power[t-1h] + solar_actual[t-1h] 
                  - gshp_power[t-1h] - leaf_power[t-1h]
```

**For 24-hour lag:**
```
baseload_lag_24h = baseload_power[t-24h]
```

### 4. Leaf (EV) Features

**Power lag (1h):**
```
leaf_power_lag_1h = leaf_power[t-1h] (in Watts)
```

**Energy integral (24h):**
```
leaf_energy_24h = (average_leaf_power_last_24h * 24 hours) / 1000
                = Wh to kWh conversion
```

### 5. Net Load for Battery Planning

```
net_without_battery = predicted_baseload_kw 
                     + planned_gshp_kw 
                     - solar_forecast_kw

If net > 0: import from grid or battery
If net < 0: export to grid or battery
```

### 6. Battery State of Charge (SOC)

```
soc_kwh[t+1] = soc_kwh[t] 
              + (charge_from_solar * charge_eff) 
              + (charge_from_grid * charge_eff)
              - (discharge_to_load / discharge_eff)
              - (discharge_to_export / discharge_eff)
              
Constrained to: min_soc_kwh ≤ soc_kwh ≤ max_soc_kwh
```

---

## Known Issues with Current Approach

### 1. **AAHP (Heat Pump Assist) Not Separated**

Even though `aahp_living_power` and `aahp_cabin_power` are extracted, they are **not subtracted** from baseload. They're treated as part of the load to be modeled.

**Why:** These are consumption, not separate loads like GSHP. But if they ever need to be controlled (similar to GSHP), the formula will need updating.

### 2. **Mummun Power Not Separated**

`mummun_power` is extracted but not identified or subtracted. It appears to be "everything else" load.

**Recommendation:** Investigate what mummun is. If it's a known appliance, consider separating it like GSHP.

### 3. **Battery Power NOT Extracted**

The most critical gap: When a battery is installed, its power flow must be extracted as a sensor. Currently:

```
What we measure:  grid_power, solar_actual
What we don't:    battery_charge_power, battery_discharge_power
Consequence:      baseload calculations become WRONG
```

### 4. **Battery SOC Only Read Real-Time**

`sensor.be_soc` is not extracted into raw_data.csv. This means:
- Historical SOC is not available for analysis
- Model cannot learn SOC-dependent patterns
- No backward-compatibility checks

### 5. **Circular Dependency: Lagged Features**

The model uses `baseload_lag_1h` and `baseload_lag_24h` as features, but these are calculated on-the-fly in `predict_future.py`:

```python
def get_baseload_at_lag(hours_back):
    # Recalculates: grid + solar - gshp - leaf
    # Doesn't use the pre-calculated baseload_lag_* from processed_data.csv
    # Why? For real-time predictions when processed_data might be stale
```

If baseload calculation changes (e.g., adding battery), **both** places must be updated.

---

## What Changes When Battery Is Installed

### Before Battery
```
Raw Sensors → Process → Baseload Calculation → Model Training → Predictions OK
(No battery power to measure)
```

### After Battery (If not adjusted)
```
Raw Sensors → Process → Baseload Calculation → Model Training → PREDICTIONS WRONG
(Battery power ignored)         (wrong formula)  (learns wrong patterns)
```

### After Battery (If properly adjusted)
```
Raw Sensors → New Battery Sensor Extract → Updated Process → Correct Baseload → Model Retrain → Predictions OK
(grid, solar, battery_power)               (subtract battery_net) → Accurate learning
```

---

## Testing & Validation Strategy

### 1. Unit Tests for Load Calculation
```python
def test_baseload_without_battery():
    # grid +2 kW, solar +3 kW, gshp +1 kW, leaf +0.5 kW
    # Expected: 2 + 3 - 1 - 0.5 = 3.5 kW
    
def test_baseload_with_battery_charging():
    # grid -2 kW, solar +5 kW, battery charging +3 kW, 
    # gshp +1 kW, leaf +0.5 kW
    # Current (wrong): -2 + 5 - 1 - 0.5 = 1.5 kW
    # Correct (with fix): -2 + 5 - 3 - 1 - 0.5 = -1.5 kW (impossible, clip to 0)
    # Actually correct: 0 kW (battery is storing all excess)
```

### 2. Integration Tests
- Extract data with battery present
- Run process_data.py
- Verify baseload < baseload_without_battery_formula
- Verify model MAE improves after retraining on corrected data

### 3. Performance Monitoring
- Track bias_kw metric in performance_analysis table
- If battery present but not measured: bias should increase
- If battery properly measured: bias should decrease

---

## Summary: The Power Balance

| Scenario | Formula | Result |
|---|---|---|
| **No battery, importing** | 2 + 3 - 1 - 0.5 | Home uses 3.5 kW ✅ |
| **No battery, exporting** | -3 + 5 - 0 - 0 | Home uses 2 kW, export 3 kW ✅ |
| **Battery charging, current formula** | 1 + 4 - 1 - 0 | Says 4 kW used ❌ |
| **Battery charging, correct formula** | 1 + 4 - 2 - 1 - 0 | Says 2 kW used ✅ |
| **Battery discharging, current formula** | 1 + 0 - 1 - 0 | Says 0 kW used ❌ |
| **Battery discharging, correct formula** | 1 + 0 + 2 - 1 - 0 | Says 2 kW used ✅ |

Where:
- First value: grid_power (kW)
- Second value: solar_actual (kW)
- Third value (when shown): battery_net_power (kW)
- Fourth value: gshp_power (kW)
- Fifth value: leaf_power (kW)

