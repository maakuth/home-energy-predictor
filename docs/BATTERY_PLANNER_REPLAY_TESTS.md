# Battery Planner Replay Tests

This document describes the battery planner replay test system for validating planner behavior over realistic historical data.

## Overview

The replay test system simulates battery planner performance by:

1. **Loading fixture data** from pickle files containing:
   - Current Home Assistant entity states
   - Prediction archive with `generated_at` timestamps
   - Actual measurements (grid power, solar, GSHP, etc.)
   - Market prices

2. **Time-aware visibility enforcement** ensuring planners don't cheat:
   - Forecasts only visible if `generated_at <= planning_time`
   - Spot prices only visible through end of current day until 15:00 local time, then through next day
   - No future measurement leakage

3. **Step-by-step simulation** of battery dispatch:
   - For each 15-minute interval, call planner with visible data
   - Execute first interval of plan
   - Update simulated SoC
   - Compute realized costs using actual measurements
   - Check for constraint violations

4. **Parametrized test execution** across all planner implementations and fixture combinations

## Fixture Format

Fixtures are pickle files in `tests/fixtures/` containing a dictionary:

```python
{
    'metadata': {
        'dumped_at': ISO timestamp,
        'period_start': ISO timestamp,
        'period_end': ISO timestamp,
        'description': 'Battery planning test data',
        'model_version': str,
    },
    'ha_states': {
        'entity_id': {'state': str, 'attributes': dict, 'last_updated': str},
        ...
    },
    'predictions': [...],  # Current prediction (optional)
    'market_prices': [...],  # Current prices (optional)
    'history': {
        'predictions_archive': [
            {
                'target_timestamp': ISO timestamp,
                'generated_at': ISO timestamp,  # CRITICAL: when this forecast was made
                'predicted_usage_kw': float,
                'solar_forecast_kw': float,
                'import_price': float,
                'export_price': float,
                'is_fallback_price': int,
                'battery_action': str,        # Optional: actual planner output
                'battery_power_kw': float,    # Optional
                'battery_soc_pct': float,     # Optional
                'grid_import_kwh': float,     # Optional
                'grid_export_kwh': float,     # Optional
                ...
            },
            ...
        ],
        'measurements': [
            {
                'timestamp': ISO timestamp,
                'total_power_kw': float,      # Grid power (kW, positive=import)
                'solar_actual_kw': float,     # Solar actual power (kW)
                'gshp_power_kw': float,       # GSHP power (kW)
                'leaf_power_kw': float,       # Leaf power (kW)
                'outside_temp_c': float,      # Outside temperature (°C)
            },
            ...
        ],
    },
    'battery_config': {
        'capacity_kwh': float,
        'min_soc_pct': float,
        'max_soc_pct': float,
        'charge_rate_kw': float,
        'discharge_rate_kw': float,
        'enabled': bool,
    },
    'gshp_config': {
        'enabled': bool,
        'max_power_kw': float,
    },
}
```

## Regenerating Fixtures

The fixture files are generated using `dump_battery_data.py`, which has been updated to:

1. Preserve `generated_at` in the predictions archive (critical for time-aware replay)
2. Include actual HA measurement history (grid power, solar, etc.)
3. Keep all forecast versions (not just the latest per target timestamp)

**Prerequisites:**
- SQLite database with prediction history
- Home Assistant API access for entity snapshots
- PostgreSQL connectivity to Home Assistant database (optional, for measurements)

**To regenerate fixtures:**

```bash
# Last 7 days
python dump_battery_data.py --days 7 --output battery_test_data.pkl --verbose

# Specific date range
python dump_battery_data.py --start "2026-01-01" --end "2026-01-31" --output tests/fixtures/jan.pkl --verbose

# Available options
python dump_battery_data.py --help
```

**Note on measurements:** 
- The dumper will include actual HA measurement history if PostgreSQL is available
- If PostgreSQL is not available, the fixture will only contain predictions archive
- Tests will skip measurement-based assertions if the `history['measurements']` list is empty
- For full replay testing with cost validation, ensure PostgreSQL connectivity when dumping fixtures

## Running Tests

```bash
# Run all replay tests
pytest tests/test_battery_planner_replay.py -v

# Run tests for specific planner
pytest tests/test_battery_planner_replay.py -k "heuristic" -v

# Run tests for specific fixture
pytest tests/test_battery_planner_replay.py -k "jan" -v

# Run with detailed output
pytest tests/test_battery_planner_replay.py -v -s

# Skip slow SARIMA tests (faster iteration)
pytest tests/ -k 'not sarima' -v
```

## Test Assertions

The parametrized tests validate:

1. **SoC Constraint Compliance**
   - SoC never drops below `min_soc_pct` (default 10%)
   - SoC never exceeds `max_soc_pct` (default 90%)
   - Any violation is reported with timestamp and values

2. **Output Structure Validity**
   - All required fields present
   - All numeric values are finite (not NaN/inf)
   - Grid import/export never negative
   - Battery power within inverter limits

3. **Performance Baseline**
   - Cost with battery planner ≤ baseline no-battery cost + 10% tolerance
   - Excessive degradation indicates a problem (e.g., always charging, never discharging)

4. **No Future Data Leakage**
   - Only forecasts with `generated_at <= planning_time` are visible
   - Spot prices respect 15:00 local time visibility rule
   - No actual measurements from future intervals are used

## Replay Semantics

### Time-Aware Forecast Visibility

For each planning interval at time `T`:

1. Load all forecasts where `generated_at <= T`
2. For each `target_timestamp` in the horizon, use **only the latest** forecast generated before or at `T`
3. If no forecast exists for a target timestamp, pad with edge values

### Spot Price Visibility

Before 15:00 local time:
- Prices available through end of current day (23:59)

At or after 15:00 local time:
- Prices available through end of next day (23:59)

This models the real-world scenario where day-ahead prices are published around 15:00 CET.

### Measurement Integration

Realized costs are computed using **actual measurements**, not forecasts:

```python
actual_load_kw = total_power_kw + solar_actual_kw
grid_import_kwh = max(0, actual_load_kw - battery_discharge_kwh)
grid_export_kwh = max(0, battery_discharge_kwh - actual_load_kw)
realized_cost = grid_import_kwh * import_price - grid_export_kwh * export_price
```

## Adding New Planners

To automatically include a new planner in replay tests:

1. Implement `BatteryPlanner` subclass and register it:

```python
class MyBatteryPlanner(BatteryPlanner):
    def plan(self, ...):
        ...

BatteryPlannerFactory.register('my_planner', MyBatteryPlanner)
```

2. The new planner is automatically discovered and included in parametrized tests via `BatteryPlannerFactory.names()`

3. Run tests to validate against all fixtures:

```bash
pytest tests/test_battery_planner_replay.py -k "my_planner" -v
```

## Performance Metrics

After simulation, metrics are collected per fixture/planner combination:

- `intervals_run`: Number of 15-minute intervals simulated
- `soc_violations`: Count of constraint violations
- `cost_with_battery_eur`: Total cost including planner decisions
- `cost_no_battery_eur`: Baseline cost (no battery optimization)
- `savings_eur`: Absolute savings (EUR)
- `savings_pct`: Percentage savings
- `final_soc_pct`: Final battery state of charge

## Debugging Failed Tests

### SoC Violations

If a test fails due to SoC constraint violations:

1. Check the planner's charge/discharge logic
2. Verify `battery_initial_soc_pct` environment variable is set correctly
3. Inspect `soc_violation_details` in test output
4. Review the heuristic for over-aggressive charging/discharging

### Cost Degradation

If planner cost exceeds baseline + 10%:

1. Verify price visibility (check 15:00 rule is implemented correctly)
2. Check if forecasts are being used correctly
3. Ensure battery isn't being charged and discharged simultaneously
4. Validate that solar surplus handling is correct

### No Visible Forecasts

If a test is skipped with "No visible forecasts":

1. Check if the fixture has prediction archive data
2. Regenerate the fixture with the updated `dump_battery_data.py`
3. Verify that `generated_at` is populated in predictions archive

## Future Enhancements

- [ ] Store replay metrics in performance database for tracking over time
- [ ] Add visualization of planner behavior (SoC trajectory, costs)
- [ ] Support replay with different battery capacities/constraints
- [ ] Compare multiple planners on the same fixture (head-to-head)
- [ ] Sensitivity analysis on tunable parameters
