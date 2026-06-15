# Battery Planning Test Data Dumper

## Overview

This toolset allows you to extract and serialize battery planning data from your Home Assistant instance and SQLite database into pickle files for use in tests. This enables:

- **Reproducible testing** with real-world data
- **Offline testing** without needing Home Assistant or database connectivity
- **Data sharing** for debugging battery planning issues
- **Historical analysis** of battery performance across different scenarios

## Scripts

### `dump_battery_data.py` - Extract and Serialize Data

Dumps battery planning relevant variables from Home Assistant API and local SQLite database to a pickle file.

#### Installation

No additional installation needed - uses existing dependencies.

#### Usage

```bash
# Last 7 days (default)
python dump_battery_data.py --output my_data.pkl

# Last 3 days
python dump_battery_data.py --days 3 --output last_3days.pkl

# Specific date range
python dump_battery_data.py --start "2026-01-01" --end "2026-01-15" --output jan_data.pkl

# With verbose output
python dump_battery_data.py --days 7 --output data.pkl --verbose
```

#### Command Line Arguments

- `--days N` (default: 7): Number of days to go back
- `--start DATE` (YYYY-MM-DD or ISO format): Start of time range
- `--end DATE` (YYYY-MM-DD or ISO format): End of time range
- `--output FILE` (default: battery_test_data.pkl): Output file path
- `--verbose` / `-v`: Print detailed progress information

#### Requirements

The script requires:
- `.env` file with Home Assistant credentials and database connection details
- Home Assistant API accessible at `HA_HOST` with valid `HA_TOKEN`
- SQLite database at the configured path (usually `hepo.db`)

#### Output Format

The pickle file contains a dictionary with the following structure:

```python
{
    'metadata': {
        'dumped_at': ISO timestamp,
        'period_start': ISO timestamp,
        'period_end': ISO timestamp,
        'description': 'Battery planning test data',
        'model_version': '1.2.3',  # From VERSION file
    },
    
    'ha_states': {
        # Current state snapshot from Home Assistant
        'sensor.battery_soc_pct': {
            'state': '85.5',
            'attributes': {...},
            'last_updated': ISO timestamp,
        },
        # ... other sensors ...
    },
    
    'predictions': [
        {
            'timestamp': ISO timestamp,
            'predicted_baseload': 2.5,  # kW
            'solar_forecast': 1.2,      # kW
            'outside_temp': 15.3,       # °C
            'is_sauna_active': 0,
            'is_fallback_price': 0,
        },
        # ... more predictions ...
    ],
    
    'market_prices': [
        {
            'timestamp': ISO timestamp,
            'import_price': 0.125,  # EUR/kWh
            'export_price': 0.095,  # EUR/kWh
        },
        # ... more prices ...
    ],
    
    'history': {
        # Historical state changes from database
        'predictions_archive': [
            {
                'timestamp': ISO timestamp,
                'predicted_usage_kw': 3.2,
                'battery_soc_pct': 85.0,
                'battery_power_kw': 1.5,
                'grid_import_kwh': 0.75,
                'grid_export_kwh': 0.0,
                'import_price': 0.125,
                'export_price': 0.095,
                'battery_action': 'charge_solar',
            },
            # ... more records ...
        ],
    },
    
    'battery_config': {
        'capacity_kwh': 13.5,
        'min_soc_pct': 10.0,
        'max_soc_pct': 95.0,
        'charge_rate_kw': 5.0,
        'discharge_rate_kw': 5.0,
        'enabled': True,
    },
    
    'gshp_config': {
        'enabled': True,
        'max_power_kw': 7.0,
    },
}
```

## Loading Test Data in Tests

### Using `BatteryTestData` Class

The `utils/battery_test_data.py` module provides a convenient `BatteryTestData` class for loading and accessing pickled data:

```python
from utils.battery_test_data import BatteryTestData, load_battery_test_data

# Load from pickle file
data = load_battery_test_data('battery_test_data.pkl')

# Or with explicit class
data = BatteryTestData.load('battery_test_data.pkl')
```

### Accessing Data

```python
# Home Assistant states
soc_pct = data.ha_state_float('sensor.battery_soc_pct')
battery_enabled = data.battery_config.get('enabled')

# Predictions as list
predictions = data.predictions_list()

# Predictions as pandas DataFrame (time-indexed)
df_pred = data.predictions_df()
avg_load = df_pred['predicted_baseload'].mean()

# Market prices
df_prices = data.market_prices_df()
import_prices = df_prices['import_price'].values

# Historical data from database
df_history = data.archive_predictions_df()

# Metadata
period_start = data.period_start()
period_end = data.period_end()
model_version = data.model_version()
```

### Example Test

```python
import unittest
from utils.battery_test_data import load_battery_test_data

class TestBatteryPlanning(unittest.TestCase):
    def setUp(self):
        """Load test data from pickle file."""
        self.data = load_battery_test_data('battery_test_data.pkl')
    
    def test_battery_planning_with_real_data(self):
        """Test battery planner with real-world data."""
        # Extract inputs
        predictions = self.data.predictions_list()
        df_prices = self.data.market_prices_df()
        config = self.data.battery_config
        
        # Convert to arrays for planner
        import numpy as np
        predictions_kw = np.array([p['predicted_baseload'] for p in predictions])
        solar_kw = np.array([p['solar_forecast'] for p in predictions])
        import_prices = df_prices['import_price'].values
        export_prices = df_prices['export_price'].values
        
        # Run planning algorithm
        from battery_planners import BatteryPlannerFactory
        planner = BatteryPlannerFactory.create('heuristic')
        plan = planner.plan(
            predictions_kw,
            solar_kw,
            import_prices,
            export_prices,
            prediction_timestamps=[],
        )
        
        # Verify plan is reasonable
        self.assertEqual(len(plan), len(predictions_kw))
        for entry in plan:
            self.assertGreaterEqual(entry.soc_pct, 0)
            self.assertLessEqual(entry.soc_pct, 100)
```

## Entities Captured

The script captures the following Home Assistant entities (if available):

### Battery
- `sensor.battery_soc_kwh` - State of charge in kWh
- `sensor.battery_soc_pct` - State of charge as percentage
- `sensor.battery_power_kw` - Battery power output

### Solar
- `sensor.solar_power_kw` - Current solar generation
- `sensor.solar_today_kwh` - Today's solar generation

### Grid
- `sensor.grid_import_kw` - Current grid import
- `sensor.grid_export_kw` - Current grid export
- `sensor.grid_import_today_kwh` - Today's grid import
- `sensor.grid_export_today_kwh` - Today's grid export

### Load
- `sensor.house_load_kw` - Total house load
- `sensor.total_power_kw` - Total power

### Temperature
- `sensor.outside_temperature` - Outside temperature
- `sensor.inside_temperature` - Inside temperature
- `sensor.mlp_lampotila` - GSHP temperature

### Controls
- `switch.battery_enabled` - Battery enabled/disabled
- `switch.gshp_enabled` - GSHP enabled/disabled
- `input_boolean.sauna_active` - Sauna active status

### Prices
- `sensor.nordpool_import_price` - Current import price
- `sensor.nordpool_export_price` - Current export price

## Best Practices

### When to Use Pickled Test Data

✅ **Good use cases:**
- Testing battery planning algorithms with realistic data
- Verifying performance metrics against historical data
- Debugging battery behavior in specific scenarios
- Sharing test cases for issue reproduction
- Offline development and testing

❌ **Not ideal for:**
- Testing real-time interaction with Home Assistant
- Testing database schema changes
- Performance benchmarking (data volumes may differ)

### Data Privacy

The pickled data contains:
- Current state values from your Home Assistant
- Historical predictions and prices
- Battery configuration

**Consider:** Remove or anonymize sensitive data before sharing pickle files with others.

### File Size

A typical 7-day dump is 50-200 KB depending on:
- Number of entities captured
- Prediction frequency
- Database history size

### Updating Test Data

When your system's configuration or behavior changes significantly:

```bash
# Create a new dump with an expressive name
python dump_battery_data.py --days 7 --output battery_data_summer_2026.pkl

# Update test fixtures
cp battery_data_summer_2026.pkl tests/fixtures/
```

## Troubleshooting

### Script Cannot Connect to Home Assistant

```
⚠️ Error fetching sensor.battery_soc_pct: ...
```

**Fix:** Ensure `HA_HOST` and `HA_TOKEN` in `.env` are correct.

### Script Cannot Connect to Database

```
⚠️ Could not fetch SQLite history: ...
```

**Fix:** Check that the database path is correct in `.env` and the file is readable.

### Output File is Too Large

If your pickle file is unexpectedly large:

1. Check the prediction file has reasonable size:
   ```bash
   wc -l future_predictions.json
   ```

2. Reduce the date range:
   ```bash
   python dump_battery_data.py --days 1 --output small_data.pkl
   ```

3. Post-compress the pickle:
   ```bash
   gzip battery_test_data.pkl
   ```

## Integration with Tests

### Store in Version Control

Add pickle files to your test fixtures:

```bash
mkdir -p tests/fixtures
python dump_battery_data.py --days 7 --output tests/fixtures/battery_data_ref.pkl
git add tests/fixtures/battery_data_ref.pkl
```

### Load in conftest.py

```python
# tests/conftest.py
import pytest
from utils.battery_test_data import load_battery_test_data
from pathlib import Path

@pytest.fixture(scope='session')
def battery_test_data():
    """Load reference battery test data."""
    data_path = Path(__file__).parent / 'fixtures' / 'battery_data_ref.pkl'
    if data_path.exists():
        return load_battery_test_data(str(data_path))
    return None

# In your test:
def test_something(battery_test_data):
    if battery_test_data is None:
        pytest.skip("No battery test data available")
    
    # Use battery_test_data...
```

## See Also

- `battery_planners/` - Battery planning algorithms
- `optimize_plan.py` - Main optimization logic
- `BATTERY_PLANNER_ARCHITECTURE.md` - Planner design
- `.env.template` - Configuration options
