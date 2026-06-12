# Quick Start: Battery Test Data

## 5-Minute Setup

### Step 1: Dump Your Data

```bash
# With your Home Assistant and database running, extract 7 days of data:
python dump_battery_data.py --days 7 --output battery_test_data.pkl --verbose
```

You'll see output like:
```
📊 Dumping battery planning data
   Period: 2026-06-05 ... to 2026-06-12 ...
   Output: battery_test_data.pkl

1️⃣ Fetching Home Assistant snapshots...
   ✓ sensor.battery_soc_pct: 85.5
   ...

2️⃣ Loading predictions...
   ✓ Loaded 2016 predictions

✅ Successfully dumped data to battery_test_data.pkl
   File size: 145.3 KB
```

### Step 2: Use in Tests

```python
# tests/test_my_battery_logic.py
from utils.battery_test_data import load_battery_test_data

class TestBatteryLogic(unittest.TestCase):
    def setUp(self):
        self.data = load_battery_test_data('battery_test_data.pkl')
    
    def test_something(self):
        # Get current battery state
        soc = self.data.ha_state_float('sensor.battery_soc_pct')
        
        # Get predictions as DataFrame
        df = self.data.predictions_df()
        
        # Use in your test
        self.assertGreater(soc, 0)
```

### Step 3: Run Tests

```bash
python -m pytest tests/test_my_battery_logic.py -v
```

---

## Common Recipes

### Get Battery Current State

```python
from utils.battery_test_data import load_battery_test_data

data = load_battery_test_data('battery_test_data.pkl')

# As float
soc_pct = data.ha_state_float('sensor.battery_soc_pct')

# Raw state object
state = data.ha_state('sensor.battery_soc_pct')
print(state['state'])  # "85.5"
```

### Get Predictions as DataFrame

```python
df = data.predictions_df()

# Access columns
avg_load = df['predicted_baseload'].mean()
solar_total = df['solar_forecast'].sum()

# Filter by time
df_morning = df.between_time('06:00', '12:00')
```

### Get Market Prices

```python
df_prices = data.market_prices_df()

# Summary
print(f"Avg import: {df_prices['import_price'].mean():.3f} EUR/kWh")
print(f"Max price: {df_prices['import_price'].max():.3f} EUR/kWh")

# Use in algorithm
import_prices = df_prices['import_price'].values
export_prices = df_prices['export_price'].values
```

### Get Configuration

```python
# Battery config
config = data.battery_config
print(f"Capacity: {config['capacity_kwh']} kWh")
print(f"Enabled: {config['enabled']}")

# GSHP config
gshp = data.gshp_config
print(f"GSHP power: {gshp['max_power_kw']} kW")
```

### Get Database History

```python
# Archived predictions from database
archive = data.archive_predictions_df()

# What was actually stored
print(archive[['battery_soc_pct', 'battery_action', 'import_price']])
```

---

## Organizing Test Data

### Store with Your Project

```bash
# Create a fixtures directory
mkdir -p tests/fixtures

# Move your data there
mv battery_test_data.pkl tests/fixtures/

# Track it in git
git add tests/fixtures/battery_test_data.pkl
```

### Different Scenarios

Create multiple dumps for different conditions:

```bash
# Summer day
python dump_battery_data.py --start "2026-07-01" --end "2026-07-07" \
  --output tests/fixtures/battery_summer.pkl

# Winter day
python dump_battery_data.py --start "2026-01-01" --end "2026-01-07" \
  --output tests/fixtures/battery_winter.pkl

# High price period
python dump_battery_data.py --start "2025-12-15" --end "2025-12-22" \
  --output tests/fixtures/battery_expensive_period.pkl
```

Use in tests:

```python
import pytest

@pytest.mark.parametrize('data_file', [
    'tests/fixtures/battery_summer.pkl',
    'tests/fixtures/battery_winter.pkl',
    'tests/fixtures/battery_expensive_period.pkl',
])
def test_planning_all_scenarios(data_file):
    data = load_battery_test_data(data_file)
    # Your test...
```

---

## Troubleshooting

### Script Can't Connect to Home Assistant

**Error:** `⚠️ Error fetching sensor.battery_soc_pct`

**Fix:** Check `.env` file has correct `HA_HOST` and `HA_TOKEN`:
```bash
grep HA_HOST .env
grep HA_TOKEN .env
```

### Script Can't Connect to Database

**Error:** `⚠️ Could not fetch SQLite history`

**Fix:** Ensure database file exists and is readable:
```bash
ls -lah hepo.db
```

### Test Data File Too Large

Compress it:
```bash
gzip battery_test_data.pkl
# Now use in tests:
import gzip
import pickle

with gzip.open('battery_test_data.pkl.gz', 'rb') as f:
    raw = pickle.load(f)
```

### No Data for Time Period

The script silently continues if:
- No predictions exist in date range
- Some HA entities don't exist
- Database is empty

This is OK - tests can skip gracefully:
```python
def setUp(self):
    self.data = load_battery_test_data('battery_test_data.pkl')
    if len(self.data.predictions_list()) == 0:
        self.skipTest("No predictions in test data")
```

---

## Next Steps

- Read full docs: [BATTERY_TEST_DATA.md](BATTERY_TEST_DATA.md)
- See examples: [tests/test_battery_data_example.py](tests/test_battery_data_example.py)
- Check API: [utils/battery_test_data.py](utils/battery_test_data.py)
