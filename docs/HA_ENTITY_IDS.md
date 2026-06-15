# Home Assistant Entity IDs Used by HEPO

This document lists the actual entity IDs used by the home-energy-predictor system. These are used to extract data for battery planning tests.

## Entity Mapping

### Battery System
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.be_soc` | Home battery state of charge | % | Battery planning, real-time optimization |
| `sensor.be_stat_batt_power` | Home battery power | W | Battery planning (positive=charging) |

### Grid & Solar
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.sahkokauppa_nyt` | Grid meter (import/export) | kW | Load calculations, optimization |
| `sensor.solarh_63038_real_power_kw` | Solar panel actual output | kW | Load calculations, predictions |
| `sensor.solcast_pv_forecast_forecast_tomorrow` | Solar forecast | kW | Predictions, optimization |

### Heat Systems (GSHP)
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.mlp_teho` | GSHP power consumption | W | Load calculations, optimization |
| `sensor.mlp_varaajan_lampotila` | GSHP accumulator temp | °C | Temperature monitoring, heat load |
| `sensor.mlp_pumpun_lampotla` | GSHP pump outlet temp | °C | Heat system diagnostics |

### Heat Systems (AAHP - Air-to-Air Heat Pump)
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.saikaan_olohuone_current_power` | AAHP living room power | W | Load calculations |
| `sensor.mokkimokin_ilp_power` | AAHP cabin power | W | Load calculations |

### EV (Electric Vehicle)
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.xpz_491_battery_level` | Nissan Leaf battery SOC | % | EV charging optimization |
| `sensor.tasmota_energy_power_3` | EV charger power | W | Load calculations |
| `device_tracker.xpz_491_position` | Leaf location (Home/Away) | enum | EV charging logic |

### Other Loads
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.mummun_energy` | Additional load (from energy meter) | Wh | Load calculations |

### Temperature
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.ulkona_temperature_2` | Outside air temperature | °C | Heat load predictions |
| `sensor.sauna_temperature_2` | Sauna heating element temp | °C | Sauna activity detection |

### Market Data
| Entity ID | Description | Unit | Used By |
|-----------|-------------|------|---------|
| `sensor.nordpool_total` | Nordpool electricity prices | EUR/kWh | Price-based optimization |

---

## How to Use This List

### Option 1: Use Default Entities
If your entity IDs match the ones above, just run:
```bash
python dump_battery_data.py --days 7 --output battery_test_data.pkl --verbose
```

### Option 2: Customize for Your Setup
If your entity IDs are different, either:

**A) Set environment variable:**
```bash
export BATTERY_TEST_HA_ENTITIES="sensor.your_battery_soc,sensor.your_grid_power,..."
python dump_battery_data.py --days 7 --output battery_test_data.pkl --verbose
```

**B) Edit dump_battery_data.py:**
Update the `get_ha_relevant_entities()` function with your entity IDs.

### Option 3: Find Your Entity IDs
```bash
python find_ha_entities.py --search battery
python find_ha_entities.py --search solar
python find_ha_entities.py --search grid
```

---

## Entity ID Naming Conventions

The entities follow these patterns:
- **Sensors:** `sensor.<name>`
- **Switches:** `switch.<name>`
- **Input Booleans:** `input_boolean.<name>`
- **Device Trackers:** `device_tracker.<name>`

---

## Critical Entities for Battery Planning

These are the most important for battery planning tests:

1. **Battery SOC:** `sensor.be_soc` (%)
2. **Grid Power:** `sensor.sahkokauppa_nyt` (kW)
3. **Solar Power:** `sensor.solarh_63038_real_power_kw` (kW)
4. **Outside Temp:** `sensor.ulkona_temperature_2` (°C)
5. **Market Prices:** `sensor.nordpool_total` (EUR/kWh)

If you only have these 5, you can still create meaningful test data.

---

## Debugging Entity Issues

If the script says entities are not found:

```bash
# Check HA connectivity
python -c "from utils.ha_utils import get_ha_state; print(get_ha_state('sensor.be_soc'))"

# List all entities in HA
python find_ha_entities.py

# Search for similar names
python find_ha_entities.py --search soc
python find_ha_entities.py --search battery
```

---

## Database Table Schema

The SQLite database stores predictions with these columns:

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER,
    predicted_usage_kw REAL,
    battery_soc_pct REAL,
    battery_power_kw REAL,
    grid_import_kwh REAL,
    grid_export_kwh REAL,
    import_price REAL,
    export_price REAL,
    battery_action TEXT,
    version TEXT
);
```

The dump script will automatically adapt to whatever columns exist in your database.
