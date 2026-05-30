#!/bin/bash

set -e

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== HEPO Quick Update: $(date) ==="

# Pull current sensor states and push battery intent
python3 -c "
from utils.ha_utils import get_ha_state, push_ha_state
import json

# Read current states
soc = get_ha_state('sensor.be_soc')
power = get_ha_state('sensor.be_stat_batt_power')
grid = get_ha_state('sensor.sahkokauppa_nyt')

print(f'Battery SoC: {soc.get(\"state\") if soc else \"unavailable\"}%')
print(f'Battery Power: {power.get(\"state\") if power else \"unavailable\"}W')
print(f'Grid Power: {grid.get(\"state\") if grid else \"unavailable\"}W')

# Load current plan and push intent
try:
    with open('optimization_plan.json') as f:
        plan = json.load(f)
    if plan:
        battery_power_kw = plan[0].get('battery_power_kw', 0.0)
        battery_control_w = int(-battery_power_kw * 1000)
        push_ha_state('number.hoymiles_remote_control_hoymiles_battery_power', battery_control_w, {
            'friendly_name': 'HEPO Battery Control',
            'unit_of_measurement': 'W',
            'device_class': 'power',
            'battery_action': plan[0].get('battery_action', 'idle'),
            'battery_soc_pct': plan[0].get('soc_pct')
        })
        print(f'✅ Battery Control: {battery_control_w}W')
except FileNotFoundError:
    print('⚠️ No optimization_plan.json found')
"

echo "=== Done: $(date) ==="
