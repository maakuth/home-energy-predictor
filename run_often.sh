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

# Pull current sensor states and push battery control
python3 -c "
from utils.ha_utils import get_ha_state
from utils.battery_utils import push_battery_control
import json

# Read current states
soc = get_ha_state('sensor.be_soc')
power = get_ha_state('sensor.be_stat_batt_power')
grid = get_ha_state('sensor.sahkokauppa_nyt')

print(f'Battery SoC: {soc.get(\"state\") if soc else \"unavailable\"}%')
print(f'Battery Power: {power.get(\"state\") if power else \"unavailable\"}W')
print(f'Grid Power: {grid.get(\"state\") if grid else \"unavailable\"}W')

# Load current plan and push control
try:
    with open('optimization_plan.json') as f:
        plan = json.load(f)
    if plan:
        battery_power_kw = plan[0].get('battery_power_kw', 0.0)
        battery_control_w = int(-battery_power_kw * 1000)
        push_battery_control(
            battery_power_w=battery_control_w,
            battery_action=plan[0].get('battery_action', 'idle'),
            battery_soc_pct=plan[0].get('soc_pct')
        )
except FileNotFoundError:
    print('⚠️ No optimization_plan.json found')
"

echo "=== Done: $(date) ==="
