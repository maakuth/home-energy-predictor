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

# Pull current sensor states, apply load-following, and push battery control
python3 -c "
from utils.ha_utils import get_ha_state
from utils.battery_utils import push_battery_control, compute_load_following_setpoint
import json

def _get_float(state):
    '''Safely extract float from HA state dict.'''
    if state and state.get('state') not in ['unknown', 'unavailable', None]:
        try:
            return float(state['state'])
        except (ValueError, TypeError):
            pass
    return None

# Read current states from HA
soc = get_ha_state('sensor.be_soc')
battery_power = get_ha_state('sensor.be_stat_batt_power')
grid_power = get_ha_state('sensor.sahkokauppa_20s')
solar = get_ha_state('sensor.solarh_63038_real_power_kw')
gshp = get_ha_state('sensor.mlp_teho')
leaf = get_ha_state('sensor.tasmota_energy_power_3')

# Parse values
soc_pct = _get_float(soc)
battery_w = _get_float(battery_power) or 0.0
grid_w = (_get_float(grid_power) or 0.0) * 1000.0  # sensor reports kW, convert to W
solar_kw = _get_float(solar) or 0.0
gshp_kw = (_get_float(gshp) or 0.0) / 1000.0  # sensor reports W, convert to kW
leaf_kw = (_get_float(leaf) or 0.0) / 1000.0  # sensor reports W, convert to kW

soc_str = f'{soc_pct:.1f}' if soc_pct is not None else 'unavailable'
print(f'Battery SoC: {soc_str}%')
print(f'Battery Power: {battery_w:.0f}W')
print(f'Grid Power: {grid_w:.0f}W')
print(f'Solar: {solar_kw:.2f}kW')
print(f'GSHP: {gshp_kw:.2f}kW')
print(f'Leaf: {leaf_kw:.2f}kW')

# Load current plan
try:
    with open('optimization_plan.json') as f:
        plan = json.load(f)
except FileNotFoundError:
    print('⚠️ No optimization_plan.json found')
    plan = None

if plan and plan[0]:
    current = plan[0]
    planned_battery_kw = current.get('battery_power_kw', 0.0)
    planned_action = current.get('battery_action', 'idle')
    planned_soc = current.get('soc_pct')

    adjusted_battery_kw, log_msg = compute_load_following_setpoint(
        planned_battery_kw=planned_battery_kw,
        planned_action=planned_action,
        solar_kw=solar_kw,
        grid_w=grid_w,
        battery_w=battery_w,
        gshp_kw=gshp_kw,
        leaf_kw=leaf_kw
    )

    if log_msg:
        print(f'📊 Load follow: {log_msg}')

    battery_control_w = int(-adjusted_battery_kw * 1000)
    push_battery_control(
        battery_power_w=battery_control_w,
        battery_action=planned_action,
        battery_soc_pct=planned_soc
    )
"

echo "=== Done: $(date) ==="
