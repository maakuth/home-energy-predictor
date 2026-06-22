#!/bin/bash

set -e

# Change to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

echo "=== HEPO Quick Update: $(date) ==="

# Pull current sensor states, apply load-following, and push battery control
python3 -c "
from utils.ha_utils import get_ha_state
from utils.battery_utils import push_battery_control, compute_load_following_setpoint, compute_net_metering_setpoint, get_current_plan_entry, adjust_charge_solar_for_real_time
import json
import os
from datetime import datetime

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
p1 = get_ha_state('sensor.current_phase_1')
p2 = get_ha_state('sensor.current_phase_2')
p3 = get_ha_state('sensor.current_phase_3')

# Cumulative energy sensors for net metering
import_meter = get_ha_state('sensor.cumulative_active_import')
export_meter = get_ha_state('sensor.cumulative_active_export')

# Parse values
soc_pct = _get_float(soc)
battery_w = _get_float(battery_power) or 0.0
grid_w = (_get_float(grid_power) or 0.0) * 1000.0  # sensor reports kW, convert to W
solar_kw = _get_float(solar) or 0.0
gshp_kw = (_get_float(gshp) or 0.0) / 1000.0  # sensor reports W, convert to kW
leaf_kw = (_get_float(leaf) or 0.0) / 1000.0  # sensor reports W, convert to kW
i_p1 = _get_float(p1)
i_p2 = _get_float(p2)
i_p3 = _get_float(p3)

import_kwh = _get_float(import_meter)
export_kwh = _get_float(export_meter)

soc_str = f'{soc_pct:.1f}' if soc_pct is not None else 'unavailable'
print(f'Battery SoC: {soc_str}%')
print(f'Battery Power: {battery_w:.0f}W')
print(f'Grid Power: {grid_w:.0f}W')
print(f'Solar: {solar_kw:.2f}kW')
print(f'GSHP: {gshp_kw:.2f}kW')
print(f'Leaf: {leaf_kw:.2f}kW')

phase_str = f'L1: {i_p1 if i_p1 is not None else \"?\"}, L2: {i_p2 if i_p2 is not None else \"?\"}, L3: {i_p3 if i_p3 is not None else \"?\"}'
print(f'Phase Currents: {phase_str}')

# Load current plan
try:
    with open('optimization_plan.json') as f:
        plan = json.load(f)
except FileNotFoundError:
    print('⚠️ No optimization_plan.json found')
    plan = None

if plan:
    current = get_current_plan_entry(plan)
    if current is None:
        print('⚠️ No current plan entry found')
        current = None
    planned_battery_kw = current.get('battery_power_kw', 0.0) if current else 0.0
    planned_action = current.get('battery_action', 'idle') if current else 'idle'
    planned_soc = current.get('soc_pct') if current else None

    # Pre-processing: if plan says charge_solar but real-time has no solar surplus
    # (unexpected load exceeded forecast), switch to discharge to cover the net load
    # instead of importing from grid. The battery will recharge later from future
    # solar surplus as planned.
    planned_battery_kw, planned_action = adjust_charge_solar_for_real_time(
        planned_battery_kw=planned_battery_kw,
        planned_action=planned_action,
        solar_kw=solar_kw,
        grid_w=grid_w,
        battery_w=battery_w,
        battery_soc_pct=soc_pct,
    )

    # Check if net metering mode is enabled
    net_metering = os.getenv('BATTERY_NET_METERING', '').strip().lower() in {'1', 'true', 'yes', 'on'}

    if net_metering and import_kwh is not None and export_kwh is not None:
        # Net metering mode: use cumulative energy sensors to match quarterly average
        now = datetime.now()
        elapsed_minutes = now.minute % 15 + now.second / 60.0
        interval_minutes = 15

        planned_grid_import_kwh = current.get('grid_import_kwh', 0.0) if current else 0.0
        planned_grid_export_kwh = current.get('grid_export_kwh', 0.0) if current else 0.0

        adjusted_battery_kw, log_msg = compute_net_metering_setpoint(
            planned_battery_kw=planned_battery_kw,
            planned_grid_import_kwh=planned_grid_import_kwh,
            planned_grid_export_kwh=planned_grid_export_kwh,
            cumulative_import_kwh=import_kwh,
            cumulative_export_kwh=export_kwh,
            elapsed_minutes=elapsed_minutes,
            interval_minutes=interval_minutes,
        )
        planned_action = 'net_metering'  # Override action for logging

    else:
        # Standard load-following mode
        adjusted_battery_kw, log_msg = compute_load_following_setpoint(
            planned_battery_kw=planned_battery_kw,
            planned_action=planned_action,
            solar_kw=solar_kw,
            grid_w=grid_w,
            battery_w=battery_w,
            gshp_kw=gshp_kw,
            leaf_kw=leaf_kw,
            phase_currents=[i_p1, i_p2, i_p3]
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
