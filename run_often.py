from __future__ import annotations
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from utils.ha_utils import get_ha_state, push_ha_state
from typing import cast
from utils.type_defs import BatteryAction
from utils.battery_utils import (
    push_battery_control,
    compute_load_following_setpoint,
    compute_net_metering_setpoint,
    get_current_plan_entry,
    adjust_charge_solar_for_real_time,
    smooth_planned_setpoint,
    apply_ramp_rate,
)

load_dotenv(override=True)


def _get_float(state):
    if state and state.get('state') not in ['unknown', 'unavailable', None]:
        try:
            return float(state['state'])
        except (ValueError, TypeError):
            pass
    return None


def main():
    soc = get_ha_state('sensor.be_soc')
    battery_power = get_ha_state('sensor.be_stat_batt_power')
    grid_power = get_ha_state('sensor.sahkokauppa_20s')
    solar = get_ha_state('sensor.solarh_63038_real_power_kw')
    gshp = get_ha_state('sensor.mlp_teho')
    leaf = get_ha_state('sensor.tasmota_energy_power_3')
    p1 = get_ha_state('sensor.current_phase_1')
    p2 = get_ha_state('sensor.current_phase_2')
    p3 = get_ha_state('sensor.current_phase_3')

    import_meter = get_ha_state('sensor.cumulative_active_import')
    export_meter = get_ha_state('sensor.cumulative_active_export')

    soc_pct = _get_float(soc)
    battery_w = _get_float(battery_power) or 0.0
    grid_w = (_get_float(grid_power) or 0.0) * 1000.0
    solar_kw = _get_float(solar) or 0.0
    gshp_kw = (_get_float(gshp) or 0.0) / 1000.0
    leaf_kw = (_get_float(leaf) or 0.0) / 1000.0
    i_p1 = _get_float(p1)
    i_p2 = _get_float(p2)
    i_p3 = _get_float(p3)

    import_kwh = _get_float(import_meter)
    export_kwh = _get_float(export_meter)

    soc_str = f'{soc_pct:.1f}' if soc_pct is not None else 'unavailable'
    print(f'Battery SoC: {soc_str}%')
    direction = 'charging' if battery_w >= 0 else 'discharging'
    print(f'Battery Power: {abs(battery_w):.0f}W ({direction})')
    print(f'Grid Power: {grid_w:.0f}W')
    print(f'Solar: {solar_kw:.2f}kW')
    print(f'GSHP: {gshp_kw:.2f}kW')
    print(f'Leaf: {leaf_kw:.2f}kW')

    phase_str = f'L1: {i_p1 if i_p1 is not None else "?"}, L2: {i_p2 if i_p2 is not None else "?"}, L3: {i_p3 if i_p3 is not None else "?"}'
    print(f'Phase Currents: {phase_str}')

    try:
        with open('state/optimization_plan.json') as f:
            plan = json.load(f)
    except FileNotFoundError:
        print('No optimization_plan.json found')
        plan = None

    if not plan:
        return

    current = get_current_plan_entry(plan)
    if current is None:
        print('No current plan entry found')

    planned_battery_kw = current.get('battery_power_kw', 0.0) if current else 0.0
    planned_action = current.get('battery_action', 'idle') if current else 'idle'
    planned_soc = current.get('soc_pct') if current else None

    max_battery_kw = float(os.getenv('BATTERY_MAX_CHARGE_KW', '10.0'))

    # Smooth setpoint across interval boundaries using prior interval's average
    planned_battery_kw = smooth_planned_setpoint(
        planned_battery_kw=planned_battery_kw,
        planned_action=planned_action,
        actual_battery_w=battery_w,
        plan=plan,
        max_battery_kw=max_battery_kw,
    )

    planned_battery_kw, planned_action = adjust_charge_solar_for_real_time(
        planned_battery_kw=planned_battery_kw,
        planned_action=planned_action,
        solar_kw=solar_kw,
        grid_w=grid_w,
        battery_w=battery_w,
        battery_soc_pct=soc_pct,
    )

    net_metering = os.getenv('BATTERY_NET_METERING', '').strip().lower() in {'1', 'true', 'yes', 'on'}

    if net_metering and import_kwh is not None and export_kwh is not None:
        now = datetime.now()
        elapsed_minutes = now.minute % 15 + now.second / 60.0
        interval_minutes = 15

        planned_grid_import_kwh = current.get('grid_import_kwh', 0.0) if current else 0.0
        planned_grid_export_kwh = current.get('grid_export_kwh', 0.0) if current else 0.0

        adjusted_battery_kw, log_msg = compute_net_metering_setpoint(
            planned_battery_kw=planned_battery_kw,
            planned_action=planned_action,
            planned_grid_import_kwh=planned_grid_import_kwh,
            planned_grid_export_kwh=planned_grid_export_kwh,
            cumulative_import_kwh=import_kwh,
            cumulative_export_kwh=export_kwh,
            elapsed_minutes=elapsed_minutes,
            interval_minutes=interval_minutes,
        )
        planned_action = 'net_metering'

        net_state_file = os.getenv('HEPO_NET_METERING_STATE_FILE', 'state/net_metering_state.json')
        try:
            with open(net_state_file) as f:
                net_state = json.load(f)
            if 'import_start' not in net_state:
                net_state['import_start'] = import_kwh
                net_state['export_start'] = export_kwh
                os.makedirs(os.path.dirname(net_state_file), exist_ok=True)
                with open(net_state_file, 'w') as f:
                    json.dump(net_state, f)
            i_start = net_state['import_start']
            e_start = net_state['export_start']
            interval_import = import_kwh - i_start
            interval_export = export_kwh - e_start
            interval_net = interval_import - interval_export
            planned_net = planned_grid_import_kwh - planned_grid_export_kwh
            if abs(interval_net) < 0.001:
                direction = 'balanced'
            elif interval_net > 0:
                direction = 'net import'
            else:
                direction = 'net export'
            print(f'Net metering interval: {direction}, import={interval_import:.3f}kWh, export={interval_export:.3f}kWh, net={interval_net:+.3f}kWh, target={planned_net:+.3f}kWh')

            push_ha_state('sensor.hepo_period_balance', f"{interval_net:.3f}", {
                'friendly_name': 'HEPO Period Power Balance',
                'unit_of_measurement': 'kWh',
                'import_kwh': round(interval_import, 3),
                'export_kwh': round(interval_export, 3),
                'net_kw': round(interval_net * 4.0, 3),
                'target_net_kwh': round(planned_net, 3),
            })
        except (FileNotFoundError, KeyError, TypeError):
            pass

    else:
        adjusted_battery_kw, log_msg = compute_load_following_setpoint(
            planned_battery_kw=planned_battery_kw,
            planned_action=planned_action,
            solar_kw=solar_kw,
            grid_w=grid_w,
            battery_w=battery_w,
            gshp_kw=gshp_kw,
            leaf_kw=leaf_kw,
            phase_currents=[i_p1, i_p2, i_p3],
        )

    if log_msg:
        print(f'Load follow: {log_msg}')

    ramp_rate = float(os.getenv('BATTERY_RAMP_RATE_KW_PER_MIN', '3.0'))
    adjusted_battery_kw = apply_ramp_rate(
        target_setpoint_kw=adjusted_battery_kw,
        actual_battery_kw=battery_w / 1000.0,
        ramp_rate_kw_per_min=ramp_rate,
    )

    battery_control_w = int(-adjusted_battery_kw * 1000)
    push_battery_control(
        battery_power_w=battery_control_w,
        battery_action=cast(BatteryAction, planned_action),
        battery_soc_pct=planned_soc,
    )


if __name__ == '__main__':
    main()
