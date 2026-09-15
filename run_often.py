from __future__ import annotations
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from utils.ha_utils import call_ha_service, get_ha_state, parse_ha_bool, push_ha_state
from typing import cast
from utils.type_defs import BatteryAction
from utils.gshp_health import health_attributes, update_gshp_health
from utils.battery_utils import (
    push_battery_control,
    compute_load_following_setpoint,
    compute_net_metering_setpoint,
    get_current_plan_entry,
    adjust_charge_solar_for_real_time,
    adjust_idle_for_cheap_export,
    smooth_planned_setpoint,
    apply_ramp_rate,
    apply_discharge_budget,
    accumulate_interval_discharge,
    apply_phase_current_cap,
)

load_dotenv(override=True)


def _get_float(state):
    if state and state.get('state') not in ['unknown', 'unavailable', None]:
        try:
            return float(state['state'])
        except (ValueError, TypeError):
            pass
    return None


def _get_interval_minutes() -> int:
    try:
        return max(int(os.getenv('PLAN_INTERVAL_MINUTES', '15')), 1)
    except ValueError:
        return 15


def _fuse_still_overloaded(
    battery_kw: float,
    battery_w: float,
    phase_currents: list[float | None],
) -> bool:
    """Check the projected phase currents after the requested battery response."""
    fuse_a = float(os.getenv('MAIN_FUSE_SIZE_A', '25.0'))
    battery_delta_w = battery_kw * 1000.0 - battery_w
    phase_delta_a = battery_delta_w / (3.0 * 230.0)
    return any(
        current is not None and abs(current + phase_delta_a) > fuse_a + 0.01
        for current in phase_currents
    )


def control_resistive_heater(
    current_plan: dict | None,
    accumulator_temp: float | None,
    now: datetime | None = None,
    plan_mtime: float | None = None,
) -> None:
    """Deliver the planned resistive energy while failing closed on unsafe state."""
    entity_id = os.getenv('RESISTIVE_HEATER_ENTITY', 'switch.mlp_vastus_output_0')
    max_temp = float(os.getenv('RESISTIVE_HEATER_MAX_TEMP', '60.0'))
    heater_kw = max(0.0, float(os.getenv('RESISTIVE_HEATER_POWER_KW', '6.0')))
    plan_is_current = False
    plan_is_fresh = False
    slot_id = None
    elapsed_seconds = 0.0
    now = now or datetime.now().astimezone()
    if current_plan is not None:
        try:
            timestamp = datetime.fromisoformat(str(current_plan['timestamp'])).astimezone()
            interval_minutes = _get_interval_minutes()
            current_slot = now.replace(
                minute=(now.minute // interval_minutes) * interval_minutes,
                second=0,
                microsecond=0,
            )
            plan_is_current = timestamp.replace(second=0, microsecond=0) == current_slot
            plan_is_fresh = plan_mtime is None or plan_mtime >= current_slot.timestamp()
            slot_id = current_slot.isoformat()
            elapsed_seconds = max(0.0, (now - current_slot).total_seconds())
        except (KeyError, TypeError, ValueError):
            plan_is_current = False
    state_file = os.getenv(
        'RESISTIVE_HEATER_CONTROL_STATE_FILE',
        'state/resistive_heater_control.json',
    )
    completed_slot = None
    try:
        with open(state_file) as f:
            completed_slot = json.load(f).get('completed_slot')
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
        pass

    reached_cutoff = (
        current_plan is not None
        and plan_is_current
        and plan_is_fresh
        and current_plan.get('resistive_heater_intent') == 'ON'
        and accumulator_temp is not None
        and accumulator_temp >= max_temp
    )
    if reached_cutoff and slot_id is not None:
        try:
            parent = os.path.dirname(state_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump({'completed_slot': slot_id}, f)
            completed_slot = slot_id
        except OSError as exc:
            print(f'Could not persist resistive heater cutoff: {exc}')

    planned_kw = 0.0
    if current_plan is not None:
        try:
            planned_kw = max(0.0, min(heater_kw, float(current_plan.get('planned_resistive_kw', 0.0))))
        except (TypeError, ValueError):
            pass
    planned_runtime_seconds = (
        _get_interval_minutes() * 60.0 * planned_kw / heater_kw
        if heater_kw > 0 else 0.0
    )

    should_heat = (
        current_plan is not None
        and plan_is_current
        and plan_is_fresh
        and completed_slot != slot_id
        and current_plan.get('resistive_heater_intent') == 'ON'
        and accumulator_temp is not None
        and accumulator_temp < max_temp
        and elapsed_seconds < planned_runtime_seconds
    )
    if current_plan is not None and plan_is_current and not plan_is_fresh:
        print('Resistive heater held off: plan predates current interval')
    call_ha_service(
        'switch', 'turn_on' if should_heat else 'turn_off',
        {'entity_id': entity_id}, return_response=False,
    )


def control_bulk_heater(
    current_plan: dict | None,
    accumulator_temp: float | None,
    now: datetime | None = None,
    plan_mtime: float | None = None,
) -> None:
    """Deliver the whole-reservoir element plan with the same fail-closed rules."""
    entity_id = os.getenv('BULK_HEATER_ENTITY', 'switch.mlp_vastus_output_1')
    max_temp = float(os.getenv('BULK_HEATER_MAX_TEMP', '60.0'))
    heater_kw = max(0.0, float(os.getenv('BULK_HEATER_POWER_KW', '6.0')))
    now = now or datetime.now().astimezone()
    is_current = False
    is_fresh = False
    elapsed_seconds = 0.0
    if current_plan is not None:
        try:
            timestamp = datetime.fromisoformat(str(current_plan['timestamp'])).astimezone()
            interval_minutes = _get_interval_minutes()
            slot = now.replace(minute=(now.minute // interval_minutes) * interval_minutes, second=0, microsecond=0)
            is_current = timestamp.replace(second=0, microsecond=0) == slot
            is_fresh = plan_mtime is None or plan_mtime >= slot.timestamp()
            elapsed_seconds = max(0.0, (now - slot).total_seconds())
        except (KeyError, TypeError, ValueError):
            pass
    try:
        planned_kw = max(0.0, min(heater_kw, float((current_plan or {}).get('planned_bulk_heater_kw', 0.0))))
    except (TypeError, ValueError):
        planned_kw = 0.0
    should_heat = (
        current_plan is not None and is_current and is_fresh
        and current_plan.get('bulk_heater_intent') == 'ON'
        and accumulator_temp is not None and accumulator_temp < max_temp
        and heater_kw > 0
        and elapsed_seconds < _get_interval_minutes() * 60.0 * planned_kw / heater_kw
    )
    call_ha_service('switch', 'turn_on' if should_heat else 'turn_off', {'entity_id': entity_id}, return_response=False)


def control_leaf_charger(
    current_plan: dict | None,
    manual_charging: bool = False,
    now: datetime | None = None,
    plan_mtime: float | None = None,
) -> None:
    """Apply the current Leaf plan entry without interrupting manual charging."""
    entity_id = os.getenv('LEAF_CHARGING_ENTITY', 'switch.tasmota_3')
    now = now or datetime.now().astimezone()
    plan_is_current = False
    plan_is_fresh = False
    if current_plan is not None:
        try:
            timestamp = datetime.fromisoformat(str(current_plan['timestamp'])).astimezone()
            interval_minutes = _get_interval_minutes()
            current_slot = now.replace(
                minute=(now.minute // interval_minutes) * interval_minutes,
                second=0,
                microsecond=0,
            )
            plan_is_current = timestamp.replace(second=0, microsecond=0) == current_slot
            plan_is_fresh = plan_mtime is None or plan_mtime >= current_slot.timestamp()
        except (KeyError, TypeError, ValueError):
            pass

    should_charge = (
        plan_is_current
        and plan_is_fresh
        and current_plan is not None
        and current_plan.get('leaf_intent') == 'ON'
    )
    if should_charge:
        call_ha_service('switch', 'turn_on', {'entity_id': entity_id}, return_response=False)
    elif not manual_charging:
        call_ha_service('switch', 'turn_off', {'entity_id': entity_id}, return_response=False)


def main():
    soc = get_ha_state('sensor.be_soc')
    battery_power = get_ha_state('sensor.be_stat_batt_power')
    grid_power = get_ha_state('sensor.sahkokauppa_20s')
    solar = get_ha_state(os.getenv('SOLAR_PRODUCTION_ENTITY', 'sensor.solarh_63038_real_power_kw'))
    gshp = get_ha_state('sensor.mlp_teho')
    accumulator = get_ha_state('sensor.mlp_varaajan_lampotila')
    leaf = get_ha_state('sensor.tasmota_energy_power_3')
    p1 = get_ha_state('sensor.current_phase_1')
    p2 = get_ha_state('sensor.current_phase_2')
    p3 = get_ha_state('sensor.current_phase_3')

    import_meter = get_ha_state('sensor.cumulative_active_import')
    export_meter = get_ha_state('sensor.cumulative_active_export')

    soc_pct = _get_float(soc)
    battery_w = _get_float(battery_power) or 0.0
    grid_w = (_get_float(grid_power) or 0.0) * 1000.0
    solar_raw = _get_float(solar)
    solar_kw = solar_raw if solar_raw is not None else 0.0
    gshp_kw = (_get_float(gshp) or 0.0) / 1000.0
    accumulator_temp = _get_float(accumulator)
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

    plan_mtime = None
    try:
        with open('state/optimization_plan.json') as f:
            plan = json.load(f)
            plan_mtime = os.fstat(f.fileno()).st_mtime
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        print('No readable optimization_plan.json found')
        plan = None

    manual_leaf_state = get_ha_state(
        os.getenv('LEAF_MANUAL_CHARGING_ENTITY', 'input_boolean.leaf_manuaalinen_lataus')
    )
    manual_leaf_charging = parse_ha_bool(manual_leaf_state, default=False)

    if not plan:
        if os.getenv('RESISTIVE_HEATER_OPTIMIZE_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            control_resistive_heater(None, accumulator_temp)
        if os.getenv('BULK_HEATER_OPTIMIZE_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            control_bulk_heater(None, accumulator_temp)
        control_leaf_charger(None, manual_charging=manual_leaf_charging)
        return

    current = get_current_plan_entry(plan)
    if current is None:
        print('No current plan entry found')

    upper = get_ha_state(os.getenv('RESISTIVE_HEATER_ENTITY', 'switch.mlp_vastus_output_0'))
    bulk = get_ha_state(os.getenv('BULK_HEATER_ENTITY', 'switch.mlp_vastus_output_1'))
    element_kw = (
        (float(os.getenv('RESISTIVE_HEATER_POWER_KW', '6.0')) if str((upper or {}).get('state', '')).lower() == 'on' else 0.0)
        + (float(os.getenv('BULK_HEATER_POWER_KW', '6.0')) if str((bulk or {}).get('state', '')).lower() == 'on' else 0.0)
    )
    print(
        'GSHP health inputs: '
        f'shared={gshp_kw:.2f}kW, '
        f'upper={str((upper or {}).get("state", "unknown"))} '
        f'({os.getenv("RESISTIVE_HEATER_POWER_KW", "6.0")}kW), '
        f'bulk={str((bulk or {}).get("state", "unknown"))} '
        f'({os.getenv("BULK_HEATER_POWER_KW", "6.0")}kW), '
        f'deducted={element_kw:.2f}kW'
    )
    health = update_gshp_health(current, gshp_kw, element_kw, accumulator_temp=accumulator_temp)
    push_ha_state('sensor.hepo_gshp_health', health.get('status', 'normal'), health_attributes(health))
    if health.get('status') == 'failed':
        print(f"GSHP electrical fault latched: compressor={health.get('compressor_kw', 0.0):.2f}kW")

    if os.getenv('RESISTIVE_HEATER_OPTIMIZE_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        control_resistive_heater(current, accumulator_temp, plan_mtime=plan_mtime)
    if os.getenv('BULK_HEATER_OPTIMIZE_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        control_bulk_heater(current, accumulator_temp, plan_mtime=plan_mtime)
    control_leaf_charger(
        current,
        manual_charging=manual_leaf_charging,
        plan_mtime=plan_mtime,
    )

    planned_battery_kw = current.get('battery_power_kw', 0.0) if current else 0.0
    planned_action = current.get('battery_action', 'idle') if current else 'idle'
    planned_soc = current.get('soc_pct') if current else None
    planned_budget_kwh = current.get('discharge_budget_kwh') if current else None
    if planned_budget_kwh is not None:
        print(f'Period budget: {planned_budget_kwh:.2f} kWh')

    solar_fallback_enabled = os.getenv('SOLAR_FALLBACK_TO_FORECAST', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
    if solar_raw is None and solar_fallback_enabled and current:
        forecast_kw = current.get('solar_forecast_kw', 0.0)
        if forecast_kw > 0:
            solar_kw = forecast_kw
            print(f'Solar sensor unavailable, using forecast: {solar_kw:.2f}kW')

    max_battery_kw = float(os.getenv('BATTERY_MAX_CHARGE_KW', '10.0'))
    min_soc_pct = float(os.getenv('BATTERY_MIN_SOC_PCT', '10.0'))
    floor_pct = max(min_soc_pct, float(os.getenv('BATTERY_RESERVE_SOC_PCT', str(min_soc_pct))))

    _MANUAL_ACTIONS = {
        'idle', 'follow', 'charge_solar', 'charge_grid', 'charge_mixed',
        'discharge_load', 'discharge_export', 'discharge_mixed',
    }
    manual_override = False
    override_mode = 'auto'
    battery_action_state = get_ha_state(
        os.getenv('BATTERY_ACTION_SELECT_ENTITY', 'input_select.hepo_battery_action')
    )
    if battery_action_state:
        action_state = battery_action_state.get('state', 'auto')
        if action_state not in ('unknown', 'unavailable', 'auto') and action_state in _MANUAL_ACTIONS:
            manual_override = True
            override_mode = action_state
            planned_action = action_state
            if planned_action == 'idle':
                planned_battery_kw = 0.0
            elif planned_action in ('charge_solar', 'charge_grid', 'charge_mixed'):
                planned_battery_kw = max_battery_kw
            elif planned_action in ('discharge_load', 'discharge_export', 'discharge_mixed'):
                planned_battery_kw = -max_battery_kw
            elif planned_action == 'follow':
                planned_battery_kw = 0.0
            print(f'Manual battery override: {planned_action}')

    # Smooth setpoint across interval boundaries using prior interval's average
    planned_battery_kw = smooth_planned_setpoint(
        planned_battery_kw=planned_battery_kw,
        planned_action=planned_action,
        actual_battery_w=battery_w,
        plan=plan,
        max_battery_kw=max_battery_kw,
        plan_mtime=plan_mtime,
        override_mode=override_mode,
    )

    planned_battery_kw, planned_action = adjust_charge_solar_for_real_time(
        planned_battery_kw=planned_battery_kw,
        planned_action=planned_action,
        solar_kw=solar_kw,
        grid_w=grid_w,
        battery_w=battery_w,
        battery_soc_pct=soc_pct,
        min_soc_pct=min_soc_pct,
    )

    if not manual_override:
        planned_battery_kw, planned_action = adjust_idle_for_cheap_export(
            planned_battery_kw=planned_battery_kw,
            planned_action=planned_action,
            export_price=current.get('export_unit_price') if current else None,
            grid_w=grid_w,
            battery_w=battery_w,
            battery_soc_pct=soc_pct,
            max_soc_pct=float(os.getenv('BATTERY_MAX_SOC_PCT', '90.0')),
            max_battery_kw=max_battery_kw,
        )

    net_metering = os.getenv('BATTERY_NET_METERING', '').strip().lower() in {'1', 'true', 'yes', 'on'}

    # Battery control: net metering PI only when applicable and not manually overridden
    if net_metering and import_kwh is not None and export_kwh is not None and not manual_override:
        now = datetime.now()
        interval_minutes = _get_interval_minutes()
        elapsed_minutes = now.minute % interval_minutes + now.second / 60.0

        planned_grid_import_kwh = current.get('grid_import_kwh', 0.0) if current else 0.0
        planned_grid_export_kwh = current.get('grid_export_kwh', 0.0) if current else 0.0

        if planned_action == 'follow':
            adjusted_battery_kw, log_msg = compute_load_following_setpoint(
                planned_battery_kw=planned_battery_kw,
                planned_action=planned_action,
                solar_kw=solar_kw,
                grid_w=grid_w,
                battery_w=battery_w,
                gshp_kw=gshp_kw,
                leaf_kw=leaf_kw,
                phase_currents=[i_p1, i_p2, i_p3],
                battery_soc_pct=soc_pct,
                min_soc_pct=min_soc_pct,
            )
        else:
            adjusted_battery_kw, log_msg = compute_net_metering_setpoint(
                planned_battery_kw=planned_battery_kw,
                planned_action=planned_action,
                planned_grid_import_kwh=planned_grid_import_kwh,
                planned_grid_export_kwh=planned_grid_export_kwh,
                cumulative_import_kwh=import_kwh,
                cumulative_export_kwh=export_kwh,
                elapsed_minutes=elapsed_minutes,
                interval_minutes=interval_minutes,
                battery_soc_pct=soc_pct,
                min_soc_pct=min_soc_pct,
            )
        planned_action = 'net_metering'
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
            battery_soc_pct=soc_pct,
            min_soc_pct=min_soc_pct,
        )

    # Period balance reporting: always publish when meter data is available
    if import_kwh is not None and export_kwh is not None:
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
            planned_grid_import_kwh = current.get('grid_import_kwh', 0.0) if current else 0.0
            planned_grid_export_kwh = current.get('grid_export_kwh', 0.0) if current else 0.0
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

    if log_msg:
        print(f'Load follow: {log_msg}')

    ramp_rate = float(os.getenv('BATTERY_RAMP_RATE_KW_PER_MIN', '3.0'))
    adjusted_battery_kw = apply_ramp_rate(
        target_setpoint_kw=adjusted_battery_kw,
        actual_battery_kw=battery_w / 1000.0,
        ramp_rate_kw_per_min=ramp_rate,
    )

    # Per-interval discharge budget: cap load-following discharge so the battery
    # doesn't drain faster than planned during cheap intervals (e.g. EV charging
    # spikes), conserving energy for higher-profit periods.
    plan_action = current.get('battery_action', 'idle') if current else 'idle'
    discharge_budget_kwh = current.get('discharge_budget_kwh') if current else None
    if (
        not manual_override
        and discharge_budget_kwh is not None
        and plan_action in ('follow', 'discharge_load')
    ):
        interval_minutes = _get_interval_minutes()
        discharge_used_kwh = accumulate_interval_discharge(
            battery_w, interval_minutes=interval_minutes,
        )
        adjusted_battery_kw, budget_msg = apply_discharge_budget(
            adjusted_battery_kw=adjusted_battery_kw,
            discharge_budget_kwh=discharge_budget_kwh,
            discharge_used_kwh=discharge_used_kwh,
            interval_minutes=interval_minutes,
        )
        if budget_msg:
            print(f'Discharge budget: {budget_msg}')

    # Fuse safety: final clamp so no phase exceeds the main fuse rating.
    # Applied after the ramp limiter and discharge budget so nothing (net metering
    # corrections, ramping, budgets) can push the setpoint back over the fuse.
    # Reduces charge when a phase is near the import limit and forces discharge
    # when the non-battery load alone already exceeds the fuse.
    adjusted_battery_kw, fuse_msg = apply_phase_current_cap(
        adjusted_battery_kw, battery_w, [i_p1, i_p2, i_p3], max_battery_kw)

    # SoC floor guard: never discharge below the configured minimum, even if the
    # fuse cap would otherwise force a discharge.
    if soc_pct is not None and adjusted_battery_kw < 0 and soc_pct <= floor_pct:
        adjusted_battery_kw = 0.0
        guard_msg = f"soc guard: discharge blocked at SoC {soc_pct:.1f}% (floor {floor_pct:.0f}%)"
        fuse_msg = f"{fuse_msg}; {guard_msg}" if fuse_msg else guard_msg

    # Last-resort fuse protection: after exhausting safe battery discharge, shed
    # planned heating loads rather than leave a phase above its fuse rating.
    if (
        os.getenv('RESISTIVE_HEATER_OPTIMIZE_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}
        and current is not None
        and current.get('resistive_heater_intent') == 'ON'
        and _fuse_still_overloaded(
            adjusted_battery_kw, battery_w, [i_p1, i_p2, i_p3],
        )
    ):
        call_ha_service(
            'switch', 'turn_off',
            {'entity_id': os.getenv('RESISTIVE_HEATER_ENTITY', 'switch.mlp_vastus_output_0')},
            return_response=False,
        )
        priority_msg = 'resistive heater shed after battery fuse response'
        fuse_msg = f"{fuse_msg}; {priority_msg}" if fuse_msg else priority_msg

    if (
        os.getenv('BULK_HEATER_OPTIMIZE_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}
        and current is not None
        and current.get('bulk_heater_intent') == 'ON'
        and _fuse_still_overloaded(
            adjusted_battery_kw, battery_w, [i_p1, i_p2, i_p3],
        )
    ):
        call_ha_service(
            'switch', 'turn_off',
            {'entity_id': os.getenv('BULK_HEATER_ENTITY', 'switch.mlp_vastus_output_1')},
            return_response=False,
        )
        priority_msg = 'bulk heater shed after battery fuse response'
        fuse_msg = f"{fuse_msg}; {priority_msg}" if fuse_msg else priority_msg

    if fuse_msg:
        print(f'Fuse cap: {fuse_msg}')

    battery_control_w = int(-adjusted_battery_kw * 1000)
    push_battery_control(
        battery_power_w=battery_control_w,
        battery_action=cast(BatteryAction, planned_action),
        battery_soc_pct=planned_soc,
    )


if __name__ == '__main__':
    main()
