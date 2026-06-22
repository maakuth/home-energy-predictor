"""Battery control utilities for HEPO.

Handles pushing battery control setpoint to Hoymiles inverter via Home Assistant.
Uses the number.set_value service to only touch the numeric value without
modifying entity structure or MQTT subscriptions.

Gracefully handles battery unavailability - if the entity doesn't exist,
silently continues without error (degradation mode for testing).
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv
from utils.ha_utils import call_ha_service, get_ha_state

load_dotenv(override=True)

# Battery control entity ID — configurable via environment variable
BATTERY_CONTROL_ENTITY_ID = os.getenv(
    'BATTERY_CONTROL_ENTITY_ID',
    'number.hoymiles_remote_bridge_hoymiles_power_control_v5'
)


def get_current_plan_entry(plan, interval_minutes=15):
    """
    Find the plan entry that matches the current time interval.

    The plan is generated at 15-minute intervals. This function finds the
    entry whose timestamp falls within the current 15-minute slot, to avoid
    executing the wrong interval's plan (e.g., using the 10:00 plan at 09:46).

    Args:
        plan (list): List of dict entries, each with 'timestamp' key (ISO format).
        interval_minutes (int): Plan interval in minutes (default 15).

    Returns:
        dict or None: The matching plan entry, or None if plan is empty.
                      Falls back to plan[0] if no entry matches current time.
    """
    if not plan:
        return None

    now = datetime.now().astimezone()
    current_slot = now.replace(
        minute=(now.minute // interval_minutes) * interval_minutes,
        second=0,
        microsecond=0
    )
    for entry in plan:
        if 'timestamp' not in entry:
            continue
        ts = datetime.fromisoformat(entry['timestamp'])
        if ts.replace(second=0, microsecond=0) == current_slot:
            return entry

    # Fallback: return the first entry if no timestamp matches
    return plan[0]


def is_battery_available():
    """
    Check if battery is available in Home Assistant.
    
    Uses the battery SoC sensor (sensor.be_soc) as the source of truth,
    which is more reliable than the Hoymiles control entity that can
    report unavailable while the battery itself is online.
    
    Returns:
        bool: True if battery SoC sensor exists and is not unavailable, False otherwise
    """
    entity_id = 'sensor.be_soc'
    state = get_ha_state(entity_id)
    
    if state and state.get('state') not in ['unknown', 'unavailable', None]:
        return True
    return False


def push_battery_control(battery_power_w, battery_action='follow', battery_soc_pct=None):
    """
    Push battery control setpoint to Hoymiles inverter.
    
    Args:
        battery_power_w (int): Control setpoint in Watts
            Positive = discharge (provide power to home)
            Negative = charge (draw from grid/solar)
        battery_action (str): Current planned action (for logging)
        battery_soc_pct (float): Current battery SoC percentage (for logging)
    
    Returns:
        bool: True if successful, False otherwise (still True if battery unavailable - degradation mode)
    
    Note:
        Uses number.set_value service to only update the numeric value,
        preserving the entity's MQTT subscription and attributes.
        
        If battery entity is unavailable, silently skips push (degradation mode).
    """
    entity_id = BATTERY_CONTROL_ENTITY_ID
    
    # Check if battery is available before attempting push
    if not is_battery_available():
        # Degradation mode: battery not available, skip gracefully
        action_str = f"({battery_action})" if battery_action != 'idle' else ""
        if battery_soc_pct is not None:
            print(f'⊘ Battery Unavailable: {battery_power_w}W {action_str} [SoC {battery_soc_pct:.1f}%] (skipped)')
        else:
            print(f'⊘ Battery Unavailable: {battery_power_w}W {action_str} (skipped)')
        return True  # Still return True to not break the plan
    
    try:
        # Use number.set_value service (only touches value, preserves MQTT)
        # Note: return_response=False because number.set_value doesn't support responses
        result = call_ha_service(
            domain='number',
            service='set_value',
            service_data={
                'entity_id': entity_id,
                'value': battery_power_w
            },
            return_response=False
        )
        
        # With return_response=False, the service call either succeeds (returns dict/json)
        # or fails (returns None from the except/error handlers)
        if result is not None:
            action_str = f"({battery_action})" if battery_action != 'idle' else ""
            if battery_soc_pct is not None:
                print(f'✅ Battery Control: {battery_power_w}W {action_str} [SoC {battery_soc_pct:.1f}%]')
            else:
                print(f'✅ Battery Control: {battery_power_w}W {action_str}')
            return True
        else:
            # result is None means error was already printed by call_ha_service
            return False
            
    except Exception as e:
        print(f'❌ Error pushing battery control: {e}')
        return False


def get_env_float(name, default):
    """Safely get a float from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def compute_load_following_setpoint(planned_battery_kw, planned_action,
                                   solar_kw, grid_w, battery_w,
                                   gshp_kw=0.0, leaf_kw=0.0,
                                   max_battery_kw=10.0,
                                   phase_currents=None):
    """
    Adjust planned battery setpoint based on real-time sensor readings.

    For 'charge_solar': limits charge to actual solar surplus.
    For 'discharge_load': limits discharge to actual house load.
    For 'follow': opportunistically charges/discharges to minimise grid flow
        (capped by BATTERY_FOLLOW_MAX_KW).
    For 'idle': battery does nothing — setpoint is forced to 0 kW.
    For price-arbitrage actions (charge_mixed, charge_grid, discharge_mixed,
    discharge_export): returns the planned setpoint unchanged.

    Additionally, caps the setpoint to prevent exceeding the main fuse limit
    on any of the three phases (if phase_currents are provided).

    Args:
        planned_battery_kw (float): Planned battery power in kW
            (positive = charging, negative = discharging).
        planned_action (str): Planned battery action string.
        solar_kw (float): Actual solar production in kW.
        grid_w (float): Net grid power in Watts
            (positive = importing, negative = exporting).
        battery_w (float): Current battery power in Watts
            (positive = charging, negative = discharging).
        gshp_kw (float): Actual GSHP power in kW.
        leaf_kw (float): Actual Leaf charging power in kW.
        max_battery_kw (float): Maximum battery power in kW.
        phase_currents (list of float, optional): Current flow (Amps) per phase
            at the utility meter (positive = import, negative = export).

    Returns:
        tuple: (adjusted_battery_kw, log_message)
            adjusted_battery_kw is the new setpoint in kW.
            log_message is a human-readable description of the adjustment
            (empty string if no adjustment was made).
    """
    # Calculate actual total house load excluding battery
    actual_load_kw = solar_kw + (grid_w / 1000.0) - (battery_w / 1000.0)
    adjusted_battery_kw = planned_battery_kw
    log_message = ""

    if planned_action == 'charge_solar':
        actual_surplus_kw = solar_kw - actual_load_kw
        if actual_surplus_kw > 0:
            adjusted_battery_kw = min(max_battery_kw, actual_surplus_kw)
        else:
            adjusted_battery_kw = 0.0
        if adjusted_battery_kw != planned_battery_kw:
            log_message = (f'charge_solar planned {planned_battery_kw:.2f}kW -> '
                           f'adjusted {adjusted_battery_kw:.2f}kW '
                           f'(surplus {actual_surplus_kw:.2f}kW)')

    elif planned_action == 'discharge_load':
        planned_discharge_kw = abs(planned_battery_kw)
        adjusted_discharge_kw = min(planned_discharge_kw, actual_load_kw)
        adjusted_battery_kw = -adjusted_discharge_kw
        if adjusted_battery_kw != planned_battery_kw:
            log_message = (f'discharge_load planned {planned_battery_kw:.2f}kW -> '
                           f'adjusted {adjusted_battery_kw:.2f}kW '
                           f'(load {actual_load_kw:.2f}kW)')

    elif planned_action == 'idle':
        # True idle: battery does nothing — setpoint forced to 0 kW.
        adjusted_battery_kw = 0.0
        if abs(planned_battery_kw) > 0.01 or abs(battery_w) > 10:
            log_message = (f'idle: battery forced to 0 kW '
                           f'(planned {planned_battery_kw:.2f}kW, actual {battery_w / 1000.0:.2f}kW)')

    elif planned_action == 'follow':
        # Proportional control: adjust battery power to zero out grid,
        # with a small deadband to prevent noise-induced hunting.
        # target = current_battery + (-grid) = battery - grid.
        # This naturally brings grid toward zero in one step.
        # Capped by BATTERY_FOLLOW_MAX_KW to prevent large corrections that
        # would override the planner's decision to leave the battery alone.
        grid_kw = grid_w / 1000.0
        battery_kw = battery_w / 1000.0
        max_follow_kw = get_env_float('BATTERY_FOLLOW_MAX_KW', 2.0)
        
        if abs(grid_w) > 200:  # Deadband: ignore small grid deviations
            target_kw = battery_kw - grid_kw
            if abs(target_kw) > max_follow_kw:
                # Grid flow too large — maintain current setpoint
                adjusted_battery_kw = battery_kw
            else:
                adjusted_battery_kw = max(-max_battery_kw, min(max_battery_kw, target_kw))
        else:
            adjusted_battery_kw = battery_kw  # Maintain current setpoint
        
        if abs(adjusted_battery_kw) > 0.5:
            direction = 'charge' if adjusted_battery_kw > 0 else 'discharge'
            log_message = (f'follow -> {direction} {abs(adjusted_battery_kw):.2f}kW '
                           f'(grid {grid_w:.0f}W)')

    # Phase current capping
    if phase_currents and any(c is not None for c in phase_currents):
        # Check all phases for fuse limit (25A default)
        main_fuse_a = get_env_float('MAIN_FUSE_SIZE_A', 25.0)
        
        # Power limits in Watts
        # For a 3-phase inverter, we assume power is distributed equally.
        # So each phase gets 1/3 of the total battery power.
        # Current change on one phase = (ΔP_total / 3) / 230V
        # => ΔP_total = ΔI_phase * 3 * 230V
        
        phase_caps_w = []
        for i, Ip in enumerate(phase_currents):
            if Ip is None: continue
            
            # Max power increase (charging more) before hitting import fuse limit
            # Ip + (P_extra / 3) / 230 <= fuse
            # P_extra <= (fuse - Ip) * 3 * 230
            # P_max = P_current + P_extra
            p_max_p = battery_w + (main_fuse_a - Ip) * 3 * 230.0
            
            # Max power decrease (discharging more) before hitting export fuse limit
            # Ip + (P_extra / 3) / 230 >= -fuse
            # P_extra >= (-fuse - Ip) * 3 * 230
            # P_min = P_current + P_extra
            p_min_p = battery_w + (-main_fuse_a - Ip) * 3 * 230.0
            
            phase_caps_w.append((p_min_p, p_max_p))
            
        if phase_caps_w:
            combined_min_w = max([c[0] for c in phase_caps_w])
            combined_max_w = min([c[1] for c in phase_caps_w])
            
            # Convert to kW for comparison with adjusted_battery_kw
            combined_min_kw = combined_min_w / 1000.0
            combined_max_kw = combined_max_w / 1000.0
            
            old_setpoint = adjusted_battery_kw
            adjusted_battery_kw = max(combined_min_kw, min(combined_max_kw, adjusted_battery_kw))
            
            if abs(adjusted_battery_kw - old_setpoint) > 0.01:
                cap_msg = f" (phase cap: {combined_min_kw:.2f}..{combined_max_kw:.2f}kW)"
                if log_message:
                    log_message += cap_msg
                else:
                    log_message = f"Phase cap applied: {old_setpoint:.2f}kW -> {adjusted_battery_kw:.2f}kW{cap_msg}"

    # Clamp to physical limits
    adjusted_battery_kw = max(-max_battery_kw, min(max_battery_kw, adjusted_battery_kw))

    return adjusted_battery_kw, log_message


def _load_net_metering_state(state_file=None):
    """Load the persisted net metering state for interval tracking."""
    if state_file is None:
        state_file = os.getenv('HEPO_NET_METERING_STATE_FILE', 'net_metering_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_net_metering_state(state, state_file=None):
    """Persist the net metering state for interval tracking."""
    if state_file is None:
        state_file = os.getenv('HEPO_NET_METERING_STATE_FILE', 'net_metering_state.json')
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f)
    except IOError:
        pass


def compute_net_metering_setpoint(
    planned_battery_kw,
    planned_grid_import_kwh,
    planned_grid_export_kwh,
    cumulative_import_kwh,
    cumulative_export_kwh,
    elapsed_minutes,
    interval_minutes=15,
    max_battery_kw=10.0,
    state_file=None,
):
    """
    Adjust battery power to match the planned net energy for the current interval
    using cumulative energy meter readings.

    With net metering, the grid meter only cares about net energy per quarter.
    This function uses a feedback loop: it compares actual net energy so far
    against the planned net energy, and adjusts battery power so that the
    remaining energy needed is spread evenly over the remaining interval.

    Args:
        planned_battery_kw: Planned battery power (positive=charge, negative=discharge)
        planned_grid_import_kwh: Planned grid import for this interval
        planned_grid_export_kwh: Planned grid export for this interval
        cumulative_import_kwh: Current cumulative active import reading (kWh)
        cumulative_export_kwh: Current cumulative active export reading (kWh)
        elapsed_minutes: Minutes elapsed in current interval
        interval_minutes: Total interval length (default 15)
        max_battery_kw: Maximum battery power in kW
        state_file: Path to JSON file for persisting interval state

    Returns:
        tuple: (adjusted_battery_kw, log_message)
    """
    state = _load_net_metering_state(state_file)
    
    # Interval start detection
    interval_start = state.get('interval_start')
    if interval_start is None:
        interval_start = cumulative_import_kwh + cumulative_export_kwh
        state['interval_start'] = interval_start
        state['import_start'] = cumulative_import_kwh
        state['export_start'] = cumulative_export_kwh
        state['planned_battery_kw'] = planned_battery_kw
        _save_net_metering_state(state, state_file)
        return planned_battery_kw, f"net metering baseline: import={cumulative_import_kwh:.3f}, export={cumulative_export_kwh:.3f}"

    # Check if this is a new interval (cumulative meters have been reset)
    # Actually cumulative meters are monotonic, so we detect reset by checking if
    # current reading is less than stored start (which would indicate a meter reset)
    if cumulative_import_kwh < state.get('import_start', 0) or cumulative_export_kwh < state.get('export_start', 0):
        # Meter reset: capture new baseline
        state['import_start'] = cumulative_import_kwh
        state['export_start'] = cumulative_export_kwh
        state['planned_battery_kw'] = planned_battery_kw
        _save_net_metering_state(state, state_file)
        return planned_battery_kw, f"net metering reset: new baseline import={cumulative_import_kwh:.3f}"

    # Compute actual net energy so far
    import_start = state.get('import_start', cumulative_import_kwh)
    export_start = state.get('export_start', cumulative_export_kwh)
    
    actual_import = cumulative_import_kwh - import_start
    actual_export = cumulative_export_kwh - export_start
    actual_net = actual_import - actual_export
    
    # Planned net energy
    planned_net = planned_grid_import_kwh - planned_grid_export_kwh
    
    # Deviation: positive = imported too much (or exported too little)
    deviation = actual_net - planned_net
    
    # Compute remaining time
    remaining_minutes = max(interval_minutes - elapsed_minutes, 0.5)
    remaining_hours = remaining_minutes / 60.0
    
    # Correction power: opposite sign to deviation
    correction = -deviation / remaining_hours
    
    adjusted = planned_battery_kw + correction
    
    # Clamp
    clamped = max(-max_battery_kw, min(max_battery_kw, adjusted))
    
    if abs(deviation) < 0.001:
        log_msg = ""
    else:
        log_msg = (
            f"net metering: actual_net={actual_net:.3f}kWh, planned={planned_net:.3f}kWh, "
            f"deviation={deviation:.3f}kWh, correction={correction:.2f}kW, "
        )
        if abs(clamped - adjusted) > 0.01:
            log_msg += f"clamped {adjusted:.2f}kW -> {clamped:.2f}kW"
        else:
            log_msg += f"adjusted {planned_battery_kw:.2f}kW -> {clamped:.2f}kW"
    
    # Update state
    state['planned_battery_kw'] = planned_battery_kw
    _save_net_metering_state(state, state_file)

    return clamped, log_msg


def adjust_charge_solar_for_real_time(
    planned_battery_kw: float,
    planned_action: str,
    solar_kw: float,
    grid_w: float,
    battery_w: float,
    battery_soc_pct: float | None = None,
    min_soc_pct: float = 10.0,
    max_battery_kw: float = 10.0,
):
    """Adjust charge_solar planned action based on real-time conditions.

    When the plan expects solar surplus (``charge_solar``) but real-time
    shows no surplus (load > solar), switch to discharging to cover the
    net load instead of idling and importing from grid. The battery will
    recharge later from future solar surplus as predicted by the plan.

    Returns
    -------
    tuple
        (adjusted_battery_kw, adjusted_action) — unchanged if no adjustment.
    """
    if planned_action != 'charge_solar':
        return planned_battery_kw, planned_action

    actual_load_kw = solar_kw + (grid_w / 1000.0) - (battery_w / 1000.0)
    actual_surplus_kw = solar_kw - actual_load_kw

    surplus_deadband_kw = 0.1
    epsilon = 1e-9
    if actual_surplus_kw >= -surplus_deadband_kw - epsilon:
        return planned_battery_kw, planned_action

    if battery_soc_pct is None or battery_soc_pct <= min_soc_pct + 5.0:
        return planned_battery_kw, planned_action

    net_load_kw = -actual_surplus_kw
    discharge_kw = min(max_battery_kw, net_load_kw)
    return -discharge_kw, 'discharge_load'


def estimate_follow_dispatch(net_kw, interval_hours, max_follow_kw=2.0,
                              deadband_kw=0.2):
    """Estimate real-time load-following dispatch for a single interval.
    
    Returns (follow_kwh, is_discharge) or (0.0, None) if no follow activity
    is expected (net below deadband or above follow cap).
    
    follow_kwh:  energy (kWh) the battery would charge/discharge
    is_discharge: True = battery discharges (net import being cancelled),
                  False = battery charges (net export being absorbed)
    """
    abs_net = abs(net_kw)
    if abs_net < deadband_kw or abs_net > max_follow_kw:
        return 0.0, None
    return abs_net * interval_hours, (net_kw > 0)
