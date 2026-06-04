"""Battery control utilities for HEPO.

Handles pushing battery control setpoint to Hoymiles inverter via Home Assistant.
Uses the number.set_value service to only touch the numeric value without
modifying entity structure or MQTT subscriptions.

Gracefully handles battery unavailability - if the entity doesn't exist,
silently continues without error (degradation mode for testing).
"""

import os
from dotenv import load_dotenv
from utils.ha_utils import call_ha_service, get_ha_state

load_dotenv(override=True)

# Battery control entity ID — configurable via environment variable
BATTERY_CONTROL_ENTITY_ID = os.getenv(
    'BATTERY_CONTROL_ENTITY_ID',
    'number.hoymiles_remote_bridge_hoymiles_power_control_v5'
)


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


def push_battery_control(battery_power_w, battery_action='idle', battery_soc_pct=None):
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


def compute_load_following_setpoint(planned_battery_kw, planned_action,
                                   solar_kw, grid_w, battery_w,
                                   gshp_kw=0.0, leaf_kw=0.0,
                                   max_battery_kw=10.0):
    """
    Adjust planned battery setpoint based on real-time sensor readings.

    For 'charge_solar': limits charge to actual solar surplus.
    For 'discharge_load': limits discharge to actual house load.
    For 'idle': opportunistically charges/discharges to minimize grid flow.
    For price-arbitrage actions (charge_mixed, charge_grid, discharge_mixed,
    discharge_export): returns the planned setpoint unchanged.

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
            adjusted_battery_kw = min(planned_battery_kw, actual_surplus_kw)
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
        if grid_w < -500:  # Exporting >500W
            adjusted_battery_kw = min(abs(grid_w / 1000.0), max_battery_kw)
            log_message = (f'idle -> opportunistic charge {adjusted_battery_kw:.2f}kW '
                           f'(export {abs(grid_w):.0f}W)')
        elif grid_w > 500:  # Importing >500W
            adjusted_battery_kw = -min(abs(grid_w / 1000.0), max_battery_kw)
            log_message = (f'idle -> opportunistic discharge {abs(adjusted_battery_kw):.2f}kW '
                           f'(import {grid_w:.0f}W)')
        else:
            adjusted_battery_kw = 0.0

    # Clamp to physical limits
    adjusted_battery_kw = max(-max_battery_kw, min(max_battery_kw, adjusted_battery_kw))

    return adjusted_battery_kw, log_message
