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
    entity_id = 'number.hoymiles_remote_control_hoymiles_battery_power'
    
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
