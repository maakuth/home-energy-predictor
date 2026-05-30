"""Battery control utilities for HEPO.

Handles pushing battery control setpoint to Hoymiles inverter via Home Assistant.
Uses the number.set_value service to only touch the numeric value without
modifying entity structure or MQTT subscriptions.
"""

from utils.ha_utils import call_ha_service


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
        bool: True if successful, False otherwise
    
    Note:
        Uses number.set_value service to only update the numeric value,
        preserving the entity's MQTT subscription and attributes.
    """
    entity_id = 'number.hoymiles_remote_control_hoymiles_battery_power'
    
    try:
        # Use number.set_value service (only touches value, preserves MQTT)
        result = call_ha_service(
            domain='number',
            service='set_value',
            service_data={
                'entity_id': entity_id,
                'value': battery_power_w
            }
        )
        
        if result:
            action_str = f"({battery_action})" if battery_action != 'idle' else ""
            if battery_soc_pct is not None:
                print(f'✅ Battery Control: {battery_power_w}W {action_str} [SoC {battery_soc_pct:.1f}%]')
            else:
                print(f'✅ Battery Control: {battery_power_w}W {action_str}')
            return True
        else:
            print(f'⚠️ Failed to set battery control to {battery_power_w}W')
            return False
            
    except Exception as e:
        print(f'❌ Error pushing battery control: {e}')
        return False
