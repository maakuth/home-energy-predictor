from __future__ import annotations
from utils.ha_utils import get_ha_state
from dotenv import load_dotenv

load_dotenv()

print('=== Home Battery Status ===')
soc = get_ha_state('sensor.be_soc')
if soc:
    print(f'SoC: {soc.get("state")}%')
else:
    print('SoC: unavailable')

power = get_ha_state('sensor.be_stat_batt_power')
if power and power.get('state') not in ['unknown', 'unavailable']:
    power_w = float(power['state'])
    direction = "Charging" if power_w > 0 else "Discharging"
    print(f'Power: {power_w:.0f}W ({direction} at {abs(power_w):.0f}W)')
else:
    print('Power: unavailable')
