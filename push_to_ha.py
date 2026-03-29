import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

def push_plan():
    print('Loading optimization plan...')
    try:
        with open('optimization_plan.json', 'r') as f:
            plan = json.load(f)
    except FileNotFoundError:
        print('Error: optimization_plan.json not found.')
        return

    host = os.getenv('HA_HOST')
    token = os.getenv('HA_TOKEN')
    if host and not host.startswith(('http://', 'https://')):
        host = f'http://{host}'
        
    url = f'{host}/api/states/sensor.hepo_optimization_plan'
    
    headers = {
        'Authorization': f'Bearer {token}',
        'content-type': 'application/json',
    }
    
    # State: Total predicted energy from current hour to end of tomorrow (sum of hourly predictions)
    total_energy = sum(p['predicted_usage_kwh'] for p in plan)
    
    # Calculate 24h estimate (first 24 hours of the plan)
    # Each interval is PLAN_INTERVAL_MINUTES (usually 15). 24h = 96 intervals.
    intervals_in_24h = 96 
    usage_24h = sum(p['predicted_usage_kwh'] for p in plan[:intervals_in_24h])
    
    state = f"{total_energy:.2f}"
    
    # Attributes: The full plan
    payload = {
        'state': state,
        'attributes': {
            'friendly_name': 'HEPO Optimization Plan',
            'plan': plan,
            'unit_of_measurement': 'kWh',
            'device_class': 'energy',
            'predicted_24h_usage': round(usage_24h, 2),
            'last_updated_ts': datetime.now().isoformat()
        }
    }
    
    print(f'Pushing plan to {url}...')
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code in [200, 201]:
            print('✅ Plan successfully pushed to Home Assistant!')
        else:
            print(f'❌ Error pushing plan: {response.status_code}')
    except Exception as e:
        print(f'❌ Connection failed: {e}')

    # Also push 24h usage as a standalone sensor for easier history tracking
    url_24h = f'{host}/api/states/sensor.hepo_predicted_24h_usage'
    payload_24h = {
        'state': f"{usage_24h:.2f}",
        'attributes': {
            'friendly_name': 'HEPO Predicted 24h Consumption',
            'unit_of_measurement': 'kWh',
            'device_class': 'energy'
        }
    }
    try:
        requests.post(url_24h, headers=headers, json=payload_24h, timeout=10)
    except: pass

if __name__ == '__main__':
    push_plan()
