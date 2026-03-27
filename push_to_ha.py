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
    state = f"{total_energy:.2f}"
    
    # Attributes: The full plan
    payload = {
        'state': state,
        'attributes': {
            'friendly_name': 'HEPO Optimization Plan',
            'plan': plan,
            'unit_of_measurement': 'kWh',
            'device_class': 'energy',
            'last_updated_ts': datetime.now().isoformat()
        }
    }
    
    print(f'Pushing plan to {url}...')
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code in [200, 201]:
            print('✅ Plan successfully pushed to Home Assistant!')
            print(f'Response: {response.json()}')
        else:
            print(f'❌ Error pushing plan: {response.status_code}')
            print(response.text)
    except Exception as e:
        print(f'❌ Connection failed: {e}')

if __name__ == '__main__':
    push_plan()
