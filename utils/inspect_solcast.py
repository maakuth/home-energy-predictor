from __future__ import annotations
import os
import requests
import json
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)

def inspect_solcast() -> None:
    host = os.getenv('HA_HOST')
    token = os.getenv('HA_TOKEN')
    if host and not host.startswith(('http://', 'https://')):
        host = f'http://{host}'
    
    url = f'{host}/api/states/sensor.solcast_pv_forecast_forecast_tomorrow'
    headers = {
        'Authorization': f'Bearer {token}',
        'content-type': 'application/json',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print('State:', data.get('state'))
            print('Attributes Keys:', list(data.get('attributes', {}).keys()))
            print(json.dumps(data.get('attributes', {}), indent=2))
        else:
            print(f'Error: {response.status_code}')
            print(response.text)
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    inspect_solcast()