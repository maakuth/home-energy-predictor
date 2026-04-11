import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# Centralized configuration
HA_HOST = os.getenv('HA_HOST')
HA_TOKEN = os.getenv('HA_TOKEN')

if HA_HOST and not HA_HOST.startswith(('http://', 'https://')):
    HA_HOST = f'http://{HA_HOST}'

HEADERS = {
    'Authorization': f'Bearer {HA_TOKEN}',
    'content-type': 'application/json',
}

def get_ha_state(entity_id):
    """Fetch the state and attributes of a Home Assistant entity."""
    url = f'{HA_HOST}/api/states/{entity_id}'
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f'⚠️ Error fetching {entity_id}: {e}')
    return None

def call_ha_service(domain, service, service_data=None):
    """Call a Home Assistant service."""
    url = f'{HA_HOST}/api/services/{domain}/{service}'
    try:
        response = requests.post(url, headers=HEADERS, json=service_data or {}, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            print(f'❌ Error calling service {domain}.{service}: {response.status_code} - {response.text}')
    except Exception as e:
        print(f'❌ Service call failed for {domain}.{service}: {e}')
    return None

def push_ha_state(entity_id, state, attributes=None):
    """Push a state and attributes to a Home Assistant sensor."""
    url = f'{HA_HOST}/api/states/{entity_id}'
    payload = {
        'state': str(state),
        'attributes': attributes or {}
    }
    # Ensure last_updated is always present if not provided
    if 'last_updated' not in payload['attributes']:
        payload['attributes']['last_updated'] = datetime.now().isoformat()

    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=10)
        if response.status_code in [200, 201]:
            return True
        else:
            print(f'❌ Error pushing {entity_id}: {response.status_code} - {response.text}')
    except Exception as e:
        print(f'❌ Connection failed for {entity_id}: {e}')
    return False
