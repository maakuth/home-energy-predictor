import os
import requests
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(override=True)

def inspect_nordpool():
    host = os.getenv("HA_HOST")
    token = os.getenv("HA_TOKEN")

    if host and not host.startswith(('http://', 'https://')):
        host = f"http://{host}"

    url = f"{host}/api/states/sensor.nordpool_total"
    print(f"Checking {url}...")

    headers = {
        "Authorization": f"Bearer {token}",
        "content-type": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            attr = data.get("attributes", {})
            print("State:", data.get("state"))
            print("Attributes Keys:", list(attr.keys()))

            raw_today = attr.get("raw_today", [])
            print(f"\nraw_today length: {len(raw_today)}")
            if raw_today:
                print("First item of raw_today:")
                print(json.dumps(raw_today[0], indent=2))

            raw_tomorrow = attr.get("raw_tomorrow", [])
            print(f"\nraw_tomorrow length: {len(raw_tomorrow)}")
            if raw_tomorrow:
                print("First item of raw_tomorrow:")
                print(json.dumps(raw_tomorrow[0], indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_nordpool()