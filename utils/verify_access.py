from __future__ import annotations
import os
import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

def verify_postgres() -> None:
    print("Verifying PostgreSQL access...")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    
    print(f"Connecting to {user}@{host}:{port}/{dbname}...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=os.getenv("DB_PASSWORD")
        )
        print("✅ PostgreSQL connection successful!")
        cur = conn.cursor()
        cur.execute("SELECT version();")
        row = cur.fetchone()
        print(f"PostgreSQL version: {row[0] if row else 'unknown'}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")

def verify_home_assistant() -> None:
    print("\nVerifying Home Assistant API access...")
    host = os.getenv("HA_HOST")
    token = os.getenv("HA_TOKEN")
    
    if host and not host.startswith(('http://', 'https://')):
        print(f"⚠️  HA_HOST '{host}' missing scheme. Adding http://")
        host = f"http://{host}"
    
    url = f"{host}/api/"
    print(f"Connecting to {url}...")
    headers = {
        "Authorization": f"Bearer {token}",
        "content-type": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Home Assistant API connection successful!")
            print(f"Message: {response.json().get('message')}")
        else:
            print(f"❌ Home Assistant API returned status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Home Assistant API connection failed: {e}")

if __name__ == "__main__":
    verify_postgres()
    verify_home_assistant()
