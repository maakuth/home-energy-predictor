from __future__ import annotations
import os
import psycopg2
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)

def find_wind() -> None:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    
    print("Searching for entities containing 'wind'...")
    cur.execute("SELECT entity_id FROM states_meta WHERE entity_id LIKE %s", ('%wind%',))
    results = cur.fetchall()
    
    if results:
        for res in results:
            print(f" - {res[0]}")
    else:
        print("No entities found containing 'wind'.")
        
    print("\nSearching for 'weather' entities...")
    cur.execute("SELECT entity_id FROM states_meta WHERE entity_id LIKE %s", ('weather.%',))
    results = cur.fetchall()
    for res in results:
        print(f" - {res[0]}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    find_wind()
