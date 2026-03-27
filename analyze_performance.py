import os
import pandas as pd
import numpy as np
import psycopg2
import sqlite3
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

def get_ha_state(entity_id):
    host = os.getenv('HA_HOST')
    token = os.getenv('HA_TOKEN')
    if host and not host.startswith(('http://', 'https://')):
        host = f'http://{host}'
    url = f'{host}/api/states/{entity_id}'
    headers = {'Authorization': f'Bearer {token}', 'content-type': 'application/json'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except: pass
    return None

def fetch_actuals(days=2):
    print(f"Fetching actuals for last {days} days...")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    
    entities = {
        'sensor.sahkokauppa_nyt': 'total_power',
        'sensor.solarh_63038_real_power_kw': 'solar_actual'
    }
    
    query = "SELECT metadata_id, entity_id FROM states_meta WHERE entity_id IN %s"
    cur.execute(query, (tuple(entities.keys()),))
    meta = {row[1]: row[0] for row in cur.fetchall()}
    
    start_ts = (datetime.now() - timedelta(days=days)).timestamp()
    all_data = []
    
    for entity_id, col_name in entities.items():
        if entity_id not in meta: continue
        cur.execute("SELECT last_updated_ts, state FROM states WHERE metadata_id = %s AND last_updated_ts > %s", (meta[entity_id], start_ts))
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=['ts', col_name])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        df = df.set_index('timestamp').drop(columns=['ts'])
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        all_data.append(df.resample('15min').mean())
    
    cur.close()
    conn.close()
    
    df_actual = pd.concat(all_data, axis=1).fillna(0)
    df_actual['actual_usage'] = df_actual['total_power'] + df_actual['solar_actual']
    return df_actual[['actual_usage']]

def analyze():
    db_file = 'hepo.db'
    if not os.path.exists(db_file):
        print("No prediction history (hepo.db) found.")
        return

    print("Loading prediction history from SQLite...")
    try:
        conn = sqlite3.connect(db_file)
        # We want the MOST RECENT prediction for each target timestamp
        query = """
            SELECT target_timestamp, predicted_usage_kw as predicted_usage
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY target_timestamp ORDER BY generated_at DESC) as rn
                FROM predictions
            )
            WHERE rn = 1
        """
        df_history = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"Error reading SQLite: {e}")
        return

    if df_history.empty:
        print("Prediction history is empty.")
        return

    df_history['target_timestamp'] = pd.to_datetime(df_history['target_timestamp'], utc=True)
    df_history = df_history.set_index('target_timestamp')
    
    df_actual = fetch_actuals()
    # actual_usage in df_actual is average Power (kW) over the interval.
    
    # Merge
    comparison = df_history.join(df_actual[['actual_usage']], how='inner')
    if comparison.empty:
        print("No overlapping data between history and actuals.")
        return
    
    # Aggregating to 3-hour blocks (AVERAGE of kW in each block)
    # This makes the analysis much more robust to random human behavior.
    comparison_resampled = comparison.resample('3h').mean().dropna()

    if comparison_resampled.empty:
        print("No overlapping data after 3h resampling.")
        return

    comparison_resampled['error'] = comparison_resampled['predicted_usage'] - comparison_resampled['actual_usage']
    comparison_resampled['abs_error'] = comparison_resampled['error'].abs()
    
    mae = comparison_resampled['abs_error'].mean()
    bias = comparison_resampled['error'].mean()
    rmse = np.sqrt((comparison_resampled['error']**2).mean())
    
    print(f"Analysis Results (3-Hour Windows, N={len(comparison_resampled)}):")
    print(f"  MAE (Power): {mae:.3f} kW")
    print(f"  Bias (Power): {bias:.3f} kW")
    print(f"  RMSE (Power): {rmse:.3f} kW")
    
    # Push to HA
    host = os.getenv('HA_HOST')
    token = os.getenv('HA_TOKEN')
    if host and not host.startswith(('http://', 'https://')): host = f'http://{host}'
    url = f'{host}/api/states/sensor.hepo_accuracy'
    
    payload = {
        'state': f"{mae:.3f}",
        'attributes': {
            'friendly_name': 'HEPO Prediction Accuracy (MAE, 3h windows)',
            'unit_of_measurement': 'kW',
            'bias': float(bias),
            'rmse': float(rmse),
            'sample_count': len(comparison_resampled),
            'window_size': '3h',
            'last_updated': datetime.now().isoformat()
        }
    }
    requests.post(url, headers={'Authorization': f'Bearer {token}', 'content-type': 'application/json'}, json=payload)
    print("✅ Accuracy metrics pushed to Home Assistant.")

if __name__ == "__main__":
    analyze()
