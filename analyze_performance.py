import os
import pandas as pd
import numpy as np
import psycopg2
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
    if not os.path.exists('prediction_history.csv'):
        print("No prediction history found.")
        return

    print("Loading prediction history...")
    df_history = pd.read_csv('prediction_history.csv')
    df_history['target_timestamp'] = pd.to_datetime(df_history['target_timestamp'], utc=True)
    df_history['generated_at'] = pd.to_datetime(df_history['generated_at'], utc=True)
    
    # We want the MOST RECENT prediction for each target timestamp
    df_history = df_history.sort_values('generated_at').drop_duplicates('target_timestamp', keep='last')
    df_history = df_history.set_index('target_timestamp')
    
    df_actual = fetch_actuals()
    # actual_usage in df_actual is average Power (kW) over the interval.
    # Convert kW to kWh per 15-min interval.
    df_actual['actual_usage_kwh'] = df_actual['actual_usage'] * 0.25
    
    # Merge
    comparison = df_history.join(df_actual[['actual_usage_kwh']], how='inner')
    if comparison.empty:
        print("No overlapping data between history and actuals.")
        return
    
    # Aggregating to 3-hour blocks (sum of kWh in each block)
    # This makes the analysis much more robust to random human behavior.
    comparison_resampled = comparison.resample('3h').agg({
        'predicted_usage': 'sum',
        'actual_usage_kwh': 'sum'
    }).dropna()

    if comparison_resampled.empty:
        print("No overlapping data after 3h resampling.")
        return

    comparison_resampled['error'] = comparison_resampled['predicted_usage'] - comparison_resampled['actual_usage_kwh']
    comparison_resampled['abs_error'] = comparison_resampled['error'].abs()
    
    mae = comparison_resampled['abs_error'].mean()
    bias = comparison_resampled['error'].mean()
    rmse = np.sqrt((comparison_resampled['error']**2).mean())
    
    print(f"Analysis Results (3-Hour Windows, N={len(comparison_resampled)}):")
    print(f"  MAE (Energy): {mae:.3f} kWh")
    print(f"  Bias (Energy): {bias:.3f} kWh")
    print(f"  RMSE (Energy): {rmse:.3f} kWh")
    
    # Optional: Report Power Bias if requested
    # bias_kw = bias / 3.0
    # print(f"  Avg Power Bias: {bias_kw:.3f} kW")
    
    # Push to HA
    host = os.getenv('HA_HOST')
    token = os.getenv('HA_TOKEN')
    if host and not host.startswith(('http://', 'https://')): host = f'http://{host}'
    url = f'{host}/api/states/sensor.hepo_accuracy'
    
    payload = {
        'state': f"{mae:.3f}",
        'attributes': {
            'friendly_name': 'HEPO Prediction Accuracy (MAE, 3h windows)',
            'unit_of_measurement': 'kWh',
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
