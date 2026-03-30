import os
import pandas as pd
import numpy as np
import psycopg2
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.ha_utils import push_ha_state

load_dotenv(override=True)

def fetch_actuals(days=2):
    print(f"Fetching actuals for last {days} days from PostgreSQL...")
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
    
    # Merge
    comparison = df_history.join(df_actual[['actual_usage']], how='inner')
    if comparison.empty:
        print("No overlapping data between history and actuals.")
        return
    
    # --- 3-Hour Analysis (Average Power kW) ---
    comparison_resampled_3h = comparison.resample('3h').mean().dropna()
    if not comparison_resampled_3h.empty:
        comparison_resampled_3h['error'] = comparison_resampled_3h['predicted_usage'] - comparison_resampled_3h['actual_usage']
        comparison_resampled_3h['abs_error'] = comparison_resampled_3h['error'].abs()
        mae_3h = comparison_resampled_3h['abs_error'].mean()
        bias_3h = comparison_resampled_3h['error'].mean()
        rmse_3h = np.sqrt((comparison_resampled_3h['error']**2).mean())
        print(f"Analysis Results (3-Hour Power, N={len(comparison_resampled_3h)}):")
        print(f"  MAE: {mae_3h:.3f} kW, Bias: {bias_3h:.3f} kW")
        
        # Push 3h metrics
        attributes_3h = {
            'friendly_name': 'HEPO Prediction Accuracy (3h Power)',
            'unit_of_measurement': 'kW',
            'bias': float(bias_3h),
            'rmse': float(rmse_3h),
            'sample_count': len(comparison_resampled_3h),
            'window_size': '3h'
        }
        push_ha_state('sensor.hepo_accuracy', f"{mae_3h:.3f}", attributes_3h)

    # --- 24-Hour Analysis (Total Energy kWh) ---
    # Convert kW to kWh per 15-min interval (kW * 0.25h)
    comparison_kwh = comparison.copy()
    comparison_kwh['predicted_usage'] *= 0.25
    comparison_kwh['actual_usage'] *= 0.25
    
    # Aggregating to 24-hour blocks (sum of kWh)
    comparison_resampled_24h = comparison_kwh.resample('24h').sum().dropna()
    # Filter out partial days (at least 90 out of 96 intervals)
    counts = comparison_kwh.resample('24h').count()
    comparison_resampled_24h = comparison_resampled_24h[counts['actual_usage'] >= 90]

    if not comparison_resampled_24h.empty:
        comparison_resampled_24h['error'] = comparison_resampled_24h['predicted_usage'] - comparison_resampled_24h['actual_usage']
        comparison_resampled_24h['abs_error'] = comparison_resampled_24h['error'].abs()
        
        mae_24h = comparison_resampled_24h['abs_error'].mean()
        bias_24h = comparison_resampled_24h['error'].mean()
        rmse_24h = np.sqrt((comparison_resampled_24h['error']**2).mean())
        
        print(f"Analysis Results (24-Hour Energy, N={len(comparison_resampled_24h)}):")
        print(f"  MAE: {mae_24h:.3f} kWh, Bias: {bias_24h:.3f} kWh")
        
        # Push 24h metrics
        attributes_24h = {
            'friendly_name': 'HEPO Prediction Accuracy (24h Energy)',
            'unit_of_measurement': 'kWh',
            'bias': float(bias_24h),
            'rmse': float(rmse_24h),
            'sample_count': len(comparison_resampled_24h),
            'window_size': '24h'
        }
        push_ha_state('sensor.hepo_accuracy_24h', f"{mae_24h:.3f}", attributes_24h)
        
    print("✅ Accuracy metrics pushed to Home Assistant.")

if __name__ == "__main__":
    analyze()
