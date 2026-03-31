import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.ha_utils import push_ha_state
from utils.db_utils import fetch_states_history

load_dotenv(override=True)

def fetch_actuals(days=2):
    print(f"Fetching actuals for last {days} days from PostgreSQL...")
    entities = {
        'sensor.sahkokauppa_nyt': 'total_power',
        'sensor.solarh_63038_real_power_kw': 'solar_actual'
    }
    
    # Use central utility
    hist_data = fetch_states_history(list(entities.keys()), hours=days*24)
    
    all_resampled = []
    for eid, col_name in entities.items():
        df = hist_data.get(eid)
        if df is not None and not df.empty:
            df = df.rename(columns={'state': col_name})
            all_resampled.append(df.set_index('timestamp').resample('15min').mean())
    
    if not all_resampled:
        print("⚠️ No data fetched from PostgreSQL.")
        return pd.DataFrame(columns=['actual_usage'])

    df_actual = pd.concat(all_resampled, axis=1).fillna(0)
    df_actual['actual_usage'] = df_actual.get('total_power', 0) + df_actual.get('solar_actual', 0)
    return df_actual[['actual_usage']]

def analyze():
    db_file = 'hepo.db'
    if not os.path.exists(db_file):
        print("No prediction history (hepo.db) found.")
        return

    print("Loading prediction history from SQLite...")
    try:
        conn = sqlite3.connect(db_file)
        # Check if column exists
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(predictions)")
        cols = [c[1] for c in cur.fetchall()]
        has_fallback_col = 'is_fallback_price' in cols
        
        fallback_select = ", is_fallback_price" if has_fallback_col else ", 0 as is_fallback_price"

        # We want the MOST RECENT prediction for each target timestamp
        query = f"""
            SELECT target_timestamp, predicted_usage_kw as predicted_usage {fallback_select}
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
    
    # Filter out fallback prices for interval analysis
    comparison_clean = comparison[comparison['is_fallback_price'] == 0]
    
    # --- 3-Hour Analysis (Average Power kW) ---
    comparison_resampled_3h = comparison_clean.resample('3h').mean().dropna()
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
    # We only want to include days that have ZERO fallback intervals to avoid partial sums
    daily_fallback_max = comparison['is_fallback_price'].resample('24h').max()
    comparison_resampled_24h = comparison_kwh.resample('24h').sum().dropna()
    
    # Filter by fallback and also ensure we have enough data points for a full day (at least 90 out of 96 intervals)
    counts = comparison_kwh.resample('24h').count()
    comparison_resampled_24h = comparison_resampled_24h[(daily_fallback_max == 0) & (counts['actual_usage'] >= 90)]

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
