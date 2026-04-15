import pandas as pd
import numpy as np
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from sarimax_predictor import load_historical_data as load_actual_baseload

# Comparison Script: compare_models.py
# Compares the accuracy of the main XGBoost model vs the SARIMA benchmark.

def get_archived_xgboost_predictions(db_file='hepo.db', days=2):
    """Fetch archived XGBoost predictions from SQLite."""
    if not os.path.exists(db_file):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_file)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    query = f"""
        SELECT target_timestamp, predicted_usage_kw as predicted_baseload
        FROM (
            SELECT target_timestamp, predicted_usage_kw, 
                   ROW_NUMBER() OVER (PARTITION BY target_timestamp ORDER BY generated_at DESC) as rn
            FROM predictions
            WHERE target_timestamp >= '{cutoff}'
        )
        WHERE rn = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
    return df.set_index('timestamp')[['predicted_baseload']]

def load_sarima_latest_forecast(filename='sarimax_predictions.json'):
    """Load the latest SARIMA forecast (note: this is only the latest one run)."""
    if not os.path.exists(filename):
        return pd.DataFrame()
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.set_index('timestamp')[['predicted_baseload']]

def get_archived_sarima_predictions(db_file='hepo.db', days=2):
    """Fetch archived SARIMA predictions from SQLite."""
    if not os.path.exists(db_file):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_file)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        query = f"""
            SELECT target_timestamp, predicted_baseload_kw as predicted_baseload
            FROM (
                SELECT target_timestamp, predicted_baseload_kw, 
                       ROW_NUMBER() OVER (PARTITION BY target_timestamp ORDER BY generated_at DESC) as rn
                FROM sarimax_predictions
                WHERE target_timestamp >= '{cutoff}'
            )
            WHERE rn = 1
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
        return df.set_index('timestamp')[['predicted_baseload']]
    except Exception:
        conn.close()
        return pd.DataFrame()

def compare():
    print("=== Model Comparison Analysis ===")
    
    # 1. Load actual historical baseload (Ground Truth)
    actual_ts = load_actual_baseload(file_path='processed_data.csv', last_n_days=2)
    if actual_ts is None:
        return
    
    actual_df = pd.DataFrame({'actual_baseload': actual_ts})
    
    # 2. XGBoost Performance
    xgb_df = get_archived_xgboost_predictions(days=2)
    if not xgb_df.empty:
        xgb_comp = xgb_df.join(actual_df, how='inner').dropna()
        if not xgb_comp.empty:
            mae = (xgb_comp['predicted_baseload'] - xgb_comp['actual_baseload']).abs().mean()
            print(f"XGBoost (Main) MAE: {mae:.3f} kW ({len(xgb_comp)} samples)")
        else:
            print("⚠️ XGBoost: No overlapping actuals yet.")

    # 3. SARIMA Performance
    sarima_df = get_archived_sarima_predictions(days=2)
    if not sarima_df.empty:
        sarima_comp = sarima_df.join(actual_df, how='inner').dropna()
        if not sarima_comp.empty:
            mae = (sarima_comp['predicted_baseload'] - sarima_comp['actual_baseload']).abs().mean()
            print(f"SARIMA (Benchmark) MAE: {mae:.3f} kW ({len(sarima_comp)} samples)")
        else:
            print("⚠️ SARIMA: No overlapping actuals yet.")
    else:
        print("⚠️ No archived SARIMA predictions found.")

    print("\nNote: MAE is only calculated for timestamps where both actual data and a prior prediction exist.")

if __name__ == "__main__":
    compare()
