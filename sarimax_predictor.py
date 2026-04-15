import pandas as pd
import numpy as np
import json
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timezone

# SARIMA Predictor Module: sarimax_predictor.py
# Module designed to predict future home energy load using SARIMA based on historical baseload.

def load_historical_data(file_path='processed_data.csv', target_col='baseload_power', last_n_days=7):
    """
    Loads historical baseload data for training.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run process_data.py first.")
        return None
    
    try:
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        
        # Sort index and handle duplicates
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
        # Ensure target column exists
        if target_col not in df.columns:
            print(f"Error: {target_col} missing in {file_path}.")
            return None
            
        # Select recent data to keep training fast (SARIMA is slow)
        cutoff = df.index.max() - pd.Timedelta(days=last_n_days)
        ts_data = df.loc[df.index >= cutoff, target_col]
        
        # Resample to ensure fixed frequency (crucial for SARIMA)
        ts_data = ts_data.resample('15min').mean().ffill()
        
        print(f"Loaded {len(ts_data)} points from {file_path} (Last {last_n_days} days).")
        return ts_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None

def predict_sarimax(ts_data, forecast_steps=96):
    """
    Trains a SARIMA model and forecasts future values.
    Uses daily seasonality (s=96 for 15-min intervals).
    """
    if ts_data is None or len(ts_data) < 192: # Need at least 2 seasonal cycles
        print("SARIMA Prediction failed: Insufficient data (Need at least 2 days of history).")
        return None

    try:
        print("Training SARIMA model (Daily seasonality, s=96)...")
        # (1,1,1) x (1,1,1,96) is often slow. Let's start with a simpler (1,0,0) seasonal if it's too slow.
        # But for benchmarking, let's try standard daily SARIMA.
        model = SARIMAX(
            ts_data, 
            order=(1, 1, 1), 
            seasonal_order=(1, 0, 0, 96), 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        print(f"Forecasting {forecast_steps} steps ahead...")
        forecast = results.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        
        # Ensure no negative predictions
        forecast_mean = np.clip(forecast_mean, 0, None)
        
        return forecast_mean
    except Exception as e:
        print(f"SARIMA Model Error: {e}")
        return None

def save_benchmark_results(sarimax_forecast, filename="sarimax_predictions.json"):
    """Saves SARIMA forecast in a format comparable to future_predictions.json."""
    if sarimax_forecast is not None:
        results = []
        for ts, val in sarimax_forecast.items():
            results.append({
                "timestamp": ts.isoformat(),
                "predicted_baseload": float(val),
                "model": "SARIMA"
            })
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ SARIMA forecast saved to {filename}")
    else:
        print("Warning: No predictions generated to save.")

import sqlite3

def archive_sarimax_predictions(sarimax_forecast, db_file='hepo.db'):
    """Archiving SARIMA predictions to SQLite for later benchmarking."""
    if sarimax_forecast is None:
        return
        
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS sarimax_predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_baseload_kw REAL,
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')
        
        generated_at = datetime.now(timezone.utc).isoformat()
        data_to_insert = [
            (ts.isoformat(), generated_at, float(val))
            for ts, val in sarimax_forecast.items()
        ]
        
        cur.executemany('''
            INSERT OR REPLACE INTO sarimax_predictions 
            (target_timestamp, generated_at, predicted_baseload_kw)
            VALUES (?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f"✅ Archived {len(data_to_insert)} SARIMA points to {db_file}")
    except Exception as e:
        print(f"⚠️ Error archiving SARIMA to SQLite: {e}")

def main():
    # 1. Load historical data
    ts_data = load_historical_data()
    if ts_data is None:
        return

    # 2. Predict (Forecast for 24h = 96 steps of 15min)
    forecast = predict_sarimax(ts_data, forecast_steps=96)
    
    # 3. Save & Archive
    if forecast is not None:
        save_benchmark_results(forecast)
        archive_sarimax_predictions(forecast)

if __name__ == "__main__":
    main()
