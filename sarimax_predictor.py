import pandas as pd
import numpy as np
import json
import os
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from datetime import datetime, timezone

# SARIMA Prediction Module: sarimax_predictor.py
# Updated for fast frequent execution: loads a daily-trained model and forecasts.

def load_historical_data(file_path='processed_data.csv', target_col='baseload_power', last_n_days=14):
    """
    Loads historical baseload data for prediction anchoring.
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
        
        # Select recent data
        cutoff = df.index.max() - pd.Timedelta(days=last_n_days)
        ts_data = df.loc[df.index >= cutoff, target_col]
        
        # Resample to ensure fixed frequency (crucial for SARIMA)
        ts_data = ts_data.resample('15min').mean().ffill()
        
        return ts_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None

def save_benchmark_results(forecast_mean, forecast_ci, filename="sarimax_predictions.json"):
    """Saves SARIMA forecast and confidence intervals."""
    if forecast_mean is not None:
        results = []
        for ts, val in forecast_mean.items():
            low = forecast_ci.loc[ts, 'lower baseload_power']
            high = forecast_ci.loc[ts, 'upper baseload_power']
            
            results.append({
                "timestamp": ts.isoformat(),
                "predicted_baseload": float(val),
                "lower_95": float(max(0, low)),
                "upper_95": float(max(0, high)),
                "model": "SARIMA"
            })
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ SARIMA forecast with CI saved to {filename}")

def archive_sarimax_predictions(forecast_mean, forecast_ci, db_file='hepo.db'):
    """Archiving SARIMA predictions and CI to SQLite for later benchmarking."""
    if forecast_mean is None:
        return
        
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        
        # Add columns if they don't exist
        cur.execute('''
            CREATE TABLE IF NOT EXISTS sarimax_predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_baseload_kw REAL,
                lower_95 REAL,
                upper_95 REAL,
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')
        
        # Migration: check for new columns
        cur.execute("PRAGMA table_info(sarimax_predictions)")
        cols = [c[1] for c in cur.fetchall()]
        if 'lower_95' not in cols:
            cur.execute("ALTER TABLE sarimax_predictions ADD COLUMN lower_95 REAL")
        if 'upper_95' not in cols:
            cur.execute("ALTER TABLE sarimax_predictions ADD COLUMN upper_95 REAL")

        generated_at = datetime.now(timezone.utc).isoformat()
        data_to_insert = [
            (
                ts.isoformat(), 
                generated_at, 
                float(val), 
                float(max(0, forecast_ci.loc[ts, 'lower baseload_power'])),
                float(max(0, forecast_ci.loc[ts, 'upper baseload_power']))
            )
            for ts, val in forecast_mean.items()
        ]
        
        cur.executemany('''
            INSERT OR REPLACE INTO sarimax_predictions 
            (target_timestamp, generated_at, predicted_baseload_kw, lower_95, upper_95)
            VALUES (?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f"✅ Archived {len(data_to_insert)} SARIMA points with CI to {db_file}")
    except Exception as e:
        print(f"⚠️ Error archiving SARIMA to SQLite: {e}")

def main():
    model_path = 'sarima_model.pkl'
    train_end_path = 'sarima_train_end.txt'

    if not os.path.exists(model_path):
        print(f"⚠️ No SARIMA model found at {model_path}. Please run train_sarima.py first.")
        return

    # 1. Load the pre-trained model
    print("Loading pre-trained SARIMA model...")
    results = SARIMAXResults.load(model_path)
    
    # 2. Load latest data for anchoring (extend model to 'now')
    # Fetch enough to cover the gap since training (usually 1 day) plus some context.
    ts_data = load_historical_data(last_n_days=3)
    
    if ts_data is None:
        return

    # 3. Synchronize model to latest data without full retraining
    # We use .apply() to update the model state with the latest 15-minute intervals.
    # This is much faster than full retraining.
    print(f"Extending SARIMA model with {len(ts_data)} recent data points...")
    updated_results = results.apply(ts_data, refit=False)
    
    # 4. Predict
    forecast_steps = 96 # 24 hours
    print(f"Forecasting {forecast_steps} steps ahead using SARIMA...")
    forecast = updated_results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05) # 95% CI
    
    # Ensure no negative predictions
    forecast_mean = np.clip(forecast_mean, 0, None)
    
    # 5. Save & Archive
    save_benchmark_results(forecast_mean, forecast_ci)
    archive_sarimax_predictions(forecast_mean, forecast_ci)

if __name__ == "__main__":
    main()
