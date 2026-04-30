import pandas as pd
import numpy as np
import json
import os
import sqlite3
import pickle
import fcntl
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timezone
from utils.sqlite_utils import get_db_connection, get_db_path

# SARIMA Prediction Module: sarimax_predictor.py
# Updated for fast frequent execution: loads pre-trained parameters and forecasts.

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

def predict_sarimax(ts_data, forecast_steps=96, params_path='sarima_model_params.pkl'):
    """
    Predicts future values using SARIMA model.
    If params_path exists, uses pre-trained parameters.
    Otherwise, fits a default SARIMA model on the data.
    """
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            model_data = pickle.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        
        model = SARIMAX(
            ts_data,
            order=model_data['order'],
            seasonal_order=model_data['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.smooth(model_data['params'])
    else:
        model = SARIMAX(
            ts_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 96),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
    
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    forecast_mean = np.clip(forecast_mean, 0, None)
    
    return forecast_mean

def save_benchmark_results(forecast_mean, forecast_ci=None, filename="sarimax_predictions.json"):
    """Saves SARIMA forecast and confidence intervals."""
    if forecast_mean is not None:
        results = []
        for ts, val in forecast_mean.items():
            # Handle potential missing CI values
            try:
                low = forecast_ci.loc[ts, 'lower baseload_power']
                high = forecast_ci.loc[ts, 'upper baseload_power']
            except (KeyError, AttributeError):
                low = val * 0.5
                high = val * 1.5
            
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

def archive_sarimax_predictions(forecast_mean, forecast_ci):
    """Archiving SARIMA predictions and CI to SQLite for later benchmarking."""
    if forecast_mean is None:
        return
        
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
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
        data_to_insert = []
        for ts, val in forecast_mean.items():
            try:
                low = float(max(0, forecast_ci.loc[ts, 'lower baseload_power']))
                high = float(max(0, forecast_ci.loc[ts, 'upper baseload_power']))
            except (KeyError, AttributeError):
                low = float(val * 0.5)
                high = float(val * 1.5)
                
            data_to_insert.append((ts.isoformat(), generated_at, float(val), low, high))
        
        cur.executemany('''
            INSERT OR REPLACE INTO sarimax_predictions 
            (target_timestamp, generated_at, predicted_baseload_kw, lower_95, upper_95)
            VALUES (?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f"✅ Archived {len(data_to_insert)} SARIMA points with CI to {get_db_path()}")
    except Exception as e:
        print(f"⚠️ Error archiving SARIMA to SQLite: {e}")

def main():
    params_path = 'sarima_model_params.pkl'

    if not os.path.exists(params_path):
        print(f"⚠️ No SARIMA parameters found at {params_path}. Please run train_sarima.py first.")
        return

    # 1. Load the pre-trained parameters
    print("Loading pre-trained SARIMA parameters...")
    with open(params_path, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        model_data = pickle.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    
    # 2. Load latest data for anchoring (at least 2 days to maintain seasonal state)
    ts_data = load_historical_data(last_n_days=3)
    if ts_data is None:
        return

    # 3. Re-instantiate model and apply parameters
    print(f"Reconstructing SARIMA model and updating state...")
    model = SARIMAX(
        ts_data,
        order=model_data['order'],
        seasonal_order=model_data['seasonal_order'],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    # This 'smooth' method allows updating the model with new data using fixed parameters
    results = model.smooth(model_data['params'])
    
    # 4. Predict
    forecast_steps = 96 # 24 hours
    print(f"Forecasting {forecast_steps} steps ahead using SARIMA...")
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    # Ensure no negative predictions
    forecast_mean = np.clip(forecast_mean, 0, None)
    
    # 5. Save & Archive
    save_benchmark_results(forecast_mean, forecast_ci)
    archive_sarimax_predictions(forecast_mean, forecast_ci)

if __name__ == "__main__":
    main()
