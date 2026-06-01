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

def _get_safe_forecast(model, forecast_steps, start_params=None, max_ci_upper=100):
    """
    Fit SARIMA model and return forecast with reasonable confidence intervals.
    
    If start_params are provided, tries a warm-start fit first.
    If the resulting CIs explode (indicating numerical instability), falls back
    to a clean fit without warm start.
    """
    if start_params is not None:
        try:
            results = model.fit(start_params=start_params, disp=False)
            forecast = results.get_forecast(steps=forecast_steps)
            ci = forecast.conf_int(alpha=0.05)
            max_upper = ci.iloc[:, 1].max()
            if max_upper <= max_ci_upper:
                return forecast
            print(f"⚠️ SARIMA warm-start CIs exploded (max upper={max_upper:.1f}), refitting without warm start...")
        except Exception as e:
            print(f"⚠️ SARIMA warm-start fit failed: {e}, refitting without warm start...")
    
    results = model.fit(disp=False)
    return results.get_forecast(steps=forecast_steps)


def _clamp_ci(val, low, high, historical_std=None):
    """Clamp confidence intervals to physically reasonable values for home baseload."""
    if historical_std is not None:
        max_reasonable = val + 3 * historical_std + 5
    else:
        max_reasonable = max(50.0, val * 5)
    upper = min(float(high), max_reasonable)
    lower = max(float(low), 0.0)
    return lower, upper


def predict_sarimax(ts_data, forecast_steps=96, params_path='sarima_model_params.pkl'):
    """
    Predicts future values using SARIMA model.
    If params_path exists, uses pre-trained parameters with fallback if CIs explode.
    Otherwise, fits a default SARIMA model on the data.
    
    Returns:
        (forecast_mean, forecast_ci): tuple of predicted mean and confidence intervals
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
        forecast = _get_safe_forecast(model, forecast_steps, start_params=model_data['params'])
    else:
        model = SARIMAX(
            ts_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 96),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        forecast = _get_safe_forecast(model, forecast_steps)
    
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    forecast_mean = np.clip(forecast_mean, 0, None)
    
    return forecast_mean, forecast_ci

def save_benchmark_results(forecast_mean, forecast_ci=None, filename="sarimax_predictions.json", historical_std=None):
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
            
            low, high = _clamp_ci(val, low, high, historical_std)
            
            results.append({
                "timestamp": ts.isoformat(),
                "predicted_baseload": float(val),
                "lower_95": low,
                "upper_95": high,
                "model": "SARIMA"
            })
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ SARIMA forecast with CI saved to {filename}")

def archive_sarimax_predictions(forecast_mean, forecast_ci, historical_std=None):
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
                low = float(forecast_ci.loc[ts, 'lower baseload_power'])
                high = float(forecast_ci.loc[ts, 'upper baseload_power'])
            except (KeyError, AttributeError):
                low = float(val * 0.5)
                high = float(val * 1.5)
            
            low, high = _clamp_ci(val, low, high, historical_std)
                
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

    historical_std = float(ts_data.std()) if len(ts_data) > 1 else None

    # 3. Re-instantiate model and apply parameters
    print(f"Reconstructing SARIMA model and updating state...")
    model = SARIMAX(
        ts_data,
        order=model_data['order'],
        seasonal_order=model_data['seasonal_order'],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    # Use safe forecast: tries warm start, falls back to clean fit if CIs explode
    forecast_steps = 96 # 24 hours
    print(f"Forecasting {forecast_steps} steps ahead using SARIMA...")
    forecast = _get_safe_forecast(model, forecast_steps, start_params=model_data['params'])
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    # Ensure no negative predictions
    forecast_mean = np.clip(forecast_mean, 0, None)
    
    # 5. Save & Archive
    save_benchmark_results(forecast_mean, forecast_ci, historical_std=historical_std)
    archive_sarimax_predictions(forecast_mean, forecast_ci, historical_std=historical_std)

if __name__ == "__main__":
    main()
