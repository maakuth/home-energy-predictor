import pandas as pd
import numpy as np
import os
import pickle
import fcntl
import argparse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sarimax_predictor import load_historical_data

# SARIMA Training Module: train_sarima.py
# Runs periodically to fit the SARIMA model on historical data.
# Default: 14 days (for frequent runs), but 30 days recommended for weekly runs.

def train_sarima(days=14):
    print(f"Loading last {days} days for SARIMA training...")
    ts_data = load_historical_data(last_n_days=days)
    
    if ts_data is None or len(ts_data) < 192:
        print("❌ Not enough data for SARIMA training.")
        return

    params_path = 'sarima_model_params.pkl'
    start_params = None
    if os.path.exists(params_path):
        try:
            with open(params_path, 'rb') as f:
                old_data = pickle.load(f)
                start_params = old_data.get('params')
                print("🔄 Found existing parameters. Using as 'Warm Start' for faster convergence.")
        except Exception:
            pass

    try:
        print(f"Fitting SARIMA model ({days}-day window, s=96, D=1)...")
        # Specification
        order = (1, 1, 1)
        seasonal_order = (1, 1, 0, 96)
        
        model = SARIMAX(
            ts_data, 
            order=order, 
            seasonal_order=seasonal_order, 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        
        # Use warm start if available
        results = model.fit(start_params=start_params, disp=False)
        
        # Manually save ONLY the parameters and specification to keep file size tiny
        model_data = {
            'params': results.params,
            'order': order,
            'seasonal_order': seasonal_order,
            'last_index': ts_data.index.max()
        }
        
        with open('sarima_model_params.pkl', 'wb') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            pickle.dump(model_data, f)
            fcntl.flock(f, fcntl.LOCK_UN)
            
        print(f"✅ SARIMA parameters saved to sarima_model_params.pkl ({len(results.params)} params)")
        
    except Exception as e:
        print(f"❌ SARIMA Training Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SARIMA model")
    parser.add_argument("--days", type=int, default=14, help="Number of days of historical data to use (default: 14)")
    args = parser.parse_args()
    
    train_sarima(days=args.days)
