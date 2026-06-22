from __future__ import annotations
import pandas as pd
import numpy as np
import os
import pickle
import fcntl
import argparse
from typing import Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sarimax_predictor import load_historical_data, BASELOAD_MAX_KW

def train_sarima(days: int = 14, params_path: Optional[str] = None) -> None:
    print(f"Loading last {days} days for SARIMA training...")
    ts_data = load_historical_data(last_n_days=days)
    
    if ts_data is None or len(ts_data) < 192:
        print("❌ Not enough data for SARIMA training.")
        return

    # Support environment variable override for testing
    if params_path is None:
        params_path = os.getenv('TEST_SARIMA_PARAMS', 'sarima_model_params.pkl')
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
        
        # Validate: do a quick forecast to ensure the model isn't producing insane values
        forecast = results.get_forecast(steps=10)
        forecast_mean = forecast.predicted_mean
        max_mean = float(forecast_mean.max())
        min_mean = float(forecast_mean.min())
        
        if min_mean < 0 or max_mean > BASELOAD_MAX_KW:
            print(f"⚠️ SARIMA model validation failed: forecast mean range [{min_mean:.2f}, {max_mean:.2f}] kW exceeds physical limits [0, {BASELOAD_MAX_KW}] kW.")
            print(f"⚠️ Keeping existing params. Not overwriting {params_path}.")
            return
        
        # Manually save ONLY the parameters and specification to keep file size tiny
        model_data = {
            'params': results.params,
            'order': order,
            'seasonal_order': seasonal_order,
            'last_index': ts_data.index.max()
        }
        
        with open(params_path, 'wb') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            pickle.dump(model_data, f)
            fcntl.flock(f, fcntl.LOCK_UN)
            
        print(f"✅ SARIMA parameters saved to {params_path} ({len(results.params)} params)")
        
    except Exception as e:
        print(f"❌ SARIMA Training Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SARIMA model")
    parser.add_argument("--days", type=int, default=14, help="Number of days of historical data to use (default: 14)")
    args = parser.parse_args()
    
    train_sarima(days=args.days)
