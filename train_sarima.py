import pandas as pd
import numpy as np
import os
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sarimax_predictor import load_historical_data

# SARIMA Training Module: train_sarima.py
# Runs daily to fit the SARIMA model on 60 days of history.

def train_sarima(days=60):
    print(f"Loading last {days} days for SARIMA training...")
    ts_data = load_historical_data(last_n_days=days)
    
    if ts_data is None or len(ts_data) < 192:
        print("❌ Not enough data for SARIMA training.")
        return

    try:
        print(f"Fitting SARIMA model (60-day window, s=96, D=1)... This may take a few minutes.")
        # Using the same config as sarimax_predictor.py
        model = SARIMAX(
            ts_data, 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 0, 96), 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Save the results object (statsmodels has its own save/load)
        results.save('sarima_model.pkl')
        print("✅ SARIMA model trained and saved to sarima_model.pkl")
        
        # Also save the training data's end index to know where to start 'extend' from
        with open('sarima_train_end.txt', 'w') as f:
            f.write(ts_data.index.max().isoformat())
            
    except Exception as e:
        print(f"❌ SARIMA Training Error: {e}")

if __name__ == "__main__":
    train_sarima(days=60)
