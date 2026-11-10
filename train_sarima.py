from __future__ import annotations
import pandas as pd
import numpy as np
import os
import pickle
import fcntl
import argparse
from typing import Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sarimax_predictor import load_historical_data, BASELOAD_MAX_KW, add_history_arguments, history_input

def train_sarima(
    days: int | None = 14,
    params_path: Optional[str] = None,
    data_path: str | pd.DataFrame = 'state/processed_data.csv',
    *, _outcome: dict[str, bool] | None = None,
) -> None:
    if _outcome is not None:
        _outcome['saved'] = False
    window = f'last {days} days' if days is not None else 'explicit input window'
    print(f"Loading {window} for SARIMA training...")
    ts_data = load_historical_data(file_path=data_path, last_n_days=days)
    
    if ts_data is None or ts_data.notna().sum() < 192:
        print("❌ Not enough data for SARIMA training.")
        return

    # Support environment variable override for testing
    if params_path is None:
        params_path = os.getenv('TEST_SARIMA_PARAMS', 'state/sarima_model_params.pkl')
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
        print(f"Fitting SARIMA model ({window}, s=96, D=1)...")
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
        
        if min_mean < -0.5 or max_mean > BASELOAD_MAX_KW:
            print(f"⚠️ SARIMA model validation failed: forecast mean range [{min_mean:.2f}, {max_mean:.2f}] kW exceeds physical limits [-0.5, {BASELOAD_MAX_KW}] kW.")
            print(f"⚠️ Keeping existing params. Not overwriting {params_path}.")
            return
        
        # Manually save ONLY the parameters and specification to keep file size tiny
        model_data = {
            'params': results.params,
            'order': order,
            'seasonal_order': seasonal_order,
            'last_index': ts_data.index.max(),
            'training_metadata': dict(ts_data.attrs),
        }
        
        with open(params_path, 'wb') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            pickle.dump(model_data, f)
            fcntl.flock(f, fcntl.LOCK_UN)
            
        print(f"✅ SARIMA parameters saved to {params_path} ({len(results.params)} params)")
        if _outcome is not None:
            _outcome['saved'] = True
        
    except Exception as e:
        print(f"❌ SARIMA Training Error: {e}")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train SARIMA model")
    parser.add_argument('--days', type=int, default=None, help='CSV lookback days (default: 14); archive uses explicit bounds')
    parser.add_argument('--params-path', default=None)
    parser.add_argument('--require-success', action='store_true', help='Exit nonzero when no new model parameters were saved')
    add_history_arguments(parser, 'state/processed_data.csv')
    args = parser.parse_args(argv)
    if args.archive and args.days is not None:
        parser.error('--days is for CSV mode; archive mode uses explicit --from/--to')
    if args.days is not None and args.days <= 0:
        parser.error('--days must be positive')
    data = history_input(parser, args)
    outcome: dict[str, bool] = {}
    train_sarima(days=None if args.archive else (args.days or 14),
                  data_path=data, params_path=args.params_path, _outcome=outcome)
    if args.require_success and not outcome.get('saved', False):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
