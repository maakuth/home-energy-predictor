import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from sklearn.metrics import mean_absolute_error
import joblib
import json

def train(holdout_days=0):
    print('Loading processed data...')
    df = pd.read_csv('processed_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  - Raw rows: {len(df)}")
    
    # Target (Y): baseload = total home power - gshp_power
    target = 'baseload_power'
    
    # Features (X)
    features = [
        'outside_temp', 'wind_speed', 'solar_forecast', 
        'accumulator_temp', 'acc_roc', 'is_fireplace_lag1', 
        'ev_soc', 'ev_position',
        'baseload_lag_1h', 'baseload_lag_24h',
        'is_extended_complex',
        'hour', 'minute', 'quarter_hour', 'day_of_week', 'month'
    ]

    # Drop rows where critical new features are missing
    df = df.dropna(subset=['baseload_lag_1h', 'baseload_lag_24h'])
    print(f"  - Rows after dropna: {len(df)}")

    # --- TRUE BACKTEST LOGIC ---
    if holdout_days > 0:
        cutoff = df.index.max() - pd.Timedelta(days=holdout_days)
        print(f'  - Excluding everything after {cutoff} for a true hold-out test.')
        df = df[df.index <= cutoff]
        print(f"  - Rows after holdout: {len(df)}")

    if len(df) < 10:
        print("❌ Error: Not enough data points to train a model.")
        return

    X = df[features]
    y = df[target]

    # Weights: Give more weight to recent data (last 6 months)
    weights = np.where(df['is_extended_complex'] == 1, 3.0, 1.0)
    
    # Temporal Split: No random shuffling for time-series!
    split_idx = int(len(df) * 0.8)
    
    if split_idx == 0 or split_idx == len(df):
        print("⚠️ Warning: Dataset too small for 80/20 split. Using all data for training/testing.")
        X_train = X_test = X
        y_train = y_test = y
        w_train = w_test = weights
    else:
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        w_train = weights[:split_idx]
        w_test = weights[split_idx:]
    
    print(f"  - Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

    # Model Specification
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42
    )
    
    print('Fitting model...')
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test],
        verbose=False
    )
    
    # Evaluation
    predictions = model.predict(X_test)
    if len(y_test) > 0:
        mae = mean_absolute_error(y_test, predictions)
        print(f'✅ Model Training Complete. Test Set MAE: {mae:.4f}')
    else:
        print('✅ Model Training Complete (Evaluation skipped due to empty test set).')
    
    # Save model
    model.save_model('energy_model.json')
    # Save feature list for inference
    with open('model_features.json', 'w') as f:
        json.dump(features, f)
    print('Model saved to energy_model.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout-days', type=int, default=0, help='Number of recent days to exclude from training')
    args = parser.parse_args()
    train(holdout_days=args.holdout_days)
