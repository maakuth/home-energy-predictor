from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from sklearn.metrics import mean_absolute_error
import json
import os
from utils.baseload_quality import BASELOAD_FEATURES, select_training_rows

def train(holdout_days: int = 0, data_path: str = 'state/processed_data.csv', min_coverage_days: float = 14.0) -> None:
    print('Loading processed data...')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    if df.index.has_duplicates:
        print('❌ Duplicate timestamps; keeping existing model.')
        return
    step_seconds = float(df.index.to_series().diff().dt.total_seconds().median())
    print(f"  - Raw rows: {len(df)}")
    
    # Target (Y): baseload = total home power - gshp_power
    target = 'baseload_power'
    
    # Features (X)
    features = list(BASELOAD_FEATURES)

    # A partial/corrupt extract cannot provide a usable model.  Check columns
    # before dropna so this path fails gracefully instead of raising KeyError.
    required_columns = [target, *features]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print(f"❌ Error: Missing required columns: {', '.join(missing_columns)}")
        return

    # Drop rows where critical new features are missing
    # Anchor holdout to the input end, even if recent measurements are rejected.
    dataset_end = df.index.max()
    try:
        df = select_training_rows(df)
    except ValueError as error:
        print(f'❌ {error}; keeping existing model.')
        return
    print(f"  - Rows after dropna: {len(df)}")
    if 'heating_regime' in df:
        print('  - Eligible rows by heating regime:\n' + df['heating_regime'].value_counts().to_string())

    # --- TRUE BACKTEST LOGIC ---
    if holdout_days > 0:
        cutoff = dataset_end - pd.Timedelta(days=holdout_days)
        print(f'  - Excluding everything after {cutoff} for a true hold-out test.')
        df = df[df.index <= cutoff]
        print(f"  - Rows after holdout: {len(df)}")

    if len(df) < 10:
        print("❌ Error: Not enough data points to train a model.")
        return
    if (not np.isfinite(step_seconds) or step_seconds <= 0 or step_seconds > 900
            or not np.isfinite(min_coverage_days) or min_coverage_days <= 0
            or len(df) * step_seconds < min_coverage_days * 86400):
        print(f'❌ Insufficient eligible coverage (requires {min_coverage_days:g} sample-days); keeping existing model.')
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
    os.makedirs('state', exist_ok=True)
    model.save_model('state/energy_model.json')
    with open('state/model_features.json', 'w') as f:
        json.dump(features, f)
    print('Model saved to energy_model.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout-days', type=int, default=0, help='Number of recent days to exclude from training')
    parser.add_argument('--data-path', default='state/processed_data.csv', help='Processed training-data CSV path')
    parser.add_argument('--min-coverage-days', type=float, default=14.0, help='Minimum eligible sample-days before fitting (default 14)')
    args = parser.parse_args()
    train(holdout_days=args.holdout_days, data_path=args.data_path, min_coverage_days=args.min_coverage_days)
