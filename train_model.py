import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import json

def train():
    print('Loading processed data...')
    df = pd.read_csv('processed_data.csv', index_col=0)
    
    # Target (Y): baseload = total home power - gshp_power
    target = 'baseload_power'
    
    # Features (X)
    # Features mentioned in PLAN.md:
    # Weather: outside_temp, wind_speed, solar_forecast
    # Thermal State: accumulator_temp, acc_roc, is_fireplace_lag1
    # EV State: ev_soc, ev_position
    # Temporal: hour, quarter_hour, day_of_week, month
    # Anchors: baseload_lag_1h, baseload_lag_24h
    features = [
        'outside_temp', 'wind_speed', 'solar_forecast', 
        'accumulator_temp', 'acc_roc', 'is_fireplace_lag1', 
        'ev_soc', 'ev_position',
        'baseload_lag_1h', 'baseload_lag_24h',
        'is_extended_complex',
        'hour', 'minute', 'quarter_hour', 'day_of_week', 'month'
    ]

    # Drop rows where critical new features are missing (e.g. at the start of history)
    df = df.dropna(subset=['baseload_lag_1h', 'baseload_lag_24h'])

    X = df[features]
    y = df[target]


    # Calculate weights: Give more weight to recent data (last 6 months)
    # This helps the model anchor to the new building's consumption levels.
    weights = np.where(df['is_extended_complex'] == 1, 3.0, 1.0)
    
    print(f'Training with features: {features}')
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
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
        sample_weight_eval=[w_test],
        verbose=False
    )
    
    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'✅ Model Training Complete. MAE: {mae:.4f}')
    
    # Save model
    model.save_model('energy_model.json')
    # Save feature list for inference
    with open('model_features.json', 'w') as f:
        json.dump(features, f)
    print('Model saved to energy_model.json')

if __name__ == '__main__':
    train()
