import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

def get_ha_state(entity_id):
    host = os.getenv('HA_HOST')
    token = os.getenv('HA_TOKEN')
    if host and not host.startswith(('http://', 'https://')):
        host = f'http://{host}'
    
    url = f'{host}/api/states/{entity_id}'
    headers = {
        'Authorization': f'Bearer {token}',
        'content-type': 'application/json',
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('state')
    except Exception as e:
        print(f'⚠️ Error fetching {entity_id}: {e}')
    return None

def predict():
    print('Syncing with Home Assistant...')
    solar_tomorrow = get_ha_state('sensor.solcast_pv_forecast_forecast_tomorrow')
    try:
        solar_val = float(solar_tomorrow) if solar_tomorrow and solar_tomorrow not in ['unknown', 'unavailable'] else 0.0
    except (ValueError, TypeError):
        solar_val = 0.0
    current_temp = get_ha_state('sensor.ulkona_temperature_2')
    try:
        temp_val = float(current_temp) if current_temp and current_temp not in ['unknown', 'unavailable'] else 5.0
    except (ValueError, TypeError):
        temp_val = 5.0
    acc_temp = get_ha_state('sensor.mlp_varaajan_lampotila')
    try:
        acc_val = float(acc_temp) if acc_temp and acc_temp not in ['unknown', 'unavailable'] else 45.0
    except (ValueError, TypeError):
        acc_val = 45.0
    ev_soc = get_ha_state('sensor.xpz_491_battery_level')
    try:
        soc_val = float(ev_soc) if ev_soc and ev_soc not in ['unknown', 'unavailable'] else 80.0
    except (ValueError, TypeError):
        soc_val = 80.0
    model = xgb.XGBRegressor()
    try:
        model.load_model('energy_model.json')
        with open('model_features.json', 'r') as f:
            features = json.load(f)
    except FileNotFoundError:
        print('Error: Model files not found. Run train_model.py first.')
        return
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    print(f'Predicting for {tomorrow.date()} using Solar Forecast: {solar_val} kWh')
    inference_data = []
    for h in range(24):
        ts = tomorrow + timedelta(hours=h)
        row = {
            'outside_temp': temp_val,
            'solar_forecast': solar_val,
            'accumulator_temp': acc_val,
            'is_fireplace_lag1': 0,
            'ev_soc': soc_val,
            'ev_position': 1,
            'hour': ts.hour,
            'day_of_week': ts.weekday(),
            'month': ts.month
        }
        inference_data.append(row)
    X_inference = pd.DataFrame(inference_data)[features]
    predictions = model.predict(X_inference)
    print('\nPredicted Hourly Usage (kWh):')
    for h, p in enumerate(predictions):
        print(f' {h:02d}:00 -> {p:.2f} kWh')
    np.save('tomorrow_predictions.npy', predictions)
    print('\n✅ Predictions saved to tomorrow_predictions.npy')

if __name__ == "__main__":
    predict()