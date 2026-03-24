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
            return response.json()
    except Exception as e:
        print(f'⚠️ Error fetching {entity_id}: {e}')
    return None

def predict():
    print('Syncing with Home Assistant...')
    
    # fetch solar data
    solar_today_state = get_ha_state('sensor.solcast_pv_forecast_forecast_today')
    solar_tomorrow_state = get_ha_state('sensor.solcast_pv_forecast_forecast_tomorrow')
    
    # Combine solar forecasts
    solar_data = []
    
    if solar_today_state:
        raw = solar_today_state.get('attributes', {}).get('detailedHourly', [])
        for entry in raw:
            entry['source'] = 'today'
        solar_data.extend(raw)
        
    if solar_tomorrow_state:
        raw = solar_tomorrow_state.get('attributes', {}).get('detailedHourly', [])
        for entry in raw:
            entry['source'] = 'tomorrow'
        solar_data.extend(raw)
        
    # Convert to DataFrame
    if not solar_data:
        print('Error: No solar forecast data found.')
        return
        
    df_solar = pd.DataFrame(solar_data)
    df_solar['period_start'] = pd.to_datetime(df_solar['period_start'])
    df_solar = df_solar.sort_values('period_start').set_index('period_start')
    
    # Current states
    current_temp = get_ha_state('sensor.ulkona_temperature_2')
    try:
        temp_val = float(current_temp.get('state')) if current_temp and current_temp.get('state') not in ['unknown', 'unavailable'] else 5.0
    except (ValueError, TypeError, AttributeError):
        temp_val = 5.0
        
    acc_temp = get_ha_state('sensor.mlp_varaajan_lampotila')
    try:
        acc_val = float(acc_temp.get('state')) if acc_temp and acc_temp.get('state') not in ['unknown', 'unavailable'] else 45.0
    except (ValueError, TypeError, AttributeError):
        acc_val = 45.0

    ev_soc = get_ha_state('sensor.xpz_491_battery_level')
    try:
        soc_val = float(ev_soc.get('state')) if ev_soc and ev_soc.get('state') not in ['unknown', 'unavailable'] else 80.0
    except (ValueError, TypeError, AttributeError):
        soc_val = 80.0

    # Load model and features
    model = xgb.XGBRegressor()
    try:
        model.load_model('energy_model.json')
        with open('model_features.json', 'r') as f:
            features = json.load(f)
    except FileNotFoundError:
        print('Error: Model files not found. Run train_model.py first.')
        return

    # Determine prediction window: From next hour until end of tomorrow
    now = datetime.now().astimezone() # Use aware datetime
    start_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    
    # Find the latest timestamp in solar data (usually end of tomorrow)
    # If solar data is short, limit to what we have
    end_time = df_solar.index.max()
    
    print(f'Predicting from {start_time} to {end_time}')
    
    inference_data = []
    timestamps = []
    
    current_ts = start_time
    while current_ts <= end_time:
        # Get solar for this hour
        # Resample logic: take the nearest or interpolate?
        # Solar data is hourly or 30min? Let's check. 
        # Usually detailedHourly is 30min or hourly. Let's use asof/nearest for simplicity or resample.
        # But for now, let's just use the index lookup with tolerance.
        
        try:
            # Look up solar forecast for this specific hour
            # We use truncate to hour just in case
            lookup_ts = current_ts.replace(minute=0)
            # Find closest index
            idx = df_solar.index.get_indexer([lookup_ts], method='nearest')[0]
            solar_val = df_solar.iloc[idx]['pv_estimate']
        except Exception:
            solar_val = 0.0
            
        row = {
            'outside_temp': temp_val,
            'solar_forecast': solar_val,
            'accumulator_temp': acc_val,
            'is_fireplace_lag1': 0,
            'ev_soc': soc_val,
            'ev_position': 1, # Assume home
            'hour': current_ts.hour,
            'day_of_week': current_ts.weekday(),
            'month': current_ts.month
        }
        inference_data.append(row)
        timestamps.append(current_ts.isoformat())
        
        current_ts += timedelta(hours=1)
        
    X_inference = pd.DataFrame(inference_data)[features]
    predictions = model.predict(X_inference)
    
    # Combine predictions with timestamps
    results = []
    for i, p in enumerate(predictions):
        results.append({
            'timestamp': timestamps[i],
            'predicted_usage': float(p),
            'solar_forecast': float(inference_data[i]['solar_forecast'])
        })
        
    print(f'\nGenerated {len(results)} hourly predictions.')
    
    with open('future_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\n✅ Predictions saved to future_predictions.json')

if __name__ == '__main__':
    predict()
