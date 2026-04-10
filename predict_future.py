import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.ha_utils import get_ha_state
from utils.price_utils import fetch_market_prices
from utils.db_utils import fetch_states_history

load_dotenv(override=True)

PREDICTION_INTERVAL_MINUTES = int(os.getenv('PREDICTION_INTERVAL_MINUTES', '15'))

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
    df_solar['period_start'] = pd.to_datetime(df_solar['period_start'], utc=True)
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

    sauna_temp = get_ha_state('sensor.sauna_temperature_2')
    try:
        s_temp_val = float(sauna_temp.get('state')) if sauna_temp and sauna_temp.get('state') not in ['unknown', 'unavailable'] else 20.0
    except (ValueError, TypeError, AttributeError):
        s_temp_val = 20.0
    
    # Is sauna currently heating up? (Rising temperature check)
    is_sauna_detected = False
    if s_temp_val > 30.0:
        # Fetch last 30 mins to see if it's rising
        s_history_dict = fetch_states_history('sensor.sauna_temperature_2', hours=0.5)
        s_history = s_history_dict.get('sensor.sauna_temperature_2', pd.DataFrame())
        if not s_history.empty and len(s_history) > 1:
            # Check if current is higher than 30 mins ago
            first_val = s_history.iloc[0]['state']
            if s_temp_val > (first_val + 1.0):
                is_sauna_detected = True
                print(f"🔥 Sauna heating detected: {first_val:.1f}C -> {s_temp_val:.1f}C")
            else:
                print(f"♨️ Sauna is warm but not rising: {s_temp_val:.1f}C (steady or cooling)")

    # Heuristic for tomorrow's sauna: Was it warm yesterday evening?
    # Check history from yesterday 18:00 to 22:00
    now_local = datetime.now()
    yesterday_evening_start = (now_local - timedelta(days=1)).replace(hour=18, minute=0, second=0)
    # We need to fetch enough history to cover that period
    hours_to_fetch = (now_local - yesterday_evening_start).total_seconds() / 3600.0 + 4.0
    s_yesterday_dict = fetch_states_history('sensor.sauna_temperature_2', hours=hours_to_fetch)
    s_yesterday = s_yesterday_dict.get('sensor.sauna_temperature_2', pd.DataFrame())
    
    was_warm_yesterday = False
    if not s_yesterday.empty:
        # Filter to 18-22 window
        s_y_window = s_yesterday[
            (s_yesterday['timestamp'].dt.hour >= 18) & 
            (s_yesterday['timestamp'].dt.hour <= 22) &
            (s_yesterday['timestamp'].dt.day == yesterday_evening_start.day)
        ]
        if not s_y_window.empty and s_y_window['state'].max() > 50.0:
            was_warm_yesterday = True
            print("📅 Sauna was used yesterday evening.")

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

    # Determine prediction window: from next interval until end of available solar horizon.
    now = datetime.now().astimezone() # Use aware datetime
    interval = max(PREDICTION_INTERVAL_MINUTES, 1)
    minutes_to_next = interval - (now.minute % interval)
    if minutes_to_next == interval and now.second == 0 and now.microsecond == 0:
        minutes_to_next = 0
    start_time = (now + timedelta(minutes=minutes_to_next)).replace(second=0, microsecond=0)
    
    # Find the latest timestamp in solar data (usually end of tomorrow)
    # If solar data is short, limit to what we have
    end_time = df_solar.index.max()
    
    print(f'Predicting from {start_time} to {end_time}')
    
    inference_data = []
    timestamps = []
    
    current_ts = start_time
    while current_ts <= end_time:
        # Get nearest solar estimate for this interval.
        try:
            idx = df_solar.index.get_indexer([current_ts], method='nearest')[0]
            solar_val = df_solar.iloc[idx]['pv_estimate']
        except Exception:
            solar_val = 0.0
            
        # Sauna Heuristic Projection
        # 1. Real-time detection (4-hour window from now if already hot)
        is_sauna_proj = 0
        if is_sauna_detected and current_ts < (now + timedelta(hours=4)):
            is_sauna_proj = 1
        
        # 2. Schedule Heuristic (Sept-May)
        month = current_ts.month
        if month in [9,10,11,12,1,2,3,4,5]:
            is_weekend = current_ts.weekday() >= 5
            hour = current_ts.hour
            # Weekend evenings: 18-22
            if is_weekend and 18 <= hour <= 21:
                is_sauna_proj = 1
            # Weekday heuristic: If it WASN'T warm yesterday, assume it might be today
            elif not is_weekend and 18 <= hour <= 21 and not was_warm_yesterday:
                is_sauna_proj = 1

        row = {
            'outside_temp': temp_val,
            'solar_forecast': solar_val,
            'accumulator_temp': acc_val,
            'acc_roc': 0, # Placeholder for future prediction
            'is_fireplace_lag1': 0,
            'ev_soc': soc_val,
            'ev_position': 1, # Assume home
            'hour': current_ts.hour,
            'minute': current_ts.minute,
            'quarter_hour': current_ts.minute // 15,
            'day_of_week': current_ts.weekday(),
            'month': current_ts.month,
            'is_sauna_active': is_sauna_proj
        }
        inference_data.append(row)
        timestamps.append(current_ts.isoformat())
        
        current_ts += timedelta(minutes=interval)
        
    X_inference = pd.DataFrame(inference_data)[features]
    predictions = model.predict(X_inference)
    
    # Combine predictions with timestamps
    # Model output is average Power (kW) for the interval
    results = []
    generated_at = datetime.now().astimezone().isoformat()
    
    # Fetch market prices to identify fallbacks
    print('Fetching market prices for fallback detection...')
    prediction_timestamps_dt = [datetime.fromisoformat(ts) for ts in timestamps]
    market_prices, is_fallback_price, _ = fetch_market_prices(prediction_timestamps_dt, PREDICTION_INTERVAL_MINUTES)
    if is_fallback_price is None:
        is_fallback_price = [0] * len(predictions)

    for i, p in enumerate(predictions):
        p_kw = float(p)
        results.append({
            'timestamp': timestamps[i],
            'predicted_baseload': p_kw,     # house usage without GSHP (kW)
            'predicted_usage': p_kw,        # backward compatibility
            'solar_forecast': float(inference_data[i]['solar_forecast']),
            'outside_temp': float(inference_data[i]['outside_temp']),
            'is_sauna_active': int(inference_data[i].get('is_sauna_active', 0)),
            'is_fallback_price': int(is_fallback_price[i])
        })
        
    print(f'\nGenerated {len(results)} predictions at {interval}-minute resolution.')
    
    with open('future_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\n✅ Predictions saved to future_predictions.json')

    # Archive predictions for feedback loop using SQLite
    db_file = 'hepo.db'
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_usage_kw REAL,
                solar_forecast_kw REAL,
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')
        
        # Schema migration: add is_fallback_price if missing
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        if 'is_fallback_price' not in columns:
            print("Adding is_fallback_price column to predictions table...")
            cur.execute("ALTER TABLE predictions ADD COLUMN is_fallback_price INTEGER DEFAULT 0")

        # Insert predictions.
        data_to_insert = [
            (res['timestamp'], generated_at, res['predicted_usage'], res['solar_forecast'], res['is_fallback_price'])
            for res in results
        ]
        cur.executemany('''
            INSERT OR REPLACE INTO predictions 
            (target_timestamp, generated_at, predicted_usage_kw, solar_forecast_kw, is_fallback_price)
            VALUES (?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f'✅ Archived {len(results)} points to {db_file} (SQLite)')
    except Exception as e:
        print(f'⚠️ Error archiving to SQLite: {e}')

if __name__ == '__main__':
    predict()
