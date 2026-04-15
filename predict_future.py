import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from utils.ha_utils import get_ha_state, call_ha_service
from utils.price_utils import fetch_market_prices
from utils.db_utils import fetch_states_history
from utils.git_utils import get_git_version

load_dotenv(override=True)

PREDICTION_INTERVAL_MINUTES = int(os.getenv('PREDICTION_INTERVAL_MINUTES', '15'))

def generate_inference_data(start_time, end_time, interval_minutes, df_solar, df_weather, current_states, sauna_states):
    """
    Generate inference data rows for the model.
    Extracted for testability.
    """
    inference_data = []
    timestamps = []
    current_ts = start_time
    
    temp_val = current_states.get('temp_val', 5.0)
    wind_val = current_states.get('wind_val', 0.0)
    acc_val = current_states.get('acc_val', 45.0)
    soc_val = current_states.get('soc_val', 80.0)
    lag1h_val = current_states.get('lag1h_val', 1.0)
    lag24h_val = current_states.get('lag24h_val', 1.0)
    
    is_sauna_detected = sauna_states.get('is_sauna_detected', False)
    was_warm_yesterday = sauna_states.get('was_warm_yesterday', False)
    now = sauna_states.get('now', datetime.now().astimezone())

    while current_ts <= end_time:
        # Get nearest solar estimate
        try:
            idx = df_solar.index.get_indexer([current_ts], method='nearest')[0]
            solar_val = df_solar.iloc[idx]['pv_estimate']
        except Exception:
            solar_val = 0.0
            
        # Get nearest weather forecast estimate
        forecast_temp = temp_val
        forecast_wind = wind_val
        try:
            if not df_weather.empty:
                w_idx = df_weather.index.get_indexer([current_ts], method='nearest')[0]
                forecast_dt = df_weather.index[w_idx]
                
                # Force both to UTC for comparison to avoid offset-naive vs offset-aware issues
                ts_utc = current_ts.astimezone(timezone.utc) if current_ts.tzinfo else current_ts.replace(tzinfo=timezone.utc)
                f_utc = forecast_dt.astimezone(timezone.utc) if forecast_dt.tzinfo else forecast_dt.replace(tzinfo=timezone.utc)
                
                if abs((f_utc - ts_utc).total_seconds()) < 7200:
                    forecast_temp = float(df_weather.iloc[w_idx].get('temperature', temp_val))
                    forecast_wind = float(df_weather.iloc[w_idx].get('wind_speed', wind_val))
        except Exception:
            pass

        # Sauna Heuristic Projection
        is_sauna_proj = 0
        is_weekend = current_ts.weekday() >= 5
        hour = current_ts.hour

        if is_sauna_detected and current_ts < (now + timedelta(hours=4)):
            is_sauna_proj = 1
        
        month = current_ts.month
        if month in [9,10,11,12,1,2,3,4,5]:
            if is_weekend and 18 <= hour <= 21:
                is_sauna_proj = 1
            elif not is_weekend and 18 <= hour <= 21 and not was_warm_yesterday:
                is_sauna_proj = 1

        # EV Position Heuristic Projection: XPZ is rarely home during workdays (08:00-17:00)
        ev_pos_proj = 1
        if not is_weekend and 8 <= hour < 17:
            ev_pos_proj = 0
        
        # Override with current state for the first 2 hours
        if current_ts < (now + timedelta(hours=2)):
            ev_pos_proj = current_states.get('ev_pos_val', ev_pos_proj)

        row = {
            'outside_temp': forecast_temp,
            'wind_speed': forecast_wind,
            'solar_forecast': solar_val,
            'accumulator_temp': acc_val,
            'acc_roc': 0,
            'is_fireplace_lag1': 0,
            'ev_soc': soc_val,
            'ev_position': ev_pos_proj,
            'baseload_lag_1h': lag1h_val,
            'baseload_lag_24h': lag24h_val,
            'is_extended_complex': 1,
            'hour': current_ts.hour,
            'minute': current_ts.minute,
            'quarter_hour': current_ts.minute // 15,
            'day_of_week': current_ts.weekday(),
            'month': current_ts.month,
            'is_sauna_active': is_sauna_proj
        }
        inference_data.append(row)
        timestamps.append(current_ts.isoformat())
        current_ts += timedelta(minutes=interval_minutes)
        
    return inference_data, timestamps

def predict():
    print('Syncing with Home Assistant...')
    
    # fetch solar data
    solar_today_state = get_ha_state('sensor.solcast_pv_forecast_forecast_today')
    solar_tomorrow_state = get_ha_state('sensor.solcast_pv_forecast_forecast_tomorrow')
    
    # fetch weather forecast (hourly)
    print('Fetching weather forecast...')
    weather_forecast_data = call_ha_service('weather', 'get_forecasts', {'entity_id': 'weather.home', 'type': 'hourly'})
    df_weather = pd.DataFrame()
    if weather_forecast_data and 'weather.home' in weather_forecast_data:
        raw_forecast = weather_forecast_data['weather.home'].get('forecast', [])
        if raw_forecast:
            df_weather = pd.DataFrame(raw_forecast)
            df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], utc=True)
            df_weather = df_weather.sort_values('datetime').set_index('datetime')
            print(f"✅ Fetched {len(df_weather)} hourly weather forecast points.")
    
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

    current_wind = get_ha_state('sensor.outside_wind_speed')
    try:
        wind_val = float(current_wind.get('state')) if current_wind and current_wind.get('state') not in ['unknown', 'unavailable'] else 0.0
    except (ValueError, TypeError, AttributeError):
        wind_val = 0.0

    # Anchors: Fetch historical baseload for lags
    # Fetch last 25 hours to cover both 1h and 24h lags
    print('Fetching historical data for anchors...')
    anchor_entities = [
        'sensor.sahkokauppa_nyt', 
        'sensor.solarh_63038_real_power_kw', 
        'sensor.mlp_teho',
        'sensor.tasmota_energy_power_3'
    ]
    anchor_data = fetch_states_history(anchor_entities, hours=25)
    
    def get_baseload_at_lag(hours_back):
        target_ts = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        try:
            total_df = anchor_data.get('sensor.sahkokauppa_nyt')
            solar_df = anchor_data.get('sensor.solarh_63038_real_power_kw')
            gshp_df = anchor_data.get('sensor.mlp_teho')
            leaf_df = anchor_data.get('sensor.tasmota_energy_power_3')
            
            # Find nearest values
            def get_nearest(df, ts):
                if df is None or df.empty: return 0.0
                # Ensure we use a DatetimeIndex for get_indexer
                if not isinstance(df.index, pd.DatetimeIndex):
                    temp_df = df.set_index('timestamp')
                else:
                    temp_df = df
                
                idx = temp_df.index.get_indexer([ts], method='nearest')[0]
                if idx == -1: return 0.0
                return float(temp_df.iloc[idx]['state'])

            total = get_nearest(total_df, target_ts)
            solar = get_nearest(solar_df, target_ts)
            gshp = get_nearest(gshp_df, target_ts) / 1000.0 # W to kW
            leaf = get_nearest(leaf_df, target_ts) / 1000.0 # W to kW
            
            return max(0.0, total + solar - gshp - leaf)
        except Exception as e:
            print(f"⚠️ Error calculating anchor at lag {hours_back}h: {e}")
            return 1.0 # Fallback

    lag1h_val = get_baseload_at_lag(1)
    lag24h_val = get_baseload_at_lag(24)
    print(f"⚓ Anchors - 1h: {lag1h_val:.2f}kW, 24h: {lag24h_val:.2f}kW")
        
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

    ev_pos = get_ha_state('device_tracker.xpz_491_position')
    pos_val = 1 if ev_pos and ev_pos.get('state') == 'home' else 0
    print(f"🚗 EV Status - SOC: {soc_val}%, Position: {'Home' if pos_val else 'Away'}")

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
    
    current_states = {
        'temp_val': temp_val, 
        'wind_val': wind_val,
        'acc_val': acc_val, 
        'soc_val': soc_val,
        'ev_pos_val': pos_val,
        'lag1h_val': lag1h_val,
        'lag24h_val': lag24h_val
    }
    sauna_states = {'is_sauna_detected': is_sauna_detected, 'was_warm_yesterday': was_warm_yesterday, 'now': now}
    
    inference_data, timestamps = generate_inference_data(
        start_time, end_time, interval, df_solar, df_weather, current_states, sauna_states
    )
        
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
    git_version = get_git_version()
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_usage_kw REAL,
                solar_forecast_kw REAL,
                version TEXT,
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')
        
        # Schema migration: add is_fallback_price if missing
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        if 'is_fallback_price' not in columns:
            print("Adding is_fallback_price column to predictions table...")
            cur.execute("ALTER TABLE predictions ADD COLUMN is_fallback_price INTEGER DEFAULT 0")
        if 'version' not in columns:
            print("Adding version column to predictions table...")
            cur.execute("ALTER TABLE predictions ADD COLUMN version TEXT DEFAULT 'unknown'")

        # Insert predictions.
        data_to_insert = [
            (res['timestamp'], generated_at, res['predicted_usage'], res['solar_forecast'], res['is_fallback_price'], git_version)
            for res in results
        ]
        cur.executemany('''
            INSERT OR REPLACE INTO predictions 
            (target_timestamp, generated_at, predicted_usage_kw, solar_forecast_kw, is_fallback_price, version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f'✅ Archived {len(results)} points to {db_file} (SQLite)')
    except Exception as e:
        print(f'⚠️ Error archiving to SQLite: {e}')

if __name__ == '__main__':
    predict()
