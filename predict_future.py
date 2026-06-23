from __future__ import annotations
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from dotenv import load_dotenv
from utils.ha_utils import get_ha_state, call_ha_service
from utils.price_utils import fetch_market_prices
from utils.db_utils import fetch_states_history
from utils.git_utils import get_model_version
from utils.sqlite_utils import get_db_connection, get_db_path

load_dotenv(override=True)

PREDICTION_INTERVAL_MINUTES: int = int(os.getenv('PREDICTION_INTERVAL_MINUTES', '15'))


def compute_baseload_at_lag(anchor_data: dict[str, pd.DataFrame], hours_back: float) -> float:
    """
    Compute baseload power at a given historical lag using combined DataFrame
    + forward-fill alignment across all sensors.

    The old approach used ``get_nearest`` per sensor independently, which could
    pick values from *different* timestamps for different sensors (time-skew).
    This became particularly harmful when battery power changed rapidly: the
    grid meter value and the battery sensor value could be tens of seconds
    apart, producing an incorrect instantaneous baseload estimate.

    The fix aligns all five sensor streams to a common timeline by
    forward-filling gaps, then picks values from the same row.  This ensures
    the five quantities (grid, solar, GSHP, EV, battery) represent
    approximately the same moment.

    Returns
    -------
    float
        Baseload power in kW, clipped to ``[0.0, inf)``.  Returns ``1.0``
        as a safe fallback when no sensor data is available.
    """
    target_ts = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    try:
        sensors_def = [
            ('sensor.sahkokauppa_nyt', 'total', 1.0),
            ('sensor.solarh_63038_real_power_kw', 'solar', 1.0),
            ('sensor.mlp_teho', 'gshp', 1 / 1000.0),
            ('sensor.tasmota_energy_power_3', 'leaf', 1 / 1000.0),
            ('sensor.be_stat_batt_power', 'battery', 1 / 1000.0),
        ]

        series_list = []
        for entity_id, name, scale in sensors_def:
            df = anchor_data.get(entity_id)
            if df is None or df.empty:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('timestamp')
            s = pd.to_numeric(df['state'], errors='coerce').dropna()
            s = s * scale
            s.name = name
            series_list.append(s)

        if not series_list:
            return 1.0

        combined = pd.concat(series_list, axis=1, sort=True)
        combined = combined.sort_index()

        all_ts = combined.index.union(pd.DatetimeIndex([target_ts]))
        combined = combined.reindex(all_ts).sort_index().ffill()

        row = combined.loc[target_ts]

        total = row.get('total', 0.0)
        total = float(total) if pd.notna(total) else 0.0
        solar = row.get('solar', 0.0)
        solar = float(solar) if pd.notna(solar) else 0.0
        gshp = row.get('gshp', 0.0)
        gshp = float(gshp) if pd.notna(gshp) else 0.0
        leaf = row.get('leaf', 0.0)
        leaf = float(leaf) if pd.notna(leaf) else 0.0
        battery = row.get('battery', 0.0)
        battery = float(battery) if pd.notna(battery) else 0.0

        if 'battery' not in combined.columns:
            print(f"⚠️ Battery sensor data unavailable for baseload lag at {hours_back}h — baseload may include battery charging")

        return max(0.0, total + solar - gshp - leaf - battery)
    except Exception as e:
        print(f"⚠️ Error calculating anchor at lag {hours_back}h: {e}")
        return 1.0


def generate_inference_data(
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
    df_solar: pd.DataFrame,
    df_weather: pd.DataFrame,
    current_states: dict[str, Any],
    sauna_states: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
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
    p_temp_val = current_states.get('p_temp_val', np.nan)
    leaf_lag_val = current_states.get('leaf_lag_val', 0.0)
    leaf_energy_val = current_states.get('leaf_energy_val', 0.0)
    lag1h_val = current_states.get('lag1h_val', 1.0)
    lag24h_val = current_states.get('lag24h_val', 1.0)
    
    is_sauna_detected = sauna_states.get('is_sauna_detected', False)
    was_warm_yesterday = sauna_states.get('was_warm_yesterday', False)
    now = sauna_states.get('now', datetime.now().astimezone())

    while current_ts <= end_time:
        # Get nearest solar estimate
        try:
            idx = df_solar.index.get_indexer([current_ts], method='nearest')[0]  # type: ignore[arg-type]
            solar_val = df_solar.iloc[idx]['pv_estimate']
        except Exception:
            solar_val = 0.0
            
        # Get nearest weather forecast estimate
        forecast_temp = temp_val
        forecast_wind = wind_val
        try:
            if not df_weather.empty:
                w_idx = df_weather.index.get_indexer([current_ts], method='nearest')[0]  # type: ignore[arg-type]
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
            'gshp_pump_temp': p_temp_val,
            'is_gshp_pump_running': 1 if not np.isnan(p_temp_val) else 0,
            'leaf_power_lag_1h': leaf_lag_val,
            'leaf_energy_24h': leaf_energy_val,
            'acc_roc': 0,
            'is_fireplace_lag1': 0,
            'ev_position': ev_pos_proj,
            'baseload_lag_1h': lag1h_val,
            'baseload_lag_24h': lag24h_val,
            'is_extended_complex': 1,
            'hour': current_ts.hour,
            'minute': current_ts.minute,
            'quarter_hour': current_ts.minute // 15,
            'day_of_week': current_ts.weekday(),
            'month': current_ts.month,
            'hour_sin': np.sin(2 * np.pi * current_ts.hour / 24),
            'hour_cos': np.cos(2 * np.pi * current_ts.hour / 24),
            'day_sin': np.sin(2 * np.pi * current_ts.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * current_ts.weekday() / 7),
            'month_sin': np.sin(2 * np.pi * (current_ts.month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (current_ts.month - 1) / 12),
            'is_sauna_active': is_sauna_proj
        }
        inference_data.append(row)
        timestamps.append(current_ts.isoformat())
        current_ts += timedelta(minutes=interval_minutes)
        
    return inference_data, timestamps

def predict() -> None:
    print('Syncing with Home Assistant...')
    
    # fetch solar data
    solar_today_state = get_ha_state('sensor.solcast_pv_forecast_forecast_today')
    solar_tomorrow_state = get_ha_state('sensor.solcast_pv_forecast_forecast_tomorrow')
    
    # fetch weather forecast (hourly)
    print('Fetching weather forecast...')
    weather_forecast_data = call_ha_service('weather', 'get_forecasts', {'entity_id': 'weather.home', 'type': 'hourly'})
    df_weather = pd.DataFrame()
    if weather_forecast_data and 'weather.home' in weather_forecast_data:
        raw_forecast = weather_forecast_data['weather.home'].get('forecast', [])  # type: ignore[arg-type]
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
    temp_val = 5.0
    if current_temp is not None:
        try:
            temp_state = current_temp.get('state')
            if temp_state is not None and temp_state not in ['unknown', 'unavailable']:
                temp_val = float(temp_state)
        except (ValueError, TypeError, AttributeError):
            pass

    current_wind = get_ha_state('sensor.outside_wind_speed')
    wind_val = 0.0
    if current_wind is not None:
        try:
            wind_state = current_wind.get('state')
            if wind_state is not None and wind_state not in ['unknown', 'unavailable']:
                wind_val = float(wind_state)
        except (ValueError, TypeError, AttributeError):
            pass

    # Anchors: Fetch historical baseload for lags
    # Fetch last 25 hours to cover both 1h and 24h lags
    print('Fetching historical data for anchors...')
    anchor_entities = [
        'sensor.sahkokauppa_nyt', 
        'sensor.solarh_63038_real_power_kw', 
        'sensor.mlp_teho',
        'sensor.tasmota_energy_power_3',
        'sensor.be_stat_batt_power'
    ]
    anchor_data = fetch_states_history(anchor_entities, hours=25)
    
    def get_baseload_at_lag(hours_back):
        return compute_baseload_at_lag(anchor_data, hours_back)

    def get_leaf_features():
        # Get Leaf power 1h ago and energy sum for last 24h
        try:
            leaf_df = anchor_data.get('sensor.tasmota_energy_power_3')
            if leaf_df is None or leaf_df.empty:
                return 0.0, 0.0

            # 1h lag
            target_ts_1h = datetime.now(timezone.utc) - timedelta(hours=1)
            # fetch_states_history now returns DataFrames with timestamp as index
            temp_df = leaf_df if isinstance(leaf_df.index, pd.DatetimeIndex) else leaf_df.set_index('timestamp')
            idx = temp_df.index.get_indexer([target_ts_1h], method='nearest')[0]  # type: ignore[arg-type]
            leaf_lag_1h = float(temp_df.iloc[idx]['state']) if idx != -1 else 0.0

            # 24h energy proxy (kWh)
            # Power is in Watts. Average power * 24 hours / 1000 = kWh
            leaf_energy_24h = (temp_df['state'].astype(float).mean() * 24.0) / 1000.0

            return leaf_lag_1h, leaf_energy_24h
        except Exception as e:
            print(f"⚠️ Error calculating Leaf features: {e}")
            return 0.0, 0.0

    lag1h_val = get_baseload_at_lag(1)
    lag24h_val = get_baseload_at_lag(24)
    leaf_lag1h, leaf_energy_24h = get_leaf_features()

    print(f"⚓ Anchors - 1h: {lag1h_val:.2f}kW, 24h: {lag24h_val:.2f}kW, Leaf 24h: {leaf_energy_24h:.1f}kWh")

        
    acc_temp = get_ha_state('sensor.mlp_varaajan_lampotila')
    acc_val = 45.0
    if acc_temp is not None:
        try:
            acc_state = acc_temp.get('state')
            if acc_state is not None and acc_state not in ['unknown', 'unavailable']:
                acc_val = float(acc_state)
        except (ValueError, TypeError, AttributeError):
            pass

    # GSHP Pump Temperature logic
    gshp_pump_temp_state = get_ha_state('sensor.mlp_pumpun_lampotla')
    gshp_power_state = get_ha_state('sensor.mlp_teho')
    p_temp_val = np.nan
    p_power_val = 0.0
    if gshp_pump_temp_state is not None and gshp_power_state is not None:
        try:
            p_temp_raw = gshp_pump_temp_state.get('state')
            p_power_raw = gshp_power_state.get('state')
            if p_temp_raw is not None and p_temp_raw not in ['unknown', 'unavailable']:
                p_temp_val = float(p_temp_raw)
            if p_power_raw is not None and p_power_raw not in ['unknown', 'unavailable']:
                p_power_val = float(p_power_raw)
            
            # User specified: if pump power < 100W, the sensor is invalid/weird.
            if p_power_val < 100:
                p_temp_val = np.nan
        except (ValueError, TypeError, AttributeError):
            p_temp_val = np.nan
            p_power_val = 0.0

    sauna_temp = get_ha_state('sensor.sauna_temperature_2')
    s_temp_val = 20.0
    if sauna_temp is not None:
        try:
            sauna_state = sauna_temp.get('state')
            if sauna_state is not None and sauna_state not in ['unknown', 'unavailable']:
                s_temp_val = float(sauna_state)
        except (ValueError, TypeError, AttributeError):
            pass
    
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
        # timestamp is now the index, so we use the index directly
        timestamp_series = s_yesterday.index if isinstance(s_yesterday.index, pd.DatetimeIndex) else s_yesterday['timestamp']
        s_y_window = s_yesterday[
            (timestamp_series.hour >= 18) & 
            (timestamp_series.hour <= 22) &
            (timestamp_series.day == yesterday_evening_start.day)
        ]
        if not s_y_window.empty and s_y_window['state'].max() > 50.0:
            was_warm_yesterday = True
            print("📅 Sauna was used yesterday evening.")

    ev_pos = get_ha_state('device_tracker.xpz_491_position')
    pos_val = 1 if ev_pos and ev_pos.get('state') == 'home' else 0
    print(f"🚗 EV Position: {'Home' if pos_val else 'Away'}")

    # Load model and features
    model = xgb.XGBRegressor()
    try:
        model.load_model('state/energy_model.json')
        with open('state/model_features.json', 'r') as f:
            features = json.load(f)
    except FileNotFoundError:
        print('Error: Model files not found. Run train_model.py first.')
        return

    # Determine prediction window: from current interval until end of available solar horizon.
    now = datetime.now().astimezone() # Use aware datetime, local tz
    interval = max(PREDICTION_INTERVAL_MINUTES, 1)
    # Round down to the current interval boundary so the plan includes the current slot.
    # This ensures run_often/push_to_ha always find the correct current entry.
    start_time = now.replace(
        minute=(now.minute // interval) * interval,
        second=0,
        microsecond=0
    )
    
    # Find the latest timestamp in solar data (usually end of tomorrow)
    # If solar data is short, limit to what we have
    end_time = df_solar.index.max()
    
    # Ensure end_time matches the same timezone as start_time
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize('UTC').astimezone(start_time.tzinfo)
    else:
        end_time = end_time.astimezone(start_time.tzinfo)
    
    print(f'Predicting from {start_time} to {end_time}')
    
    current_states = {
        'temp_val': temp_val, 
        'wind_val': wind_val,
        'acc_val': acc_val, 
        'p_temp_val': p_temp_val,
        'ev_pos_val': pos_val,
        'leaf_lag_val': leaf_lag1h,
        'leaf_energy_val': leaf_energy_24h,
        'lag1h_val': lag1h_val,
        'lag24h_val': lag24h_val
    }
    sauna_states = {'is_sauna_detected': is_sauna_detected, 'was_warm_yesterday': was_warm_yesterday, 'now': now}
    
    inference_data, timestamps = generate_inference_data(
        start_time, end_time, interval, df_solar, df_weather, current_states, sauna_states
    )
        
    X_inference = pd.DataFrame(inference_data)
    missing = set(features) - set(X_inference.columns)
    if missing:
        defaults = {'ev_soc': 80.0}
        print(f"⚠ Model expects features not in inference data: {missing}. Filling with defaults.")
        for f in missing:
            X_inference[f] = defaults.get(f, 0)
    X_inference = X_inference[features]
    predictions = model.predict(X_inference)
    
    # Combine predictions with timestamps
    # Model output is average Power (kW) for the interval
    results = []
    generated_at = datetime.now().astimezone().isoformat()
    
    # Fetch market prices to identify fallbacks
    print('Fetching market prices for fallback detection...')
    market_prices, is_fallback_price, _, _, _, _ = fetch_market_prices(timestamps, PREDICTION_INTERVAL_MINUTES)
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
            'ev_position': int(inference_data[i].get('ev_position', 1)),
            'is_sauna_active': int(inference_data[i].get('is_sauna_active', 0)),
            'is_fallback_price': int(is_fallback_price[i])
        })
        
    print(f'\nGenerated {len(results)} predictions at {interval}-minute resolution.')
    
    # Support environment variable override for testing
    predictions_file = os.getenv('TEST_PREDICTIONS_FILE', 'state/future_predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ Predictions saved to {predictions_file}')

    # Archive predictions for feedback loop using SQLite
    git_version = get_model_version()
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_usage_kw REAL,
                solar_forecast_kw REAL,
                version TEXT,
                battery_action TEXT,
                battery_power_kw REAL,
                battery_soc_pct REAL,
                import_price REAL,
                export_price REAL,
                grid_import_kwh REAL,
                grid_export_kwh REAL,
                charge_from_solar_kwh REAL,
                charge_from_grid_kwh REAL,
                discharge_to_load_kwh REAL,
                discharge_to_export_kwh REAL,
                planned_gshp_kw REAL,
                gshp_intent TEXT,
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')
        
        # Schema migration: add missing columns
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        
        new_cols = {
            'is_fallback_price': 'INTEGER DEFAULT 0',
            'version': "TEXT DEFAULT 'unknown'",
            'battery_action': 'TEXT',
            'battery_power_kw': 'REAL',
            'battery_soc_pct': 'REAL',
            'import_price': 'REAL',
            'export_price': 'REAL',
            'grid_import_kwh': 'REAL',
            'grid_export_kwh': 'REAL',
            'charge_from_solar_kwh': 'REAL',
            'charge_from_grid_kwh': 'REAL',
            'discharge_to_load_kwh': 'REAL',
            'discharge_to_export_kwh': 'REAL',
            'planned_gshp_kw': 'REAL',
            'gshp_intent': 'TEXT'
        }
        
        for col, col_type in new_cols.items():
            if col not in columns:
                print(f"Adding {col} column to predictions table...")
                cur.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")

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
        print(f'✅ Archived {len(results)} points to {get_db_path()} (SQLite)')
    except Exception as e:
        print(f'⚠️ Error archiving to SQLite: {e}')

if __name__ == '__main__':
    predict()
