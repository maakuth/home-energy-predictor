from __future__ import annotations
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import sqlite3
import argparse
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from dotenv import load_dotenv
from utils.ha_utils import get_ha_state, call_ha_service
from utils.price_utils import fetch_market_prices
from utils.db_utils import fetch_states_history
from utils.git_utils import get_model_version
from utils.sqlite_utils import get_db_connection, get_db_path
from utils.archive_history import ArchiveHistory, add_history_arguments, history_from_arguments
from utils.archive_quality import METERS, apply_training_policy
from utils.baseload_model import validate_baseload_model

load_dotenv(override=True)

PREDICTION_INTERVAL_MINUTES: int = int(os.getenv('PREDICTION_INTERVAL_MINUTES', '15'))


def _safe_float(value: Any, fallback: float) -> float:
    """Convert a value to float, falling back to ``fallback`` on NaN/invalid."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return f if np.isfinite(f) else fallback


def compute_baseload_at_lag(
    anchor_data: dict[str, pd.DataFrame],
    hours_back: float,
    solar_forecast_df: pd.DataFrame | None = None,
) -> float:
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

    When ``SOLAR_FALLBACK_TO_FORECAST`` is enabled and the real-time solar
    sensor has no data at the target timestamp, the Solcast forecast
    (``solar_forecast_df``) is used as a substitute.

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
            (os.getenv('SOLAR_PRODUCTION_ENTITY', 'sensor.solarh_63038_real_power_kw'), 'solar', 1.0),
            (os.getenv('GSHP_POWER_ENTITY', 'sensor.mlp_teho'), 'gshp', 1 / 1000.0),
            (os.getenv('RESISTIVE_HEATER_ENTITY', 'switch.mlp_vastus_output_0'), 'resistive', 1.0),
            (os.getenv('BULK_HEATER_ENTITY', 'switch.mlp_vastus_output_1'), 'bulk_heater', 1.0),
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
            if name in {'resistive', 'bulk_heater'}:
                heater_kw = float(os.getenv(
                    'RESISTIVE_HEATER_POWER_KW' if name == 'resistive' else 'BULK_HEATER_POWER_KW', '6.0',
                ))
                s = df['state'].astype(str).str.lower().eq('on').astype(float) * heater_kw
            else:
                s = pd.to_numeric(df['state'], errors='coerce').dropna() * scale
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

        if solar == 0.0 and os.getenv('SOLAR_FALLBACK_TO_FORECAST', 'true').strip().lower() in {'1', 'true', 'yes', 'on'} and solar_forecast_df is not None and not solar_forecast_df.empty:
            try:
                f_idx = solar_forecast_df.index.get_indexer([target_ts], method='nearest')[0]
                if f_idx >= 0:
                    from_forecast = float(solar_forecast_df.iloc[f_idx].get('pv_estimate', 0.0))
                    if from_forecast > 0:
                        solar = from_forecast
            except Exception:
                pass

        # sensor.mlp_teho is the shared meter for GSHP and resistive heating,
        # so its reading already includes the switched 6 kW element.
        heating = row.get('gshp', 0.0)
        heating = float(heating) if pd.notna(heating) else 0.0
        resistive = row.get('resistive', 0.0)
        resistive = float(resistive) if pd.notna(resistive) else 0.0
        bulk_heater = row.get('bulk_heater', 0.0)
        bulk_heater = float(bulk_heater) if pd.notna(bulk_heater) else 0.0
        heating = max(heating, resistive + bulk_heater)
        leaf = row.get('leaf', 0.0)
        leaf = float(leaf) if pd.notna(leaf) else 0.0
        battery = row.get('battery', 0.0)
        battery = float(battery) if pd.notna(battery) else 0.0

        if 'battery' not in combined.columns:
            print(f"⚠️ Battery sensor data unavailable for baseload lag at {hours_back}h — baseload may include battery charging")

        return max(0.0, total + solar - heating - leaf - battery)
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
    Model calendar features use UTC, matching process_frame training inputs.
    Heuristics, interval stepping and output timestamps retain the caller's clock
    (Europe/Helsinki in predict). Naive model timestamps are interpreted as UTC,
    as in process_frame, without changing existing bare-datetime callers.
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
    now = sauna_states.get('now')
    if now is None:
        now = datetime.now().astimezone()

    while current_ts <= end_time:
        # Get nearest solar estimate
        try:
            idx = df_solar.index.get_indexer([current_ts], method='nearest')[0]  # type: ignore[arg-type]
            nearest_solar = df_solar.iloc[idx]
            solar_val = float(nearest_solar['pv_estimate'])
        except Exception:
            nearest_solar = None
            solar_val = 0.0

        # Worst/best case estimates (estimate10/estimate90) for monitoring.
        # Fall back to the nominal (pv_estimate) value when not available.
        solar_p10_val = solar_val
        solar_p90_val = solar_val
        if nearest_solar is not None:
            solar_p10_val = _safe_float(nearest_solar.get('pv_estimate10'), solar_val)
            solar_p90_val = _safe_float(nearest_solar.get('pv_estimate90'), solar_val)
            
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

        # Separate the model clock from local sauna/EV projection above. Never
        # call astimezone on a naive datetime: that would depend on the host TZ.
        model_ts = (current_ts.astimezone(timezone.utc) if current_ts.tzinfo is not None
                    else current_ts.replace(tzinfo=timezone.utc))
        row = {
            'outside_temp': forecast_temp,
            'wind_speed': forecast_wind,
            'solar_forecast': solar_val,
            'solar_forecast_p10': solar_p10_val,
            'solar_forecast_p90': solar_p90_val,
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
            'hour': model_ts.hour,
            'minute': model_ts.minute,
            'quarter_hour': model_ts.minute // 15,
            'day_of_week': model_ts.weekday(),
            'month': model_ts.month,
            'hour_sin': np.sin(2 * np.pi * model_ts.hour / 24),
            'hour_cos': np.cos(2 * np.pi * model_ts.hour / 24),
            'day_sin': np.sin(2 * np.pi * model_ts.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * model_ts.weekday() / 7),
            'month_sin': np.sin(2 * np.pi * (model_ts.month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (model_ts.month - 1) / 12),
            'is_sauna_active': is_sauna_proj
        }
        inference_data.append(row)
        timestamps.append(current_ts.isoformat())
        current_ts += timedelta(minutes=interval_minutes)
        
    return inference_data, timestamps

def archive_inference_anchors(history: ArchiveHistory) -> tuple[float, float, float, float]:
    """Exact cutoff-relative meter bins; no smoothing, relay subtraction or fill.

    The synthetic cutoff row is only a lag-query position, never a measurement.
    Applying the training contract there gives the same complete prior-day Leaf
    energy and exact lags used during fitting without reading future raw data.
    """
    raw = history.raw(history.cutoff - 86400, history.cutoff, measurements=list(METERS))
    cutoff = pd.to_datetime(history.cutoff, unit='s', utc=True)
    expected = pd.date_range(cutoff - pd.Timedelta(days=1), cutoff, freq='min', inclusive='left')
    if raw.index.has_duplicates:
        raise ValueError('Duplicate archive anchor timestamps')
    raw = raw.reindex(expected.append(pd.DatetimeIndex([cutoff])))
    targets = apply_training_policy(raw, pd.DataFrame(index=raw.index), history.legacy_policy)
    row = targets.loc[cutoff]
    names = ['baseload_lag_1h', 'baseload_lag_24h', 'leaf_power_lag_1h', 'leaf_energy_24h']
    if not all(pd.notna(row[name]) and np.isfinite(row[name]) for name in names):
        raise ValueError('Archive inference anchors/Leaf prior-day coverage unavailable; refusing prediction')
    return (float(targets['baseload_lag_1h'].iloc[-1]),
            float(targets['baseload_lag_24h'].iloc[-1]),
            float(targets['leaf_power_lag_1h'].iloc[-1]),
            float(targets['leaf_energy_24h'].iloc[-1]))


def archive_sauna_history(history: ArchiveHistory, current_temp: float) -> tuple[bool, bool]:
    """Historical sauna evidence, with calendar windows in Europe/Helsinki.

    Missing eligible evidence means no heuristic activation (not measured cold).
    Archive I/O/configuration errors still propagate rather than falling back.
    """
    now = datetime.fromtimestamp(history.cutoff, ZoneInfo('Europe/Helsinki'))
    yesterday = now.date() - timedelta(days=1)
    start = datetime.combine(yesterday, datetime.min.time(), tzinfo=now.tzinfo)
    raw = history.raw(start.timestamp(), history.cutoff, measurements=['sauna_temp'])
    values = pd.to_numeric(raw['sauna_temp'], errors='coerce').where(history.eligible(raw, 'sauna_temp'))
    local_index = pd.DatetimeIndex(raw.index).tz_convert('Europe/Helsinki')
    evening = (local_index.date == yesterday) & (local_index.hour >= 18) & (local_index.hour <= 22)
    warm = bool(values.loc[evening].gt(50).any())
    # Compare with the exact 30-minute bin, not an arbitrarily stale sample.
    target = pd.to_datetime(history.cutoff - 1800, unit='s', utc=True)
    previous = values.get(target, np.nan)
    rising = bool(current_temp > 30 and pd.notna(previous) and current_temp > previous + 1)
    return rising, warm


def predict(*, history: ArchiveHistory | None = None) -> None:
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
        if history is not None:
            raise ValueError('No solar forecast data; refusing archive prediction')
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
        os.getenv('SOLAR_PRODUCTION_ENTITY', 'sensor.solarh_63038_real_power_kw'), 
        os.getenv('GSHP_POWER_ENTITY', 'sensor.mlp_teho'),
        os.getenv('RESISTIVE_HEATER_ENTITY', 'switch.mlp_vastus_output_0'),
        'sensor.tasmota_energy_power_3',
        'sensor.be_stat_batt_power'
    ]
    anchor_data = fetch_states_history(anchor_entities, hours=25) if history is None else {}
    
    def get_baseload_at_lag(hours_back):
        return compute_baseload_at_lag(anchor_data, hours_back, solar_forecast_df=df_solar)

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

    if history is None:
        lag1h_val = get_baseload_at_lag(1)
        lag24h_val = get_baseload_at_lag(24)
        leaf_lag1h, leaf_energy_24h = get_leaf_features()
    else:
        lag1h_val, lag24h_val, leaf_lag1h, leaf_energy_24h = archive_inference_anchors(history)

    print(f"⚓ Anchors - 1h: {lag1h_val:.2f}kW, 24h: {lag24h_val:.2f}kW, Leaf 24h: {leaf_energy_24h:.1f}kWh")

        
    accumulator_entity = os.getenv(
        'GSHP_ACCUMULATOR_TEMP_ENTITY', 'sensor.mlp_varaajan_lampotila',
    ).strip()
    supply_entity = os.getenv(
        'GSHP_SUPPLY_TEMP_ENTITY', 'sensor.gshp_supply_temp',
    ).strip()

    def _valid_temp_state(state):
        if state is None:
            return None
        try:
            value = state.get('state') if isinstance(state, dict) else state
            if value is None or value in ('unknown', 'unavailable'):
                return None
            return float(value)
        except (TypeError, ValueError, AttributeError):
            return None

    acc_val = None
    acc_temp = get_ha_state(accumulator_entity) if accumulator_entity else None
    value = _valid_temp_state(acc_temp)
    if value is not None:
        acc_val = value
    else:
        if accumulator_entity:
            print(f"⚠️ Physical accumulator temperature unavailable/invalid ({accumulator_entity}); trying supply temperature.")
        supply_state = get_ha_state(supply_entity) if supply_entity else None
        value = _valid_temp_state(supply_state)
        if value is not None:
            acc_val = value
            print(f"⚠️ Using supply temperature fallback for prediction accumulator: {value:.1f}°C")
        else:
            try:
                from utils.sg_ready import load_virtual_accumulator_state
                virtual_state = load_virtual_accumulator_state()
                if isinstance(virtual_state, dict):
                    v = virtual_state.get('temp')
                    if v is not None:
                        acc_val = float(v)
                        print(f"⚠️ Using persisted virtual accumulator fallback for prediction: {acc_val:.1f}°C")
            except Exception:
                acc_val = None
    if acc_val is None:
        acc_val = 45.0
        print(f"⚠️ Using configured static default for prediction accumulator: {acc_val:.1f}°C")

    # GSHP Pump Temperature logic
    gshp_pump_temp_state = get_ha_state('sensor.mlp_pumpun_lampotla')
    gshp_power_state = get_ha_state(os.getenv('GSHP_POWER_ENTITY', 'sensor.mlp_teho'))
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
    if history is None and s_temp_val > 30.0:
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
    now_local = (datetime.fromtimestamp(history.cutoff, ZoneInfo('Europe/Helsinki'))
                 if history is not None else datetime.now())
    yesterday_evening_start = (now_local - timedelta(days=1)).replace(hour=18, minute=0, second=0)
    # We need to fetch enough history to cover that period
    hours_to_fetch = (now_local - yesterday_evening_start).total_seconds() / 3600.0 + 4.0
    s_yesterday_dict = (fetch_states_history('sensor.sauna_temperature_2', hours=hours_to_fetch)
                        if history is None else {})
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

    if history is not None:
        is_sauna_detected, was_warm_yesterday = archive_sauna_history(history, s_temp_val)

    ev_pos = get_ha_state('device_tracker.xpz_491_position')
    pos_val = 1 if ev_pos and ev_pos.get('state') == 'home' else 0
    print(f"🚗 EV Position: {'Home' if pos_val else 'Away'}")

    # Load model and features
    model = xgb.XGBRegressor()
    try:
        model.load_model('state/energy_model.json')
        with open('state/model_features.json', 'r') as f:
            features = json.load(f)
        if history is not None:
            with open('state/model_training_metadata.json') as f:
                metadata = json.load(f)
            booster = model.get_booster()
            validate_baseload_model(features, metadata, booster.attr('baseload_feature_contract'), booster.feature_names)
    except FileNotFoundError:
        if history is not None:
            raise
        print('Error: Model files not found. Run train_model.py first.')
        return

    # Determine prediction window: from current interval until end of available solar horizon.
    now = (datetime.fromtimestamp(history.cutoff, ZoneInfo('Europe/Helsinki'))
           if history is not None else datetime.now().astimezone())
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
    if history is not None and not inference_data:
        raise ValueError('Empty prediction horizon at history cutoff')
        
    X_inference = pd.DataFrame(inference_data)
    missing = set(features) - set(X_inference.columns)
    if missing:
        if history is not None:
            raise ValueError(f'Missing archive inference features: {sorted(missing)}; refusing fabricated defaults')
        defaults = {'ev_soc': 80.0}
        print(f"⚠ Model expects features not in inference data: {missing}. Filling with defaults.")
        for f in missing:
            X_inference[f] = defaults.get(f, 0)
    X_inference = X_inference[features]
    predictions = model.predict(X_inference)
    if history is not None and (len(predictions) != len(timestamps) or not np.isfinite(predictions).all()):
        raise ValueError('Missing or nonfinite model predictions')
    
    # Combine predictions with timestamps
    # Model output is average Power (kW) for the interval
    results = []
    generated_at = (now if history is not None else datetime.now().astimezone()).isoformat()
    
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
            'solar_forecast_p10': float(inference_data[i].get('solar_forecast_p10', inference_data[i]['solar_forecast'])),
            'solar_forecast_p90': float(inference_data[i].get('solar_forecast_p90', inference_data[i]['solar_forecast'])),
            'outside_temp': float(inference_data[i]['outside_temp']),
            'ev_position': int(inference_data[i].get('ev_position', 1)),
            'is_sauna_active': int(inference_data[i].get('is_sauna_active', 0)),
            'is_fallback_price': int(is_fallback_price[i])
        })
        
    print(f'\nGenerated {len(results)} predictions at {interval}-minute resolution.')
    
    # Support environment variable override for testing
    predictions_file = os.getenv('TEST_PREDICTIONS_FILE', 'state/future_predictions.json')
    if history is None:
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        temporary = Path(predictions_file + '.tmp')
        try:
            with temporary.open('w') as f:
                json.dump(results, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, predictions_file)
        finally:
            temporary.unlink(missing_ok=True)
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
                solar_forecast_p10_kw REAL,
                solar_forecast_p90_kw REAL,
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
                planned_resistive_kw REAL,
                resistive_heater_intent TEXT,
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')
        
        # Schema migration: add missing columns
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        
        new_cols = {
            'is_fallback_price': 'INTEGER DEFAULT 0',
            'solar_forecast_p10_kw': 'REAL',
            'solar_forecast_p90_kw': 'REAL',
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
            'gshp_intent': 'TEXT',
            'planned_resistive_kw': 'REAL',
            'resistive_heater_intent': 'TEXT'
        }
        
        for col, col_type in new_cols.items():
            if col not in columns:
                print(f"Adding {col} column to predictions table...")
                cur.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")

        # Insert predictions.
        data_to_insert = [
            (res['timestamp'], generated_at, res['predicted_usage'], res['solar_forecast'], res.get('solar_forecast_p10'), res.get('solar_forecast_p90'), res['is_fallback_price'], git_version)
            for res in results
        ]
        cur.executemany('''
            INSERT OR REPLACE INTO predictions 
            (target_timestamp, generated_at, predicted_usage_kw, solar_forecast_kw, solar_forecast_p10_kw, solar_forecast_p90_kw, is_fallback_price, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f'✅ Archived {len(results)} points to {get_db_path()} (SQLite)')
    except Exception as e:
        print(f'⚠️ Error archiving to SQLite: {e}')

    if history is not None:
        metadata = {
            'history_cutoff': history.cutoff,
            'archive_path': str(Path(history.path).resolve()),
            'sources_path': str(Path(history.sources_path).resolve() if history.sources_path else
                                Path(__file__).resolve().parent / 'config/measurement_sources.yaml'),
            'measurement_policy_path': str(Path(history.policy_path).resolve()) if history.policy_path else None,
            'legacy_policy': history.legacy_policy,
            'allow_legacy': history.allow_legacy,
            'preferred_imports': list(history.preferred_imports),
            'model_version': git_version,
            'generated_at': generated_at,
            'current_telemetry_and_forecasts': 'live',
        }
        training_metadata = Path('state/model_training_metadata.json')
        if training_metadata.exists():
            metadata['model_training'] = json.loads(training_metadata.read_text())
        sidecar = Path(predictions_file).with_name('future_predictions_metadata.json')
        temporary = sidecar.with_suffix('.json.tmp')
        try:
            with temporary.open('w') as f:
                json.dump(metadata, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, sidecar)
        finally:
            temporary.unlink(missing_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate baseload predictions')
    add_history_arguments(parser)
    args = parser.parse_args()
    predict(history=history_from_arguments(args))
