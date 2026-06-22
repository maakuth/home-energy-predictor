#!/usr/bin/env python3
from __future__ import annotations
"""
Dump battery planning relevant variables from Home Assistant API and local database.

This script extracts data needed for battery planning tests into a pickle file
for easy deserialization and replay in test scenarios.

Usage:
    # Dump last 7 days
    python dump_battery_data.py --days 7 --output battery_test_data.pkl
    
    # Dump specific date range
    python dump_battery_data.py --start "2026-01-01" --end "2026-01-15" --output battery_data_jan.pkl
    
    # Dump with verbose output
    python dump_battery_data.py --days 3 --output data.pkl --verbose

The output pickle file contains a dictionary with:
    {
        'metadata': {
            'dumped_at': ISO timestamp,
            'period_start': ISO timestamp,
            'period_end': ISO timestamp,
            'description': 'Battery planning test data',
            'model_version': str,
        },
        'ha_states': {
            'entity_id': {'state': str, 'attributes': dict, 'last_updated': str},
            ...
        },
        'predictions': [
            {
                'timestamp': ISO timestamp,
                'predicted_baseload': float (kW),
                'solar_forecast': float (kW),
                'outside_temp': float (°C),
                'is_sauna_active': int (0/1),
                'is_fallback_price': int (0/1),
            },
            ...
        ],
        'market_prices': [
            {
                'timestamp': ISO timestamp,
                'import_price': float (EUR/kWh),
                'export_price': float (EUR/kWh),
            },
            ...
        ],
        'history': {
            'predictions_archive': [
                {
                    'target_timestamp': ISO timestamp,
                    'generated_at': ISO timestamp,
                    'predicted_usage_kw': float,
                    'solar_forecast_kw': float,
                    'import_price': float (EUR/kWh),
                    'export_price': float (EUR/kWh),
                    'is_fallback_price': int (0/1),
                    'battery_action': str,
                    'battery_power_kw': float,
                    'battery_soc_pct': float,
                    ...
                },
                ...
            ],
            'measurements': [
                {
                    'timestamp': ISO timestamp,
                    'total_power_kw': float,
                    'solar_actual_kw': float,
                    'gshp_power_kw': float,
                    'leaf_power_kw': float,
                    'outside_temp_c': float,
                },
                ...
            ],
        },
        'battery_config': {
            'capacity_kwh': float,
            'min_soc_pct': float,
            'max_soc_pct': float,
            'charge_rate_kw': float,
            'discharge_rate_kw': float,
            'enabled': bool,
        },
        'gshp_config': {
            'enabled': bool,
            'max_power_kw': float,
        }
    }
"""

import argparse
import json
import pickle
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from utils.type_defs import BatteryConfig, GshpConfig, FuturePredictionRecord, SqlitePredictionRecord, PriceRecord

sys.path.insert(0, str(Path(__file__).parent))

from utils.ha_utils import get_ha_state
from utils.price_utils import fetch_market_prices
from utils.sqlite_utils import get_db_connection
from utils.git_utils import get_model_version


def load_json_file(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not load {path}: {e}")
        return None


def get_ha_relevant_entities() -> list[str]:
    load_dotenv(override=True)
    
    # Try to get custom entity list from environment
    custom_entities_str = os.getenv('BATTERY_TEST_HA_ENTITIES')
    if custom_entities_str:
        return [e.strip() for e in custom_entities_str.split(',')]
    
    # Default entities from HEPO codebase
    return [
        # Battery state
        'sensor.be_soc',                          # Home battery SOC (%)
        'sensor.be_stat_batt_power',              # Home battery power (W)
        
        # Grid and Solar
        'sensor.sahkokauppa_nyt',                 # Grid meter (kW, positive=import)
        'sensor.solarh_63038_real_power_kw',      # Solar actual power (kW)
        'sensor.solcast_pv_forecast_forecast_tomorrow',  # Solar forecast
        
        # Heat Pumps and Loads
        'sensor.mlp_teho',                        # GSHP power (W)
        'sensor.mlp_varaajan_lampotila',          # GSHP accumulator temp (°C)
        'sensor.mlp_pumpun_lampotla',             # GSHP pump temp (°C)
        'sensor.saikaan_olohuone_current_power',  # AAHP living room (W)
        'sensor.mokkimokin_ilp_power',            # AAHP cabin (W)
        'sensor.tasmota_energy_power_3',          # EV charging power (W) - Nissan Leaf
        'sensor.xpz_491_battery_level',           # EV SOC (%)
        'sensor.mummun_energy',                   # Other load (energy)
        
        # Temperature
        'sensor.ulkona_temperature_2',            # Outside temperature (°C)
        'sensor.sauna_temperature_2',             # Sauna temperature (°C)
        
        # Market Prices
        'sensor.nordpool_total',                  # Nordpool prices
    ]


def fetch_ha_snapshot(verbose: bool = False) -> dict[str, dict[str, Any]]:
    """Fetch current state snapshot from Home Assistant."""
    entities = get_ha_relevant_entities()
    snapshot = {}
    
    if verbose:
        print(f"\nFetching {len(entities)} entities from Home Assistant...")
    
    for entity_id in entities:
        state = get_ha_state(entity_id)
        if state:
            snapshot[entity_id] = {
                'state': state.get('state'),
                'attributes': state.get('attributes', {}),
                'last_updated': state.get('last_updated'),
            }
            if verbose:
                print(f"  ✓ {entity_id}: {state.get('state')}")
        else:
            if verbose:
                print(f"  ✗ {entity_id}: not found")
    
    return snapshot


def fetch_predictions(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> list[FuturePredictionRecord]:
    """Load predictions from future_predictions.json."""
    preds = load_json_file('future_predictions.json')
    if not preds:
        return []
    
    if not isinstance(preds, list):
        return []
    
    # Filter by time range if specified
    if start_time or end_time:
        filtered = []
        for pred in preds:
            try:
                ts = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                filtered.append(pred)
            except (KeyError, ValueError):
                continue
        return filtered
    
    return preds


def fetch_nordpool_prices_from_ha(
    start_time: datetime,
    end_time: datetime,
    verbose: bool = False,
) -> list[PriceRecord]:
    """Fetch historical Nordpool prices from HA PostgreSQL database.
    
    Queries the states table for sensor.nordpool_total history.
    """
    prices = []
    
    try:
        import psycopg2
        from utils.db_utils import fetch_states_history
    except ImportError:
        if verbose:
            print(f"  ℹ️ PostgreSQL client not available for HA price history")
        return prices
    
    try:
        # Fetch nordpool price history
        delta = end_time - start_time
        hours = delta.total_seconds() / 3600.0
        
        hist_data = fetch_states_history(['sensor.nordpool_total'], hours=hours)
        
        if 'sensor.nordpool_total' not in hist_data or hist_data['sensor.nordpool_total'] is None:
            if verbose:
                print(f"  ℹ️ No nordpool_total history found in HA")
            return prices
        
        df = hist_data['sensor.nordpool_total']
        if df.empty:
            return prices
        
        # Parse the state values (they're stored as JSON or as numbers)
        import json
        
        for ts, row in df.iterrows():
            try:
                state_value = row.get('state', 0.0)
                
                # Try to convert directly (if it's a number string)
                try:
                    import_price = float(state_value)
                except (ValueError, TypeError):
                    # If that fails, it might be JSON. Try to parse it
                    try:
                        data = json.loads(state_value)
                        import_price = float(data.get('current_price', data))
                    except:
                        continue
                
                # Skip if outside time range (shouldn't happen but be safe)
                if ts < start_time or ts > end_time:  # type: ignore[operator]
                    continue
                
                prices.append({
                    'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    'import_price': import_price,
                    'export_price': max(0, import_price - 0.05),  # Typical spread
                })
            except (ValueError, KeyError, TypeError):
                continue
        
        if verbose and prices:
            print(f"  ✓ Loaded {len(prices)} price records from HA Nordpool history")
        
        return prices
    
    except Exception as e:
        if verbose:
            print(f"  ℹ️ Error fetching prices from HA: {e}")
        return prices


def fetch_market_prices_range(
    start_time: datetime,
    end_time: datetime,
    verbose: bool = False,
) -> list[PriceRecord]:
    """Fetch market prices for the given time range.
    
    Tries multiple sources in order:
    1. future_predictions.json file
    2. SQLite predictions table (import/export prices)
    3. HA PostgreSQL states table (sensor.nordpool_total)
    """
    prices = []
    
    if verbose:
        print(f"\nFetching market prices from {start_time} to {end_time}...")
    
    try:
        # Try to load from predictions file if it has price data
        preds = load_json_file('future_predictions.json')
        if preds and isinstance(preds, list) and len(preds) > 0:
            if 'import_price' in preds[0] or 'export_price' in preds[0]:
                for pred in preds:
                    try:
                        ts = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                        if start_time and ts < start_time:
                            continue
                        if end_time and ts > end_time:
                            continue
                        
                        prices.append({
                            'timestamp': pred['timestamp'],
                            'import_price': pred.get('import_price', 0.0),
                            'export_price': pred.get('export_price', 0.0),
                        })
                    except (KeyError, ValueError):
                        continue
                
                if verbose and prices:
                    print(f"  ✓ Loaded {len(prices)} price records from predictions file")
                    return prices
        
        # Try to get from SQLite predictions table
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("PRAGMA table_info(predictions)")
            columns = {row[1]: row[0] for row in cur.fetchall()}
            
            timestamp_col = 'target_timestamp' if 'target_timestamp' in columns else 'timestamp'
            
            if 'import_price' in columns and 'export_price' in columns:
                # Timestamps are TEXT in the database, use string comparison
                start_str = start_time.isoformat()
                end_str = end_time.isoformat()
                
                query = f"""
                    SELECT {timestamp_col}, import_price, export_price
                    FROM predictions
                    WHERE {timestamp_col} >= ? AND {timestamp_col} <= ?
                    ORDER BY {timestamp_col}
                """
                cur.execute(query, (start_str, end_str))
                
                for row in cur.fetchall():
                    prices.append({
                        'timestamp': row[0],  # Already in ISO format
                        'import_price': row[1] if row[1] is not None else 0.0,
                        'export_price': row[2] if row[2] is not None else 0.0,
                    })
                
                if verbose and prices:
                    print(f"  ✓ Loaded {len(prices)} price records from SQLite predictions table")
                    cur.close()
                    conn.close()
                    return prices
            
            cur.close()
            conn.close()
        except Exception as db_err:
            if verbose:
                print(f"  ℹ️ Could not fetch from SQLite: {db_err}")
        
        # Try to get from HA PostgreSQL (sensor.nordpool_total historical data)
        try:
            prices = fetch_nordpool_prices_from_ha(start_time, end_time, verbose=verbose)
            if prices:
                return prices
        except Exception as ha_err:
            if verbose:
                print(f"  ℹ️ Could not fetch from HA PostgreSQL: {ha_err}")
        
        if not prices and verbose:
            print(f"  ℹ️ No market prices found")
    
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Error fetching prices: {e}")
    
    return prices


def synthesize_predictions(
    measurements: list[dict[str, Any]],
    start_time: datetime,
    end_time: datetime,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Synthesize realistic prediction archive from measurement data.
    
    Creates predictions offset by MAE (mean absolute error) to simulate
    realistic forecast inaccuracy. Uses actual measurements as ground truth
    but shifts them forward in time as if they were predictions made earlier.
    """
    if not measurements:
        return []
    
    predictions = []
    
    # Typical forecast error from performance analysis
    mae_kw = 1.2  # Mean absolute error in kW
    bias_kw = 0.3  # Slight bias toward underestimation
    
    # Convert measurements to DataFrame for easier handling
    df_meas = pd.DataFrame(measurements)
    df_meas['timestamp'] = pd.to_datetime(df_meas['timestamp'], utc=True)
    df_meas = df_meas.sort_values('timestamp').set_index('timestamp')
    
    # Generate realistic price variation (Nordic Pool typical range)
    # Daily factor: varies by day for realistic market movement
    num_days = (df_meas.index[-1] - df_meas.index[0]).days + 1
    day_factors = np.random.normal(1.0, 0.15, num_days)  # Some days expensive, some cheap
    
    for i, (ts, row) in enumerate(df_meas.iterrows()):
        # This measurement is the forecast target
        target_ts = ts
        
        # The forecast was "generated" 24 hours before
        generated_ts = ts - pd.Timedelta(hours=24)  # type: ignore[operator]
        
        # Skip if generated time is before start of range
        if generated_ts < start_time:
            continue
        
        # Get the actual usage
        actual_usage = row.get('total_power_kw', 0.0)
        
        # Add realistic forecast error
        np.random.seed(int(ts.timestamp()))  # Reproducible randomness per timestamp
        random_error = np.random.normal(0, mae_kw / 2)
        predicted_usage = actual_usage + bias_kw + random_error
        predicted_usage = max(0, predicted_usage)
        
        # Synthesize realistic Nordic Pool market prices
        hour = ts.hour
        day_of_period = (ts - df_meas.index[0]).days
        
        # Base hourly pattern (peak during daytime, low at night)
        if 6 <= hour <= 22:
            base_hourly = 0.15 + 0.08 * np.sin((hour - 6) * np.pi / 16)
        else:
            base_hourly = 0.08
        
        # Add day-to-day variation
        day_factor = day_factors[min(day_of_period, len(day_factors) - 1)]
        
        # Add small random noise for intraday variability
        np.random.seed(int(ts.timestamp()))
        noise = np.random.normal(0, 0.01)
        
        import_price = max(0.05, base_hourly * day_factor + noise)
        export_price = max(0, import_price - 0.05)
        
        # Get solar data if available, default to 0 if missing
        solar_forecast = row.get('solar_actual_kw', 0.0)
        if pd.isna(solar_forecast):
            solar_forecast = 0.0
        
        record = {
            'target_timestamp': target_ts.isoformat(),
            'generated_at': generated_ts.isoformat(),
            'predicted_usage_kw': float(predicted_usage),
            'solar_forecast_kw': float(solar_forecast),
            'is_fallback_price': 0,
            'import_price': float(import_price),
            'export_price': float(export_price),
            'battery_action': None,
            'battery_power_kw': None,
            'battery_soc_pct': None,
            'grid_import_kwh': None,
            'grid_export_kwh': None,
            'charge_from_solar_kwh': None,
            'charge_from_grid_kwh': None,
            'discharge_to_load_kwh': None,
            'discharge_to_export_kwh': None,
            'planned_gshp_kw': None,
            'gshp_intent': None,
        }
        predictions.append(record)
    
    if verbose:
        print(f"  ✓ Synthesized {len(predictions)} prediction records (MAE={mae_kw:.1f}kW, Bias={bias_kw:.1f}kW)")
    
    return predictions


def fetch_sqlite_predictions(
    start_time: datetime,
    end_time: datetime,
    verbose: bool = False,
) -> list[SqlitePredictionRecord]:
    """Fetch prediction archive from SQLite database, preserving generated_at.
    
    Returns list of prediction records with target_timestamp, generated_at,
    and other forecast/plan data. Keeps ALL records to allow time-aware filtering
    during replay (filter by generated_at <= planning_time).
    """
    predictions = []
    
    if verbose:
        print(f"\nFetching predictions archive from SQLite for {start_time} to {end_time}...")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check what columns exist in predictions table
        cur.execute("PRAGMA table_info(predictions)")
        columns = {row[1]: row[0] for row in cur.fetchall()}
        
        if not columns:
            if verbose:
                print(f"  ⚠️ 'predictions' table not found in database")
            cur.close()
            conn.close()
            return predictions
        
        # Determine timestamp column name
        timestamp_col = 'target_timestamp' if 'target_timestamp' in columns else 'timestamp'
        
        # Build query: select target_timestamp, generated_at, and forecast/plan data
        available_cols = [timestamp_col, 'generated_at']
        for col in ['predicted_usage_kw', 'solar_forecast_kw', 'is_fallback_price',
                    'import_price', 'export_price', 
                    'battery_action', 'battery_power_kw', 'battery_soc_pct',
                    'grid_import_kwh', 'grid_export_kwh',
                    'charge_from_solar_kwh', 'charge_from_grid_kwh',
                    'discharge_to_load_kwh', 'discharge_to_export_kwh',
                    'planned_gshp_kw', 'gshp_intent']:
            if col in columns:
                available_cols.append(col)
        
        select_clause = ', '.join(available_cols)
        query = f"""
            SELECT {select_clause}
            FROM predictions
            WHERE {timestamp_col} >= ? AND {timestamp_col} <= ?
            ORDER BY {timestamp_col}, generated_at
        """
        
        start_str = start_time.isoformat()
        end_str = end_time.isoformat()
        
        cur.execute(query, (start_str, end_str))
        rows = cur.fetchall()
        
        if rows:
            for row in rows:
                record = {}
                for i, col in enumerate(available_cols):
                    value = row[i]
                    if col in ('timestamp', 'target_timestamp'):
                        # Normalize to 'target_timestamp' and ensure ISO format
                        if isinstance(value, str) and 'T' not in value:
                            record['target_timestamp'] = value + 'T00:00:00+00:00'
                        else:
                            record['target_timestamp'] = value
                    elif col == 'generated_at':
                        # Ensure generated_at is ISO format
                        if isinstance(value, str) and 'T' not in value:
                            record['generated_at'] = value + 'T00:00:00+00:00'
                        else:
                            record['generated_at'] = value
                    else:
                        record[col] = value
                
                predictions.append(record)
            
            if verbose:
                print(f"  ✓ Fetched {len(rows)} prediction records")
        else:
            if verbose:
                print(f"  ℹ️ No records found in time range")
        
        cur.close()
        conn.close()
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Could not fetch predictions: {e}")
    
    return predictions


def fetch_ha_measurements(
    start_time: datetime,
    end_time: datetime,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fetch actual measurements from Home Assistant PostgreSQL database.
    
    Returns list of measurement records for grid power, solar, GSHP, Leaf, and other
    actual sensor readings for the given time range.
    """
    measurements = []
    
    if verbose:
        print(f"\nFetching HA measurements from PostgreSQL for {start_time} to {end_time}...")
    
    # Try to import PostgreSQL client
    try:
        import psycopg2
        from utils.db_utils import fetch_states_history
    except ImportError:
        if verbose:
            print(f"  ℹ️ PostgreSQL client not available; skipping HA measurements")
        return measurements
    
    try:
        # Fetch measurements for key entities
        entities = [
            'sensor.sahkokauppa_nyt',                      # Grid power (kW)
            'sensor.solarh_63038_real_power_kw',           # Solar actual (kW)
            'sensor.mlp_teho',                              # GSHP power (W)
            'sensor.tasmota_energy_power_3',                # Leaf power (W)
            'sensor.ulkona_temperature_2',                  # Outside temp (°C)
        ]
        
        # Calculate hours to fetch
        delta = end_time - start_time
        hours = delta.total_seconds() / 3600.0
        
        hist_data = fetch_states_history(entities, hours=hours)
        
        # Combine all measurements into a unified timeline
        import pandas as pd
        all_dfs = []
        
        for entity_id, col_name in [
            ('sensor.sahkokauppa_nyt', 'total_power_kw'),
            ('sensor.solarh_63038_real_power_kw', 'solar_actual_kw'),
            ('sensor.mlp_teho', 'gshp_power_w'),
            ('sensor.tasmota_energy_power_3', 'leaf_power_w'),
            ('sensor.ulkona_temperature_2', 'outside_temp_c'),
        ]:
            df = hist_data.get(entity_id)
            if df is not None and not df.empty:
                # fetch_states_history returns DataFrame with 'timestamp' as index
                # Rename 'state' to our column name and convert to numeric
                if 'state' in df.columns:
                    df = df.rename(columns={'state': col_name}).copy()
                else:
                    # If 'state' is not there, assume column is already named correctly
                    df = df.copy()
                
                try:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                except:
                    pass
                
                # Reset index to make timestamp a column, then set it back
                if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    df = df[['timestamp', col_name]].set_index('timestamp')
                else:
                    df = df[[col_name]]
                
                all_dfs.append(df)
        
        if all_dfs:
            # Merge all measurements, resample to 15-min intervals
            df_merged = pd.concat(all_dfs, axis=1)
            df_merged = df_merged.resample('15min').mean()
            
            # Convert back to list of dicts
            for ts, row in df_merged.iterrows():
                record = {
                    'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                }
                # Convert W to kW where needed
                if 'gshp_power_w' in row.index and pd.notna(row['gshp_power_w']):
                    record['gshp_power_kw'] = row['gshp_power_w'] / 1000.0
                if 'leaf_power_w' in row.index and pd.notna(row['leaf_power_w']):
                    record['leaf_power_kw'] = row['leaf_power_w'] / 1000.0
                
                # Add other measurements as-is
                for col in ['total_power_kw', 'solar_actual_kw', 'outside_temp_c']:
                    if col in row.index and pd.notna(row[col]):
                        record[col] = float(row[col])
                
                measurements.append(record)
            
            if verbose:
                print(f"  ✓ Fetched {len(measurements)} measurement records")
        else:
            if verbose:
                print(f"  ℹ️ No measurement data found")
    
    except Exception as e:
        if verbose:
            print(f"  ℹ️ Could not fetch HA measurements: {e}")
    
    return measurements


def fetch_sqlite_history(
    start_time: datetime,
    end_time: datetime,
    verbose: bool = False,
) -> dict[str, Any]:
    """Wrapper to maintain backward compatibility. Combines predictions and measurements."""
    history: dict[str, Any] = {}
    
    if verbose:
        print(f"\nFetching history from SQLite and PostgreSQL for {start_time} to {end_time}...")
    
    # Fetch actual measurements first (needed for synthesis if predictions are missing)
    measurements = fetch_ha_measurements(start_time, end_time, verbose=verbose)
    if measurements:
        history['measurements'] = measurements
    
    # Fetch prediction archive with generated_at preserved
    predictions = fetch_sqlite_predictions(start_time, end_time, verbose=verbose)
    
    # If no predictions found and we have measurements, synthesize predictions
    if not predictions and measurements:
        if verbose:
            print(f"\nSynthesizing predictions from measurements...")
        predictions = synthesize_predictions(measurements, start_time, end_time, verbose=verbose)
    
    if predictions:
        history['predictions_archive'] = predictions
    
    return history


def get_battery_config() -> BatteryConfig:
    """Get battery configuration from environment."""
    load_dotenv(override=True)
    
    return {
        'capacity_kwh': float(os.getenv('BATTERY_CAPACITY_KWH', '0')),
        'min_soc_pct': float(os.getenv('BATTERY_MIN_SOC_PCT', '10')),
        'max_soc_pct': float(os.getenv('BATTERY_MAX_SOC_PCT', '95')),
        'charge_rate_kw': float(os.getenv('BATTERY_CHARGE_RATE_KW', '0')),
        'discharge_rate_kw': float(os.getenv('BATTERY_DISCHARGE_RATE_KW', '0')),
        'enabled': os.getenv('BATTERY_ENABLED', '0').lower() in {'1', 'true', 'yes'},
    }


def get_gshp_config() -> GshpConfig:
    """Get GSHP configuration from environment."""
    load_dotenv(override=True)
    
    return {
        'enabled': os.getenv('GSHP_ENABLED', '0').lower() in {'1', 'true', 'yes'},
        'max_power_kw': float(os.getenv('GSHP_MAX_POWER_KW', '0')),
    }


def dump_battery_data(
    days: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    output: str = 'battery_test_data.pkl',
    verbose: bool = False,
) -> bool:
    """
    Dump battery planning data to pickle file.
    
    Args:
        days: Number of days to go back (if start/end not specified)
        start: Start date string (YYYY-MM-DD or ISO format)
        end: End date string (YYYY-MM-DD or ISO format)
        output: Output pickle file path
        verbose: Print progress information
    """
    load_dotenv(override=True)
    
    # Determine time range
    now = datetime.now(tz=timezone.utc)
    
    if start and end:
        # Both start and end provided
        try:
            start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except ValueError:
            start_time = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        try:
            end_time = datetime.fromisoformat(end.replace('Z', '+00:00'))
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
        except ValueError:
            end_time = datetime.strptime(end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    elif start:
        # Only start provided - calculate end from days
        try:
            start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except ValueError:
            start_time = datetime.strptime(start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        days = days or 7
        end_time = start_time + timedelta(days=days)
    else:
        # Neither start nor end provided - use relative days from now
        days = days or 7
        start_time = now - timedelta(days=days)
        end_time = now
    
    if verbose:
        print(f"📊 Dumping battery planning data")
        print(f"   Period: {start_time} to {end_time}")
        print(f"   Output: {output}")
    
    # Collect data
    data: dict = {
        'metadata': {
            'dumped_at': now.isoformat(),
            'period_start': start_time.isoformat(),
            'period_end': end_time.isoformat(),
            'description': 'Battery planning test data',
            'model_version': get_model_version(),
        },
    }
    
    # Fetch current HA state
    if verbose:
        print("\n1️⃣ Fetching Home Assistant snapshots...")
    data['ha_states'] = fetch_ha_snapshot(verbose=verbose)
    
    # Fetch predictions
    if verbose:
        print("\n2️⃣ Loading predictions...")
    predictions_list = fetch_predictions(start_time=start_time, end_time=end_time)
    data['predictions'] = predictions_list
    if verbose:
        print(f"   ✓ Loaded {len(predictions_list)} predictions")
    
    # Fetch market prices
    if verbose:
        print("\n3️⃣ Fetching market prices...")
    prices_list = fetch_market_prices_range(start_time, end_time, verbose=verbose)
    data['market_prices'] = prices_list
    
    # Fetch SQLite history
    if verbose:
        print("\n4️⃣ Fetching SQLite history...")
    data['history'] = fetch_sqlite_history(start_time, end_time, verbose=verbose)
    
    # Get configuration
    if verbose:
        print("\n5️⃣ Loading configuration...")
    data['battery_config'] = get_battery_config()
    data['gshp_config'] = get_gshp_config()
    if verbose:
        print(f"   ✓ Battery: {data['battery_config']}")
        print(f"   ✓ GSHP: {data['gshp_config']}")
    
    # Write to pickle file
    try:
        with open(output, 'wb') as f:
            pickle.dump(data, f)
        if verbose:
            print(f"\n✅ Successfully dumped data to {output}")
            print(f"   File size: {os.path.getsize(output) / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"❌ Error writing pickle file: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dump battery planning data from Home Assistant and database to pickle file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Last 7 days (default)
  python dump_battery_data.py --output my_data.pkl
  
  # Last 3 days
  python dump_battery_data.py --days 3 --output last_3days.pkl
  
  # Specific date range
  python dump_battery_data.py --start "2026-01-01" --end "2026-01-15" --output jan_data.pkl
  
  # With verbose output
  python dump_battery_data.py --days 7 --output data.pkl --verbose
        """
    )
    
    parser.add_argument('--days', type=int, default=7, 
                        help='Number of days to go back (default: 7)')
    parser.add_argument('--start', type=str, 
                        help='Start date (YYYY-MM-DD or ISO format)')
    parser.add_argument('--end', type=str,
                        help='End date (YYYY-MM-DD or ISO format)')
    parser.add_argument('--output', type=str, default='battery_test_data.pkl',
                        help='Output pickle file path (default: battery_test_data.pkl)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress information')
    
    args = parser.parse_args()
    
    success = dump_battery_data(
        days=args.days if not args.start else None,
        start=args.start,
        end=args.end,
        output=args.output,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
