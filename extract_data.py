import os
import psycopg2
import pandas as pd
import argparse
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv(override=True)

RESAMPLE_INTERVAL = os.getenv('DATA_RESAMPLE_INTERVAL', '1min')

ENTITIES = {
    'sensor.ulkona_temperature_2': 'outside_temp',
    'sensor.mlp_teho': 'gshp_power',
    'sensor.saikaan_olohuone_current_power': 'aahp_living_power',
    'sensor.mokkimokin_ilp_power': 'aahp_cabin_power',
    'sensor.mummun_energy': 'mummun_energy',
    'sensor.mlp_varaajan_lampotila': 'accumulator_temp',
    'sensor.xpz_491_battery_level': 'ev_soc',
    'device_tracker.xpz_491_position': 'ev_position',
    'sensor.tasmota_energy_power_3': 'leaf_power',
    'sensor.sahkokauppa_nyt': 'total_power',
    'sensor.solcast_pv_forecast_forecast_tomorrow': 'solar_forecast',
    'sensor.solarh_63038_real_power_kw': 'solar_actual',
    'sensor.sauna_temperature_2': 'sauna_temp',
    'weather.home': 'wind_speed' # Special handling for attribute
}

def get_metadata_ids(cur):
    query = "SELECT metadata_id, entity_id FROM states_meta WHERE entity_id IN %s"
    cur.execute(query, (tuple(ENTITIES.keys()),))
    return {row[1]: row[0] for row in cur.fetchall()}

def extract_states(cur, metadata_id, days=365):
    start_ts = (datetime.now() - timedelta(days=days)).timestamp()
    query = """
        SELECT last_updated_ts, state 
        FROM states 
        WHERE metadata_id = %s AND last_updated_ts > %s
        ORDER BY last_updated_ts ASC
    """
    cur.execute(query, (metadata_id, start_ts))
    return cur.fetchall()

def extract_attribute(cur, metadata_id, attr_name, days=365):
    """Extract a specific attribute from JSON for entities like weather.home."""
    start_ts = (datetime.now() - timedelta(days=days)).timestamp()
    # Try the modern schema (post 2023.4)
    try:
        query = """
            SELECT s.last_updated_ts, sa.shared_attrs::json->>%s
            FROM states s
            JOIN state_attributes sa ON s.attributes_id = sa.attributes_id
            WHERE s.metadata_id = %s AND s.last_updated_ts > %s
            ORDER BY s.last_updated_ts ASC
        """
        cur.execute(query, (attr_name, metadata_id, start_ts))
        return cur.fetchall()
    except Exception:
        # Fallback to older schema if needed
        query = """
            SELECT last_updated_ts, attributes::json->>%s
            FROM states 
            WHERE metadata_id = %s AND last_updated_ts > %s
            ORDER BY last_updated_ts ASC
        """
        cur.execute(query, (attr_name, metadata_id, start_ts))
        return cur.fetchall()

def main():
    parser = argparse.ArgumentParser(description='Extract historical data from Home Assistant DB.')
    parser.add_argument('--days', type=int, default=365, help='Number of days to look back.')
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()
    
    metadata_map = get_metadata_ids(cur)
    all_dfs = []
    
    for entity_id, col_name in ENTITIES.items():
        if entity_id not in metadata_map:
            print(f"⚠️ Warning: Metadata ID for {entity_id} not found.")
            continue
            
        print(f"Extracting {entity_id} (last {args.days} days)...")
        
        if entity_id == 'weather.home':
            # Extract wind_speed attribute
            rows = extract_attribute(cur, metadata_map[entity_id], 'wind_speed', days=args.days)
        else:
            rows = extract_states(cur, metadata_map[entity_id], days=args.days)
            
        if not rows:
            print(f"No data for {entity_id}")
            continue
            
        df = pd.DataFrame(rows, columns=['ts', col_name])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        df = df.set_index('timestamp').drop(columns=['ts'])
        
        # Numeric conversion (except for position)
        if col_name != 'ev_position':
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Resample to configured interval (default 15 minutes).
        if col_name == 'ev_position':
            # Is home if any state in the interval says 'home'.
            resampled = df[col_name].resample(RESAMPLE_INTERVAL).apply(lambda x: 'home' in x.values if not x.empty else None)
        elif col_name == 'mummun_energy':
            # Handle energy total to power conversion later
            resampled = df[col_name].resample(RESAMPLE_INTERVAL).mean()
        else:
            resampled = df[col_name].resample(RESAMPLE_INTERVAL).mean()
            
        all_dfs.append(resampled)
    
    cur.close()
    conn.close()
    
    print("Merging data...")
    final_df = pd.concat(all_dfs, axis=1, sort=False)
    
    # Convert cumulative energy delta to average power over each interval.
    # kW = delta_kWh / delta_hours.
    if 'mummun_energy' in final_df.columns:
        delta_hours = pd.Timedelta(RESAMPLE_INTERVAL).total_seconds() / 3600.0
        final_df['mummun_power'] = (final_df['mummun_energy'].diff() / max(delta_hours, 1e-9)).clip(lower=0)
        final_df = final_df.drop(columns=['mummun_energy'])
    
    # Gap Filling
    # Linear interpolation for short gaps.
    # Only interpolate numeric columns
    numeric_cols = final_df.select_dtypes(include=['number']).columns
    final_df[numeric_cols] = final_df[numeric_cols].interpolate(method='linear', limit=15)
    
    # Create final dataset
    final_df.to_csv('raw_data.csv')
    print(f"✅ Data extraction complete. Saved to raw_data.csv. Shape: {final_df.shape}")

if __name__ == "__main__":
    main()
