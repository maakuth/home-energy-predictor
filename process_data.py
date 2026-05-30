# Data processing and feature engineering
import pandas as pd
import numpy as np

def process_data():
    print('Loading raw data...')
    try:
        df = pd.read_csv('raw_data.csv', index_col=0, low_memory=False)
    except FileNotFoundError:
        print('Error: raw_data.csv not found.')
        return
    
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Fundamental step: Sort and deduplicate
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    print('Denoising and filling gaps...')
    fill_zero_cols = ['gshp_power', 'aahp_living_power', 'aahp_cabin_power', 'mummun_power', 'solar_forecast', 'solar_actual', 'leaf_power']
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    if 'ev_soc' in df.columns:
        df['ev_soc'] = df['ev_soc'].ffill().fillna(0)
    if 'ev_position' in df.columns:
        if df['ev_position'].dtype == object:
            df['ev_position'] = (df['ev_position'] == 'home').astype(int)
        df['ev_position'] = df['ev_position'].ffill().fillna(0).astype(int)

    # Denoise on high resolution if available (e.g. 1-minute)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 15-point median filter is ~15 mins if 1-min data.
    rolled = df[numeric_cols].rolling(window=15, center=True).median()
    df[numeric_cols] = rolled.fillna(df[numeric_cols])
    
    # --- RESOLUTION AGNOSTIC STEP ---
    # Detect frequency to set correct lag shifts
    # (Using median diff to be robust against small gaps)
    if len(df) > 1:
        median_interval = pd.Series(df.index).diff().median()
        interval_minutes = int(round(median_interval.total_seconds() / 60.0))
        print(f"Detected data resolution: {interval_minutes} minutes")
    else:
        interval_minutes = 15 # Fallback
    
    # Compute total home consumption BEFORE clipping
    # With a home battery: Battery power is positive when charging, negative when discharging
    # Since we measure grid power (what flows to/from grid), we must subtract net battery power
    # to get true home load: Home Load = Grid Import + Solar - (Battery Discharge - Battery Charge)
    #                                  = Grid + Solar - Battery Power (our convention)
    if 'total_power' in df.columns and 'solar_actual' in df.columns:
        df['total_home_power'] = df['total_power'] + df['solar_actual']
        # Subtract battery power to get true home load (only if battery sensor exists)
        if 'battery_power' in df.columns:
            battery_kw = (df['battery_power'] / 1000.0)  # Convert W to kW
            df['total_home_power'] = df['total_home_power'] - battery_kw
    elif 'total_power' in df.columns:
        df['total_home_power'] = df['total_power']

    # Baseload: House consumption excluding the GSHP and other known high-power loads
    if 'total_home_power' in df.columns:
        gshp_kw = (df['gshp_power'] / 1000.0) if 'gshp_power' in df.columns else 0.0
        leaf_kw = (df['leaf_power'] / 1000.0) if 'leaf_power' in df.columns else 0.0
        df['baseload_power'] = df['total_home_power'] - gshp_kw - leaf_kw
        df['baseload_power'] = df['baseload_power'].clip(lower=0)
    else:
        df['baseload_power'] = 0.0

    # Clip component power meters
    power_cols = [c for c in df.columns if 'power' in c or 'teho' in c or 'energy' in c]
    exclude_from_clip = {'total_power'} 
    for col in power_cols:
        if col in df.columns and col not in exclude_from_clip:
            df[col] = df[col].clip(lower=0)
    
    if 'outside_temp' in df.columns:
        df['outside_temp'] = df['outside_temp'].ffill().clip(lower=-50, upper=50)
    if 'wind_speed' in df.columns:
        df['wind_speed'] = df['wind_speed'].ffill().clip(lower=0, upper=100)
    if 'accumulator_temp' in df.columns:
        df['accumulator_temp'] = df['accumulator_temp'].ffill().clip(lower=0, upper=100)
    
    if 'gshp_pump_temp' in df.columns:
        # User specified: if pump power < 100W, the sensor is invalid/weird.
        # Set to NaN so XGBoost treats it as 'missing' or disregards it.
        # This also naturally handles the pre-June 2025 missing data.
        if 'gshp_power' in df.columns:
            df['is_gshp_pump_running'] = (df['gshp_power'] >= 100).astype(int)
            df.loc[df['is_gshp_pump_running'] == 0, 'gshp_pump_temp'] = np.nan
        else:
            df['is_gshp_pump_running'] = 0
            
    if 'sauna_temp' in df.columns:
        df['sauna_temp'] = df['sauna_temp'].ffill().clip(lower=0, upper=120)
        df['is_sauna_active'] = (df['sauna_temp'] > 30).astype(int)

    print('Adding lagged features (time-aware)...')
    if 'baseload_power' in df.columns:
        # Calculate shifts based on detected resolution
        # 1 hour = 60 mins. 24 hours = 1440 mins.
        shift_1h = max(1, 60 // interval_minutes)
        shift_24h = max(1, 1440 // interval_minutes)
        
        print(f"Applying lags: 1h = shift({shift_1h}), 24h = shift({shift_24h})")
        df['baseload_lag_1h'] = df['baseload_power'].shift(shift_1h).ffill()
        df['baseload_lag_24h'] = df['baseload_power'].shift(shift_24h).ffill()
        
        if 'leaf_power' in df.columns:
            # Leaf features: power lag and 24h cumulative energy proxy
            df['leaf_power_lag_1h'] = df['leaf_power'].shift(shift_1h).ffill()
            
            # 24h rolling sum of power (divided by resolution to approximate kWh)
            # Power is in Watts, convert to kW and then sum over intervals
            leaf_kw = df['leaf_power'] / 1000.0
            # Energy = power * time. For 1-min data, each point is 1/60 kWh.
            # Rolling sum of (kW * (interval_min/60))
            df['leaf_energy_24h'] = leaf_kw.rolling(window=shift_24h, min_periods=1).sum() * (interval_minutes / 60.0)

    print('Applying fireplace logic...')
    if 'accumulator_temp' in df.columns:
        df['acc_roc'] = df['accumulator_temp'].diff().fillna(0)
        hp_cols = ['gshp_power', 'aahp_living_power', 'aahp_cabin_power']
        available_hp = [c for c in hp_cols if c in df.columns]
        df['total_hp_power'] = df[available_hp].sum(axis=1) / 1000.0
        
        # Fireplace heuristic: temperature rising even though HP is mostly off
        # Threshold (0.3) was designed for 15-min intervals. 
        # For 1-min, we should scale it or check a window. 
        # Let's check a 15-min window for fireplace logic to be robust.
        acc_roc_15m = df['accumulator_temp'].diff(periods=max(1, 15//interval_minutes)).fillna(0)
        df['is_fireplace_active'] = ((acc_roc_15m > 0.3) & (df['total_hp_power'] < 0.5)).astype(int)
        df['is_fireplace_lag1'] = df['is_fireplace_active'].shift(1).fillna(0)
    
    print('Adding structural change features...')
    structural_change_date = pd.to_datetime('2025-10-01', utc=True)
    df['is_extended_complex'] = (df.index >= structural_change_date).astype(int)

    print('Adding temporal features...')
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['quarter_hour'] = df.index.minute // 15
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Cyclical Encoding (Trigonometric features)
    # Hour of day (24h cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (7d cycle)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month of year (12m cycle)
    df['month_sin'] = np.sin(2 * np.pi * (df['month']-1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month']-1) / 12)
    
    critical_cols = ['total_home_power', 'outside_temp', 'accumulator_temp']
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
    
    df.to_csv('processed_data.csv')
    print(f'✅ Processing complete. Saved to processed_data.csv. Shape: {df.shape}')

if __name__ == '__main__':
    process_data()
