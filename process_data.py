# Data processing and feature engineering
import pandas as pd
import numpy as np

def process_data():
    print('Loading raw data...')
    try:
        df = pd.read_csv('raw_data.csv', index_col=0)
    except FileNotFoundError:
        print('Error: raw_data.csv not found.')
        return
    
    df.index = pd.to_datetime(df.index, utc=True)
    
    print('Denoising and filling gaps...')
    fill_zero_cols = ['gshp_power', 'aahp_living_power', 'aahp_cabin_power', 'mummun_power', 'solar_forecast', 'solar_actual']
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    if 'ev_soc' in df.columns:
        df['ev_soc'] = df['ev_soc'].ffill().fillna(0)
    if 'ev_position' in df.columns:
        df['ev_position'] = df['ev_position'].ffill().fillna(False).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rolled = df[numeric_cols].rolling(window=15, center=True).median()
    df[numeric_cols] = rolled.fillna(df[numeric_cols])
    
    # Compute total home consumption BEFORE clipping, because total_power (grid meter)
    # is legitimately negative when solar export exceeds load.
    # total_home_power = grid_power + solar_production
    if 'total_power' in df.columns and 'solar_actual' in df.columns:
        df['total_home_power'] = df['total_power'] + df['solar_actual']
    elif 'total_power' in df.columns:
        df['total_home_power'] = df['total_power']

    # Baseload: House consumption excluding the GSHP
    # Note: gshp_power sensor (mlp_teho) is in Watts, total_home_power is in kW.
    if 'total_home_power' in df.columns and 'gshp_power' in df.columns:
        df['baseload_power'] = df['total_home_power'] - (df['gshp_power'] / 1000.0)
        # Ensure baseload isn't negative due to sensor noise, but keep it realistic
        df['baseload_power'] = df['baseload_power'].clip(lower=0)
    else:
        df['baseload_power'] = df.get('total_home_power', 0)

    # Clip component power meters (not the bidirectional grid meter total_power)
    power_cols = [c for c in df.columns if 'power' in c or 'teho' in c or 'energy' in c]
    exclude_from_clip = {'total_power'}  # grid meter, can be negative during solar export
    for col in power_cols:
        if col in df.columns and col not in exclude_from_clip:
            df[col] = df[col].clip(lower=0)
    
    if 'outside_temp' in df.columns:
        df['outside_temp'] = df['outside_temp'].ffill().clip(lower=-50, upper=50)
    if 'accumulator_temp' in df.columns:
        df['accumulator_temp'] = df['accumulator_temp'].ffill().clip(lower=0, upper=100)
    if 'sauna_temp' in df.columns:
        df['sauna_temp'] = df['sauna_temp'].ffill().clip(lower=0, upper=120)
        df['is_sauna_active'] = (df['sauna_temp'] > 30).astype(int)

    print('Applying fireplace logic...')
    if 'accumulator_temp' in df.columns:
        df['acc_roc'] = df['accumulator_temp'].diff().fillna(0)
        hp_cols = ['gshp_power', 'aahp_living_power', 'aahp_cabin_power']
        available_hp = [c for c in hp_cols if c in df.columns]
        # HP sensors are in Watts, convert to kW for logic consistency
        df['total_hp_power'] = df[available_hp].sum(axis=1) / 1000.0
        df['is_fireplace_active'] = ((df['acc_roc'] > 0.3) & (df['total_hp_power'] < 0.5)).astype(int)
        df['is_fireplace_lag1'] = df['is_fireplace_active'].shift(1).fillna(0)
    
    print('Adding structural change features...')
    # New building added roughly Oct 1st 2025
    structural_change_date = pd.to_datetime('2025-10-01', utc=True)
    df['is_extended_complex'] = (df.index >= structural_change_date).astype(int)

    print('Adding temporal features...')
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['quarter_hour'] = df.index.minute // 15
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    critical_cols = ['total_home_power', 'outside_temp', 'accumulator_temp']
    df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
    
    df.to_csv('processed_data.csv')
    print(f'✅ Processing complete. Saved to processed_data.csv. Shape: {df.shape}')

if __name__ == '__main__':
    process_data()
