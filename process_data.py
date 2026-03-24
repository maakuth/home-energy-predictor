import pandas as pd
import numpy as np

def process_data():
    print('Loading raw data...')
    try:
        df = pd.read_csv('raw_data.csv', index_col=0)
    except FileNotFoundError:
        print('Error: raw_data.csv not found.')
        return
    
    df.index = pd.to_datetime(df.index)
    
    print('Denoising data...')
    numeric_df = df.select_dtypes(include=[np.number])
    rolled = numeric_df.rolling(window=3, center=True).median()
    df[numeric_df.columns] = rolled.fillna(numeric_df)
    
    power_cols = [c for c in df.columns if 'power' in c or 'teho' in c or 'energy' in c]
    for col in power_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    
    if 'outside_temp' in df.columns:
        df['outside_temp'] = df['outside_temp'].clip(lower=-50, upper=50)
    if 'accumulator_temp' in df.columns:
        df['accumulator_temp'] = df['accumulator_temp'].clip(lower=0, upper=100)

    print('Applying fireplace logic...')
    if 'accumulator_temp' in df.columns:
        df['acc_roc'] = df['accumulator_temp'].diff()
        hp_cols = ['gshp_power', 'aahp_living_power', 'aahp_cabin_power']
        available_hp = [c for c in hp_cols if c in df.columns]
        df['total_hp_power'] = df[available_hp].sum(axis=1)
        df['is_fireplace_active'] = ((df['acc_roc'] > 0.3) & (df['total_hp_power'] < 0.5)).astype(int)
        df['is_fireplace_lag1'] = df['is_fireplace_active'].shift(1)
    
    print('Adding temporal features...')
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df = df.dropna()
    df.to_csv('processed_data.csv')
    print(f'✅ Processing complete. Saved to processed_data.csv. Shape: {df.shape}')

if __name__ == '__main__':
    process_data()
