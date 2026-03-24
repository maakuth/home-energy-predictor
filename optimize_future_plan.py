import os
import json
import numpy as np
import pandas as pd
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

def optimize():
    print('Loading future predictions...')
    try:
        with open('future_predictions.json', 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        print('Error: future_predictions.json not found. Run predict_future.py first.')
        return

    print('Fetching spot prices...')
    nordpool_state = get_ha_state('sensor.nordpool_kwh_fi_eur_3_10_0')
    if not nordpool_state:
        # Fallback to total if specific sensor fails
        nordpool_state = get_ha_state('sensor.nordpool_total')

    if not nordpool_state:
        print('Error: Could not fetch spot prices.')
        return

    # Process Prices
    # Fetch both today and tomorrow to cover the full prediction range
    raw_prices = []
    
    raw_today = nordpool_state.get('attributes', {}).get('raw_today', [])
    if raw_today:
        raw_prices.extend(raw_today)
        
    raw_tomorrow = nordpool_state.get('attributes', {}).get('raw_tomorrow', [])
    if raw_tomorrow:
        raw_prices.extend(raw_tomorrow)

    if not raw_prices:
        print('Error: No price data available.')
        return

    # Create DataFrame for easier resampling
    try:
        df_prices = pd.DataFrame(raw_prices)
        df_prices['start'] = pd.to_datetime(df_prices['start'])
        df_prices = df_prices.set_index('start')
        
        # Resample to hourly mean
        hourly_prices = df_prices['value'].resample('1h').mean()
    except Exception as e:
        print(f'Error processing prices: {e}')
        return

    # Align predictions with prices
    final_plan = []
    
    print('\nOptimization Plan (Future):')
    print('Time | Pred (kWh) | Price (€) | Solar (kWh) | Actions')
    print('-----|------------|-----------|-------------|--------')
    
    pred_df = pd.DataFrame(predictions)
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
    pred_df = pred_df.set_index('timestamp')
    
    # Use reindex to align prices
    aligned_prices = hourly_prices.reindex(pred_df.index, method='nearest')
    pred_df['spot_price'] = aligned_prices
    
    # Fill missing prices
    pred_df['spot_price'] = pred_df['spot_price'].fillna(method='ffill')
    
    for date, group in pred_df.groupby(pred_df.index.date):
        # Daily logic
        daily_prices = group['spot_price'].values
        daily_solar = group['solar_forecast'].values
        
        # EV Strategy: N cheapest hours in this day's window
        n_hours = min(len(daily_prices), 4)
        cheapest_indices = np.argsort(daily_prices)[:n_hours]
        
        ev_flags = np.zeros(len(daily_prices), dtype=bool)
        ev_flags[cheapest_indices] = True
        
        # Heating Strategy: Effective price logic
        effective_prices = np.where(daily_solar > 0.5, 0.0, daily_prices)
        if len(effective_prices) > 0:
            price_threshold = np.percentile(effective_prices, 20)
        else:
            price_threshold = 0.0
            
        heat_flags = effective_prices <= price_threshold
        
        # Add to final plan
        for i, (ts, row) in enumerate(group.iterrows()):
            actions = []
            if ev_flags[i]: actions.append('⚡ CHARGE')
            if heat_flags[i]: actions.append('🔥 BOOST')
            if row['solar_forecast'] > 0.5: actions.append('☀️ SOLAR')
            
            print(f'{ts} | {row["predicted_usage"]:10.2f} | {row["spot_price"]:9.3f} | {row["solar_forecast"]:11.2f} | {" ".join(actions)}')
            
            final_plan.append({
                'timestamp': ts.isoformat(),
                'hour': ts.hour,
                'predicted_usage': row['predicted_usage'],
                'spot_price': row['spot_price'],
                'solar_forecast': row['solar_forecast'],
                'ev_charge': bool(ev_flags[i]),
                'heat_boost': bool(heat_flags[i])
            })

    # Save plan
    with open('optimization_plan.json', 'w') as f:
        json.dump(final_plan, f, indent=2)
    print('\n✅ Plan saved to optimization_plan.json')

if __name__ == '__main__':
    optimize()
