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
    print('Loading predictions...')
    try:
        predictions = np.load('tomorrow_predictions.npy')
    except FileNotFoundError:
        print('Error: tomorrow_predictions.npy not found. Run predict_tomorrow.py first.')
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
    # Data is 15-min intervals. We need hourly.
    raw_tomorrow = nordpool_state.get('attributes', {}).get('raw_tomorrow', [])
    
    if not raw_tomorrow:
        print('⚠️ Warning: No price data for tomorrow. Using today\'s prices as fallback.')
        raw_tomorrow = nordpool_state.get('attributes', {}).get('raw_today', [])

    if not raw_tomorrow:
        print('Error: No price data available at all.')
        return

    # Create DataFrame for easier resampling
    try:
        df_prices = pd.DataFrame(raw_tomorrow)
        df_prices['start'] = pd.to_datetime(df_prices['start'])
        df_prices = df_prices.set_index('start')
        
        # Resample to hourly mean
        hourly_prices = df_prices['value'].resample('1h').mean()
        
        # Ensure we have 24 hours of prices aligning with predictions
        prices = hourly_prices.values
        if len(prices) > 24:
            prices = prices[:24]
        elif len(prices) < 24:
            print(f'⚠️ Warning: Only have {len(prices)} hours of price data. Padding with mean.')
            pad = np.full(24 - len(prices), prices.mean())
            prices = np.concatenate([prices, pad])
    except Exception as e:
        print(f'Error processing prices: {e}')
        return

    # --- Optimization Logic ---
    
    # 1. EV Strategy: Charge during N cheapest hours
    # Assume we need 4 hours of charging
    HOURS_TO_CHARGE = 4
    # Get indices of sorted prices (ascending)
    cheapest_indices = np.argsort(prices)[:HOURS_TO_CHARGE]
    ev_plan = [1 if i in cheapest_indices else 0 for i in range(24)]
    
    # 2. Thermal Strategy: Boost if price < 20th percentile
    price_threshold = np.percentile(prices, 20)
    heating_plan = [1 if p <= price_threshold else 0 for p in prices]
    
    # Output Plan
    print(f'\nOptimization Plan for Tomorrow:')
    print(f'Price Threshold (20th percentile): {price_threshold:.3f} €/kWh')
    
    final_plan = []
    print('Hour | Pred (kWh) | Price (€) | Actions')
    print('-----|------------|-----------|--------')
    for h in range(24):
        p_pred = predictions[h] if h < len(predictions) else 0
        p_price = prices[h]
        actions = []
        if ev_plan[h]: actions.append('⚡ CHARGE')
        if heating_plan[h]: actions.append('🔥 BOOST')
        action_str = ' '.join(actions)
        
        print(f'{h:02d}:00 | {p_pred:10.2f} | {p_price:9.3f} | {action_str}')
        
        final_plan.append({
            'hour': h,
            'predicted_usage': float(p_pred),
            'spot_price': float(p_price),
            'ev_charge': bool(ev_plan[h]),
            'heat_boost': bool(heating_plan[h])
        })
        
    # Save plan
    with open('optimization_plan.json', 'w') as f:
        json.dump(final_plan, f, indent=2)
    print('\n✅ Plan saved to optimization_plan.json')

if __name__ == '__main__':
    optimize()
