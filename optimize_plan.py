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
        with open('future_predictions.json', 'r') as f:
            predictions_data = json.load(f)
        predictions = [p['predicted_usage'] for p in predictions_data]
        prediction_timestamps = [datetime.fromisoformat(p['timestamp']) for p in predictions_data]
        prediction_solar = [p['solar_forecast'] for p in predictions_data]
    except FileNotFoundError:
        print('Error: future_predictions.json not found. Run predict_future.py first.')
        return

    print("Fetching spot prices...")
    nordpool_state = get_ha_state('sensor.nordpool_kwh_fi_eur_3_10_0')
    if not nordpool_state:
        nordpool_state = get_ha_state('sensor.nordpool_total')

    if not nordpool_state:
        print("Error: Could not fetch spot prices.")
        return

    raw_today = nordpool_state.get('attributes', {}).get('raw_today', [])
    raw_tomorrow = nordpool_state.get('attributes', {}).get('raw_tomorrow', [])

    if not raw_today and not raw_tomorrow:
        print("Error: No price data available at all.")
        return

    try:
        all_raw = raw_today + raw_tomorrow
        df_prices = pd.DataFrame(all_raw)
        df_prices['start'] = pd.to_datetime(df_prices['start'])
        df_prices = df_prices.drop_duplicates(subset='start').set_index('start').sort_index()
        hourly_prices_series = df_prices['value'].resample('1h').mean()
        
        prices = []
        for ts in prediction_timestamps:
            if ts in hourly_prices_series.index:
                prices.append(hourly_prices_series[ts])
            else:
                prices.append(np.nan)
        prices = np.array(prices)
        
        nan_count = np.isnan(prices).sum()
        if nan_count > 0:
            print(f"⚠️ Warning: {nan_count} hours without price data. Padding with mean.")
            mean_price = np.nanmean(prices)
            prices = np.nan_to_num(prices, nan=mean_price)
    except Exception as e:
        print(f"Error processing prices: {e}")
        return

    solar_array = np.array(prediction_solar)

    HOURS_TO_CHARGE = 4
    cheapest_indices = np.argsort(prices)[:HOURS_TO_CHARGE]
    ev_plan = [1 if i in cheapest_indices else 0 for i in range(len(prices))]
    
    effective_prices = np.where(solar_array > 0.5, 0.0, prices)
    price_threshold = np.percentile(effective_prices, 20)
    heating_plan = [1 if p <= price_threshold else 0 for p in effective_prices]

    print(f"\nOptimization Plan from {prediction_timestamps[0]} to {prediction_timestamps[-1]}:")
    print(f"Price Threshold (20th percentile effective): {price_threshold:.3f} €/kWh")
    
    final_plan = []
    print("Time | Pred (kWh) | Price (€) | Solar (kWh) | Actions")
    print("-----|------------|-----------|-------------|--------")
    for i, ts in enumerate(prediction_timestamps):
        p_pred = predictions[i]
        p_price = prices[i]
        p_solar = solar_array[i]
        
        actions = []
        if ev_plan[i]: actions.append('⚡ CHARGE')
        if heating_plan[i]: actions.append('🔥 BOOST')
        if p_solar > 0.5: actions.append('☀️ SOLAR')
        action_str = ' '.join(actions)
        
        print(f"{ts.strftime('%m-%d %H:%M')} | {p_pred:10.2f} | {p_price:9.3f} | {p_solar:11.2f} | {action_str}")
        
        final_plan.append({
            'timestamp': ts.isoformat(),
            'predicted_usage': float(p_pred),
            'spot_price': float(p_price),
            'solar_forecast': float(p_solar),
            'ev_charge': bool(ev_plan[i]),
            'heat_boost': bool(heating_plan[i])
        })
        
    with open('optimization_plan.json', 'w') as f:
        json.dump(final_plan, f, indent=2)
    print('\n✅ Plan saved to optimization_plan.json')

if __name__ == '__main__':
    optimize()
