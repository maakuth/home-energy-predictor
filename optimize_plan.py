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


def get_env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ Invalid float for {name}='{raw}', using default {default}")
        return float(default)


def get_env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def load_predictions(file_path='future_predictions.json'):
    with open(file_path, 'r') as f:
        predictions_data = json.load(f)

    predictions = np.array([p['predicted_usage'] for p in predictions_data], dtype=float)
    prediction_timestamps = [datetime.fromisoformat(p['timestamp']) for p in predictions_data]
    prediction_solar = np.array([p['solar_forecast'] for p in predictions_data], dtype=float)

    return predictions_data, predictions, prediction_timestamps, prediction_solar


def align_hourly_prices(raw_today, raw_tomorrow, prediction_timestamps):
    all_raw = raw_today + raw_tomorrow
    if not all_raw:
        return None

    df_prices = pd.DataFrame(all_raw)
    if 'start' not in df_prices.columns or 'value' not in df_prices.columns:
        return None

    df_prices['start'] = pd.to_datetime(df_prices['start'])
    df_prices = df_prices.drop_duplicates(subset='start').set_index('start').sort_index()
    hourly_prices_series = df_prices['value'].resample('1h').mean()

    aligned = []
    for ts in prediction_timestamps:
        if ts in hourly_prices_series.index:
            aligned.append(hourly_prices_series[ts])
        else:
            aligned.append(np.nan)
    return np.array(aligned, dtype=float)


def fetch_market_prices(prediction_timestamps):
    # Primary source: price without tax + grid transfer.
    candidate_sensors = [
        'sensor.current_electricity_market_price',
        'sensor.nordpool_kwh_fi_eur_3_10_0',
        'sensor.nordpool_total',
    ]

    for sensor in candidate_sensors:
        state = get_ha_state(sensor)
        if not state:
            continue

        attrs = state.get('attributes', {})
        raw_today = attrs.get('raw_today', []) or []
        raw_tomorrow = attrs.get('raw_tomorrow', []) or []
        aligned = align_hourly_prices(raw_today, raw_tomorrow, prediction_timestamps)

        if aligned is not None:
            nan_count = int(np.isnan(aligned).sum())
            if nan_count > 0:
                print(f"⚠️ Warning: {nan_count} hours without market data from {sensor}. Padding with local mean.")
                mean_price = np.nanmean(aligned)
                if np.isnan(mean_price):
                    mean_price = 0.0
                aligned = np.nan_to_num(aligned, nan=mean_price)
            return aligned, sensor

        # Fallback: single numeric state (constant across horizon).
        try:
            base_price = float(state.get('state'))
            if np.isfinite(base_price):
                return np.full(len(prediction_timestamps), base_price, dtype=float), sensor
        except (TypeError, ValueError):
            pass

    return None, None


def build_tariff_prices(market_prices):
    grid_transfer = get_env_float('GRID_TRANSFER_EUR_PER_KWH', 0.0)
    electricity_tax = get_env_float('ELECTRICITY_TAX_EUR_PER_KWH', 0.0)
    import_fixed_adders = get_env_float('IMPORT_FIXED_ADDERS_EUR_PER_KWH', 0.0)
    import_vat_multiplier = get_env_float('IMPORT_VAT_MULTIPLIER', 1.0)
    export_deduction = get_env_float('EXPORT_DEDUCTION_EUR_PER_KWH', 0.0)

    market_prices = np.array(market_prices, dtype=float)
    import_unit_prices = (market_prices + grid_transfer + electricity_tax + import_fixed_adders) * import_vat_multiplier
    export_unit_prices = np.maximum(0.0, market_prices - export_deduction)

    return import_unit_prices, export_unit_prices


def plan_battery_dispatch(predictions, solar_array, import_prices, export_prices):
    capacity_kwh = get_env_float('BATTERY_CAPACITY_KWH', 40.0)
    min_soc_pct = get_env_float('BATTERY_MIN_SOC_PCT', 10.0)
    max_soc_pct = get_env_float('BATTERY_MAX_SOC_PCT', 90.0)
    reserve_soc_pct = get_env_float('BATTERY_RESERVE_SOC_PCT', min_soc_pct)
    initial_soc_pct = get_env_float('BATTERY_INITIAL_SOC_PCT', 50.0)
    max_charge_kw = get_env_float('BATTERY_MAX_CHARGE_KW', 10.0)
    max_discharge_kw = get_env_float('BATTERY_MAX_DISCHARGE_KW', 10.0)
    charge_eff = get_env_float('BATTERY_CHARGE_EFFICIENCY', 0.95)
    discharge_eff = get_env_float('BATTERY_DISCHARGE_EFFICIENCY', 0.95)
    allow_export = get_env_bool('BATTERY_ALLOW_EXPORT', True)

    charge_eff = min(max(charge_eff, 0.01), 1.0)
    discharge_eff = min(max(discharge_eff, 0.01), 1.0)
    round_trip_eff = charge_eff * discharge_eff

    min_soc_kwh = capacity_kwh * max(min_soc_pct, reserve_soc_pct) / 100.0
    max_soc_kwh = capacity_kwh * max_soc_pct / 100.0
    soc_kwh = min(max(capacity_kwh * initial_soc_pct / 100.0, min_soc_kwh), max_soc_kwh)

    horizon = len(predictions)
    net_without_battery = np.array(predictions, dtype=float) - np.array(solar_array, dtype=float)

    import_q30 = np.percentile(import_prices, 30)
    import_q70 = np.percentile(import_prices, 70)
    export_q80 = np.percentile(export_prices, 80)

    battery_plan = []

    for i in range(horizon):
        net_load = float(net_without_battery[i])
        current_import = float(import_prices[i])
        current_export = float(export_prices[i])

        future_import = import_prices[i + 1:] if i + 1 < horizon else np.array([current_import])
        future_export = export_prices[i + 1:] if i + 1 < horizon else np.array([current_export])
        best_future_value = max(float(np.max(future_import)), float(np.max(future_export)) if allow_export else -np.inf)

        charge_from_solar = 0.0
        charge_from_grid = 0.0
        discharge_to_load = 0.0
        discharge_to_export = 0.0

        # Available headroom to charge this hour (input kWh before charge efficiency).
        soc_room_kwh = max(0.0, max_soc_kwh - soc_kwh)
        charge_limit_input_kwh = min(max_charge_kw, soc_room_kwh / charge_eff)

        # Available output from battery this hour (kWh delivered after discharge efficiency).
        soc_available_kwh = max(0.0, soc_kwh - min_soc_kwh)
        discharge_limit_output_kwh = min(max_discharge_kw, soc_available_kwh * discharge_eff)

        if net_load < 0 and charge_limit_input_kwh > 0:
            solar_surplus = -net_load
            charge_from_solar = min(solar_surplus, charge_limit_input_kwh)
            soc_kwh += charge_from_solar * charge_eff
            charge_limit_input_kwh -= charge_from_solar

        if net_load > 0 and discharge_limit_output_kwh > 0 and current_import >= import_q70:
            discharge_to_load = min(net_load, discharge_limit_output_kwh)
            soc_kwh -= discharge_to_load / discharge_eff
            discharge_limit_output_kwh -= discharge_to_load

        # Grid charging only if no battery discharge this hour and arbitrage looks profitable.
        profitable_grid_charge = (best_future_value * round_trip_eff) > current_import
        is_cheap_hour = current_import <= import_q30
        if discharge_to_load == 0.0 and charge_limit_input_kwh > 0 and profitable_grid_charge and is_cheap_hour:
            charge_from_grid = charge_limit_input_kwh
            soc_kwh += charge_from_grid * charge_eff

        # Optional export arbitrage from stored energy on high-value hours.
        if (
            allow_export
            and charge_from_solar == 0.0
            and charge_from_grid == 0.0
            and discharge_to_load == 0.0
            and discharge_limit_output_kwh > 0
            and current_export >= export_q80
            and current_export >= float(np.max(future_export))
        ):
            discharge_to_export = discharge_limit_output_kwh
            soc_kwh -= discharge_to_export / discharge_eff

        soc_kwh = min(max(soc_kwh, min_soc_kwh), max_soc_kwh)

        net_after_battery = net_load + charge_from_solar + charge_from_grid - discharge_to_load - discharge_to_export
        grid_import_kwh = max(net_after_battery, 0.0)
        grid_export_kwh = max(-net_after_battery, 0.0)

        no_battery_import = max(net_load, 0.0)
        no_battery_export = max(-net_load, 0.0)

        hour_cost_no_battery = (no_battery_import * current_import) - (no_battery_export * current_export)
        hour_cost_with_battery = (grid_import_kwh * current_import) - (grid_export_kwh * current_export)

        charge_total = charge_from_solar + charge_from_grid
        discharge_total = discharge_to_load + discharge_to_export
        if charge_total > 1e-9:
            battery_action = 'charge'
        elif discharge_total > 1e-9:
            battery_action = 'discharge'
        else:
            battery_action = 'idle'

        battery_plan.append({
            'battery_action': battery_action,
            'battery_charge_kwh': float(charge_total),
            'battery_discharge_kwh': float(discharge_total),
            'battery_power_kw': float(charge_total - discharge_total),
            'soc_kwh': float(soc_kwh),
            'soc_pct': float((soc_kwh / capacity_kwh) * 100.0 if capacity_kwh > 0 else 0.0),
            'grid_import_kwh': float(grid_import_kwh),
            'grid_export_kwh': float(grid_export_kwh),
            'estimated_hour_cost': float(hour_cost_with_battery),
            'estimated_hour_savings': float(hour_cost_no_battery - hour_cost_with_battery),
            'net_load_without_battery_kwh': float(net_load),
        })

    return battery_plan

def optimize():
    print('Loading predictions...')
    try:
        predictions_data, predictions, prediction_timestamps, prediction_solar = load_predictions('future_predictions.json')
    except FileNotFoundError:
        print('Error: future_predictions.json not found. Run predict_future.py first.')
        return

    print('Fetching market prices...')
    market_prices, price_source = fetch_market_prices(prediction_timestamps)
    if market_prices is None:
        print('Error: Could not fetch market prices from Home Assistant sensors.')
        return

    print(f'Using market prices from {price_source}')
    import_prices, export_prices = build_tariff_prices(market_prices)

    solar_array = np.array(prediction_solar, dtype=float)

    HOURS_TO_CHARGE = 4
    cheapest_indices = np.argsort(import_prices)[:HOURS_TO_CHARGE]
    ev_plan = [1 if i in cheapest_indices else 0 for i in range(len(import_prices))]
    
    effective_prices = np.where(solar_array > 0.5, 0.0, import_prices)
    price_threshold = np.percentile(effective_prices, 20)
    heating_plan = [1 if p <= price_threshold else 0 for p in effective_prices]

    battery_plan = plan_battery_dispatch(predictions, solar_array, import_prices, export_prices)

    print(f"\nOptimization Plan from {prediction_timestamps[0]} to {prediction_timestamps[-1]}:")
    print(f"Import Price Threshold (20th percentile effective): {price_threshold:.3f} €/kWh")
    
    final_plan = []
    print('Time | Pred | Market | Import | Export | Solar | Grid In | Grid Out | SOC% | Actions')
    print('-----|------|--------|--------|--------|-------|---------|----------|------|--------')
    for i, ts in enumerate(prediction_timestamps):
        p_pred = float(predictions[i])
        p_market = float(market_prices[i])
        p_import = float(import_prices[i])
        p_export = float(export_prices[i])
        p_solar = solar_array[i]
        b = battery_plan[i]
        
        actions = []
        if ev_plan[i]:
            actions.append('EV_CHARGE')
        if heating_plan[i]:
            actions.append('HEAT_BOOST')
        if p_solar > 0.5:
            actions.append('SOLAR')
        if b['battery_action'] != 'idle':
            actions.append(f"BATTERY_{b['battery_action'].upper()}")
        action_str = ' '.join(actions)
        
        print(
            f"{ts.strftime('%m-%d %H:%M')} | {p_pred:4.1f} | {p_market:6.3f} | {p_import:6.3f} | "
            f"{p_export:6.3f} | {p_solar:5.2f} | {b['grid_import_kwh']:7.2f} | {b['grid_export_kwh']:8.2f} | "
            f"{b['soc_pct']:4.1f} | {action_str}"
        )
        
        final_plan.append({
            'timestamp': ts.isoformat(),
            'predicted_usage': float(p_pred),
            'spot_price': float(p_market),
            'market_base_price': float(p_market),
            'import_unit_price': float(p_import),
            'export_unit_price': float(p_export),
            'solar_forecast': float(p_solar),
            'ev_charge': bool(ev_plan[i]),
            'heat_boost': bool(heating_plan[i]),
            **b,
        })
        
    with open('optimization_plan.json', 'w') as f:
        json.dump(final_plan, f, indent=2)
    print('\n✅ Plan saved to optimization_plan.json')

if __name__ == '__main__':
    optimize()
