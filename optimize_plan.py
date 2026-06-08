import os
import json
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.ha_utils import get_ha_state, parse_ha_bool
from utils.price_utils import fetch_market_prices, align_interval_prices
from utils.git_utils import get_model_version
from utils.sqlite_utils import get_db_connection, get_db_path
from utils.db_utils import fetch_states_history

load_dotenv(override=True)

def get_plan_interval_minutes():
    return int(os.getenv('PLAN_INTERVAL_MINUTES', '15'))

def get_plan_interval_hours():
    return max(get_plan_interval_minutes(), 1) / 60.0

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


def load_predictions(file_path=None, sarima_path=None):
    # Support environment variable overrides for testing
    if file_path is None:
        file_path = os.getenv('TEST_PREDICTIONS_FILE', 'future_predictions.json')
    if sarima_path is None:
        sarima_path = os.getenv('TEST_SARIMA_FILE', 'sarimax_predictions.json')
    with open(file_path, 'r') as f:
        xgb_data = json.load(f)

    # Convert XGBoost data to DataFrame for easier alignment
    df_xgb = pd.DataFrame(xgb_data)
    df_xgb['timestamp'] = pd.to_datetime(df_xgb['timestamp'], utc=True)
    df_xgb = df_xgb.set_index('timestamp')

    # Default to 100% XGBoost if SARIMA is missing
    final_baseload = df_xgb['predicted_baseload'].copy()
    sarima_lower = pd.Series(np.nan, index=df_xgb.index)
    sarima_upper = pd.Series(np.nan, index=df_xgb.index)

    if os.path.exists(sarima_path):
        try:
            with open(sarima_path, 'r') as f:
                sarima_data = json.load(f)
            
            df_sarima = pd.DataFrame(sarima_data)
            df_sarima['timestamp'] = pd.to_datetime(df_sarima['timestamp'], utc=True)
            df_sarima = df_sarima.set_index('timestamp')
            
            # Align SARIMA (15min) to XGBoost (1min) using interpolation
            # Only blend for the overlapping period
            common_idx = df_xgb.index.union(df_sarima.index).sort_values()
            df_sarima_resampled = df_sarima[['predicted_baseload', 'lower_95', 'upper_95']].reindex(common_idx).interpolate(method='time').reindex(df_xgb.index)
            
            # Weighting: 50% SARIMA, 50% XGBoost
            # Balanced approach for seasonal stability (SARIMA) and feature awareness (XGBoost)
            print(f"Blending XGBoost with SARIMA (50/50 weight)...")
            final_baseload = (0.5 * df_xgb['predicted_baseload']) + (0.5 * df_sarima_resampled['predicted_baseload'])
            # Fill any NaNs (if SARIMA horizon is shorter) with XGBoost
            final_baseload = final_baseload.fillna(df_xgb['predicted_baseload'])
            
            sarima_lower = df_sarima_resampled['lower_95']
            sarima_upper = df_sarima_resampled['upper_95']
            
        except Exception as e:
            print(f"⚠️ Error blending SARIMA: {e}. Falling back to 100% XGBoost.")

    predictions = final_baseload.values.astype(float)
    prediction_timestamps = df_xgb.index.to_pydatetime()
    prediction_solar = df_xgb['solar_forecast'].values.astype(float)

    return xgb_data, predictions, prediction_timestamps, prediction_solar, sarima_lower, sarima_upper


def build_tariff_prices(market_prices, is_inclusive=False):
    grid_transfer = get_env_float('GRID_TRANSFER_EUR_PER_KWH', 0.0)
    electricity_tax = get_env_float('ELECTRICITY_TAX_EUR_PER_KWH', 0.0)
    import_fixed_adders = get_env_float('IMPORT_FIXED_ADDERS_EUR_PER_KWH', 0.0)
    import_vat_multiplier = get_env_float('IMPORT_VAT_MULTIPLIER', 1.0)
    export_deduction = get_env_float('EXPORT_DEDUCTION_EUR_PER_KWH', 0.0)

    market_prices = np.array(market_prices, dtype=float)
    
    # If the source is already inclusive, we don't add transfer and tax again
    effective_transfer = 0.0 if is_inclusive else grid_transfer
    effective_tax = 0.0 if is_inclusive else electricity_tax
    
    import_unit_prices = (market_prices + effective_transfer + effective_tax + import_fixed_adders) * import_vat_multiplier
    export_unit_prices = np.maximum(0.0, market_prices - export_deduction)

    return import_unit_prices, export_unit_prices


def is_battery_enabled():
    """
    Check if battery optimization should be used.
    
    Battery can be disabled for testing/degradation mode via:
    - HEPO_DISABLE_BATTERY=1 in .env
    - Missing sensor.be_soc entity in Home Assistant
    
    Returns:
        bool: True if battery optimization should be used
    """
    # Check environment variable (for testing/degradation mode)
    if get_env_bool('HEPO_DISABLE_BATTERY', False):
        print("⚠️ Battery optimization DISABLED (HEPO_DISABLE_BATTERY=1)")
        return False
    
    # Check if battery SoC sensor exists
    batt_state = get_ha_state('sensor.be_soc')
    if not batt_state or batt_state.get('state') in ['unknown', 'unavailable', None]:
        print("⚠️ Battery SoC sensor unavailable - disabling battery optimization")
        return False
    
    return True


def _has_spill_risk(soc_kwh, start_idx, horizon, net_without_battery, min_soc_kwh, max_soc_kwh,
                     charge_eff, discharge_eff, max_charge_kw, max_discharge_kw, interval_hours):
    """Simulate battery from start_idx forward with no grid charging.
    Returns True if battery would hit max_soc with solar still spilling."""
    sim_soc = soc_kwh
    for j in range(start_idx, horizon):
        net_j = float(net_without_battery[j])
        if net_j < 0:
            charge = min(-net_j, (max_soc_kwh - sim_soc) / charge_eff,
                         max_charge_kw * interval_hours)
            sim_soc += charge * charge_eff
            if sim_soc >= max_soc_kwh - 1e-6 and net_j + charge < -1e-6:
                return True
        else:
            discharge = min(net_j, (sim_soc - min_soc_kwh) * discharge_eff,
                            max_discharge_kw * interval_hours)
            sim_soc -= discharge / discharge_eff
    return False


def _compute_opportunity_cost(sim_soc, start_idx, horizon, net_without_battery,
                              import_prices, export_prices, allow_export,
                              min_soc_kwh, max_soc_kwh,
                              charge_eff, discharge_eff,
                              max_charge_kw, max_discharge_kw, interval_hours):
    """Forward-simulate battery greedily from start_idx and compute the mean
    value (EUR per stored kWh) of future discharges.

    Returns 0.0 if the battery never discharges in the simulation,
    which signals the caller to fall back to existing logic.
    """
    total_revenue = 0.0
    total_discharged_kwh = 0.0
    soc = sim_soc

    for j in range(start_idx, horizon):
        net_j = float(net_without_battery[j])

        # 1. Charge from solar surplus
        if net_j < 0:
            solar_surplus = -net_j
            room = max(0.0, max_soc_kwh - soc)
            charge_limit = min(max_charge_kw * interval_hours, room / charge_eff)
            charge = min(solar_surplus, charge_limit)
            soc += charge * charge_eff

        # 2. Discharge to load
        if net_j > 0:
            soc_available = max(0.0, soc - min_soc_kwh)
            discharge_limit = min(max_discharge_kw * interval_hours, soc_available * discharge_eff)
            discharge_to_load = min(net_j, discharge_limit)
            if discharge_to_load > 0:
                revenue = discharge_to_load * float(import_prices[j])
                total_revenue += revenue
                total_discharged_kwh += discharge_to_load
                soc -= discharge_to_load / discharge_eff

        # 3. Discharge to export (remaining inverter capacity)
        if allow_export and net_j >= 0:
            soc_available = max(0.0, soc - min_soc_kwh)
            remaining_limit = min(max_discharge_kw * interval_hours, soc_available * discharge_eff)
            discharge_to_export = max(0.0, remaining_limit)
            if discharge_to_export > 0:
                revenue = discharge_to_export * float(export_prices[j])
                total_revenue += revenue
                total_discharged_kwh += discharge_to_export
                soc -= discharge_to_export / discharge_eff

    if total_discharged_kwh < 1e-9:
        return 0.0
    return total_revenue / total_discharged_kwh


def plan_no_battery_dispatch(predictions, solar_array, import_prices, export_prices, committed_load_kwh=None):
    """
    Create a no-op battery plan when battery is disabled.
    
    Returns a list of plans with no battery action, allowing the system
    to continue without battery optimization during testing/degradation.
    """
    horizon = len(predictions)
    interval_hours = get_plan_interval_hours()
    
    if committed_load_kwh is None:
        committed_load_kwh = np.zeros(horizon)
    
    no_battery_plan = []
    for i in range(horizon):
        net_load = float(predictions[i]) - float(solar_array[i])
        committed = float(committed_load_kwh[i]) if i < len(committed_load_kwh) else 0.0
        total_net = net_load + committed
        grid_import_kwh = max(total_net, 0.0)
        grid_export_kwh = max(-total_net, 0.0)
        hour_cost = (grid_import_kwh * import_prices[i]) - (grid_export_kwh * export_prices[i])
        
        no_battery_plan.append({
            'battery_action': 'idle',
            'battery_power_kw': 0.0,
            'charge_from_solar_kwh': 0.0,
            'charge_from_grid_kwh': 0.0,
            'discharge_to_load_kwh': 0.0,
            'discharge_to_export_kwh': 0.0,
            'soc_kwh': 0.0,
            'soc_pct': 0.0,
            'grid_import_kwh': float(grid_import_kwh),
            'grid_export_kwh': float(grid_export_kwh),
            'estimated_hour_cost': float(hour_cost),
            'estimated_hour_savings': 0.0,
            'net_load_without_battery_kwh': float(net_load),
        })
    
    return no_battery_plan


def plan_battery_dispatch(predictions, solar_array, import_prices, export_prices, committed_load_kwh=None, allow_export=None):
    capacity_kwh = get_env_float('BATTERY_CAPACITY_KWH', 40.0)
    min_soc_pct = get_env_float('BATTERY_MIN_SOC_PCT', 10.0)
    max_soc_pct = get_env_float('BATTERY_MAX_SOC_PCT', 90.0)
    reserve_soc_pct = get_env_float('BATTERY_RESERVE_SOC_PCT', min_soc_pct)
    
    # Try to get current House Battery SoC from HA
    batt_state = get_ha_state('sensor.be_soc')
    initial_soc_pct = get_env_float('BATTERY_INITIAL_SOC_PCT', 50.0)
    if batt_state and batt_state.get('state') not in ['unknown', 'unavailable']:
        try:
            initial_soc_pct = float(batt_state['state'])
            print(f"House Battery SoC from HA: {initial_soc_pct}%")
        except (ValueError, TypeError):
            pass
    else:
        print(f"House Battery SoC using .env/default: {initial_soc_pct}%")

    print(f"Battery Config: Capacity={capacity_kwh}kWh, MinSOC={min_soc_pct}%, Reserve={reserve_soc_pct}%")
    
    max_charge_kw = get_env_float('BATTERY_MAX_CHARGE_KW', 10.0)
    max_discharge_kw = get_env_float('BATTERY_MAX_DISCHARGE_KW', 10.0)
    charge_eff = get_env_float('BATTERY_CHARGE_EFFICIENCY', 0.95)
    discharge_eff = get_env_float('BATTERY_DISCHARGE_EFFICIENCY', 0.95)
    
    if allow_export is None:
        allow_export_entity = os.getenv('BATTERY_ALLOW_EXPORT_ENTITY', 'input_boolean.battery_allow_export')
        ha_state = get_ha_state(allow_export_entity)
        allow_export = parse_ha_bool(ha_state, default=get_env_bool('BATTERY_ALLOW_EXPORT', True))
        print(f"Battery allow_export from HA ({allow_export_entity}): {allow_export}")
    
    # Grid connection limit (3-phase, 230V)
    main_fuse_a = get_env_float('MAIN_FUSE_SIZE_A', 25.0)
    max_grid_import_kw = main_fuse_a * 3 * 0.230
    
    interval_hours = get_plan_interval_hours()

    charge_eff = min(max(charge_eff, 0.01), 1.0)
    discharge_eff = min(max(discharge_eff, 0.01), 1.0)

    min_soc_kwh = capacity_kwh * max(min_soc_pct, reserve_soc_pct) / 100.0
    max_soc_kwh = capacity_kwh * max(max_soc_pct, 0.0) / 100.0
    soc_kwh = min(max(capacity_kwh * initial_soc_pct / 100.0, min_soc_kwh), max_soc_kwh)

    horizon = len(predictions)
    net_without_battery = np.array(predictions, dtype=float) - np.array(solar_array, dtype=float)
    
    if committed_load_kwh is None:
        committed_load_kwh = np.zeros(horizon)

    battery_plan = []

    for i in range(horizon):
        net_load = float(net_without_battery[i])
        current_import = float(import_prices[i])
        current_export = float(export_prices[i])

        future_import = import_prices[i + 1:] if i + 1 < horizon else np.array([current_import])
        future_export = export_prices[i + 1:] if i + 1 < horizon else np.array([current_export])
        best_future_value = max(float(np.max(future_import)), float(np.max(future_export)) if allow_export else -np.inf)

        if i + 1 < horizon:
            min_future_import = float(np.min(future_import))
        else:
            min_future_import = current_import

        # 1. Charge from solar surplus (always first priority)
        charge_from_solar = 0.0
        if net_load < 0:
            solar_surplus = -net_load
            soc_room_kwh = max(0.0, max_soc_kwh - soc_kwh)
            charge_limit_input_kwh = min(max_charge_kw * interval_hours, soc_room_kwh / charge_eff)
            charge_from_solar = min(solar_surplus, charge_limit_input_kwh)
            soc_kwh += charge_from_solar * charge_eff

        # Compute opportunity cost of keeping energy vs. discharging now
        opportunity_cost = _compute_opportunity_cost(
            soc_kwh, i + 1, horizon, net_without_battery,
            import_prices, export_prices, allow_export,
            min_soc_kwh, max_soc_kwh,
            charge_eff, discharge_eff,
            max_charge_kw, max_discharge_kw, interval_hours
        )

        # 2. Discharge to load (only when current price is at least as good as
        # the opportunity cost of keeping the energy)
        discharge_to_load = 0.0
        if opportunity_cost > 0.0:
            should_discharge = current_import >= opportunity_cost
        else:
            should_discharge = current_import >= best_future_value
        if net_load > 0 and should_discharge:
            soc_available_kwh = max(0.0, soc_kwh - min_soc_kwh)
            discharge_limit_output_kwh = min(max_discharge_kw * interval_hours, soc_available_kwh * discharge_eff)
            discharge_to_load = min(net_load, discharge_limit_output_kwh)
            soc_kwh -= discharge_to_load / discharge_eff

        # 3. Discharge to export (when export price is best in horizon, or when
        # export-now is at least as valuable as the opportunity cost of keeping it)
        discharge_to_export = 0.0
        if allow_export and charge_from_solar == 0.0:
            soc_available_kwh = max(0.0, soc_kwh - min_soc_kwh)
            total_discharge_limit = min(max_discharge_kw * interval_hours, soc_available_kwh * discharge_eff)
            remaining_capacity = max(0.0, total_discharge_limit - discharge_to_load)

            is_best_export = current_export >= float(np.max(future_export))
            round_trip_eff = charge_eff * discharge_eff
            if opportunity_cost > 0.0:
                is_export_arbitrage = (current_export * discharge_eff) >= opportunity_cost
            else:
                is_export_arbitrage = current_export > (min_future_import / round_trip_eff)

            if remaining_capacity > 0 and (is_best_export or is_export_arbitrage):
                discharge_to_export = remaining_capacity
                soc_kwh -= discharge_to_export / discharge_eff

        # 4. Grid charge (only when profitable, no cheaper future import, and solar won't fill the battery)
        charge_from_grid = 0.0

        # Lookahead for expected solar surplus
        lookahead_steps = min(int(24 / interval_hours), horizon - i - 1)
        if lookahead_steps > 0:
            future_net = net_without_battery[i+1 : i+1+lookahead_steps]
            expected_solar_surplus_kwh = np.sum(np.maximum(0, -future_net)) * interval_hours
        else:
            expected_solar_surplus_kwh = 0.0

        remaining_room_kwh = max(0.0, max_soc_kwh - soc_kwh)
        solar_can_fill = expected_solar_surplus_kwh >= (remaining_room_kwh / charge_eff)

        # Grid capacity calculation
        committed = float(committed_load_kwh[i]) if i < len(committed_load_kwh) else 0.0
        existing_grid_import = max(net_load - discharge_to_load - discharge_to_export + committed, 0.0)
        available_grid_kwh = max(0.0, max_grid_import_kw * interval_hours - existing_grid_import)

        # Forward simulation: would adding grid charge now cause solar spill later?
        spill_risk = _has_spill_risk(soc_kwh, i, horizon, net_without_battery, min_soc_kwh, max_soc_kwh,
                                      charge_eff, discharge_eff, max_charge_kw, max_discharge_kw, interval_hours)

        profitable_grid_charge = (best_future_value * charge_eff) > current_import
        is_cheapest_window = current_import <= min_future_import + 1e-9

        if (not solar_can_fill and profitable_grid_charge and is_cheapest_window
            and charge_from_solar == 0.0 and discharge_to_load == 0.0
            and discharge_to_export == 0.0 and not spill_risk):
            soc_room_kwh = max(0.0, max_soc_kwh - soc_kwh)
            charge_limit_input_kwh = min(max_charge_kw * interval_hours, soc_room_kwh / charge_eff)
            max_grid_charge_kwh = min(charge_limit_input_kwh, available_grid_kwh)
            if max_grid_charge_kwh > 1e-4:
                charge_from_grid = max_grid_charge_kwh
                soc_kwh += charge_from_grid * charge_eff

        soc_kwh = min(max(soc_kwh, min_soc_kwh), max_soc_kwh)

        # Total grid exchange including committed loads (EV, Leaf)
        total_net_after_battery = net_load + charge_from_solar + charge_from_grid - discharge_to_load - discharge_to_export + committed
        grid_import_kwh = max(total_net_after_battery, 0.0)
        grid_export_kwh = max(-total_net_after_battery, 0.0)

        no_battery_import = max(net_load + committed, 0.0)
        no_battery_export = max(-(net_load + committed), 0.0)

        hour_cost_no_battery = (no_battery_import * current_import) - (no_battery_export * current_export)
        hour_cost_with_battery = (grid_import_kwh * current_import) - (grid_export_kwh * current_export)

        charge_total = charge_from_solar + charge_from_grid
        discharge_total = discharge_to_load + discharge_to_export

        if charge_from_solar > 1e-9 and charge_from_grid > 1e-9:
            battery_action = 'charge_mixed'
        elif charge_from_solar > 1e-9:
            battery_action = 'charge_solar'
        elif charge_from_grid > 1e-9:
            battery_action = 'charge_grid'
        elif discharge_to_load > 1e-9 and discharge_to_export > 1e-9:
            battery_action = 'discharge_mixed'
        elif discharge_to_load > 1e-9:
            battery_action = 'discharge_load'
        elif discharge_to_export > 1e-9:
            battery_action = 'discharge_export'
        else:
            battery_action = 'idle'

        battery_plan.append({
            'battery_action': battery_action,
            'battery_power_kw': float((charge_total - discharge_total) / interval_hours),
            'charge_from_solar_kwh': float(charge_from_solar),
            'charge_from_grid_kwh': float(charge_from_grid),
            'discharge_to_load_kwh': float(discharge_to_load),
            'discharge_to_export_kwh': float(discharge_to_export),
            'soc_kwh': float(soc_kwh),
            'soc_pct': float((soc_kwh / capacity_kwh) * 100.0 if capacity_kwh > 0 else 0.0),
            'grid_import_kwh': float(grid_import_kwh),
            'grid_export_kwh': float(grid_export_kwh),
            'estimated_hour_cost': float(hour_cost_with_battery),
            'estimated_hour_savings': float(hour_cost_no_battery - hour_cost_with_battery),
            'net_load_without_battery_kwh': float(net_load),
        })

    return battery_plan


def plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, export_prices=None, solar_forecast_kw=None):
    # Constants/Defaults (can be overridden by .env)
    p_min = get_env_float('GSHP_POWER_MIN_KW', 3.4)
    p_max = get_env_float('GSHP_POWER_MAX_KW', 4.2)
    # Maintain fallback for GSHP_ELECTRIC_POWER_KW if both are equal (no ramp)
    if 'GSHP_ELECTRIC_POWER_KW' in os.environ and 'GSHP_POWER_MIN_KW' not in os.environ and 'GSHP_POWER_MAX_KW' not in os.environ:
        p_min = p_max = get_env_float('GSHP_ELECTRIC_POWER_KW', 4.0)

    cop = get_env_float('GSHP_COP', 3.5)
    heating_efficiency = get_env_float('GSHP_HEATING_EFFICIENCY', 1.0)
    reservoir_l = get_env_float('GSHP_RESERVOIR_LITERS', 500)
    kwh_per_degree = (reservoir_l * 4.18) / 3600.0 
    
    min_temp = get_env_float('GSHP_MIN_TEMP', 42.0)
    max_temp = get_env_float('GSHP_MAX_TEMP', 55.0)
    heat_loss_k = get_env_float('GSHP_HEAT_LOSS_K', 0.135)
    baseline_demand_kw = get_env_float('GSHP_BASELINE_DEMAND_KW', 1.0)
    sauna_demand_kw = get_env_float('SAUNA_HOT_WATER_DEMAND_KW', 6.0)
    
    initial_temp = get_env_float('GSHP_INITIAL_TEMP', 50.0)
    is_hp_running = get_env_bool('GSHP_IS_RUNNING', False)
    layering_drop = get_env_float('GSHP_INITIAL_TEMP_DROP', 3.0)

    # Strategic stop parameters
    stop_diff_threshold = get_env_float('GSHP_STRATEGIC_STOP_DIFF_EUR', 0.02)
    # Don't strategically stop if we are too close to min_temp
    stop_temp_buffer = 1.0 

    horizon = len(prediction_timestamps)
    interval_h = get_plan_interval_hours()
    current_temp = initial_temp
    gshp_plan = []

    # Solar-aware pricing for decisions: 
    # The cost of using solar is the opportunity cost (lost export revenue)
    if solar_forecast_kw is not None and export_prices is not None:
        effective_prices = []
        for i in range(horizon):
            solar_kw = solar_forecast_kw[i]
            if solar_kw >= p_max:
                # Fully covered by solar. Cost = export price (opportunity cost)
                effective_prices.append(export_prices[i])
            elif solar_kw > 0:
                # Partially covered. Weighted average of export and import prices
                cost = (solar_kw * export_prices[i] + (p_max - solar_kw) * import_prices[i]) / p_max
                effective_prices.append(max(0.0, cost))
            else:
                effective_prices.append(import_prices[i])
        effective_prices = np.array(effective_prices)
    else:
        effective_prices = np.array(import_prices)

    # 8-hour lookahead for optimization
    lookahead_intervals = int(8.0 / interval_h)
    # 2-hour lookahead for solar-heavy decisions
    solar_lookahead = int(2.0 / interval_h)

    for i in range(horizon):
        price = effective_prices[i]
        o_temp = outside_temps[i]
        is_sauna = is_sauna_active[i]

        # Calculate base heat demand (house loss) plus baseline for DHW/circulation/standby
        demand_kw = baseline_demand_kw + max(0, (20.0 - o_temp) * heat_loss_k)
        # Add sauna-induced hot water demand
        if is_sauna:
            demand_kw += sauna_demand_kw

        # Decide if we should START/STAY ON
        if is_hp_running:
            # 1. Hardware Stop
            if current_temp >= max_temp:
                is_hp_running = False
            else:
                # 2. Strategic Stop Lookahead
                # If we are safely above min_temp, check if we should stop to wait for cheaper price
                if current_temp > (min_temp + stop_temp_buffer):
                    # Find cheapest price before we WOULD HAVE to restart if we stopped now
                    temp_sim = current_temp
                    intervals_to_min = horizon - i
                    for j in range(i, min(i + lookahead_intervals, horizon)):
                        o_j = outside_temps[j]
                        d_j = max(0, (20.0 - o_j) * heat_loss_k)
                        if is_sauna_active[j]:
                            d_j += sauna_demand_kw
                        temp_sim -= (d_j * interval_h) / kwh_per_degree
                        if temp_sim <= min_temp:
                            intervals_to_min = j - i
                            break

                    if intervals_to_min >= 1: # Allow stopping even for 1 interval if price is better
                        window_prices = effective_prices[i : i + intervals_to_min + 1]
                        min_price_in_window = np.min(window_prices)
                        if price >= (min_price_in_window + stop_diff_threshold):
                            is_hp_running = False

        if not is_hp_running:
            # Check if we MUST start because we are at min_temp
            should_start = (current_temp <= min_temp)
            # Strategic Buffer/Pre-heating
            # Fill more aggressively if we have solar (effective price is lower than import)
            has_solar = (price < import_prices[i])
            buffer_margin = 0.0 if has_solar else 1.5
            
            if not should_start and current_temp < (max_temp - buffer_margin):
                # Adaptive lookahead: if we have solar, don't wait for absolute minimum 8h away.
                # Just check if now is the cheapest in the next 2 hours.
                l_window = solar_lookahead if has_solar else lookahead_intervals
                window_prices = effective_prices[i : min(i + l_window, horizon)]
                cheapest_in_window = np.min(window_prices)
                if price <= cheapest_in_window:
                    should_start = True

            if should_start:
                is_hp_running = True
                current_temp -= layering_drop

        # Update temperature with hardware-limit awareness
        if is_hp_running:
            # Calculate current electric power based on temperature ramp
            # Linear ramp from p_min at min_temp to p_max at max_temp
            if max_temp > min_temp:
                clamped_temp = max(min_temp, min(max_temp, current_temp))
                current_electric_kw = p_min + (p_max - p_min) * (clamped_temp - min_temp) / (max_temp - min_temp)
            else:
                current_electric_kw = p_max
            
            current_heat_kw = current_electric_kw * cop
        else:
            current_electric_kw = 0
            current_heat_kw = 0

        net_heat_kw = (current_heat_kw * heating_efficiency) - demand_kw
        temp_delta = (net_heat_kw * interval_h) / kwh_per_degree
        
        new_temp = current_temp + temp_delta
        
        # Hardware Auto-Stop Logic
        actual_electric_kw = current_electric_kw if is_hp_running else 0
        if is_hp_running and new_temp > max_temp:
            temp_gain_needed = max_temp - current_temp
            total_potential_gain = new_temp - current_temp
            if total_potential_gain > 0:
                fraction_run = max(0, min(1, temp_gain_needed / total_potential_gain))
                actual_electric_kw = current_electric_kw * fraction_run
            
            new_temp = max_temp
            is_hp_running = False
            
        current_temp = new_temp
        
        gshp_plan.append({
            'gshp_intent': 'START' if (actual_electric_kw > 0) else 'STOP',
            'gshp_temp_sim': float(current_temp),
            'gshp_electric_kw': float(actual_electric_kw)
        })
        
    return gshp_plan


def optimize():
    print('Loading predictions...')
    try:
        predictions_data, predictions, prediction_timestamps, prediction_solar, sarima_lower, sarima_upper = load_predictions()
    except FileNotFoundError:
        print('Error: future_predictions.json not found. Run predict_future.py first.')
        return

    print('Fetching market prices...')
    market_prices, is_fallback_price, price_source, is_inclusive = fetch_market_prices(prediction_timestamps, get_plan_interval_minutes())
    if market_prices is None:
        print('Error: Could not fetch market prices from Home Assistant sensors.')
        return

    print(f'Using market prices from {price_source} (Inclusive of fees: {is_inclusive})')
    import_prices, export_prices = build_tariff_prices(market_prices, is_inclusive)

    solar_array = np.array(prediction_solar, dtype=float)

    # --- GSHP Optimization (Must run before battery) ---
    acc_temp_state = get_ha_state('sensor.mlp_varaajan_lampotila')
    current_acc_temp = 50.0
    try:
        current_acc_temp = float(acc_temp_state.get('state', 50.0))
    except (TypeError, ValueError):
        pass
    
    gshp_power_state = get_ha_state('sensor.mlp_teho')
    is_hp_currently_running = False
    try:
        is_hp_currently_running = float(gshp_power_state.get('state', 0)) > 100
    except (TypeError, ValueError):
        pass

    # Fireplace detection: check if accumulator temp is rising while GSHP is off
    is_fireplace_currently_on = False
    try:
        # Get recent accumulator temperature history (last 30 minutes)
        acc_temp_history = fetch_states_history('sensor.mlp_varaajan_lampotila', hours=0.5)
        acc_df = acc_temp_history.get('sensor.mlp_varaajan_lampotila', pd.DataFrame())
        
        if len(acc_df) >= 2:
            # Calculate rate of change over the most recent data
            acc_df = acc_df.sort_values('timestamp')
            time_diff = (acc_df['timestamp'].iloc[-1] - acc_df['timestamp'].iloc[-2]).total_seconds() / 60.0  # in minutes
            temp_diff = acc_df['state'].iloc[-1] - acc_df['state'].iloc[-2]
            
            if time_diff > 0:
                acc_roc = temp_diff / time_diff  # °C per minute
                # Scale to 15-minute rate for comparison with the 0.3 threshold
                acc_roc_15m = acc_roc * 15
                
                # Check if accumulator is rising fast and GSHP is mostly off
                # Using the same heuristic as in process_data.py
                is_fireplace_currently_on = (acc_roc_15m > 0.3) and (float(gshp_power_state.get('state', 0)) < 100)
    except Exception as e:
        print(f"⚠️ Error detecting fireplace status: {e}")

    os.environ['GSHP_INITIAL_TEMP'] = str(current_acc_temp)
    os.environ['GSHP_IS_RUNNING'] = '1' if is_hp_currently_running else '0'

    outside_temps = [p.get('outside_temp', 5.0) for p in predictions_data]
    is_sauna_active = [p.get('is_sauna_active', 0) for p in predictions_data]
    gshp_plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, export_prices, solar_array)

    # Combine Baseload + Planned GSHP + Planned EV (XPZ) for Battery optimization
    planned_gshp_kw = np.array([g['gshp_electric_kw'] for g in gshp_plan])
    
    # EV Strategy:
    # 1. Target SoC logic: Calculate kWh needed.
    # 2. Fallback: Fixed EV_CHARGE_HOURS logic if SoC unavailable.
    
    ev_target_soc = get_env_float('EV_TARGET_SOC_PCT', 80.0)
    ev_capacity_kwh = get_env_float('EV_CAPACITY_KWH', 60.0)
    
    # Try to get current SoC from HA
    ev_soc_state = get_ha_state('sensor.xpz_491_battery_level')
    current_soc = None
    if ev_soc_state and ev_soc_state.get('state') not in ['unknown', 'unavailable']:
        try:
            current_soc = float(ev_soc_state['state'])
        except (ValueError, TypeError):
            pass
            
    # Filter indices where EV is at home
    home_indices = [i for i, p in enumerate(predictions_data) if p.get('ev_position', 1) == 1]
    
    if not home_indices:
        print("⚠️ Warning: EV (XPZ) not predicted to be home at any time in the plan window.")
        ev_plan = [0] * len(import_prices)
    else:
        # Calculate needed slots
        ev_power_kw = get_env_float('EV_CHARGE_POWER_KW', 3.5)
        if current_soc is None:
            # Fallback to fixed duration if we don't know the SoC
            ev_charge_hours = get_env_float('EV_CHARGE_HOURS', 4.0)
            needed_slots = max(1, int(round(ev_charge_hours / get_plan_interval_hours())))
            print(f"EV SoC unknown. Falling back to fixed {ev_charge_hours}h duration.")
        elif current_soc >= ev_target_soc:
            # Battery is already at or above target
            needed_slots = 0
            print(f"EV SoC {current_soc}% is at or above target {ev_target_soc}%. No charging needed.")
        else:
            # Calculate deficit
            deficit_kwh = (ev_target_soc - current_soc) / 100.0 * ev_capacity_kwh
            needed_slots = max(1, int(np.ceil(deficit_kwh / (ev_power_kw * get_plan_interval_hours()))))
            print(f"EV SoC {current_soc}%: Need {needed_slots} slots to reach {ev_target_soc}% at {ev_power_kw}kW")

        # Sort HOME intervals by price
        home_prices = [(import_prices[i], i) for i in home_indices]
        home_prices.sort()
        
        # Take N cheapest home slots
        cheapest_home_indices = [idx for price, idx in home_prices[:needed_slots]]
        ev_plan = [1 if i in cheapest_home_indices else 0 for i in range(len(import_prices))]

    planned_ev_kw = np.array([ev_power_kw if ev else 0.0 for ev in ev_plan])

    # Leaf Strategy:
    # Keep the frequent dispatch behavior (Solar/Night/Cheap) but fix the predicted power.
    # User reports ~10kWh/day total usage, so we scale power to match that.
    leaf_backup_hours = get_env_float('LEAF_BACKUP_HOURS', 4.0)
    leaf_intervals_backup = max(1, int(round(leaf_backup_hours / get_plan_interval_hours())))
    
    night_window_indices = [
        i for i, ts in enumerate(prediction_timestamps) 
        if ts.hour >= 22 or ts.hour < 7
    ]
    night_prices = [(import_prices[i], i) for i in night_window_indices]
    night_prices.sort()
    leaf_backup_indices = [idx for price, idx in night_prices[:leaf_intervals_backup]]
    
    leaf_price_threshold_day = np.percentile(import_prices, 35)
    
    leaf_intents = []
    for i, ts in enumerate(prediction_timestamps):
        price = import_prices[i]
        solar = solar_array[i]
        is_day = 7 <= ts.hour < 22
        
        intent = 'OFF'
        if i in leaf_backup_indices:
            intent = 'ON' # Night Backup
        elif is_day and (price <= leaf_price_threshold_day or solar >= 2.0):
            intent = 'ON' # Day Opportunity
        leaf_intents.append(intent)

    # Calculate realistic average power to hit daily target (default 10kWh/day)
    num_on = sum(1 for x in leaf_intents if x == 'ON')
    leaf_daily_target = get_env_float('LEAF_DAILY_TARGET_KWH', 10.0)
    plan_hours = len(prediction_timestamps) * get_plan_interval_hours()
    target_kwh = leaf_daily_target * (plan_hours / 24.0)
    
    leaf_avg_power = (target_kwh / (num_on * get_plan_interval_hours())) if num_on > 0 else 0.0
    leaf_avg_power = min(leaf_avg_power, 3.0) # Don't exceed nominal 3kW
    
    planned_leaf_kw = np.array([leaf_avg_power if intent == 'ON' else 0.0 for intent in leaf_intents])
    
    # We only use Baseload + GSHP for battery optimization.
    # Charging an EV from a stationary battery is double-conversion loss.
    battery_optimization_load_kw = predictions + planned_gshp_kw
    total_planned_load_kw = battery_optimization_load_kw + planned_ev_kw + planned_leaf_kw

    effective_prices = np.where(solar_array > 0.5, 0.0, import_prices)
    price_threshold = np.percentile(effective_prices, 20)
    heating_plan = [1 if p <= price_threshold else 0 for p in effective_prices]

    # Battery Dispatch uses Baseload + GSHP
    predictions_kwh = (predictions + planned_gshp_kw) * get_plan_interval_hours()
    solar_kwh = solar_array * get_plan_interval_hours()
    
    # Committed loads (EV + Leaf) consume grid capacity but are not powered from house battery
    committed_load_kwh = (planned_ev_kw + planned_leaf_kw) * get_plan_interval_hours()
    
    # Fetch live battery-export toggle from Home Assistant (falls back to .env)
    allow_export_entity = os.getenv('BATTERY_ALLOW_EXPORT_ENTITY', 'input_boolean.battery_allow_export')
    allow_export_state = get_ha_state(allow_export_entity)
    allow_export = parse_ha_bool(allow_export_state, default=get_env_bool('BATTERY_ALLOW_EXPORT', True))
    print(f"Battery allow_export from HA ({allow_export_entity}): {allow_export}")
    
    # Use battery optimization if available, otherwise fall back to no-battery plan
    if is_battery_enabled():
        battery_plan = plan_battery_dispatch(predictions_kwh, solar_kwh, import_prices, export_prices, committed_load_kwh, allow_export=allow_export)
    else:
        battery_plan = plan_no_battery_dispatch(predictions_kwh, solar_kwh, import_prices, export_prices, committed_load_kwh)

    print(f"\nOptimization Plan from {prediction_timestamps[0]} to {prediction_timestamps[-1]}:")
    print(f"Interval: {get_plan_interval_minutes()} minutes")
    print(f"GSHP Initial State: {current_acc_temp:.1f}°C, {'RUNNING' if is_hp_currently_running else 'STOPPED'}")
    print(f"Fireplace: {'ON' if is_fireplace_currently_on else 'OFF'}")
    
    final_plan = []
    print('Time        | Baseload | GSHP kW | Grid kW | Solar | SOC% | P-tile | Intent | Acc Sim')
    print('------------|----------|---------|---------|-------|------|--------|--------|--------')
    for i, ts in enumerate(prediction_timestamps):
        # Ensure ts is local-aware for consistent display
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc).astimezone()
        else:
            ts = ts.astimezone()
            
        p_baseload_kw = float(predictions[i])
        p_gshp_kw = float(planned_gshp_kw[i])
        p_ev_kw = float(planned_ev_kw[i])
        p_leaf_kw = float(planned_leaf_kw[i])
        p_market = float(market_prices[i])
        p_import = float(import_prices[i])
        p_export = float(export_prices[i])
        p_solar_kw = solar_array[i]
        b = battery_plan[i]
        g = gshp_plan[i]
        
        # Calculate price percentile within the current plan window
        # (Where does current price rank among all prices in the plan)
        p_tile = (import_prices < p_import).mean() * 100.0
        
        # Net Grid Exchange: positive means importing, negative means exporting
        # grid_import and grid_export are in kWh per interval
        # power (kW) = energy (kWh) / hours
        p_grid_kw = (b['grid_import_kwh'] - b['grid_export_kwh']) / get_plan_interval_hours()
        
        print(
            f"{ts.strftime('%m-%d %H:%M')} | {p_baseload_kw:8.1f} | {p_gshp_kw:7.1f} | {p_grid_kw:7.1f} | "
            f"{p_solar_kw:5.2f} | {b['soc_pct']:4.1f} | {p_tile:5.1f}% | {g['gshp_intent']:6} | {g['gshp_temp_sim']:5.1f}"
        )
        
        final_plan.append({
            'timestamp': ts.isoformat(),
            'predicted_baseload_kw': p_baseload_kw,
            'sarima_lower_95': float(sarima_lower.iloc[i]) if not np.isnan(sarima_lower.iloc[i]) else None,
            'sarima_upper_95': float(sarima_upper.iloc[i]) if not np.isnan(sarima_upper.iloc[i]) else None,
            'planned_gshp_kw': p_gshp_kw,
            'planned_ev_kw': p_ev_kw,
            'planned_leaf_kw': p_leaf_kw,
            'leaf_intent': leaf_intents[i],
            'predicted_usage_kw': float(total_planned_load_kw[i]),
            'predicted_usage_kwh': float(predictions_kwh[i]),
            'spot_price': float(p_market),
            'market_base_price': float(p_market),
            'import_unit_price': float(p_import),
            'export_unit_price': float(p_export),
            'is_fallback_price': int(is_fallback_price[i]),
            'solar_forecast_kw': float(p_solar_kw),
            'solar_forecast_kwh': float(solar_kwh[i]),
            'ev_charge': bool(ev_plan[i]),
            'heat_boost': bool(heating_plan[i]),
            'gshp_intent': g['gshp_intent'],
            'gshp_temp_simulated': g['gshp_temp_sim'],
            **b,
        })
        
    # Support environment variable override for testing
    plan_file = os.getenv('TEST_PLAN_FILE', 'optimization_plan.json')
    with open(plan_file, 'w') as f:
        json.dump(final_plan, f, indent=2)
    print(f'\n✅ Plan saved to {plan_file}')

    # Archive the TOTAL planned usage to hepo.db for accuracy tracking
    # This ensures analyze_performance.py compares actuals against Baseload + Planned GSHP
    generated_at = datetime.now().astimezone().isoformat()
    git_version = get_model_version()
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Ensure table exists (re-using logic from predict_future.py)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_usage_kw REAL,
                solar_forecast_kw REAL,
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
                PRIMARY KEY (target_timestamp, generated_at)
            )
        ''')

        # Schema migration: add missing columns
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        
        new_cols = {
            'is_fallback_price': 'INTEGER DEFAULT 0',
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
            'gshp_intent': 'TEXT'
        }
        
        for col, col_type in new_cols.items():
            if col not in columns:
                cur.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")

        # We reuse the same table schema as predict_future.py
        data_to_insert = [
            (
                item['timestamp'], 
                generated_at, 
                item['predicted_usage_kw'], 
                item['solar_forecast_kw'], 
                item['is_fallback_price'],
                git_version,
                item.get('battery_action'),
                item.get('battery_power_kw'),
                item.get('soc_pct'),
                item.get('import_unit_price'),
                item.get('export_unit_price'),
                item.get('grid_import_kwh'),
                item.get('grid_export_kwh'),
                item.get('charge_from_solar_kwh'),
                item.get('charge_from_grid_kwh'),
                item.get('discharge_to_load_kwh'),
                item.get('discharge_to_export_kwh'),
                item.get('planned_gshp_kw'),
                item.get('gshp_intent')
            )
            for item in final_plan
        ]
        cur.executemany('''
            INSERT OR REPLACE INTO predictions 
            (
                target_timestamp, generated_at, predicted_usage_kw, solar_forecast_kw, 
                is_fallback_price, version, battery_action, battery_power_kw, 
                battery_soc_pct, import_price, export_price, grid_import_kwh, grid_export_kwh,
                charge_from_solar_kwh, charge_from_grid_kwh, discharge_to_load_kwh, discharge_to_export_kwh,
                planned_gshp_kw, gshp_intent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)

        
        conn.commit()
        conn.close()
        print(f'✅ Archived {len(final_plan)} optimized points to {get_db_path()}')
    except Exception as e:
        print(f'⚠️ Error archiving optimized plan to SQLite: {e}')

if __name__ == '__main__':
    optimize()
