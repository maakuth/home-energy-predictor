"""Heuristic-based battery planner using opportunity cost analysis.

This planner implements the original HEPO battery optimization logic,
which uses a series of heuristics based on:
- Opportunity cost of holding energy vs future discharge opportunities
- Peak shaving by discharging during expensive periods
- Solar surplus charging when it's cheaper than exporting
- Grid charging during cheap price windows
- Conservative discharge reservation for high-value future opportunities

All heuristics are designed to maximize battery ROI by minimizing grid costs.
"""

import os
import numpy as np
from typing import List, Any, Optional
from datetime import datetime

from .base import BatteryPlanner, BatteryPlanEntry, BatteryPlannerContext, should_idle_interval


def get_env_float(name, default):
    """Safely get a float from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ Invalid float for {name}='{raw}', using default {default}")
        return float(default)


def get_env_bool(name, default=False):
    """Safely get a boolean from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def get_plan_interval_hours():
    """Get planning interval in hours from environment."""
    plan_interval_minutes = int(os.getenv('PLAN_INTERVAL_MINUTES', '15'))
    return max(plan_interval_minutes, 1) / 60.0


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
                              max_charge_kw, max_discharge_kw, interval_hours,
                              max_lookahead_hours=8.0):
    """Compute the marginal value (EUR per output kWh) of keeping stored energy.

    Looks at future intervals with net load within the lookahead window,
    computes the maximum discharge value at each, sorts by value, and returns
    the value of the marginal opportunity that would be lost if we had less
    energy.

    The lookahead window is capped to avoid far-future cheap intervals
    (where forecasts are less reliable) from inappropriately lowering the
    threshold and encouraging early, low-profit discharge.

    Returns 0.0 if there are no future discharge opportunities.
    """
    total_available_output_kwh = max(0.0, (sim_soc - min_soc_kwh) * discharge_eff)
    if total_available_output_kwh < 1e-9:
        return 0.0

    max_lookahead_intervals = int(max_lookahead_hours / interval_hours)
    end_idx = min(start_idx + max_lookahead_intervals, horizon)

    opportunities = []

    for j in range(start_idx, end_idx):
        net_j = float(net_without_battery[j])
        if net_j <= 0:
            # Solar surplus: stored energy would displace free solar charging
            continue

        max_total_discharge = max_discharge_kw * interval_hours
        max_load_discharge = min(net_j, max_total_discharge)
        if max_load_discharge > 0:
            opportunities.append((float(import_prices[j]), max_load_discharge))

        if allow_export:
            remaining_capacity = max(0.0, max_total_discharge - max_load_discharge)
            if remaining_capacity > 0:
                opportunities.append((float(export_prices[j]), remaining_capacity))

    if not opportunities:
        return 0.0

    opportunities.sort(key=lambda x: x[0], reverse=True)

    remaining = total_available_output_kwh
    for value, kwh in opportunities:
        if remaining <= kwh + 1e-9:
            return value
        remaining -= kwh

    return 0.0


def _compute_reserved_kwh(sim_soc, start_idx, horizon, net_without_battery,
                        import_prices, export_prices, allow_export,
                        min_soc_kwh, discharge_eff,
                        max_discharge_kw, interval_hours,
                        threshold_price, max_lookahead_hours=8.0):
    """Compute how much stored energy must be reserved for future opportunities
    that are strictly better than threshold_price.

    Returns the amount of output kWh that should be kept for strictly better
    future opportunities.
    """
    total_available_output_kwh = max(0.0, (sim_soc - min_soc_kwh) * discharge_eff)
    if total_available_output_kwh < 1e-9:
        return 0.0

    max_lookahead_intervals = int(max_lookahead_hours / interval_hours)
    end_idx = min(start_idx + max_lookahead_intervals, horizon)

    opportunities = []

    for j in range(start_idx, end_idx):
        net_j = float(net_without_battery[j])
        if net_j <= 0:
            continue

        max_total_discharge = max_discharge_kw * interval_hours
        max_load_discharge = min(net_j, max_total_discharge)
        if max_load_discharge > 0:
            opportunities.append((float(import_prices[j]), max_load_discharge))

        if allow_export:
            remaining_capacity = max(0.0, max_total_discharge - max_load_discharge)
            if remaining_capacity > 0:
                opportunities.append((float(export_prices[j]), remaining_capacity))

    if not opportunities:
        return 0.0

    opportunities.sort(key=lambda x: x[0], reverse=True)

    reserved = 0.0
    for value, kwh in opportunities:
        if value > threshold_price + 1e-9:
            reserved += kwh
        else:
            break

    return min(reserved, total_available_output_kwh)


def _find_near_term_discharge_need(
    start_idx, horizon, net_without_battery,
    import_prices, export_prices, allow_export,
    min_soc_kwh, discharge_eff, max_discharge_kw,
    interval_hours, margin_eur_per_kwh=0.0, max_lookahead_hours=4.0
):
    """
    Find profitable discharge opportunities before the next cheaper import window.
    
    Returns (output_kwh_needed, last_profitable_idx):
      - output_kwh_needed: total kWh discharge value needed (output after discharge_eff applied)
      - last_profitable_idx: last interval where discharge is profitable
    
    The lookahead is capped to max_lookahead_hours to avoid distant cheap intervals
    from triggering unnecessary early charging.
    
    Profitable discharge at interval j (from current interval's perspective):
    - If future import_prices[j] > current_import: we save money by discharging then
    - But only if j is before the next cheaper window (where we'd rather save energy)
    """
    current_import = import_prices[start_idx]
    
    # Find next cheaper import (within lookahead window)
    max_lookahead_intervals = int(max_lookahead_hours / interval_hours)
    next_cheaper_idx = min(start_idx + max_lookahead_intervals, horizon)
    
    for j in range(start_idx + 1, next_cheaper_idx):
        if import_prices[j] < current_import - margin_eur_per_kwh - 1e-9:
            next_cheaper_idx = j
            break
    
    # Sum profitable load discharge up to next_cheaper_idx
    # Discharge is profitable if the future price is HIGHER than current (we avoid paying that high price)
    total_output_needed = 0.0
    last_profitable_idx = start_idx
    
    for j in range(start_idx + 1, next_cheaper_idx):
        net_j = float(net_without_battery[j])
        if net_j > 0:  # Load exists
            # Profitable if import_prices[j] > current_import (future is more expensive)
            if import_prices[j] > current_import + margin_eur_per_kwh + 1e-9:
                max_discharge_load = min(net_j, max_discharge_kw * interval_hours)
                total_output_needed += max_discharge_load
                last_profitable_idx = j
    
    return total_output_needed, last_profitable_idx


class HeuristicBatteryPlanner(BatteryPlanner):
    """Heuristic-based battery planner using opportunity cost analysis.
    
    This is the original HEPO battery optimization implementation.
    It uses several heuristics based on price forecasts and load predictions
    to maximize battery ROI.
    """
    
    def plan(
        self,
        predictions_kwh: np.ndarray,
        solar_kwh: np.ndarray,
        import_prices: np.ndarray,
        export_prices: np.ndarray,
        prediction_timestamps: List[Any],
        committed_load_kwh: np.ndarray = None,
        allow_export: bool = True,
        initial_soc_pct: float = None,
        max_lookahead_hours: float = 8.0,
        context: Optional['BatteryPlannerContext'] = None,
    ) -> List[BatteryPlanEntry]:
        """Generate battery dispatch plan using heuristic opportunity-cost approach."""
        
        # Load battery configuration
        capacity_kwh = get_env_float('BATTERY_CAPACITY_KWH', 40.0)
        min_soc_pct = get_env_float('BATTERY_MIN_SOC_PCT', 10.0)
        max_soc_pct = get_env_float('BATTERY_MAX_SOC_PCT', 90.0)
        reserve_soc_pct = get_env_float('BATTERY_RESERVE_SOC_PCT', min_soc_pct)
        
        # Use caller-provided live SOC if available, otherwise fall back to env default
        if initial_soc_pct is not None:
            effective_initial_soc_pct = float(initial_soc_pct)
        else:
            effective_initial_soc_pct = get_env_float('BATTERY_INITIAL_SOC_PCT', 50.0)
        
        max_charge_kw = get_env_float('BATTERY_MAX_CHARGE_KW', 10.0)
        max_discharge_kw = get_env_float('BATTERY_MAX_DISCHARGE_KW', 10.0)
        charge_eff = get_env_float('BATTERY_CHARGE_EFFICIENCY', 0.95)
        discharge_eff = get_env_float('BATTERY_DISCHARGE_EFFICIENCY', 0.95)
        
        # Grid connection limit (3-phase, 230V)
        main_fuse_a = get_env_float('MAIN_FUSE_SIZE_A', 25.0)
        max_grid_import_kw = main_fuse_a * 3 * 0.230
        
        interval_hours = get_plan_interval_hours()
        
        charge_eff = min(max(charge_eff, 0.01), 1.0)
        discharge_eff = min(max(discharge_eff, 0.01), 1.0)
        
        min_soc_kwh = capacity_kwh * max(min_soc_pct, reserve_soc_pct) / 100.0
        max_soc_kwh = capacity_kwh * max(max_soc_pct, 0.0) / 100.0
        soc_kwh = min(max(capacity_kwh * effective_initial_soc_pct / 100.0, min_soc_kwh), max_soc_kwh)
        
        horizon = len(predictions_kwh)
        net_without_battery = np.array(predictions_kwh, dtype=float) - np.array(solar_kwh, dtype=float)
        
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
            
            # Precompute opportunity cost before solar charging
            opportunity_cost_pre_solar = _compute_opportunity_cost(
                soc_kwh, i + 1, horizon, net_without_battery,
                import_prices, export_prices, allow_export,
                min_soc_kwh, max_soc_kwh,
                charge_eff, discharge_eff,
                max_charge_kw, max_discharge_kw, interval_hours,
                max_lookahead_hours=max_lookahead_hours
            )
            
            # 1. Charge from solar surplus (when it's better than exporting)
            charge_from_solar = 0.0
            if net_load < 0:
                solar_surplus = -net_load
                soc_room_kwh = max(0.0, max_soc_kwh - soc_kwh)
                charge_limit_input_kwh = min(max_charge_kw * interval_hours, soc_room_kwh / charge_eff)
                
                # If current export price is better than storing, export solar instead
                round_trip_eff = charge_eff * discharge_eff
                if allow_export and current_export > opportunity_cost_pre_solar * round_trip_eff:
                    charge_from_solar = 0.0
                else:
                    charge_from_solar = min(solar_surplus, charge_limit_input_kwh)
                
                soc_kwh += charge_from_solar * charge_eff
            
            # Recompute opportunity cost after solar charging for discharge decisions
            opportunity_cost = _compute_opportunity_cost(
                soc_kwh, i + 1, horizon, net_without_battery,
                import_prices, export_prices, allow_export,
                min_soc_kwh, max_soc_kwh,
                charge_eff, discharge_eff,
                max_charge_kw, max_discharge_kw, interval_hours,
                max_lookahead_hours=max_lookahead_hours
            )
            
            # 2. Discharge to load (partial — only the amount that doesn't sacrifice
            # strictly better future opportunities)
            discharge_to_load = 0.0
            if opportunity_cost > 0.0:
                should_discharge = current_import >= opportunity_cost
            else:
                should_discharge = current_import >= best_future_value
            if net_load > 0 and should_discharge:
                soc_available_kwh = max(0.0, soc_kwh - min_soc_kwh)
                discharge_limit_output_kwh = min(max_discharge_kw * interval_hours,
                                                 soc_available_kwh * discharge_eff)
                reserved_for_import = _compute_reserved_kwh(
                    soc_kwh, i + 1, horizon, net_without_battery,
                    import_prices, export_prices, allow_export,
                    min_soc_kwh, discharge_eff,
                    max_discharge_kw, interval_hours,
                    current_import,
                    max_lookahead_hours=max_lookahead_hours
                )
                dischargeable_kwh = max(0.0, soc_available_kwh * discharge_eff - reserved_for_import)
                discharge_to_load = min(net_load, discharge_limit_output_kwh, dischargeable_kwh)
                soc_kwh -= discharge_to_load / discharge_eff
            
            # 3. Discharge to export (partial — only when profitable and only the
            # excess that doesn't sacrifice strictly better future export opportunities)
            discharge_to_export = 0.0
            if allow_export and charge_from_solar == 0.0:
                soc_available_kwh = max(0.0, soc_kwh - min_soc_kwh)
                total_discharge_limit = min(max_discharge_kw * interval_hours,
                                             soc_available_kwh * discharge_eff)
                remaining_capacity = max(0.0, total_discharge_limit - discharge_to_load)
                
                is_best_export = current_export >= float(np.max(future_export))
                round_trip_eff = charge_eff * discharge_eff
                if opportunity_cost > 0.0:
                    is_export_arbitrage = current_export >= opportunity_cost
                else:
                    is_export_arbitrage = current_export > (min_future_import / round_trip_eff)
                
                if remaining_capacity > 0 and (is_best_export or is_export_arbitrage):
                    reserved_for_export = _compute_reserved_kwh(
                        soc_kwh, i + 1, horizon, net_without_battery,
                        import_prices, export_prices, allow_export,
                        min_soc_kwh, discharge_eff,
                        max_discharge_kw, interval_hours,
                        current_export,
                        max_lookahead_hours=max_lookahead_hours
                    )
                    exportable_kwh = max(0.0, soc_available_kwh * discharge_eff - reserved_for_export)
                    discharge_to_export = min(remaining_capacity, exportable_kwh)
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
            
            # New: Near-term arbitrage logic
            margin_eur = get_env_float('BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH', 0.005)
            output_needed, _ = _find_near_term_discharge_need(
                i, horizon, net_without_battery,
                import_prices, export_prices, allow_export,
                min_soc_kwh, discharge_eff, max_discharge_kw,
                interval_hours, margin_eur, max_lookahead_hours=4.0
            )
            has_profitable_near_term = output_needed > 1e-6
            
            # Also check if current is strictly cheapest within the near-term lookahead window
            # (not just tied, but actually cheaper than all future near-term prices)
            max_lookahead_intervals = int(4.0 / interval_hours)
            end_idx = min(i + max_lookahead_intervals, horizon)
            min_near_term_future = float(np.min(import_prices[i+1:end_idx])) if i+1 < end_idx else current_import
            is_strictly_cheapest_in_near_term = current_import < min_near_term_future - 1e-9
            
            # Grid charging decision: allow both global cheapest and near-term arbitrage scenarios
            # For near-term arbitrage: allow grid charging even if solar exists, because we want
            # to prepare for an upcoming expensive period that's sooner than solar can fill
            allow_grid_charge_near_term = has_profitable_near_term and discharge_to_load == 0.0 and discharge_to_export == 0.0
            allow_grid_charge_global = (not solar_can_fill and is_cheapest_window and discharge_to_load == 0.0 
                                        and discharge_to_export == 0.0)
            
            if (profitable_grid_charge 
                and (allow_grid_charge_near_term or allow_grid_charge_global)
                and discharge_to_load == 0.0 and discharge_to_export == 0.0
                and not spill_risk):
                
                # Decide how much to charge based on context
                if has_profitable_near_term and not is_strictly_cheapest_in_near_term:
                    # Near-term arbitrage with future cheaper/equal option: size to near-term need only
                    needed_output_kwh = output_needed
                    available_after_discharge = max(0.0, (soc_kwh - min_soc_kwh) * discharge_eff)
                    shortfall_output_kwh = max(0.0, needed_output_kwh - available_after_discharge)
                    # Convert back to input kWh through round-trip efficiency
                    needed_input_kwh = shortfall_output_kwh / (charge_eff * discharge_eff) if charge_eff * discharge_eff > 0.01 else 0.0
                else:
                    # Classic cheapest-window charging: fill remaining room
                    # Triggered when is_cheapest_window=True (globally cheapest)
                    # or when has_profitable_near_term=True AND is strictly cheapest in near-term window
                    needed_input_kwh = max(0.0, max_soc_kwh - soc_kwh) / charge_eff if charge_eff > 0.01 else 0.0
                
                soc_room_kwh = max(0.0, max_soc_kwh - soc_kwh)
                charge_limit_input_kwh = min(max_charge_kw * interval_hours, soc_room_kwh / charge_eff if charge_eff > 0.01 else 0.0)
                max_grid_charge_kwh = min(charge_limit_input_kwh, available_grid_kwh, needed_input_kwh)
                
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
                # Determine if true idle is better than load-following
                degradation_cost = get_env_float('BATTERY_DEGRADATION_COST_EUR_PER_KWH', 0.0)
                max_battery_kw = max(max_charge_kw, max_discharge_kw)
                is_idle = should_idle_interval(
                    net_load, max_battery_kw, degradation_cost, interval_hours,
                    charge_eff, discharge_eff, current_import, current_export,
                )
                battery_action = 'idle' if is_idle else 'follow'
            
            # Get timestamp - handle both datetime objects and other types
            ts = prediction_timestamps[i]
            if isinstance(ts, str):
                timestamp_str = ts
            elif hasattr(ts, 'isoformat'):
                timestamp_str = ts.isoformat()
            else:
                timestamp_str = str(ts)
            
            entry = BatteryPlanEntry(
                timestamp=timestamp_str,
                battery_action=battery_action,
                battery_power_kw=float((charge_total - discharge_total) / interval_hours),
                charge_from_solar_kwh=float(charge_from_solar),
                charge_from_grid_kwh=float(charge_from_grid),
                discharge_to_load_kwh=float(discharge_to_load),
                discharge_to_export_kwh=float(discharge_to_export),
                soc_kwh=float(soc_kwh),
                soc_pct=float((soc_kwh / capacity_kwh) * 100.0 if capacity_kwh > 0 else 0.0),
                grid_import_kwh=float(grid_import_kwh),
                grid_export_kwh=float(grid_export_kwh),
                estimated_hour_cost=float(hour_cost_with_battery),
                estimated_hour_savings=float(hour_cost_no_battery - hour_cost_with_battery),
                net_load_without_battery_kwh=float(net_load),
            )
            battery_plan.append(entry)
        
        return battery_plan
