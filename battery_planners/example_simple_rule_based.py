"""
Example Simple Rule-Based Battery Planner

This is a proof-of-concept showing how easy it is to create an alternative
battery planner using the pluggable architecture.

This planner uses simple rules instead of opportunity cost analysis:
1. Charge from solar whenever battery is not full
2. Discharge during the 3 most expensive hours of the day
3. Otherwise stay idle and let solar/grid provide power

This is intentionally simplified to show the API, but demonstrates that
alternative algorithms can be plugged in without touching optimize_plan.py
"""

import os
import numpy as np
from typing import List, Any
from .base import BatteryPlanner, BatteryPlanEntry


def get_env_float(name, default):
    """Safely get a float from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


class SimpleRuleBasedPlanner(BatteryPlanner):
    """
    Simple rule-based battery planner using basic heuristics.
    
    Rules:
    1. Charge from solar whenever possible
    2. Discharge during expensive hours
    3. Avoid discharge when we need to import anyway
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
    ) -> List[BatteryPlanEntry]:
        """Generate battery dispatch plan using simple rules."""
        
        # Load configuration
        capacity_kwh = get_env_float('BATTERY_CAPACITY_KWH', 40.0)
        if initial_soc_pct is not None:
            effective_initial_soc_pct = float(initial_soc_pct)
        else:
            effective_initial_soc_pct = get_env_float('BATTERY_INITIAL_SOC_PCT', 50.0)
        min_soc_pct = get_env_float('BATTERY_MIN_SOC_PCT', 10.0)
        max_soc_pct = get_env_float('BATTERY_MAX_SOC_PCT', 90.0)
        
        min_soc_kwh = capacity_kwh * min_soc_pct / 100.0
        max_soc_kwh = capacity_kwh * max_soc_pct / 100.0
        soc_kwh = capacity_kwh * effective_initial_soc_pct / 100.0
        
        horizon = len(predictions_kwh)
        interval_hours = 0.25  # Default 15 minutes
        
        max_charge_kw = get_env_float('BATTERY_MAX_CHARGE_KW', 10.0)
        max_discharge_kw = get_env_float('BATTERY_MAX_DISCHARGE_KW', 10.0)
        charge_eff = get_env_float('BATTERY_CHARGE_EFFICIENCY', 0.95)
        discharge_eff = get_env_float('BATTERY_DISCHARGE_EFFICIENCY', 0.95)
        
        # Identify expensive hours (top 1/3 of prices)
        expensive_threshold = np.percentile(import_prices, 66)
        is_expensive_hour = import_prices > expensive_threshold
        
        if committed_load_kwh is None:
            committed_load_kwh = np.zeros(horizon)
        
        net_without_battery = (
            np.array(predictions_kwh, dtype=float) - 
            np.array(solar_kwh, dtype=float)
        )
        
        battery_plan = []
        
        for i in range(horizon):
            net_load = float(net_without_battery[i])
            current_import = float(import_prices[i])
            current_export = float(export_prices[i])
            
            charge_from_solar = 0.0
            discharge_to_load = 0.0
            discharge_to_export = 0.0
            charge_from_grid = 0.0
            
            # Rule 1: Charge from solar
            if net_load < 0:  # Solar surplus
                solar_surplus = -net_load
                room_kwh = max_soc_kwh - soc_kwh
                if room_kwh > 0:
                    max_charge = min(solar_surplus, max_charge_kw * interval_hours)
                    charge_from_solar = min(max_charge, room_kwh / charge_eff)
                    soc_kwh += charge_from_solar * charge_eff
            
            # Rule 2: Discharge during expensive hours (peak shaving)
            if net_load > 0 and is_expensive_hour[i]:
                available_kwh = max(0.0, soc_kwh - min_soc_kwh)
                if available_kwh > 0:
                    max_discharge = min(net_load, max_discharge_kw * interval_hours)
                    discharge_to_load = min(max_discharge, available_kwh * discharge_eff)
                    soc_kwh -= discharge_to_load / discharge_eff
            
            soc_kwh = min(max(soc_kwh, min_soc_kwh), max_soc_kwh)
            
            # Calculate grid exchange
            committed = float(committed_load_kwh[i]) if i < len(committed_load_kwh) else 0.0
            net_with_battery = (
                net_load + charge_from_solar + charge_from_grid -
                discharge_to_load - discharge_to_export + committed
            )
            
            grid_import_kwh = max(net_with_battery, 0.0)
            grid_export_kwh = max(-net_with_battery, 0.0)
            
            # Calculate costs
            no_battery_import = max(net_load + committed, 0.0)
            no_battery_export = max(-(net_load + committed), 0.0)
            
            hour_cost_no_battery = (
                no_battery_import * current_import -
                no_battery_export * current_export
            )
            hour_cost_with_battery = (
                grid_import_kwh * current_import -
                grid_export_kwh * current_export
            )
            
            # Determine action
            charge_total = charge_from_solar + charge_from_grid
            discharge_total = discharge_to_load + discharge_to_export
            
            if charge_from_solar > 1e-9:
                battery_action = 'charge_solar'
            elif discharge_to_load > 1e-9:
                battery_action = 'discharge_load'
            else:
                battery_action = 'idle'
            
            # Get timestamp
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
                soc_pct=float((soc_kwh / capacity_kwh) * 100.0),
                grid_import_kwh=float(grid_import_kwh),
                grid_export_kwh=float(grid_export_kwh),
                estimated_hour_cost=float(hour_cost_with_battery),
                estimated_hour_savings=float(hour_cost_no_battery - hour_cost_with_battery),
                net_load_without_battery_kwh=float(net_load),
            )
            
            battery_plan.append(entry)
        
        return battery_plan
