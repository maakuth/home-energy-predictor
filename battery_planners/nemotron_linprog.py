from __future__ import annotations
"""
Nemotron-Linprog Battery Planner

Linear programming based battery dispatch planner using scipy.optimize.linprog.
Formulates the battery optimization as a linear program to minimize grid costs
over the planning horizon while respecting all physical constraints.
"""

import os
import numpy as np
from typing import List, Any, Optional
from scipy.optimize import linprog
from .base import BatteryPlanner, BatteryPlanEntry, BatteryPlannerContext, BatteryPlannerProtocol, should_idle_interval
from utils.type_defs import BatteryAction
from utils.battery_utils import estimate_follow_dispatch


def get_env_float(name, default):
    """Safely get a float from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def get_env_int(name, default):
    """Safely get an int from environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


class NemotronLinprogPlanner(BatteryPlanner):
    """
    Linear programming based battery planner.
    
    Uses scipy.optimize.linprog to solve the battery dispatch problem
    as a linear program minimizing total grid cost over the horizon.
    
    Decision variables per interval:
    - charge_from_solar, charge_from_grid, discharge_to_load, discharge_to_export, soc
    
    Objective: Minimize sum(grid_import * import_price - grid_export * export_price)
    + degradation_cost * (charge_total + discharge_total)
    """

    def plan(
        self,
        predictions_kwh: np.ndarray,
        solar_kwh: np.ndarray,
        import_prices: np.ndarray,
        export_prices: np.ndarray,
        prediction_timestamps: List[Any],
        committed_load_kwh: Optional[np.ndarray] = None,
        allow_export: bool = True,
        initial_soc_pct: Optional[float] = None,
        context: Optional[BatteryPlannerContext] = None,
    ) -> List[BatteryPlanEntry]:
        """Generate battery dispatch plan using linear programming."""
        
        # Load configuration from environment
        capacity_kwh = get_env_float('BATTERY_CAPACITY_KWH', 40.0)
        if initial_soc_pct is not None:
            effective_initial_soc_pct = float(initial_soc_pct)
        else:
            effective_initial_soc_pct = get_env_float('BATTERY_INITIAL_SOC_PCT', 50.0)
        min_soc_pct = get_env_float('BATTERY_MIN_SOC_PCT', 10.0)
        max_soc_pct = get_env_float('BATTERY_MAX_SOC_PCT', 90.0)
        
        min_soc_kwh = capacity_kwh * min_soc_pct / 100.0
        max_soc_kwh = capacity_kwh * max_soc_pct / 100.0
        initial_soc_kwh = capacity_kwh * effective_initial_soc_pct / 100.0
        
        # Configurable lookahead horizon (default: 192 intervals = 48 hours)
        # Fallback used when tomorrow's spot prices aren't available yet
        max_horizon = get_env_int('BATTERY_LP_HORIZON', 192)
        max_horizon_fallback = get_env_int('BATTERY_LP_HORIZON_FALLBACK', 96)

        # Use fallback horizon when tomorrow's prices aren't known from context
        tomorrow_valid = (context or {}).get('tomorrow_valid', False)
        if not tomorrow_valid:
            max_horizon = max_horizon_fallback

        horizon = min(len(predictions_kwh), max_horizon)
        interval_hours = get_env_int('PLAN_INTERVAL_MINUTES', 15) / 60.0
        
        max_charge_kw = get_env_float('BATTERY_MAX_CHARGE_KW', 10.0)
        max_discharge_kw = get_env_float('BATTERY_MAX_DISCHARGE_KW', 10.0)
        charge_eff = get_env_float('BATTERY_CHARGE_EFFICIENCY', 0.95)
        discharge_eff = get_env_float('BATTERY_DISCHARGE_EFFICIENCY', 0.95)
        
        # Degradation cost per kWh cycled (default 0 for apples-to-apples comparison)
        degradation_cost_per_kwh = get_env_float('BATTERY_DEGRADATION_COST_EUR_PER_KWH', 0.0)
        
        # Grid fuse limit
        main_fuse_a = get_env_float('MAIN_FUSE_SIZE_A', 25.0)
        max_grid_import_kw = main_fuse_a * 3 * 0.230
        max_grid_import_kwh = max_grid_import_kw * interval_hours
        
        if committed_load_kwh is None:
            committed_load_kwh = np.zeros(horizon)
        
        # Predictions and solar are already in kWh per interval (API contract)
        import_prices = np.array(import_prices, dtype=float)
        export_prices = np.array(export_prices, dtype=float)
        
        # Net load without battery (positive = load, negative = solar surplus)
        net_without_battery_kwh = np.array(predictions_kwh, dtype=float) - np.array(solar_kwh, dtype=float)
        committed_kwh = np.array(committed_load_kwh, dtype=float)
        
        # Build LP problem
        # Variables per interval: [c_solar, c_grid, d_load, d_export, soc]
        # Total variables = 5 * horizon
        n_vars_per_interval = 5
        n_vars = n_vars_per_interval * horizon
        
        # Objective: minimize sum(grid_import * import_price - grid_export * export_price)
        # grid_import = max(0, net + c_solar + c_grid - d_load - d_export + committed)
        # grid_export = max(0, -(net + c_solar + c_grid - d_load - d_export + committed))
        # In LP we can't use max(), so we introduce grid_import and grid_export as separate variables
        # with constraint: grid_import - grid_export = net + c_solar + c_grid - d_load - d_export + committed
        # and grid_import >= 0, grid_export >= 0
        # This adds 2 * horizon more variables
        
        # Add slack variables for grid import limit (soft constraint)
        n_grid_vars = 2 * horizon
        n_slack_vars = horizon  # One slack per interval for grid import overflow
        total_vars = n_vars + n_grid_vars + n_slack_vars
        
        # Variable indices
        def idx_c_solar(i): return i * n_vars_per_interval + 0
        def idx_c_grid(i): return i * n_vars_per_interval + 1
        def idx_d_load(i): return i * n_vars_per_interval + 2
        def idx_d_export(i): return i * n_vars_per_interval + 3
        def idx_soc(i): return i * n_vars_per_interval + 4
        def idx_grid_import(i): return n_vars + i * 2 + 0
        def idx_grid_export(i): return n_vars + i * 2 + 1
        def idx_slack(i): return n_vars + n_grid_vars + i
        
        # Discount factor for future costs (addresses receding horizon pathology)
        # γ < 1 makes future savings worth less, preventing over-optimistic planning
        discount = get_env_float('BATTERY_LP_DISCOUNT', 0.995)
        
        # Objective coefficients
        c = np.zeros(total_vars)
        for i in range(horizon):
            gamma_i = discount ** i
            c[idx_grid_import(i)] = import_prices[i] * gamma_i
            c[idx_grid_export(i)] = -export_prices[i] * gamma_i
            # Slight preference: solar charging over grid charging (solar is free)
            c[idx_c_solar(i)] = -1e-6
            c[idx_c_grid(i)] = 1e-6
            # Degradation cost on cycling (discounted consistently with grid costs)
            if degradation_cost_per_kwh > 0:
                c[idx_c_solar(i)] += degradation_cost_per_kwh * gamma_i
                c[idx_c_grid(i)] += degradation_cost_per_kwh * gamma_i
                c[idx_d_load(i)] = degradation_cost_per_kwh * gamma_i
                c[idx_d_export(i)] = degradation_cost_per_kwh * gamma_i
            # Large penalty for exceeding grid import limit (fuse protection)
            c[idx_slack(i)] = 1e3  # High penalty to discourage exceeding limit
        
        # Terminal value for final SoC: optional, can be disabled
        # Use a configurable percentile of import prices (default 0 = disabled)
        # When disabled, the LP only optimizes costs within the horizon
        terminal_value_percentile = get_env_float('BATTERY_TERMINAL_VALUE_PERCENTILE', 0.0)
        if terminal_value_percentile > 0:
            terminal_value = np.percentile(import_prices, terminal_value_percentile) if len(import_prices) > 0 else 0.10
            c[idx_soc(horizon - 1)] = -terminal_value * charge_eff * discharge_eff  # Round-trip efficiency adjusted
        
        # Constraints: A_ub * x <= b_ub, A_eq * x == b_eq
        # We'll use equality constraints for energy balance and SoC dynamics
        # Inequality for bounds
        
        # Collect constraints
        A_eq_rows = []
        b_eq_rows = []
        A_ub_rows = []
        b_ub_rows = []
        
        # Variable bounds
        bounds = []
        
        # Bounds for battery variables
        for i in range(horizon):
            # c_solar >= 0
            bounds.append((0, None))
            # c_grid >= 0
            bounds.append((0, None))
            # d_load >= 0
            bounds.append((0, None))
            # d_export >= 0
            bounds.append((0, None))
            # soc bounds
            bounds.append((min_soc_kwh, max_soc_kwh))
        
        # Bounds for grid variables
        for i in range(horizon):
            bounds.append((0, None))  # grid_import >= 0
            bounds.append((0, None))  # grid_export >= 0
        
        # Bounds for slack variables
        for i in range(horizon):
            bounds.append((0, None))  # slack >= 0
        
        # Constraints per interval
        for i in range(horizon):
            net_kwh = net_without_battery_kwh[i]
            committed_kwh_i = committed_kwh[i]
            
            # Energy balance: grid_import - grid_export = net_kwh + c_solar + c_grid - d_load - d_export + committed_kwh
            row = np.zeros(total_vars)
            row[idx_grid_import(i)] = 1
            row[idx_grid_export(i)] = -1
            row[idx_c_solar(i)] = -1
            row[idx_c_grid(i)] = -1
            row[idx_d_load(i)] = 1
            row[idx_d_export(i)] = 1
            A_eq_rows.append(row)
            b_eq_rows.append(net_kwh + committed_kwh_i)
            
            # SoC dynamics: soc[i] = soc[i-1] + (c_solar + c_grid) * eta_ch - (d_load + d_export) / eta_dis
            row = np.zeros(total_vars)
            row[idx_soc(i)] = 1
            row[idx_c_solar(i)] = -charge_eff
            row[idx_c_grid(i)] = -charge_eff
            row[idx_d_load(i)] = 1.0 / discharge_eff
            row[idx_d_export(i)] = 1.0 / discharge_eff
            if i == 0:
                # soc[0] = initial_soc + ...
                b_eq_rows.append(initial_soc_kwh)
            else:
                row[idx_soc(i-1)] = -1
                b_eq_rows.append(0.0)
            A_eq_rows.append(row)
            
            # Charge power limit: c_solar + c_grid <= max_charge_kw * interval_hours
            row = np.zeros(total_vars)
            row[idx_c_solar(i)] = 1
            row[idx_c_grid(i)] = 1
            A_ub_rows.append(row)
            b_ub_rows.append(max_charge_kw * interval_hours)
            
            # Discharge power limit: d_load + d_export <= max_discharge_kw * interval_hours
            row = np.zeros(total_vars)
            row[idx_d_load(i)] = 1
            row[idx_d_export(i)] = 1
            A_ub_rows.append(row)
            b_ub_rows.append(max_discharge_kw * interval_hours)
            
            # Solar charging limited by solar surplus
            solar_surplus_kwh = max(0.0, -net_kwh)  # Only when net is negative (solar > load)
            row = np.zeros(total_vars)
            row[idx_c_solar(i)] = 1
            A_ub_rows.append(row)
            b_ub_rows.append(solar_surplus_kwh)
            
            # Grid import limit (fuse limit) - soft constraint with slack
            # grid_import - slack <= max_grid_import
            # Equivalent to: grid_import <= max_grid_import + slack
            # Slack >= 0, so when grid_import exceeds limit, slack absorbs the overflow
            row = np.zeros(total_vars)
            row[idx_grid_import(i)] = 1
            row[idx_slack(i)] = -1
            A_ub_rows.append(row)
            b_ub_rows.append(max_grid_import_kwh)
            
            # If export not allowed, heavily penalize grid_export in objective
            # (soft constraint - allows curtailment when battery full)
            if not allow_export:
                c[idx_grid_export(i)] = 1e6  # Large penalty to discourage export
        
        # Solve LP
        A_eq = np.array(A_eq_rows) if A_eq_rows else None
        b_eq = np.array(b_eq_rows) if b_eq_rows else None
        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub = np.array(b_ub_rows) if b_ub_rows else None
        
        try:
            # HiGHS parallel option - only beneficial for large problems
            # Can be enabled with BATTERY_LP_PARALLEL=1 env var (default: False for small problems)
            lp_parallel = get_env_int('BATTERY_LP_PARALLEL', 0)
            options = {}
            if lp_parallel:
                options['parallel'] = True
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs', options=options)
            
            if not result.success:
                print(f"LP FAILED: {result.message}")
                # Fallback to idle plan
                return self._create_follow_plan(horizon, prediction_timestamps, initial_soc_kwh, 
                                              capacity_kwh, net_without_battery_kwh)
            
            x = result.x
            
            # Fallback: if LP solution cost >= no-battery cost, idle is better
            total_plan_cost = 0.0
            total_no_battery_cost = 0.0
            for i in range(horizon):
                net_kwh_i = net_without_battery_kwh[i] + committed_kwh[i]
                no_bat_import = max(net_kwh_i, 0.0)
                no_bat_export = max(-net_kwh_i, 0.0)
                total_no_battery_cost += no_bat_import * import_prices[i] - no_bat_export * export_prices[i]
                
                gi = max(0.0, x[idx_grid_import(i)])
                ge = max(0.0, x[idx_grid_export(i)])
                total_plan_cost += gi * import_prices[i] - ge * export_prices[i]
            
            # Adjust both sides for terminal value of remaining SOC so the
            # comparison does not discard a plan that stores energy for use
            # beyond the horizon (the LP objective includes this, but the
            # grid-cost-only calculation above does not see it).
            if terminal_value_percentile > 0:
                tv_benefit_per_kwh = terminal_value * charge_eff * discharge_eff
                total_plan_cost -= x[idx_soc(horizon - 1)] * tv_benefit_per_kwh
                total_no_battery_cost -= initial_soc_kwh * tv_benefit_per_kwh
            
            if total_plan_cost >= total_no_battery_cost - 0.001:
                return self._create_follow_plan(horizon, prediction_timestamps, initial_soc_kwh, 
                                              capacity_kwh, net_without_battery_kwh)
            
            # ---- Stage 2: classify zero-dispatch intervals and re-solve with follow energy ----
            max_follow_kw = get_env_float('BATTERY_FOLLOW_MAX_KW', 2.0)
            follow_intervals = {}
            for i in range(horizon):
                if max(0.0, x[idx_c_solar(i)]) + max(0.0, x[idx_c_grid(i)]) + max(0.0, x[idx_d_load(i)]) + max(0.0, x[idx_d_export(i)]) > 1e-6:
                    continue
                net_kw_i = net_without_battery_kwh[i] / interval_hours
                is_idle = should_idle_interval(
                    net_kw_i, max(max_charge_kw, max_discharge_kw),
                    degradation_cost_per_kwh, interval_hours,
                    charge_eff, discharge_eff,
                    import_prices[i], export_prices[i],
                )
                if is_idle:
                    continue
                follow_kwh, is_discharge = estimate_follow_dispatch(
                    net_kw_i, interval_hours, max_follow_kw)
                if follow_kwh > 0:
                    if is_discharge:
                        follow_intervals[i] = {'charge': 0.0, 'discharge': follow_kwh}
                    else:
                        follow_intervals[i] = {'charge': follow_kwh, 'discharge': 0.0}
            
            if follow_intervals:
                # Pre-clip follow energy to avoid infeasible SoC bounds
                cumulative_soc = initial_soc_kwh
                for i in sorted(follow_intervals.keys()):
                    f = follow_intervals[i]
                    soc_delta = charge_eff * f['charge'] - f['discharge'] / discharge_eff
                    new_soc = cumulative_soc + soc_delta
                    if new_soc < min_soc_kwh:
                        max_discharge = (cumulative_soc - min_soc_kwh) * discharge_eff
                        f['discharge'] = min(f['discharge'], max(max_discharge, 0.0))
                    elif new_soc > max_soc_kwh:
                        max_charge = (max_soc_kwh - cumulative_soc) / charge_eff
                        f['charge'] = min(f['charge'], max(max_charge, 0.0))
                    cumulative_soc += charge_eff * f['charge'] - f['discharge'] / discharge_eff
                    if f['charge'] < 1e-9 and f['discharge'] < 1e-9:
                        del follow_intervals[i]
                
                # Build Stage 2 LP with follow energy embedded
                c2 = c.copy()
                A_eq_rows2 = []
                b_eq_rows2 = []
                A_ub_rows2 = []
                b_ub_rows2 = []
                bounds2 = list(bounds)
                
                for i in range(horizon):
                    f = follow_intervals.get(i)
                    f_charge = f['charge'] if f else 0.0
                    f_discharge = f['discharge'] if f else 0.0
                    follow_grid_offset = f_charge - f_discharge
                    follow_soc_delta = charge_eff * f_charge - f_discharge / discharge_eff
                    net_kwh = net_without_battery_kwh[i]
                    committed_kwh_i = committed_kwh[i]
                    
                    # Energy balance (RHS includes follow offset)
                    row = np.zeros(total_vars)
                    row[idx_grid_import(i)] = 1
                    row[idx_grid_export(i)] = -1
                    row[idx_c_solar(i)] = -1
                    row[idx_c_grid(i)] = -1
                    row[idx_d_load(i)] = 1
                    row[idx_d_export(i)] = 1
                    A_eq_rows2.append(row)
                    b_eq_rows2.append(net_kwh + committed_kwh_i + follow_grid_offset)
                    
                    # SoC dynamics (RHS includes follow delta)
                    row = np.zeros(total_vars)
                    row[idx_soc(i)] = 1
                    row[idx_c_solar(i)] = -charge_eff
                    row[idx_c_grid(i)] = -charge_eff
                    row[idx_d_load(i)] = 1.0 / discharge_eff
                    row[idx_d_export(i)] = 1.0 / discharge_eff
                    if i == 0:
                        b_eq_rows2.append(initial_soc_kwh + follow_soc_delta)
                    else:
                        row[idx_soc(i - 1)] = -1
                        b_eq_rows2.append(follow_soc_delta)
                    A_eq_rows2.append(row)
                    
                    # Charge power limit
                    row = np.zeros(total_vars)
                    row[idx_c_solar(i)] = 1
                    row[idx_c_grid(i)] = 1
                    A_ub_rows2.append(row)
                    b_ub_rows2.append(max_charge_kw * interval_hours)
                    
                    # Discharge power limit
                    row = np.zeros(total_vars)
                    row[idx_d_load(i)] = 1
                    row[idx_d_export(i)] = 1
                    A_ub_rows2.append(row)
                    b_ub_rows2.append(max_discharge_kw * interval_hours)
                    
                    # Solar charging limited by solar surplus
                    solar_surplus_kwh = max(0.0, -net_kwh)
                    row = np.zeros(total_vars)
                    row[idx_c_solar(i)] = 1
                    A_ub_rows2.append(row)
                    b_ub_rows2.append(solar_surplus_kwh)
                    
                    # Grid import limit (fuse)
                    row = np.zeros(total_vars)
                    row[idx_grid_import(i)] = 1
                    row[idx_slack(i)] = -1
                    A_ub_rows2.append(row)
                    b_ub_rows2.append(max_grid_import_kwh)
                    
                    # Lock follow intervals — charge/discharge forced to zero
                    if f:
                        bounds2[idx_c_solar(i)] = (0, 0)
                        bounds2[idx_c_grid(i)] = (0, 0)
                        bounds2[idx_d_load(i)] = (0, 0)
                        bounds2[idx_d_export(i)] = (0, 0)
                
                A_eq2 = np.array(A_eq_rows2)
                b_eq2 = np.array(b_eq_rows2)
                A_ub2 = np.array(A_ub_rows2)
                b_ub2 = np.array(b_ub_rows2)
                
                result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                                 bounds=bounds2, method='highs', options=options)
                
                if result2.success:
                    x = result2.x
                else:
                    print(f"WARNING: Stage 2 LP failed ({result2.message}), using Stage 1 solution")
                    follow_intervals = {}
            
        except Exception as e:
            print(f"LP EXCEPTION: {e}")
            # Fallback to idle plan on solver error
            return self._create_follow_plan(horizon, prediction_timestamps, initial_soc_kwh, 
                                          capacity_kwh, net_without_battery_kwh)
        
        # Build plan entries from solution
        battery_plan = []
        for i in range(horizon):
            c_solar = max(0.0, x[idx_c_solar(i)])
            c_grid = max(0.0, x[idx_c_grid(i)])
            d_load = max(0.0, x[idx_d_load(i)])
            d_export = max(0.0, x[idx_d_export(i)])
            soc_kwh = x[idx_soc(i)]
            grid_import = max(0.0, x[idx_grid_import(i)])
            grid_export = max(0.0, x[idx_grid_export(i)])
            slack = max(0.0, x[idx_slack(i)])
            
            # Clamp SoC to valid range (numerical precision)
            soc_kwh = min(max(soc_kwh, min_soc_kwh), max_soc_kwh)
            
            # Include follow energy for intervals classified in Stage 2
            f = follow_intervals.get(i) if follow_intervals else None
            f_charge = f['charge'] if f else 0.0
            f_discharge = f['discharge'] if f else 0.0
            
            battery_action: BatteryAction
            if f:
                battery_action = 'follow'
                entry_c_solar = f_charge + c_solar
                entry_c_grid = c_grid
                entry_d_load = f_discharge + d_load
                entry_d_export = d_export
                battery_power_kw = (f_charge + c_solar + c_grid - f_discharge - d_load - d_export) / interval_hours
            else:
                charge_total = c_solar + c_grid
                discharge_total = d_load + d_export
                if charge_total > 1e-6 and c_grid > c_solar:
                    battery_action = 'charge_grid'
                elif charge_total > 1e-6:
                    battery_action = 'charge_solar'
                elif d_load > 1e-6 and d_export > 1e-6:
                    battery_action = 'discharge_mixed'
                elif d_load > 1e-6:
                    battery_action = 'discharge_load'
                elif d_export > 1e-6:
                    battery_action = 'discharge_export'
                else:
                    battery_action = 'idle'
                entry_c_solar = c_solar
                entry_c_grid = c_grid
                entry_d_load = d_load
                entry_d_export = d_export
                battery_power_kw = (c_solar + c_grid - d_load - d_export) / interval_hours
            
            # Override: if LP gave zero dispatch but load-following would
            # be profitable, switch to 'follow' (Stage 2 may have missed
            # these intervals when the re-solve covers only some intervals)
            if battery_action == 'idle':
                net_kw_i = net_without_battery_kwh[i] / interval_hours
                if not should_idle_interval(
                    net_kw_i, max(max_charge_kw, max_discharge_kw),
                    degradation_cost_per_kwh, interval_hours,
                    charge_eff, discharge_eff,
                    import_prices[i], export_prices[i],
                ):
                    follow_kwh, is_discharge = estimate_follow_dispatch(
                        net_kw_i, interval_hours, max_follow_kw)
                    if follow_kwh > 0:
                        battery_action = 'follow'
                        if is_discharge:
                            entry_d_load = follow_kwh
                        else:
                            entry_c_solar = follow_kwh
                        battery_power_kw = (entry_c_solar + entry_c_grid - entry_d_load - entry_d_export) / interval_hours
            
            # Calculate costs (net and committed already in kWh)
            no_battery_net = net_without_battery_kwh[i] + committed_kwh[i]
            no_battery_import = max(no_battery_net, 0.0)
            no_battery_export = max(-no_battery_net, 0.0)
            
            hour_cost_no_battery = (no_battery_import * import_prices[i] - 
                                    no_battery_export * export_prices[i])
            hour_cost_with_battery = (grid_import * import_prices[i] - 
                                       grid_export * export_prices[i])
            
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
                battery_power_kw=float(battery_power_kw),
                charge_from_solar_kwh=float(entry_c_solar),
                charge_from_grid_kwh=float(entry_c_grid),
                discharge_to_load_kwh=float(entry_d_load),
                discharge_to_export_kwh=float(entry_d_export),
                soc_kwh=float(soc_kwh),
                soc_pct=float((soc_kwh / capacity_kwh) * 100.0),
                grid_import_kwh=float(grid_import),
                grid_export_kwh=float(grid_export),
                estimated_hour_cost=float(hour_cost_with_battery),
                estimated_hour_savings=float(hour_cost_no_battery - hour_cost_with_battery),
                net_load_without_battery_kwh=float(net_without_battery_kwh[i]),
            )
            
            battery_plan.append(entry)
        
        # Pad with follow entries if LP horizon is shorter than input length
        while len(battery_plan) < len(prediction_timestamps):
            i = len(battery_plan)
            ts = prediction_timestamps[i]
            timestamp_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            net_kwh = float(net_without_battery_kwh[i] + (committed_kwh[i] if i < len(committed_kwh) else 0.0))
            net_kw = net_kwh / interval_hours
            
            follow_kwh, is_discharge = estimate_follow_dispatch(net_kw, interval_hours, max_follow_kw)
            if follow_kwh > 0:
                if is_discharge:
                    actual = min(follow_kwh, (soc_kwh - min_soc_kwh) * discharge_eff)
                    actual = max(actual, 0.0)
                    soc_kwh -= actual / discharge_eff
                    grid_import = max(net_kwh - actual, 0.0)
                    grid_export = 0.0
                    battery_pwr = -min(abs(net_kw), max_follow_kw)
                    d_load_actual = actual
                    c_solar_actual = 0.0
                else:
                    actual = min(follow_kwh, (max_soc_kwh - soc_kwh) / charge_eff)
                    actual = max(actual, 0.0)
                    soc_kwh += actual * charge_eff
                    grid_import = 0.0
                    grid_export = max(-net_kwh - actual, 0.0)
                    battery_pwr = min(abs(net_kw), max_follow_kw)
                    c_solar_actual = actual
                    d_load_actual = 0.0
            else:
                grid_import = max(net_kwh, 0.0)
                grid_export = max(-net_kwh, 0.0)
                battery_pwr = 0.0
                c_solar_actual = 0.0
                d_load_actual = 0.0
            
            entry = BatteryPlanEntry(
                timestamp=timestamp_str,
                battery_action='follow',
                battery_power_kw=float(battery_pwr),
                charge_from_solar_kwh=float(c_solar_actual),
                charge_from_grid_kwh=0.0,
                discharge_to_load_kwh=float(d_load_actual),
                discharge_to_export_kwh=0.0,
                soc_kwh=float(soc_kwh),
                soc_pct=float((soc_kwh / capacity_kwh) * 100.0),
                grid_import_kwh=float(grid_import),
                grid_export_kwh=float(grid_export),
                estimated_hour_cost=0.0,
                estimated_hour_savings=0.0,
                net_load_without_battery_kwh=float(net_without_battery_kwh[i]),
            )
            battery_plan.append(entry)
        
        return battery_plan
    
    def _create_follow_plan(self, horizon, prediction_timestamps, initial_soc_kwh, 
                            capacity_kwh, net_without_battery_kwh):
        """Create a follow plan as fallback with load-following SoC correction."""
        battery_plan = []
        soc_kwh = initial_soc_kwh
        interval_hours = get_env_int('PLAN_INTERVAL_MINUTES', 15) / 60.0
        max_follow_kw = get_env_float('BATTERY_FOLLOW_MAX_KW', 2.0)
        min_soc_kwh = capacity_kwh * get_env_float('BATTERY_MIN_SOC_PCT', 10.0) / 100.0
        max_soc_kwh = capacity_kwh * get_env_float('BATTERY_MAX_SOC_PCT', 90.0) / 100.0
        charge_eff = get_env_float('BATTERY_CHARGE_EFFICIENCY', 0.95)
        discharge_eff = get_env_float('BATTERY_DISCHARGE_EFFICIENCY', 0.95)
        
        for i in range(horizon):
            ts = prediction_timestamps[i]
            if isinstance(ts, str):
                timestamp_str = ts
            elif hasattr(ts, 'isoformat'):
                timestamp_str = ts.isoformat()
            else:
                timestamp_str = str(ts)
            
            net_kwh = float(net_without_battery_kwh[i])
            net_kw = net_kwh / interval_hours
            
            # Compute follow dispatch for this interval
            follow_kwh, is_discharge = estimate_follow_dispatch(net_kw, interval_hours, max_follow_kw)
            if follow_kwh > 0:
                if is_discharge:
                    actual = min(follow_kwh, (soc_kwh - min_soc_kwh) * discharge_eff)
                    actual = max(actual, 0.0)
                    soc_kwh -= actual / discharge_eff
                    grid_import = max(net_kwh - actual, 0.0)
                    grid_export = 0.0
                    battery_pwr = -min(abs(net_kw), max_follow_kw)
                    d_load_actual = actual
                    c_solar_actual = 0.0
                else:
                    actual = min(follow_kwh, (max_soc_kwh - soc_kwh) / charge_eff)
                    actual = max(actual, 0.0)
                    soc_kwh += actual * charge_eff
                    grid_import = 0.0
                    grid_export = max(-net_kwh - actual, 0.0)
                    battery_pwr = min(abs(net_kw), max_follow_kw)
                    c_solar_actual = actual
                    d_load_actual = 0.0
            else:
                grid_import = max(net_kwh, 0.0)
                grid_export = max(-net_kwh, 0.0)
                battery_pwr = 0.0
                c_solar_actual = 0.0
                d_load_actual = 0.0
            
            entry = BatteryPlanEntry(
                timestamp=timestamp_str,
                battery_action='follow',
                battery_power_kw=float(battery_pwr),
                charge_from_solar_kwh=float(c_solar_actual),
                charge_from_grid_kwh=0.0,
                discharge_to_load_kwh=float(d_load_actual),
                discharge_to_export_kwh=0.0,
                soc_kwh=float(soc_kwh),
                soc_pct=float((soc_kwh / capacity_kwh) * 100.0),
                grid_import_kwh=float(grid_import),
                grid_export_kwh=float(grid_export),
                estimated_hour_cost=0.0,
                estimated_hour_savings=0.0,
                net_load_without_battery_kwh=net_kwh,
            )
            battery_plan.append(entry)
        
        # Pad with follow entries if horizon is shorter than input length
        while len(battery_plan) < len(prediction_timestamps):
            i = len(battery_plan)
            ts = prediction_timestamps[i]
            timestamp_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            net_kwh = float(net_without_battery_kwh[i])
            net_kw = net_kwh / interval_hours
            
            follow_kwh, is_discharge = estimate_follow_dispatch(net_kw, interval_hours, max_follow_kw)
            if follow_kwh > 0:
                if is_discharge:
                    actual = min(follow_kwh, (soc_kwh - min_soc_kwh) * discharge_eff)
                    actual = max(actual, 0.0)
                    soc_kwh -= actual / discharge_eff
                    grid_import = max(net_kwh - actual, 0.0)
                    grid_export = 0.0
                    battery_pwr = -min(abs(net_kw), max_follow_kw)
                    d_load_actual = actual
                    c_solar_actual = 0.0
                else:
                    actual = min(follow_kwh, (max_soc_kwh - soc_kwh) / charge_eff)
                    actual = max(actual, 0.0)
                    soc_kwh += actual * charge_eff
                    grid_import = 0.0
                    grid_export = max(-net_kwh - actual, 0.0)
                    battery_pwr = min(abs(net_kw), max_follow_kw)
                    c_solar_actual = actual
                    d_load_actual = 0.0
            else:
                grid_import = max(net_kwh, 0.0)
                grid_export = max(-net_kwh, 0.0)
                battery_pwr = 0.0
                c_solar_actual = 0.0
                d_load_actual = 0.0
            
            entry = BatteryPlanEntry(
                timestamp=timestamp_str,
                battery_action='follow',
                battery_power_kw=float(battery_pwr),
                charge_from_solar_kwh=float(c_solar_actual),
                charge_from_grid_kwh=0.0,
                discharge_to_load_kwh=float(d_load_actual),
                discharge_to_export_kwh=0.0,
                soc_kwh=float(soc_kwh),
                soc_pct=float((soc_kwh / capacity_kwh) * 100.0),
                grid_import_kwh=float(grid_import),
                grid_export_kwh=float(grid_export),
                estimated_hour_cost=0.0,
                estimated_hour_savings=0.0,
                net_load_without_battery_kwh=net_kwh,
            )
            battery_plan.append(entry)
        
        return battery_plan