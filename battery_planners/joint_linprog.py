from __future__ import annotations
"""Joint LP-native co-optimization planner for Battery, GSHP, and Leaf.

Simultaneously optimizes:
- Electrical flows: grid, solar, and battery to house loads, battery storage, and export
- Thermal storage: Ground Source Heat Pump (GSHP) power and accumulator tank temperature
- Flexible EV loads: Nissan Leaf daily charging requirements

By solving a single unified linear program, the planner ensures loads consume
direct solar, utilize stored battery energy when advantageous, and avoid peak grid
tariffs without double-conversion losses, peak collisions, or fuse limit violations.
"""

import os
from typing import Any, List, Optional, Tuple, Dict
from datetime import datetime

import numpy as np
from scipy.optimize import linprog

from .base import BatteryPlanEntry, BatteryPlanner, BatteryPlannerContext
from utils.type_defs import BatteryAction


def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else float(default)
    except ValueError:
        return float(default)


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else int(default)
    except ValueError:
        return int(default)


def get_env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {'1', 'true', 'yes', 'on'}


def compute_gshp_thermal_demand(
    outside_temps: np.ndarray,
    is_sauna_active: np.ndarray,
    baseline_demand_kw: float,
    heat_loss_k: float,
    sauna_demand_kw: float,
) -> np.ndarray:
    """Compute per-interval thermal heat demand (kW) for building and DHW."""
    demands = []
    for i in range(len(outside_temps)):
        o_temp = float(outside_temps[i])
        if o_temp >= 15.0:
            envelope_kw = 0.0
            eff_baseline = baseline_demand_kw * 0.5
        elif o_temp >= 10.0:
            fraction = (o_temp - 10.0) / 5.0
            envelope_kw = max(0.0, (20.0 - o_temp) * heat_loss_k) * (1.0 - fraction)
            eff_baseline = baseline_demand_kw * (1.0 - fraction * 0.5)
        else:
            envelope_kw = max(0.0, (20.0 - o_temp) * heat_loss_k)
            eff_baseline = baseline_demand_kw

        total_demand = eff_baseline + envelope_kw
        if is_sauna_active is not None and i < len(is_sauna_active) and is_sauna_active[i]:
            total_demand += sauna_demand_kw
        demands.append(total_demand)
    return np.asarray(demands, dtype=float)


class JointLinprogPlanner(BatteryPlanner):
    """Joint LP co-optimization of Battery, GSHP, and Leaf dispatch."""

    @staticmethod
    def _hedge_solar(
        solar: np.ndarray,
        context: Optional[BatteryPlannerContext],
    ) -> np.ndarray:
        """Blend the central solar forecast toward the worst-case p10 bound."""
        alpha = np.clip(get_env_float('BATTERY_SOLAR_HEDGE_ALPHA', 1.0), 0.0, 1.0)
        if alpha >= 1.0 or not context:
            return solar
        p10 = context.get('solar_p10_kwh')
        if p10 is None:
            return solar
        p10 = np.asarray(p10, dtype=float)
        if p10.shape != solar.shape or not np.all(np.isfinite(p10)):
            print('⚠️ solar_p10_kwh in planner context is malformed; skipping solar hedge')
            return solar
        return np.maximum(alpha * solar + (1.0 - alpha) * p10, 0.0)

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
        """Generate a co-optimized battery, GSHP, and Leaf dispatch plan."""
        load = np.asarray(predictions_kwh, dtype=float)
        solar = np.asarray(solar_kwh, dtype=float)
        import_prices = np.asarray(import_prices, dtype=float)
        export_prices = np.asarray(export_prices, dtype=float)
        total_intervals = len(prediction_timestamps)
        if total_intervals == 0:
            return []
        if any(len(values) != total_intervals for values in (load, solar, import_prices, export_prices)):
            raise ValueError('predictions, solar, prices, and timestamps must have equal lengths')
        if not np.all(np.isfinite(load)) or not np.all(np.isfinite(solar)):
            raise ValueError('predictions and solar values must be finite')
        if not np.all(np.isfinite(import_prices)) or not np.all(np.isfinite(export_prices)):
            raise ValueError('prices must be finite')

        if committed_load_kwh is None:
            committed = np.zeros(total_intervals)
        else:
            committed = np.asarray(committed_load_kwh, dtype=float)
            if len(committed) != total_intervals or not np.all(np.isfinite(committed)):
                raise ValueError('committed_load_kwh must be finite and match the planning horizon')

        # Battery configuration
        capacity_kwh = get_env_float('BATTERY_CAPACITY_KWH', 40.0)
        min_soc_pct = get_env_float('BATTERY_MIN_SOC_PCT', 10.0)
        reserve_soc_pct = get_env_float('BATTERY_RESERVE_SOC_PCT', min_soc_pct)
        max_soc_pct = get_env_float('BATTERY_MAX_SOC_PCT', 90.0)
        min_soc_kwh = capacity_kwh * max(min_soc_pct, reserve_soc_pct) / 100.0
        max_soc_kwh = capacity_kwh * max_soc_pct / 100.0
        if capacity_kwh <= 0 or min_soc_kwh < 0 or max_soc_kwh <= min_soc_kwh:
            raise ValueError('battery capacity and SoC limits must define a positive usable range')

        configured_soc_pct = get_env_float('BATTERY_INITIAL_SOC_PCT', 50.0)
        effective_soc_pct = float(initial_soc_pct) if initial_soc_pct is not None else configured_soc_pct
        initial_soc_kwh = np.clip(capacity_kwh * effective_soc_pct / 100.0, min_soc_kwh, max_soc_kwh)

        interval_minutes = get_env_int('PLAN_INTERVAL_MINUTES', 15)
        if interval_minutes <= 0:
            raise ValueError('PLAN_INTERVAL_MINUTES must be positive')
        interval_hours = interval_minutes / 60.0
        max_charge_kw = get_env_float('BATTERY_MAX_CHARGE_KW', 10.0)
        max_discharge_kw = get_env_float('BATTERY_MAX_DISCHARGE_KW', 10.0)
        charge_eff = np.clip(get_env_float('BATTERY_CHARGE_EFFICIENCY', 0.95), 0.01, 1.0)
        discharge_eff = np.clip(get_env_float('BATTERY_DISCHARGE_EFFICIENCY', 0.95), 0.01, 1.0)

        max_horizon = get_env_int('BATTERY_LP_HORIZON', 192)
        if not (context or {}).get('tomorrow_valid', False):
            max_horizon = get_env_int('BATTERY_LP_HORIZON_FALLBACK', 96)
        horizon = min(total_intervals, max(1, max_horizon))

        load = np.maximum(load, 0.0)
        solar = np.maximum(solar, 0.0)
        solar = self._hedge_solar(solar, context)
        committed = np.maximum(committed, 0.0)
        import_price_floor = get_env_float('IMPORT_PRICE_FLOOR_EUR_PER_KWH', 0.0)
        if import_price_floor > 0:
            import_prices = np.maximum(import_prices, import_price_floor)

        main_fuse_a = get_env_float('MAIN_FUSE_SIZE_A', 25.0)
        max_grid_import_kwh = max(0.0, main_fuse_a * 3 * 0.230 * interval_hours)
        degradation_cost = max(0.0, get_env_float('BATTERY_DEGRADATION_COST_EUR_PER_KWH', 0.0))
        discount = np.clip(get_env_float('BATTERY_LP_DISCOUNT', 0.995), 0.0, 1.0)

        # GSHP configuration
        gshp_enabled = get_env_bool('GSHP_OPTIMIZE_ENABLED', True)
        resistive_enabled = get_env_bool('RESISTIVE_HEATER_OPTIMIZE_ENABLED', False)
        p_min = get_env_float('GSHP_POWER_MIN_KW', 3.4)
        p_max = get_env_float('GSHP_POWER_MAX_KW', 4.2)
        if 'GSHP_ELECTRIC_POWER_KW' in os.environ and 'GSHP_POWER_MIN_KW' not in os.environ and 'GSHP_POWER_MAX_KW' not in os.environ:
            p_min = p_max = get_env_float('GSHP_ELECTRIC_POWER_KW', 4.0)

        cop = get_env_float('GSHP_COP', 3.5)
        heating_eff = get_env_float('GSHP_HEATING_EFFICIENCY', 1.0)
        reservoir_l = get_env_float('GSHP_RESERVOIR_LITERS', 500.0)
        kwh_per_degree = (reservoir_l * 4.18) / 3600.0  # ~0.580556 kWh / °C
        min_temp = get_env_float('GSHP_MIN_TEMP', 42.0)
        max_temp = get_env_float('GSHP_MAX_TEMP', 55.0)
        heat_loss_k = get_env_float('GSHP_HEAT_LOSS_K', 0.135)
        baseline_demand_kw = get_env_float('GSHP_BASELINE_DEMAND_KW', 1.0)
        sauna_demand_kw = get_env_float('SAUNA_HOT_WATER_DEMAND_KW', 6.0)

        resistive_power_kw = get_env_float('RESISTIVE_HEATER_POWER_KW', 6.0)
        resistive_eff = get_env_float('RESISTIVE_HEATER_EFFICIENCY', 1.0)
        resistive_l = get_env_float('RESISTIVE_HEATER_EFFECTIVE_LITERS', 150.0)
        resistive_kwh_per_degree = (resistive_l * 4.18) / 3600.0
        if resistive_enabled:
            max_temp = max(max_temp, get_env_float('RESISTIVE_HEATER_MAX_TEMP', 60.0))

        initial_acc_temp = float((context or {}).get('current_acc_temp', get_env_float('GSHP_INITIAL_TEMP', 50.0)))
        initial_acc_temp = np.clip(initial_acc_temp, min_temp, max_temp)

        outside_temps = (context or {}).get('outside_temps')
        if outside_temps is None or len(outside_temps) < horizon:
            outside_temps = np.full(horizon, 5.0)
        else:
            outside_temps = np.asarray(outside_temps[:horizon], dtype=float)

        is_sauna = (context or {}).get('is_sauna_active')
        if is_sauna is None or len(is_sauna) < horizon:
            is_sauna = np.zeros(horizon, dtype=int)
        else:
            is_sauna = np.asarray(is_sauna[:horizon], dtype=int)

        thermal_demand_kw = compute_gshp_thermal_demand(
            outside_temps, is_sauna, baseline_demand_kw, heat_loss_k, sauna_demand_kw
        )

        # Leaf EV configuration
        leaf_enabled = get_env_bool('LEAF_OPTIMIZE_ENABLED', True)
        leaf_daily_target_kwh = get_env_float('LEAF_DAILY_TARGET_KWH', 10.0)
        leaf_max_power_kw = get_env_float('LEAF_MAX_POWER_KW', 3.0)
        leaf_target_kwh = leaf_daily_target_kwh * (horizon * interval_hours / 24.0) if leaf_enabled else 0.0

        # Build decision variables
        # Variables per interval:
        # 0: grid_house
        # 1: grid_battery
        # 2: solar_house
        # 3: solar_battery
        # 4: solar_export
        # 5: solar_curtail
        # 6: battery_house
        # 7: battery_export
        # 8: soc
        # 9: gshp_kwh (electric)
        # 10: resistive_kwh (electric)
        # 11: acc_temp (°C)
        # 12: leaf_kwh (electric)
        # 13: grid_overflow
        # 14: temp_underflow
        width = 15
        n_vars = width * horizon

        def index(i: int, offset: int) -> int:
            return i * width + offset

        (
            grid_house, grid_battery, solar_house, solar_battery,
            solar_export, solar_curtail, battery_house, battery_export,
            soc, gshp_kwh, resistive_kwh, acc_temp, leaf_kwh, overflow, temp_underflow
        ) = range(15)

        objective = np.zeros(n_vars)
        bounds = []

        for i in range(horizon):
            gamma = discount ** i
            objective[index(i, grid_house)] = import_prices[i] * gamma
            objective[index(i, grid_battery)] = import_prices[i] * gamma
            objective[index(i, solar_export)] = -export_prices[i] * gamma
            objective[index(i, battery_export)] = -export_prices[i] * gamma
            for flow in (grid_battery, solar_battery, battery_house, battery_export):
                objective[index(i, flow)] += degradation_cost * gamma
            objective[index(i, solar_curtail)] = 1e-8
            objective[index(i, overflow)] = 1e3
            objective[index(i, temp_underflow)] = 1e4  # Heavy penalty for falling below min temp

            max_gshp_interval_kwh = (p_max * interval_hours) if gshp_enabled else 0.0
            max_resistive_interval_kwh = (
                resistive_power_kw * interval_hours if resistive_enabled else 0.0
            )
            max_leaf_interval_kwh = (leaf_max_power_kw * interval_hours) if leaf_enabled else 0.0

            bounds.extend([
                (0, None),  # grid_house
                (0, None),  # grid_battery
                (0, None),  # solar_house
                (0, None),  # solar_battery
                (0, None) if allow_export else (0, 0),  # solar_export
                (0, None),  # solar_curtail
                (0, None),  # battery_house
                (0, None) if allow_export else (0, 0),  # battery_export
                (min_soc_kwh, max_soc_kwh),  # soc
                (0, max_gshp_interval_kwh),  # gshp_kwh
                (0, max_resistive_interval_kwh),  # resistive_kwh
                (min_temp, max_temp),  # acc_temp
                (0, max_leaf_interval_kwh),  # leaf_kwh
                (0, None),  # overflow
                (0, None),  # temp_underflow
            ])

        # Terminal valuation
        terminal_percentile = get_env_float('BATTERY_TERMINAL_VALUE_PERCENTILE', 0.0)
        if terminal_percentile > 0:
            terminal_price = float(np.percentile(import_prices[:horizon], terminal_percentile))
            objective[index(horizon - 1, soc)] = -terminal_price * discharge_eff * (discount ** (horizon - 1))

        # Terminal heat valuation (incentivize leaving tank warm if heated cheaply)
        avg_price = float(np.mean(import_prices[:horizon]))
        objective[index(horizon - 1, acc_temp)] = -(avg_price / cop) * (kwh_per_degree * 0.25) * (discount ** (horizon - 1))

        equal_rows: list[np.ndarray] = []
        equal_values: list[float] = []
        upper_rows: list[np.ndarray] = []
        upper_values: list[float] = []

        for i in range(horizon):
            # 1. House electrical balance:
            # Heating sources are separate loads even though they share a meter.
            row = np.zeros(n_vars)
            row[index(i, grid_house)] = 1
            row[index(i, solar_house)] = 1
            row[index(i, battery_house)] = 1
            row[index(i, gshp_kwh)] = -1
            row[index(i, resistive_kwh)] = -1
            row[index(i, leaf_kwh)] = -1
            equal_rows.append(row)
            equal_values.append(load[i])

            # 2. Solar flow balance:
            # solar_house + solar_battery + solar_export + solar_curtail = solar
            row = np.zeros(n_vars)
            for flow in (solar_house, solar_battery, solar_export, solar_curtail):
                row[index(i, flow)] = 1
            equal_rows.append(row)
            equal_values.append(solar[i])

            # 3. Battery dynamics:
            # soc[i] - soc[i-1] - charge_eff*(grid_batt + solar_batt) + (1/discharge_eff)*(batt_house + batt_export) = 0
            row = np.zeros(n_vars)
            row[index(i, soc)] = 1
            row[index(i, grid_battery)] = -charge_eff
            row[index(i, solar_battery)] = -charge_eff
            row[index(i, battery_house)] = 1.0 / discharge_eff
            row[index(i, battery_export)] = 1.0 / discharge_eff
            if i == 0:
                equal_values.append(initial_soc_kwh)
            else:
                row[index(i - 1, soc)] = -1
                equal_values.append(0.0)
            equal_rows.append(row)

            # 4. Thermal accumulator dynamics:
            # acc_temp[i] - acc_temp[i-1] - (cop * heating_eff / kwh_per_degree)*gshp_kwh[i] - temp_underflow[i] = - (demand * dt / kwh_per_degree)
            row = np.zeros(n_vars)
            row[index(i, acc_temp)] = 1
            row[index(i, gshp_kwh)] = -(cop * heating_eff) / kwh_per_degree
            row[index(i, resistive_kwh)] = -resistive_eff / resistive_kwh_per_degree
            row[index(i, temp_underflow)] = -1.0
            thermal_loss_deg = (thermal_demand_kw[i] * interval_hours) / kwh_per_degree
            if i == 0:
                equal_values.append(initial_acc_temp - thermal_loss_deg)
            else:
                row[index(i - 1, acc_temp)] = -1
                equal_values.append(-thermal_loss_deg)
            equal_rows.append(row)

            # 5. Battery power limits
            # Charging: grid_battery + solar_battery <= max_charge
            row = np.zeros(n_vars)
            row[index(i, grid_battery)] = 1
            row[index(i, solar_battery)] = 1
            upper_rows.append(row)
            upper_values.append(max_charge_kw * interval_hours)

            # Discharging: battery_house + battery_export <= max_discharge
            row = np.zeros(n_vars)
            row[index(i, battery_house)] = 1
            row[index(i, battery_export)] = 1
            upper_rows.append(row)
            upper_values.append(max_discharge_kw * interval_hours)

            # 6. Main Fuse capacity:
            # grid_house + grid_battery - overflow <= max_grid_import - committed
            row = np.zeros(n_vars)
            row[index(i, grid_house)] = 1
            row[index(i, grid_battery)] = 1
            row[index(i, overflow)] = -1
            upper_rows.append(row)
            upper_values.append(max_grid_import_kwh - committed[i])

        # 7. Leaf daily target constraint:
        # sum(leaf_kwh[i]) = leaf_target_kwh
        if leaf_enabled and leaf_target_kwh > 0:
            row = np.zeros(n_vars)
            for i in range(horizon):
                row[index(i, leaf_kwh)] = 1
            equal_rows.append(row)
            equal_values.append(leaf_target_kwh)

        # Solve Joint LP
        result = linprog(
            objective,
            A_ub=np.asarray(upper_rows),
            b_ub=np.asarray(upper_values),
            A_eq=np.asarray(equal_rows),
            b_eq=np.asarray(equal_values),
            bounds=bounds,
            method='highs',
            options={'parallel': True} if get_env_int('BATTERY_LP_PARALLEL', 0) else {},
        )

        if not result.success:
            print(f"⚠️ Joint LP solver failed: {result.message}. Falling back to idle baseline.")
            return self._fallback_plan(
                load, solar, committed, import_prices, export_prices,
                prediction_timestamps, initial_soc_kwh, capacity_kwh,
                min_soc_kwh, max_discharge_kw, discharge_eff, interval_hours,
                allow_export, initial_acc_temp,
            )

        # Headroom pass for interval 0 (real-time load-following flexibility)
        headroom_objective = np.zeros(n_vars)
        headroom_objective[index(0, battery_house)] = -1
        headroom_tolerance = max(
            0.0, get_env_float('BATTERY_LP_HEADROOM_COST_TOLERANCE_EUR', 0.001),
        )
        headroom_result = linprog(
            headroom_objective,
            A_ub=np.vstack([np.asarray(upper_rows), objective]),
            b_ub=np.append(np.asarray(upper_values), result.fun + headroom_tolerance),
            A_eq=np.asarray(equal_rows),
            b_eq=np.asarray(equal_values),
            bounds=bounds,
            method='highs',
            options={'parallel': True} if get_env_int('BATTERY_LP_PARALLEL', 0) else {},
        )
        lp_headroom_kwh = (
            max(0.0, headroom_result.x[index(0, battery_house)])
            if headroom_result.success else 0.0
        )

        plan: list[BatteryPlanEntry] = []
        x = result.x
        starting_soc = initial_soc_kwh

        for i in range(horizon):
            charge_solar = max(0.0, x[index(i, solar_battery)])
            charge_grid = max(0.0, x[index(i, grid_battery)])
            discharge_load = max(0.0, x[index(i, battery_house)])
            discharge_export = max(0.0, x[index(i, battery_export)])
            planned_gshp_kw = max(0.0, x[index(i, gshp_kwh)]) / interval_hours
            planned_resistive_kw = max(0.0, x[index(i, resistive_kwh)]) / interval_hours
            gshp_intent = 'START' if planned_gshp_kw > 0.05 else 'STOP'
            resistive_intent = 'ON' if planned_resistive_kw > 0.05 else 'OFF'
            t_acc = float(x[index(i, acc_temp)])
            p_leaf = max(0.0, x[index(i, leaf_kwh)]) / interval_hours
            leaf_intent = 'ON' if p_leaf > 0.05 else 'OFF'

            grid_import = max(0.0, x[index(i, grid_house)] + charge_grid + committed[i])
            grid_export = max(0.0, x[index(i, solar_export)] + discharge_export)
            ending_soc = float(np.clip(x[index(i, soc)], min_soc_kwh, max_soc_kwh))
            charge_total = charge_solar + charge_grid
            discharge_total = discharge_load + discharge_export

            action: BatteryAction
            if charge_total > 1e-7 and discharge_total > 1e-7:
                action = 'charge_mixed' if charge_total >= discharge_total else 'discharge_mixed'
            elif charge_total > 1e-7:
                action = 'charge_grid' if charge_grid > charge_solar else 'charge_solar'
            elif discharge_total > 1e-7:
                action = 'discharge_mixed' if discharge_export > 1e-7 and discharge_load > 1e-7 else (
                    'discharge_export' if discharge_export > 1e-7 else 'discharge_load'
                )
            else:
                action = 'idle'

            # Total load including co-optimized GSHP and Leaf
            total_net_load = load[i] + (
                planned_gshp_kw + planned_resistive_kw + p_leaf
            ) * interval_hours - solar[i]
            baseline_import = max(0.0, total_net_load) + committed[i]
            baseline_export = max(0.0, -total_net_load) if allow_export else 0.0

            plan.append(BatteryPlanEntry(
                timestamp=self._timestamp(prediction_timestamps[i]),
                battery_action=action,
                battery_power_kw=float((charge_total - discharge_total) / interval_hours),
                charge_from_solar_kwh=float(charge_solar),
                charge_from_grid_kwh=float(charge_grid),
                discharge_to_load_kwh=float(discharge_load),
                discharge_to_export_kwh=float(discharge_export),
                soc_kwh=ending_soc,
                soc_pct=float(ending_soc / capacity_kwh * 100.0),
                grid_import_kwh=float(grid_import),
                grid_export_kwh=float(grid_export),
                estimated_hour_cost=float(grid_import * import_prices[i] - grid_export * export_prices[i]),
                estimated_hour_savings=float(
                    baseline_import * import_prices[i] - baseline_export * export_prices[i]
                    - (grid_import * import_prices[i] - grid_export * export_prices[i])
                ),
                net_load_without_battery_kwh=float(total_net_load),
                discharge_budget_kwh=float(
                    max(discharge_load, lp_headroom_kwh) if i == 0 else discharge_load
                ),
                planned_gshp_kw=float(planned_gshp_kw),
                gshp_intent=gshp_intent,
                planned_resistive_kw=float(planned_resistive_kw),
                resistive_heater_intent=resistive_intent,
                gshp_temp_sim=float(t_acc),
                planned_leaf_kw=float(p_leaf),
                leaf_intent=leaf_intent,
            ))
            starting_soc = ending_soc

        if horizon < total_intervals:
            plan.extend(self._fallback_plan(
                load[horizon:], solar[horizon:], committed[horizon:],
                import_prices[horizon:], export_prices[horizon:], prediction_timestamps[horizon:],
                starting_soc, capacity_kwh, min_soc_kwh, max_discharge_kw,
                discharge_eff, interval_hours, allow_export, initial_acc_temp,
            ))

        return plan

    @staticmethod
    def _timestamp(value: Any) -> str:
        return value if isinstance(value, str) else value.isoformat() if hasattr(value, 'isoformat') else str(value)

    def _fallback_plan(
        self,
        load: np.ndarray,
        solar: np.ndarray,
        committed: np.ndarray,
        import_prices: np.ndarray,
        export_prices: np.ndarray,
        timestamps: List[Any],
        soc_kwh: float,
        capacity_kwh: float,
        min_soc_kwh: float,
        max_discharge_kw: float,
        discharge_eff: float,
        interval_hours: float,
        allow_export: bool,
        acc_temp: float,
    ) -> List[BatteryPlanEntry]:
        """Return a feasible baseline plan when the joint solver is unavailable."""
        plan = []
        for i, timestamp in enumerate(timestamps):
            solar_to_house = min(load[i], solar[i])
            surplus = solar[i] - solar_to_house
            grid_export = surplus if allow_export else 0.0
            grid_import = load[i] - solar_to_house + committed[i]
            plan.append(BatteryPlanEntry(
                timestamp=self._timestamp(timestamp),
                battery_action='idle',
                battery_power_kw=0.0,
                charge_from_solar_kwh=0.0,
                charge_from_grid_kwh=0.0,
                discharge_to_load_kwh=0.0,
                discharge_to_export_kwh=0.0,
                soc_kwh=float(soc_kwh),
                soc_pct=float(soc_kwh / capacity_kwh * 100.0),
                grid_import_kwh=float(grid_import),
                grid_export_kwh=float(grid_export),
                estimated_hour_cost=float(grid_import * import_prices[i] - grid_export * export_prices[i]),
                estimated_hour_savings=0.0,
                net_load_without_battery_kwh=float(load[i] - solar[i]),
                discharge_budget_kwh=0.0,
                planned_gshp_kw=0.0,
                gshp_intent='STOP',
                planned_resistive_kw=0.0,
                resistive_heater_intent='OFF',
                gshp_temp_sim=float(acc_temp),
                planned_leaf_kw=0.0,
                leaf_intent='OFF',
            ))
        return plan
