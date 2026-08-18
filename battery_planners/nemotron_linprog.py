from __future__ import annotations
"""LP-native battery dispatch planner using scipy.optimize.linprog."""

import os
from typing import Any, List, Optional

import numpy as np
from scipy.optimize import linprog

from .base import BatteryPlanEntry, BatteryPlanner, BatteryPlannerContext, compute_discharge_budget
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


def _interval_discharge_budget(
    soc_kwh: float,
    min_soc_kwh: float,
    discharge_eff: float,
    max_discharge_kw: float,
    interval_hours: float,
    import_prices: np.ndarray,
    i: int,
    planned_discharge_kwh: float,
) -> float:
    """Return runtime discharge headroom without undercutting the LP plan."""
    idx = min(i, len(import_prices) - 1)
    current_price = float(import_prices[idx]) if len(import_prices) else 0.0
    future_prices = import_prices[i:]
    budget = compute_discharge_budget(
        soc_kwh, min_soc_kwh, discharge_eff, max_discharge_kw, interval_hours,
        current_price, future_prices,
        min_factor=get_env_float('BATTERY_FOLLOW_BUDGET_MIN_FACTOR', 0.10),
        spread_factor=get_env_float('BATTERY_FOLLOW_BUDGET_SPREAD_FACTOR', 2.5),
    )
    return max(budget, planned_discharge_kwh)


class NemotronLinprogPlanner(BatteryPlanner):
    """Minimise grid cost with an explicit physical energy-flow network.

    Each interval has independent grid-to-house, grid-to-battery,
    solar-to-house, solar-to-battery, solar-to-export, solar-curtailment,
    battery-to-house, and battery-to-export flows. This makes every public
    ``BatteryPlanEntry`` field traceable to an LP decision variable.
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
        """Generate a physically consistent battery dispatch plan."""
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
        if max_charge_kw < 0 or max_discharge_kw < 0:
            raise ValueError('battery power limits must not be negative')

        max_horizon = get_env_int('BATTERY_LP_HORIZON', 192)
        if not (context or {}).get('tomorrow_valid', False):
            max_horizon = get_env_int('BATTERY_LP_HORIZON_FALLBACK', 96)
        horizon = min(total_intervals, max(1, max_horizon))

        load = np.maximum(load, 0.0)
        solar = np.maximum(solar, 0.0)
        committed = np.maximum(committed, 0.0)
        import_price_floor = get_env_float('IMPORT_PRICE_FLOOR_EUR_PER_KWH', 0.0)
        if import_price_floor > 0:
            import_prices = np.maximum(import_prices, import_price_floor)

        main_fuse_a = get_env_float('MAIN_FUSE_SIZE_A', 25.0)
        max_grid_import_kwh = max(0.0, main_fuse_a * 3 * 0.230 * interval_hours)
        degradation_cost = max(0.0, get_env_float('BATTERY_DEGRADATION_COST_EUR_PER_KWH', 0.0))
        discount = np.clip(get_env_float('BATTERY_LP_DISCOUNT', 0.995), 0.0, 1.0)

        # Per-interval variables: grid_house, grid_battery, solar_house,
        # solar_battery, solar_export, solar_curtail, battery_house,
        # battery_export, soc, grid_overflow.
        width = 10
        n_vars = width * horizon

        def index(i: int, offset: int) -> int:
            return i * width + offset

        grid_house, grid_battery, solar_house, solar_battery = 0, 1, 2, 3
        solar_export, solar_curtail, battery_house, battery_export, soc, overflow = range(4, 10)

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

            bounds.extend([
                (0, None),  # grid_house
                (0, None),  # grid_battery
                (0, None),  # solar_house
                (0, None),  # solar_battery
                (0, None) if allow_export else (0, 0),  # solar_export
                (0, None),  # solar_curtail
                (0, None),  # battery_house
                (0, None) if allow_export else (0, 0),  # battery_export
                (min_soc_kwh, max_soc_kwh),
                (0, None),  # grid_overflow
            ])

        terminal_percentile = get_env_float('BATTERY_TERMINAL_VALUE_PERCENTILE', 0.0)
        if terminal_percentile > 0:
            terminal_price = float(np.percentile(import_prices[:horizon], terminal_percentile))
            objective[index(horizon - 1, soc)] = -terminal_price * discharge_eff * discount ** (horizon - 1)

        equal_rows: list[np.ndarray] = []
        equal_values: list[float] = []
        upper_rows: list[np.ndarray] = []
        upper_values: list[float] = []
        for i in range(horizon):
            # The house battery may serve house demand, but never committed load.
            row = np.zeros(n_vars)
            row[index(i, grid_house)] = 1
            row[index(i, solar_house)] = 1
            row[index(i, battery_house)] = 1
            equal_rows.append(row)
            equal_values.append(load[i])

            row = np.zeros(n_vars)
            for flow in (solar_house, solar_battery, solar_export, solar_curtail):
                row[index(i, flow)] = 1
            equal_rows.append(row)
            equal_values.append(solar[i])

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

            row = np.zeros(n_vars)
            row[index(i, grid_battery)] = 1
            row[index(i, solar_battery)] = 1
            upper_rows.append(row)
            upper_values.append(max_charge_kw * interval_hours)

            row = np.zeros(n_vars)
            row[index(i, battery_house)] = 1
            row[index(i, battery_export)] = 1
            upper_rows.append(row)
            upper_values.append(max_discharge_kw * interval_hours)

            # Fixed committed load consumes fuse capacity but is not a battery flow.
            row = np.zeros(n_vars)
            row[index(i, grid_house)] = 1
            row[index(i, grid_battery)] = 1
            row[index(i, overflow)] = -1
            upper_rows.append(row)
            upper_values.append(max_grid_import_kwh - committed[i])

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
            return self._idle_plan(
                load, solar, committed, import_prices, export_prices,
                prediction_timestamps, initial_soc_kwh, capacity_kwh,
                min_soc_kwh, max_discharge_kw, discharge_eff, interval_hours,
                allow_export,
            )

        plan: list[BatteryPlanEntry] = []
        x = result.x
        starting_soc = initial_soc_kwh
        for i in range(horizon):
            charge_solar = max(0.0, x[index(i, solar_battery)])
            charge_grid = max(0.0, x[index(i, grid_battery)])
            discharge_load = max(0.0, x[index(i, battery_house)])
            discharge_export = max(0.0, x[index(i, battery_export)])
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
            baseline_import = max(0.0, load[i] - solar[i]) + committed[i]
            baseline_export = max(0.0, solar[i] - load[i]) if allow_export else 0.0
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
                net_load_without_battery_kwh=float(load[i] - solar[i]),
                discharge_budget_kwh=_interval_discharge_budget(
                    starting_soc, min_soc_kwh, discharge_eff, max_discharge_kw,
                    interval_hours, import_prices[:horizon], i, discharge_load,
                ),
            ))
            starting_soc = ending_soc

        if horizon < total_intervals:
            plan.extend(self._idle_plan(
                load[horizon:], solar[horizon:], committed[horizon:],
                import_prices[horizon:], export_prices[horizon:], prediction_timestamps[horizon:],
                starting_soc, capacity_kwh, min_soc_kwh, max_discharge_kw,
                discharge_eff, interval_hours, allow_export,
            ))
        return plan

    @staticmethod
    def _timestamp(value: Any) -> str:
        return value if isinstance(value, str) else value.isoformat() if hasattr(value, 'isoformat') else str(value)

    def _idle_plan(
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
    ) -> List[BatteryPlanEntry]:
        """Return a feasible no-battery plan if the solver is unavailable."""
        plan = []
        for i, timestamp in enumerate(timestamps):
            solar_to_house = min(load[i], solar[i])
            surplus = solar[i] - solar_to_house
            grid_export = surplus if allow_export else 0.0
            grid_import = load[i] - solar_to_house + committed[i]
            plan.append(BatteryPlanEntry(
                timestamp=self._timestamp(timestamp), battery_action='idle', battery_power_kw=0.0,
                charge_from_solar_kwh=0.0, charge_from_grid_kwh=0.0,
                discharge_to_load_kwh=0.0, discharge_to_export_kwh=0.0,
                soc_kwh=float(soc_kwh), soc_pct=float(soc_kwh / capacity_kwh * 100.0),
                grid_import_kwh=float(grid_import), grid_export_kwh=float(grid_export),
                estimated_hour_cost=float(grid_import * import_prices[i] - grid_export * export_prices[i]),
                estimated_hour_savings=0.0, net_load_without_battery_kwh=float(load[i] - solar[i]),
                discharge_budget_kwh=_interval_discharge_budget(
                    soc_kwh, min_soc_kwh, discharge_eff, max_discharge_kw,
                    interval_hours, import_prices, i, 0.0,
                ),
            ))
        return plan
