# Battery Planner Cookbook

A practical reference for implementing new battery dispatch planners in the HEPO pluggable architecture.

---

## 1. Quick Start

The fastest way to get a working planner is to copy the example and fill in your logic.

```python
# battery_planners/my_planner.py
from typing import Optional, List, Any
import numpy as np
from .base import BatteryPlanner, BatteryPlanEntry, BatteryPlannerContext

class MyPlanner(BatteryPlanner):
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
        context: Optional[BatteryPlannerContext] = None,
    ) -> List[BatteryPlanEntry]:
        horizon = len(predictions_kwh)
        entries = []
        for i in range(horizon):
            entry = BatteryPlanEntry(
                timestamp=str(prediction_timestamps[i]),
                battery_action='idle',
                battery_power_kw=0.0,
                charge_from_solar_kwh=0.0,
                charge_from_grid_kwh=0.0,
                discharge_to_load_kwh=0.0,
                discharge_to_export_kwh=0.0,
                soc_kwh=0.0,
                soc_pct=0.0,
                grid_import_kwh=0.0,
                grid_export_kwh=0.0,
                estimated_hour_cost=0.0,
                estimated_hour_savings=0.0,
                net_load_without_battery_kwh=float(predictions_kwh[i]),
            )
            entries.append(entry)
        return entries
```

Register it in `battery_planners/__init__.py` and `factory.py`, then run the tests.
See `docs/BATTERY_PLANNER_ARCHITECTURE.md` for the full registration steps.

---

## 2. `BatteryPlanner.plan()` Interface Reference

| Parameter | Type | Shape | Units | Description |
|-----------|------|-------|-------|-------------|
| `predictions_kwh` | `np.ndarray` | `(horizon,)` | kWh per interval | Baseload + GSHP forecast (net load before battery). |
| `solar_kwh` | `np.ndarray` | `(horizon,)` | kWh per interval | Solar generation forecast. |
| `import_prices` | `np.ndarray` | `(horizon,)` | EUR/kWh | Grid import tariff per interval. |
| `export_prices` | `np.ndarray` | `(horizon,)` | EUR/kWh | Grid export tariff per interval. |
| `prediction_timestamps` | `List[Any]` | `(horizon,)` | — | Timestamps (usually `datetime` or ISO strings). |
| `committed_load_kwh` | `np.ndarray` | `(horizon,)` | kWh per interval | EV + Leaf charging loads that reduce grid capacity but are **not** served by the house battery. Defaults to zeros. |
| `allow_export` | `bool` | — | — | Whether the battery inverter may export energy to the grid. |
| `initial_soc_pct` | `float` | — | % (0–100) | Battery state of charge at the start of the plan. `None` means the planner should read `BATTERY_INITIAL_SOC_PCT` from the environment. |
| `context` | `BatteryPlannerContext` | — | — | Optional extra data. See section 3. |

**Return value:** `List[BatteryPlanEntry]` — one entry per interval, in the same order as the inputs.

---

## 3. `BatteryPlannerContext` Reference

`BatteryPlannerContext` is a `TypedDict` with `total=False`. All keys are optional. When an array key is present, its length **must equal `horizon`**.

| Key | Type | Units | Range / Notes | Typical Use |
|-----|------|-------|---------------|-------------|
| `outside_temps` | `np.ndarray` | °C | Any float | Anticipate heating-load changes. |
| `is_sauna_active` | `np.ndarray` | 0/1 | `0` or `1` | Predict hot-water demand spikes. |
| `ev_position` | `np.ndarray` | 0/1 | `0` or `1` | `1` = EV at home. When away, committed load disappears. |
| `sarima_lower` | `np.ndarray` | kW | Any float | 95 % lower prediction bound for baseload. |
| `sarima_upper` | `np.ndarray` | kW | Any float | 95 % upper prediction bound. |
| `is_fallback_price` | `np.ndarray` | 0/1 | `0` or `1` | `1` = price is a fallback estimate, not a real day-ahead price. |
| `tomorrow_valid` | `bool` | — | `True` / `False` | `True` when tomorrow's day-ahead prices are known (after ~15:00 local time). |
| `planned_gshp_kw` | `np.ndarray` | kW | Any float | Dis-aggregated GSHP schedule. Battery already sees GSHP aggregated into `predictions_kwh`. |
| `current_acc_temp` | `float` | °C | Any float | Current accumulator temperature. |
| `is_fireplace_currently_on` | `bool` | — | `True` / `False` | `True` when the fireplace is heating the accumulator, reducing expected GSHP load. |
| `model_version` | `str` | — | e.g. `"1.2.0"` | Semantic version from the `VERSION` file. |

**Safe access pattern:**

```python
if context is not None:
    outside_temps = context.get('outside_temps')
    if outside_temps is not None:
        # use it
```

---

## 4. `BatteryPlanEntry` Reference

All fields are mandatory when constructing the dataclass.

| Field | Type | Units | Consistency Rule |
|-------|------|-------|------------------|
| `timestamp` | `str` | ISO-8601 | — |
| `battery_action` | `str` | — | One of: `idle`, `charge_solar`, `charge_grid`, `charge_mixed`, `discharge_load`, `discharge_export`, `discharge_mixed`. |
| `battery_power_kw` | `float` | kW | Positive = charging. Must equal `(charge_total - discharge_total) / interval_hours`. |
| `charge_from_solar_kwh` | `float` | kWh | ≥ 0 |
| `charge_from_grid_kwh` | `float` | kWh | ≥ 0 |
| `discharge_to_load_kwh` | `float` | kWh | ≥ 0 |
| `discharge_to_export_kwh` | `float` | kWh | ≥ 0 |
| `soc_kwh` | `float` | kWh | Post-action state of charge. |
| `soc_pct` | `float` | % | `(soc_kwh / capacity_kwh) * 100`. |
| `grid_import_kwh` | `float` | kWh | ≥ 0. Net grid import after battery action. |
| `grid_export_kwh` | `float` | kWh | ≥ 0. Net grid export after battery action. |
| `estimated_hour_cost` | `float` | EUR | `grid_import * import_price - grid_export * export_price`. |
| `estimated_hour_savings` | `float` | EUR | `cost_no_battery - cost_with_battery`. |
| `net_load_without_battery_kwh` | `float` | kWh | `predictions_kwh[i] - solar_kwh[i]`. |

**Important:** `grid_import_kwh` and `grid_export_kwh` should never both be positive in the same interval. They represent the *net* exchange.

---

## 5. Battery Constraints

Every planner must respect these physical limits (usually by reading them from environment variables):

| Parameter | Env Var | Default | Units | Typical Range |
|-----------|---------|---------|-------|---------------|
| Capacity | `BATTERY_CAPACITY_KWH` | 40.0 | kWh | 5–100 |
| Min SoC | `BATTERY_MIN_SOC_PCT` | 10.0 | % | 0–20 |
| Max SoC | `BATTERY_MAX_SOC_PCT` | 90.0 | % | 80–100 |
| Reserve SoC | `BATTERY_RESERVE_SOC_PCT` | same as min | % | 0–min |
| Max charge power | `BATTERY_MAX_CHARGE_KW` | 10.0 | kW | 1–50 |
| Max discharge power | `BATTERY_MAX_DISCHARGE_KW` | 10.0 | kW | 1–50 |
| Charge efficiency | `BATTERY_CHARGE_EFFICIENCY` | 0.95 | — | 0.80–0.99 |
| Discharge efficiency | `BATTERY_DISCHARGE_EFFICIENCY` | 0.95 | — | 0.80–0.99 |
| Initial SoC | `BATTERY_INITIAL_SOC_PCT` | 50.0 | % | 0–100 |
| Grid fuse limit | `MAIN_FUSE_SIZE_A` | 25.0 | A | 16–63 |

**Grid import limit:** `max_grid_import_kw = MAIN_FUSE_SIZE_A * 3 * 0.230` (3-phase, 230 V).

**Round-trip efficiency:** `charge_eff * discharge_eff`. Typical ≈ 0.90.

---

## 6. Environment Variables

Beyond the battery constraints above, these tunables affect planner behaviour:

| Env Var | Default | Description |
|-----------|---------|-------------|
| `PLAN_INTERVAL_MINUTES` | 15 | Plan interval length (minutes). |
| `BATTERY_ALLOW_EXPORT` | True | Whether exporting is allowed (fallback if HA entity unavailable). |
| `BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH` | 0.005 | Minimum arbitrage margin for grid charging. |
| `HEPO_DISABLE_BATTERY` | False | Set to `1` to disable battery optimization entirely. |

---

## 7. Common Pitfalls

### SoC drift
If you forget to clamp `soc_kwh` to `[min_soc_kwh, max_soc_kwh]` after every action, the simulated SoC will drift and eventually violate constraints.

```python
soc_kwh = min(max(soc_kwh, min_soc_kwh), max_soc_kwh)
```

### Simultaneous import and export
`grid_import_kwh` and `grid_export_kwh` must never both be positive. They represent the *net* grid exchange.

```python
net = net_load + charge_from_solar + charge_from_grid - discharge_to_load - discharge_to_export + committed
grid_import_kwh = max(net, 0.0)
grid_export_kwh = max(-net, 0.0)
```

### Efficiency inversion
Remember that charging adds less energy than the input (loss), and discharging yields less than the stored energy (loss). The energy balance is:

```python
# After charging from solar
delta_kwh = charge_from_solar * charge_efficiency
# After discharging to load
delta_kwh = -discharge_to_load / discharge_efficiency
```

### Missing `initial_soc_pct`
When `initial_soc_pct` is `None`, the planner must fall back to the environment variable. Otherwise tests that don't pass a live SoC will fail.

```python
if initial_soc_pct is not None:
    effective_initial_soc_pct = float(initial_soc_pct)
else:
    effective_initial_soc_pct = get_env_float('BATTERY_INITIAL_SOC_PCT', 50.0)
```

---

## 8. Testing Your Planner

### 8.1 Unit test

```python
import numpy as np
from battery_planners import MyPlanner, BatteryPlanEntry

planner = MyPlanner()
plan = planner.plan(
    predictions_kwh=np.array([2.0, 2.0]),
    solar_kwh=np.array([0.0, 0.0]),
    import_prices=np.array([0.15, 0.15]),
    export_prices=np.array([0.05, 0.05]),
    prediction_timestamps=["t0", "t1"],
)
assert len(plan) == 2
assert all(isinstance(e, BatteryPlanEntry) for e in plan)
```

### 8.2 Replay test

The replay harness runs your planner against historical fixture data with time-aware visibility. It checks for:

- SoC constraint violations
- NaN / infinite costs
- Cost not worse than baseline + 10 %
- Valid output structure

```bash
# Run all replay tests
venv/bin/python3 -m pytest tests/test_battery_planner_replay.py -v

# Run only your planner
venv/bin/python3 -m pytest tests/test_battery_planner_replay.py -k "my_planner" -v
```

### 8.3 Interpreting scores

After each replay test, the console prints:

- `savings_eur` — absolute savings vs no-battery baseline
- `savings_pct` — percentage savings
- `soc_violations` — number of times SoC went outside [min, max]
- `final_soc_pct` — ending battery level

A good planner should show **positive savings** and **zero SoC violations**.

---

## 9. Reference Implementations

- **`HeuristicBatteryPlanner`** (`battery_planners/heuristic.py`) — The production planner. Opportunity-cost analysis, peak shaving, solar surplus, grid charging. Uses `max_lookahead_hours` parameter.
- **`SimpleRuleBasedPlanner`** (`battery_planners/example_simple_rule_based.py`) — Minimal example. Charges from solar, discharges during expensive hours. Good starting point for custom logic.

---

## 10. Further Reading

- `docs/BATTERY_PLANNER_ARCHITECTURE.md` — High-level architecture overview.
- `docs/BATTERY_PLANNER_REPLAY_TESTS.md` — Deep dive into the replay test system.
- `tests/test_battery_planner_pluggable.py` — Interface contract tests.
- `tests/test_battery_planner_replay.py` — Parametrized replay tests.
