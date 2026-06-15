# Pluggable Battery Planner Architecture

## Overview

The battery planner has been refactored into a pluggable architecture that allows different optimization strategies to be swapped in and out without modifying the core `optimize_plan.py` logic.

## Architecture

### Abstract Base Class: `BatteryPlanner`

Located in `battery_planners/base.py`, this defines the interface all planners must implement:

```python
class BatteryPlanner(ABC):
    @abstractmethod
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
        """Generate a battery dispatch plan."""
        pass
```

### Typed Context: `BatteryPlannerContext`

`BatteryPlannerContext` is a `TypedDict` (defined in `battery_planners/base.py`) that carries optional extra data the orchestrator can provide, such as:

- `outside_temps` (np.ndarray, °C)
- `is_sauna_active` (np.ndarray, 0/1)
- `ev_position` (np.ndarray, 0/1)
- `sarima_lower` / `sarima_upper` (np.ndarray, kW)
- `is_fallback_price` (np.ndarray, 0/1)
- `tomorrow_valid` (bool)
- `planned_gshp_kw` (np.ndarray, kW)
- `current_acc_temp` (float, °C)
- `is_fireplace_currently_on` (bool)
- `model_version` (str)

Planners **must ignore keys they do not recognise** so the interface can be extended without breaking existing implementations.

### Data Structure: `BatteryPlanEntry`

A dataclass representing a single interval's battery plan, containing:
- `battery_action`: Type of action (idle, charge_solar, discharge_load, etc.)
- `soc_pct`: Battery state of charge
- `grid_import_kwh`, `grid_export_kwh`: Grid exchange
- `estimated_hour_cost`, `estimated_hour_savings`: Economic metrics
- Plus all other battery-related fields

Can be easily converted to dict for JSON serialization: `entry.to_dict()`

### Concrete Implementation: `HeuristicBatteryPlanner`

Located in `battery_planners/heuristic.py`, this is the original HEPO algorithm refactored into a pluggable class. It implements the `BatteryPlanner` interface using:
- Opportunity cost analysis
- Peak shaving
- Solar surplus charging
- Smart grid charging during cheap price windows

All the heuristics are preserved exactly as they were in the original code.

### Factory: `BatteryPlannerFactory`

Located in `battery_planners/factory.py`, provides a simple way to create planner instances:

```python
# Default: uses BATTERY_PLANNER_TYPE env var, falls back to 'heuristic'
planner = BatteryPlannerFactory.create()

# Explicit type
planner = BatteryPlannerFactory.create('heuristic')

# Case-insensitive
planner = BatteryPlannerFactory.create('HEURISTIC')
```

## Usage in `optimize_plan.py`

The main `optimize()` function now uses the factory:

```python
if is_battery_enabled():
    planner = BatteryPlannerFactory.create()
    battery_plan_entries = planner.plan(
        predictions_kwh, solar_kwh, import_prices, export_prices,
        prediction_timestamps, committed_load_kwh, allow_export=allow_export,
        initial_soc_pct=current_battery_soc_pct,
        context=battery_context,
    )
    # Convert to dicts for compatibility with rest of code
    battery_plan = [entry.to_dict() for entry in battery_plan_entries]
else:
    battery_plan = plan_no_battery_dispatch(...)
```

## How to Add a New Planner

### 1. Create a new planner class

```python
# battery_planners/my_algorithm.py
from typing import Optional
from .base import BatteryPlanner, BatteryPlanEntry, BatteryPlannerContext
import numpy as np

class MyAlgorithmPlanner(BatteryPlanner):
    def plan(self, predictions_kwh, solar_kwh, import_prices, export_prices,
             prediction_timestamps, committed_load_kwh=None, allow_export=True,
             initial_soc_pct=None, context: Optional[BatteryPlannerContext] = None):
        """Your algorithm here.

        The ``context`` dict carries optional extra data (temperature, EV
        position, prediction uncertainty, etc.).  Ignore keys you don't need
        so the interface can be extended without breaking your planner.
        """
        horizon = len(predictions_kwh)
        entries = []
        
        # Example: read a context key if it exists
        if context is not None:
            outside_temps = context.get('outside_temps')
            if outside_temps is not None:
                # Use temperature for heating-load aware decisions
                pass
        
        for i in range(horizon):
            # Your logic
            entry = BatteryPlanEntry(
                timestamp=prediction_timestamps[i],
                battery_action='idle',
                battery_power_kw=0.0,
                # ... fill in all required fields
            )
            entries.append(entry)
        
        return entries
```

### 2. Register the planner

```python
# battery_planners/__init__.py
from .my_algorithm import MyAlgorithmPlanner

# ... existing imports ...

__all__ = [
    'BatteryPlanner',
    'BatteryPlanEntry',
    'HeuristicBatteryPlanner',
    'MyAlgorithmPlanner',  # Add this
    'BatteryPlannerFactory',
]

# In factory.py after class definition:
BatteryPlannerFactory.register('my_algorithm', MyAlgorithmPlanner)
```

### 3. Test it

```python
# Test directly
planner = MyAlgorithmPlanner()
plan = planner.plan(predictions, solar, prices_import, prices_export, timestamps)

# Test via factory
planner = BatteryPlannerFactory.create('my_algorithm')
plan = planner.plan(predictions, solar, prices_import, prices_export, timestamps)

# Select via environment variable
os.environ['BATTERY_PLANNER_TYPE'] = 'my_algorithm'
planner = BatteryPlannerFactory.create()
```

## Benefits of This Architecture

1. **Testability**: Each planner can be tested in isolation
2. **Maintainability**: Algorithm logic is separated from orchestration
3. **Extensibility**: New planners can be added without touching `optimize_plan.py`
4. **Swappability**: Different algorithms can be compared on the same data
5. **Backward Compatibility**: Existing code using dicts still works
6. **Configuration-Driven**: Select planner via `BATTERY_PLANNER_TYPE` env var

## Environment Variables

- `BATTERY_PLANNER_TYPE` (default: `heuristic`): Selects which planner to use
- All existing battery configuration variables still work:
  - `BATTERY_CAPACITY_KWH`
  - `BATTERY_MIN_SOC_PCT`
  - `BATTERY_MAX_SOC_PCT`
  - etc.

## Algorithmic Opportunities

The current heuristic approach is all about **maximizing profit** through:
- Peak shaving (discharge during expensive periods)
- Cheap window charging
- Opportunity cost analysis
- Solar surplus utilization

Alternative algorithms could optimize for:
- **Grid stability**: Reduce ramps, smooth imports/exports
- **Self-consumption**: Maximize local use of solar
- **Resilience**: Keep battery charged for outages
- **Learning-based**: Use RL to discover optimal patterns
- **Linear programming**: Solve the optimization problem exactly (for smaller horizons)
- **Mixed objectives**: Balance profit with other goals

The pluggable architecture makes it easy to experiment with any of these approaches.

## Testing

Run all tests:
```bash
venv/bin/python3 -m pytest tests/test_battery_planner_pluggable.py -v
```

Run existing battery tests with backward-compat wrapper:
```bash
venv/bin/python3 -m pytest tests/test_optimize_plan.py -k battery -v
```
