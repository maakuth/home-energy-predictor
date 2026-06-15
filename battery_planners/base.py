"""Abstract base class and data structures for battery planners."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypedDict, NotRequired
from dataclasses import dataclass
import numpy as np


class BatteryPlannerContext(TypedDict, total=False):
    """Optional context data available to battery planners.

    All array fields, when present, must have the same length as the planning
    horizon (``len(predictions_kwh)``).  Planners must ignore keys they do
    not recognise so that the interface can be extended without breaking
    existing implementations.

    Keys
    ----
    outside_temps : np.ndarray
        Outside air temperature (°C). Useful for anticipating heating-load
        changes.
    is_sauna_active : np.ndarray
        Binary flag (0/1) indicating whether the sauna is expected to be on
        during each interval. Hot-water demand spikes can shift optimal
        discharge timing.
    ev_position : np.ndarray
        Binary flag (0/1) where ``1`` means the EV is at home. When the EV
        is away, its committed charging load disappears, freeing grid capacity.
    sarima_lower : np.ndarray
        95 % lower prediction bound for baseload (kW). Enables risk-aware
        / robust planning.
    sarima_upper : np.ndarray
        95 % upper prediction bound for baseload (kW).
    is_fallback_price : np.ndarray
        Binary flag (0/1) set to ``1`` when the price for the interval is a
        fallback estimate rather than a real day-ahead spot price. A cautious
        planner might avoid aggressive arbitrage on fallback prices.
    tomorrow_valid : bool
        ``True`` when tomorrow's day-ahead spot prices are already published
        (typically after 15:00 local time). A planner can cap its lookahead
        when this is ``False``.
    planned_gshp_kw : np.ndarray
        Planned GSHP electric load (kW). The battery planner already sees
        baseload + GSHP aggregated into ``predictions_kwh``, but having the
        dis-aggregated GSHP schedule lets a planner model heat-pump ramping
        constraints more precisely.
    current_acc_temp : float
        Current accumulator temperature (°C). Relevant if a planner wants to
        co-optimise battery with thermal storage.
    is_fireplace_currently_on : bool
        ``True`` when the fireplace is actively heating the accumulator. This
        reduces expected GSHP load and may create extra margin for battery
        discharge.
    model_version : str
        Semantic version string (e.g. ``"1.2.0"``) from the ``VERSION`` file.
    """
    outside_temps: NotRequired[np.ndarray]
    is_sauna_active: NotRequired[np.ndarray]
    ev_position: NotRequired[np.ndarray]
    sarima_lower: NotRequired[np.ndarray]
    sarima_upper: NotRequired[np.ndarray]
    is_fallback_price: NotRequired[np.ndarray]
    tomorrow_valid: NotRequired[bool]
    planned_gshp_kw: NotRequired[np.ndarray]
    current_acc_temp: NotRequired[float]
    is_fireplace_currently_on: NotRequired[bool]
    model_version: NotRequired[str]


@dataclass
class BatteryPlanEntry:
    """Represents a single interval's battery plan.

    All energy values are in kWh per interval.
    All prices are in EUR/kWh.
    """
    timestamp: str
    battery_action: str  # 'idle', 'charge_solar', 'charge_grid', 'discharge_load', 'discharge_export', etc.
    battery_power_kw: float  # Net power (positive = charging, negative = discharging)
    charge_from_solar_kwh: float
    charge_from_grid_kwh: float
    discharge_to_load_kwh: float
    discharge_to_export_kwh: float
    soc_kwh: float  # State of charge
    soc_pct: float  # SoC percentage
    grid_import_kwh: float
    grid_export_kwh: float
    estimated_hour_cost: float  # EUR for this interval
    estimated_hour_savings: float  # EUR saved vs no-battery baseline
    net_load_without_battery_kwh: float  # Load before battery adjustments

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'battery_action': self.battery_action,
            'battery_power_kw': float(self.battery_power_kw),
            'charge_from_solar_kwh': float(self.charge_from_solar_kwh),
            'charge_from_grid_kwh': float(self.charge_from_grid_kwh),
            'discharge_to_load_kwh': float(self.discharge_to_load_kwh),
            'discharge_to_export_kwh': float(self.discharge_to_export_kwh),
            'soc_kwh': float(self.soc_kwh),
            'soc_pct': float(self.soc_pct),
            'grid_import_kwh': float(self.grid_import_kwh),
            'grid_export_kwh': float(self.grid_export_kwh),
            'estimated_hour_cost': float(self.estimated_hour_cost),
            'estimated_hour_savings': float(self.estimated_hour_savings),
            'net_load_without_battery_kwh': float(self.net_load_without_battery_kwh),
        }


class BatteryPlanner(ABC):
    """Abstract base class for battery dispatch planning strategies.

    A battery planner receives predictions and price forecasts and produces
    a dispatch plan (list of actions) that optimizes battery usage according
    to the planner's strategy (heuristic, algorithmic, RL-based, etc).

    All planners should:
    - Respect battery physical constraints (capacity, charge/discharge rates)
    - Consider prices to minimize grid costs or maximize ROI
    - Return a list of BatteryPlanEntry objects for each interval
    """

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
        """
        Generate a battery dispatch plan.

        Args:
            predictions_kwh: Baseload predictions in kWh per interval.
            solar_kwh: Solar generation forecast in kWh per interval.
            import_prices: Grid import prices in EUR/kWh per interval.
            export_prices: Grid export prices in EUR/kWh per interval.
            prediction_timestamps: Timestamps for each interval (for logging).
            committed_load_kwh: Fixed loads (EV, Leaf) that reduce grid capacity
                but are not powered by the house battery. Optional; defaults to
                zeros if ``None``.
            allow_export: Whether the battery is allowed to export energy to the
                grid. Optional; defaults to ``True``.
            initial_soc_pct: Current battery state of charge in percent (0-100).
                If ``None``, the planner reads from the ``BATTERY_INITIAL_SOC_PCT``
                environment variable.
            context: Optional extra data (temperature, EV position, prediction
                uncertainty, etc.) provided by the orchestrator. Planners must
                ignore keys they do not recognise. See
                :class:`BatteryPlannerContext` for the full list of standard keys.

        Returns:
            List of :class:`BatteryPlanEntry` objects, one per interval.
        """
        pass
