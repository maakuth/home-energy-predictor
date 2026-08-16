from __future__ import annotations
"""Abstract base class and data structures for battery planners."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypedDict, NotRequired, Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
from utils.type_defs import BatteryAction


def should_idle_interval(
    net_kw: float,
    max_battery_kw: float,
    degradation_cost_per_kwh: float,
    interval_hours: float,
    charge_eff: float,
    discharge_eff: float,
    import_price: float,
    export_price: float,
    future_import_prices: Optional[np.ndarray] = None,
) -> bool:
    """Decide if the battery should truly idle instead of load-following.

    When a planner assigns zero dispatch for an interval, the real-time layer
    may still load-follow (try to zero out the grid meter) — this is called
    'follow'. True 'idle' means the battery does nothing, no matter what the
    grid meter says.

    This function evaluates whether load-following would do more harm than
    good based on physical feasibility and cost-benefit:
      1. If net load exceeds battery capacity → futile → idle
      2. If net load is negligible → not worth cycling → idle
      3. If no degradation cost configured → follow (legacy behaviour)
      4. If discharging at ``import_price`` is cheaper than the cost of
         refilling the battery later (cheapest future import price through
         round-trip efficiency + cycling cost) → idle
      5. If cycling cost (degradation + efficiency loss) > grid benefit → idle

    ``future_import_prices`` is the remaining import-price forecast after the
    current interval.  Passing it enables the price-aware check (rule 4):
    load-following discharges the battery now, and if the energy has to be
    bought back later at a price that makes the round trip a net loss, the
    battery should instead idle and buy from the grid now.  When omitted the
    check is skipped (legacy behaviour).
    """
    net_abs = abs(net_kw)

    # Futile: battery can't cancel this flow even at max power
    if net_abs > max_battery_kw:
        return True

    # Too small to matter (below typical deadband)
    if net_abs < 0.2:
        return True

    # No degradation cost configured → follow (legacy behaviour)
    if degradation_cost_per_kwh <= 0:
        return False

    # Price-aware refill check: discharging to load now at `import_price`
    # forces a future recharge.  If that recharge costs more than buying from
    # the grid right now, load-following is a net loss → idle.
    if (
        net_kw > 0
        and future_import_prices is not None
        and len(future_import_prices) > 0
    ):
        round_trip_eff = charge_eff * discharge_eff
        if round_trip_eff > 0.01:
            refill_price = float(np.min(future_import_prices))
            refill_cost = refill_price / round_trip_eff + 2 * degradation_cost_per_kwh
            if import_price < refill_cost:
                return True

    # Cost-benefit: what does load-following cost vs save?
    energy_kwh = net_abs * interval_hours
    round_trip_degradation = energy_kwh * degradation_cost_per_kwh * 2
    round_trip_eff = charge_eff * discharge_eff
    loss_kwh = energy_kwh * (1 - round_trip_eff)
    eff_cost = loss_kwh * max(import_price, export_price, 0.01)
    benefit = energy_kwh * (import_price if net_kw > 0 else export_price)

    return (round_trip_degradation + eff_cost) > benefit


def compute_discharge_budget(
    soc_kwh: float,
    min_soc_kwh: float,
    discharge_eff: float,
    max_discharge_kw: float,
    interval_hours: float,
    current_price: float,
    future_prices: np.ndarray,
    min_factor: float = 0.10,
    spread_factor: float = 2.5,
) -> float:
    """Compute the max discharge-to-load kWh budget for a single interval.

    The budget is scaled by how much the current import price is *below* the
    most expensive interval still ahead in the horizon (the "spread").  When a
    genuinely more expensive period is coming, the budget is tight so the
    battery conserves energy for it.  When the horizon is flat (e.g. summer
    days with plenty of sun and little intraday price difference), the spread
    is small and the budget stays generous, so the battery freely covers
    night-time loads (e.g. EV charging) from solar-charged storage instead of
    forcing a grid import.

    This replaces the old percentile-based scaling, which treated mid-tier
    night prices as "worth conserving for" even when nothing significantly
    more expensive was coming, causing needless grid imports on flat days.

    The budget is always at least ``min_factor`` of the maximum interval
    discharge, so the battery still contributes *some* load-following during
    the cheapest periods (unlike the old all-or-nothing idle logic).

    Args:
        soc_kwh: Current battery state of charge (kWh).
        min_soc_kwh: Minimum allowed SoC (kWh).
        discharge_eff: Discharge efficiency (0..1).
        max_discharge_kw: Max discharge power (kW).
        interval_hours: Interval length (hours).
        current_price: Current import price (EUR/kWh).
        future_prices: Import prices from the current interval onward.
        min_factor: Minimum budget fraction of max interval discharge
            (applied when the price spread is large).
        spread_factor: How aggressively the budget shrinks as the gap between
            the current price and the future price peak grows.  Set to 0.0 to
            disable the spread scaling entirely (budget always at full
            max interval discharge, legacy behaviour).

    Returns:
        Max discharge-to-load kWh budget for this interval.
    """
    max_interval_kwh = max_discharge_kw * interval_hours
    usable_kwh = max(0.0, (soc_kwh - min_soc_kwh) * discharge_eff)

    if usable_kwh < 1e-9:
        return 0.0

    if len(future_prices) == 0 or spread_factor <= 0.0:
        factor = 1.0
    else:
        peak_price = float(np.max(future_prices))
        if peak_price <= 0.0:
            factor = 1.0
        else:
            # spread in (0, 1]: 0 when current == peak (nothing better ahead),
            # approaching 1 when current is far below the future peak.
            spread = max(0.0, min(1.0, (peak_price - current_price) / peak_price))
            factor = 1.0 - spread_factor * spread
            factor = max(min_factor, min(1.0, factor))
    return min(max_interval_kwh, usable_kwh) * factor


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
    battery_action: BatteryAction
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
    discharge_budget_kwh: Optional[float] = None  # Max kWh battery may discharge to load this interval (None = unlimited)

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
            'discharge_budget_kwh': float(self.discharge_budget_kwh) if self.discharge_budget_kwh is not None else None,
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
        committed_load_kwh: Optional[np.ndarray] = None,
        allow_export: bool = True,
        initial_soc_pct: Optional[float] = None,
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


@runtime_checkable
class BatteryPlannerProtocol(Protocol):
    """Structural typing protocol for battery planners.

    Any object with a matching ``plan()`` method satisfies this protocol,
    enabling duck-typed planner implementations that don't need to inherit
    from :class:`BatteryPlanner`.
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
        ...
