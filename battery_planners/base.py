"""Abstract base class and data structures for battery planners."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np


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
    ) -> List[BatteryPlanEntry]:
        """
        Generate a battery dispatch plan.
        
        Args:
            predictions_kwh: Baseload predictions in kWh per interval
            solar_kwh: Solar generation forecast in kWh per interval
            import_prices: Grid import prices in EUR/kWh per interval
            export_prices: Grid export prices in EUR/kWh per interval
            prediction_timestamps: Timestamps for each interval (for logging)
            committed_load_kwh: Fixed loads (EV, Leaf) that reduce grid capacity but
                                are not powered by house battery (optional)
            allow_export: Whether battery can export to grid (optional)
        
        Returns:
            List of BatteryPlanEntry objects, one per interval
        """
        pass
