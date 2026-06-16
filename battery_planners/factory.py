"""Factory for creating battery planner instances."""

import os
from typing import Optional
from .base import BatteryPlanner


class BatteryPlannerFactory:
    """Factory for creating battery planner instances based on configuration."""
    
    _planners = {}
    
    @classmethod
    def register(cls, name: str, planner_class: type) -> None:
        """Register a planner class by name."""
        cls._planners[name.lower()] = planner_class
    
    @classmethod
    def names(cls) -> tuple:
        """Get sorted tuple of registered planner names."""
        return tuple(sorted(cls._planners.keys()))
    
    @classmethod
    def create(cls, planner_type: Optional[str] = None) -> BatteryPlanner:
        """
        Create a battery planner instance.
        
        Args:
            planner_type: Name of planner type to create. If None, reads from
                         BATTERY_PLANNER_TYPE environment variable.
                         Defaults to 'heuristic' if not specified.
        
        Returns:
            BatteryPlanner instance
        
        Raises:
            ValueError: If planner type is unknown
        """
        if planner_type is None:
            planner_type = os.getenv('BATTERY_PLANNER_TYPE', 'heuristic').lower()
        else:
            planner_type = planner_type.lower()
        
        if planner_type not in cls._planners:
            available = ', '.join(sorted(cls._planners.keys()))
            raise ValueError(
                f"Unknown battery planner type: '{planner_type}'. "
                f"Available: {available}"
            )
        
        print(f"Using battery planner: {planner_type}")
        planner_class = cls._planners[planner_type]
        return planner_class()


# Import and register planners after factory class definition
from .heuristic import HeuristicBatteryPlanner
from .nemotron_linprog import NemotronLinprogPlanner

BatteryPlannerFactory.register('heuristic', HeuristicBatteryPlanner)
BatteryPlannerFactory.register('nemotron-linprog', NemotronLinprogPlanner)
