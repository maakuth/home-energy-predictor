"""Battery planning strategies for HEPO.

This module provides pluggable battery dispatch planners with different
optimization approaches. The current heuristic-based implementation focuses
on opportunity cost and peak price avoidance, but alternative algorithmic
approaches can be plugged in.
"""

from .base import BatteryPlanner, BatteryPlanEntry, BatteryPlannerContext
from .heuristic import HeuristicBatteryPlanner
from .factory import BatteryPlannerFactory

__all__ = [
    'BatteryPlanner',
    'BatteryPlanEntry',
    'BatteryPlannerContext',
    'HeuristicBatteryPlanner',
    'BatteryPlannerFactory',
]
