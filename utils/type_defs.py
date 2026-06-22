from __future__ import annotations
from typing import TypedDict, NotRequired, Any
from typing import Literal
import numpy as np
import pandas as pd


BatteryAction = Literal[
    'idle',
    'follow',
    'charge_solar',
    'charge_grid',
    'charge_mixed',
    'discharge_load',
    'discharge_export',
    'discharge_mixed',
]
"""Valid battery dispatch actions."""


class HaState(TypedDict):
    """Shape of a Home Assistant REST API state response."""
    entity_id: NotRequired[str]
    state: str
    attributes: dict[str, Any]
    last_changed: str
    last_updated: str
    context: dict[str, Any]


class PriceEntry(TypedDict):
    """A single price entry from Nordpool/ENTSO-e."""
    start: str
    value: float


class PriceData(TypedDict, total=False):
    """Aligned price data with fallback flags."""
    import_prices: np.ndarray
    export_prices: np.ndarray
    is_fallback: np.ndarray
    sensor: str
    is_inclusive: bool
    tomorrow_valid: bool


class PriceRecord(TypedDict):
    """A single price record from Nordpool/HA."""
    timestamp: str
    import_price: float
    export_price: float


class BatteryConfig(TypedDict):
    """Battery hardware configuration from environment."""
    capacity_kwh: float
    min_soc_pct: float
    max_soc_pct: float
    charge_rate_kw: float
    discharge_rate_kw: float
    enabled: bool


class GshpConfig(TypedDict):
    """GSHP configuration from environment."""
    enabled: bool
    max_power_kw: float


class FuturePredictionRecord(TypedDict, total=False):
    """A single prediction entry from future_predictions.json."""
    timestamp: str
    predicted_baseload: float
    predicted_usage: float
    solar_forecast: float
    outside_temp: float
    ev_position: int
    is_sauna_active: int
    is_fallback_price: int


class SqlitePredictionRecord(TypedDict, total=False):
    """A single prediction/plan record from the SQLite predictions table."""
    target_timestamp: str
    generated_at: str
    predicted_usage_kw: float
    solar_forecast_kw: float
    is_fallback_price: int
    import_price: float
    export_price: float
    battery_action: str
    battery_power_kw: float
    battery_soc_pct: float
    grid_import_kwh: float
    grid_export_kwh: float
    charge_from_solar_kwh: float
    charge_from_grid_kwh: float
    discharge_to_load_kwh: float
    discharge_to_export_kwh: float
    planned_gshp_kw: float
    gshp_intent: str


class PlanEntryDict(TypedDict):
    """Serialized form of BatteryPlanEntry (as stored in optimization_plan.json)."""
    timestamp: str
    battery_action: str
    battery_power_kw: float
    charge_from_solar_kwh: float
    charge_from_grid_kwh: float
    discharge_to_load_kwh: float
    discharge_to_export_kwh: float
    soc_kwh: float
    soc_pct: float
    grid_import_kwh: float
    grid_export_kwh: float
    estimated_hour_cost: float
    estimated_hour_savings: float
    net_load_without_battery_kwh: float


class StatesHistoryDict(TypedDict):
    """Mapping from entity_id to DataFrame of historical states."""
    pass  # dict[str, pd.DataFrame] — can't type narrow further here
