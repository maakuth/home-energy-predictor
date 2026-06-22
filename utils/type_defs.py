from __future__ import annotations
from typing import TypedDict, NotRequired, Any
import numpy as np
import pandas as pd


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


class StatesHistoryDict(TypedDict):
    """Mapping from entity_id to DataFrame of historical states."""
    pass  # dict[str, pd.DataFrame] — can't type narrow further here
