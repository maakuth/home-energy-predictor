from __future__ import annotations

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Optional
from utils.ha_utils import get_ha_state

def align_interval_prices(
    raw_today: list[Any],
    raw_tomorrow: list[Any],
    prediction_timestamps: list[str],
    interval_minutes: int = 15,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    all_raw = raw_today + raw_tomorrow
    if not all_raw:
        return None, None

    # Handle if raw_today is a list of floats (like in your nordpool_total.yaml 'today' attribute)
    if all_raw and not isinstance(all_raw[0], dict):
        # Infer spacing from the list length.
        # 96 values -> 15-min spacing, 48 -> 30-min, 24 -> hourly, etc.
        # Partial lists (< 24 values) fall back to hourly as before.
        num_values = len(all_raw)
        if num_values >= 24:
            inferred_spacing_minutes = max(1, int(round(1440.0 / num_values)))
        else:
            inferred_spacing_minutes = 60
        start_date = pd.to_datetime(datetime.now().date(), utc=True)
        all_raw = [{"start": start_date + timedelta(minutes=i * inferred_spacing_minutes), "value": v} for i, v in enumerate(all_raw)]

    df_prices = pd.DataFrame(all_raw)
    if "start" not in df_prices.columns or "value" not in df_prices.columns:
        return None, None

    # Convert to datetime and ensure it's timezone-aware (matching prediction_timestamps)
    df_prices["start"] = pd.to_datetime(df_prices["start"], utc=True)
    df_prices = df_prices.drop_duplicates(subset="start").set_index("start").sort_index()

    # Resample to the target interval.
    interval_prices_series = df_prices["value"].resample(f"{interval_minutes}min").ffill()

    # Reindex to match prediction_timestamps exactly (initially WITHOUT ffill to identify NaNs)
    target_index = pd.to_datetime(prediction_timestamps, utc=True)
    aligned_series = interval_prices_series.reindex(target_index)

    # Identify where data is missing (fallback candidates)
    is_fallback = aligned_series.isna()

    # Apply 24h fallback
    for i in range(len(aligned_series)):
        if is_fallback.iloc[i]:
            ts = aligned_series.index[i]
            past_ts = ts - timedelta(days=1)
            # Try to find value for same time yesterday
            if past_ts in interval_prices_series.index:
                aligned_series.iloc[i] = interval_prices_series.loc[past_ts]

    # Final catch-all: ffill from last available price (of today or synthesized)
    # and bfill for any gaps at the very start
    aligned_series = aligned_series.ffill().bfill()

    return aligned_series.to_numpy(), is_fallback.to_numpy()


def _fetch_sensor_prices(
    entity_id: str,
    prediction_timestamps: list[str],
    interval_minutes: int,
) -> Optional[np.ndarray]:
    """Fetch and align prices from a single HA sensor entity."""
    state_data = get_ha_state(entity_id)
    if not state_data:
        return None

    attrs = state_data.get("attributes", {})
    raw_today = attrs.get("raw_today") or attrs.get("today")
    raw_tomorrow = attrs.get("raw_tomorrow") or attrs.get("tomorrow")

    # ENTSO-e format: prices_today/prices_tomorrow/prices as list of {time, price}
    if raw_today is None:
        raw_today = attrs.get("prices_today") or attrs.get("prices")
        if isinstance(raw_today, list) and len(raw_today) > 0 and isinstance(raw_today[0], dict):
            raw_today = [{"start": item.get("time") or item.get("start"), "value": item["price"]} for item in raw_today]
    if raw_tomorrow is None:
        raw_tomorrow = attrs.get("prices_tomorrow")
        if isinstance(raw_tomorrow, list) and len(raw_tomorrow) > 0 and isinstance(raw_tomorrow[0], dict):
            raw_tomorrow = [{"start": item.get("time") or item.get("start"), "value": item["price"]} for item in raw_tomorrow]

    if isinstance(raw_today, list) and len(raw_today) > 0:
        aligned, _ = align_interval_prices(raw_today, raw_tomorrow or [], prediction_timestamps, interval_minutes)
        return aligned
    return None


def fetch_market_prices(
    prediction_timestamps: list[str],
    interval_minutes: int = 15,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], bool, bool, Optional[np.ndarray]]:
    candidate_sensors = [
        "sensor.nordpool_total",
        "sensor.nordpool_kwh_fi_eur_3_10_0",
        "sensor.average_electricity_price_today",
        "sensor.current_electricity_market_price",
    ]

    for sensor in candidate_sensors:
        state_data = get_ha_state(sensor)
        if not state_data:
            continue

        attrs = state_data.get("attributes", {})
        
        # Standard format (Nordpool integration)
        raw_today = attrs.get("raw_today") or attrs.get("today")
        raw_tomorrow = attrs.get("raw_tomorrow") or attrs.get("tomorrow")
        
        # Alternative format (ENTSO-e integration)
        if raw_today is None:
            raw_today = attrs.get("prices_today")
            if raw_today:
                # Normalize ENTSO-e format to Nordpool-like list of dicts with 'start' and 'value'
                raw_today = [{"start": item["time"], "value": item["price"]} for item in raw_today]
        
        if raw_tomorrow is None:
            raw_tomorrow = attrs.get("prices_tomorrow")
            if raw_tomorrow:
                raw_tomorrow = [{"start": item["time"], "value": item["price"]} for item in raw_tomorrow]

        if isinstance(raw_today, list) and len(raw_today) > 0:
            aligned, is_fallback = align_interval_prices(raw_today, raw_tomorrow or [], prediction_timestamps, interval_minutes)
            if aligned is not None:
                # Flag if this sensor is known to include additional costs
                is_inclusive = (sensor == "sensor.nordpool_total")
                # Check tomorrow_valid for Nordpool sensors
                tomorrow_valid = False
                if sensor in ["sensor.nordpool_kwh_fi_eur_3_10_0", "sensor.nordpool_total"]:
                    tomorrow_valid = bool(attrs.get("tomorrow_valid", False))
                # When using nordpool_total (inclusive), fetch separate export prices
                # from average_electricity_price_today (raw energy only)
                export_prices_base = None
                if sensor == "sensor.nordpool_total":
                    export_prices_base = _fetch_sensor_prices("sensor.average_electricity_price_today", prediction_timestamps, interval_minutes)
                return aligned, is_fallback, sensor, is_inclusive, tomorrow_valid, export_prices_base

    return None, None, None, False, False, None
