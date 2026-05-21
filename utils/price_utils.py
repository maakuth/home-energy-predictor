
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.ha_utils import get_ha_state

def align_interval_prices(raw_today, raw_tomorrow, prediction_timestamps, interval_minutes=15):
    all_raw = raw_today + raw_tomorrow
    if not all_raw:
        return None, None

    # Handle if raw_today is a list of floats (like in your nordpool_total.yaml 'today' attribute)
    if all_raw and not isinstance(all_raw[0], dict):
        # We assume these are hourly prices starting from today at 00:00
        start_date = pd.to_datetime(datetime.now().date(), utc=True)
        all_raw = [{"start": start_date + timedelta(hours=i), "value": v} for i, v in enumerate(all_raw)]

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

    return aligned_series.values, is_fallback.values


def fetch_market_prices(prediction_timestamps, interval_minutes=15):
    candidate_sensors = [
        "sensor.average_electricity_price_today",
        "sensor.current_electricity_market_price",
        "sensor.nordpool_kwh_fi_eur_3_10_0",
        "sensor.nordpool_total",
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
                return aligned, is_fallback, sensor, is_inclusive

    return None, None, None, False
