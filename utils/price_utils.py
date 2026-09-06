from __future__ import annotations

import os
import json
import time
import pandas as pd
import numpy as np
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from utils.ha_utils import get_ha_state


NORDPOOL_PREDICTION_URL = 'https://raw.githubusercontent.com/vividfog/nordpool-predict-fi/main/deploy/prediction.json'
NORDPOOL_PREDICTION_CACHE_FILE = 'state/nordpool_prediction.json'
NORDPOOL_PREDICTION_CACHE_MAX_AGE_SECONDS = 3600


def get_grid_fees() -> float:
    return float(os.getenv('GRID_FEES_EUR_PER_KWH', '0.06'))


def estimate_export_prices(import_prices: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(0.0, np.asarray(import_prices, dtype=float) - get_grid_fees())


def estimate_import_prices(export_prices: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(export_prices, dtype=float) + get_grid_fees()


def update_monthly_spot_reference(
    settled_prices: np.ndarray,
    included_day: date,
    state_file: str = 'state/kulutusvaikutus_state.json',
) -> Optional[float]:
    """Persist each settled local day once and return its month-to-date mean."""
    state_path = Path(state_file)
    month = included_day.strftime('%Y-%m')
    day = included_day.isoformat()
    try:
        with state_path.open() as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = {}

    if state.get('month') != month:
        state = {
            'month': month,
            'latest_included_day': None,
            'spot_price_sum_eur_per_kwh': 0.0,
            'spot_price_count': 0,
            'spot_price_mean_eur_per_kwh': None,
        }

    latest_day = state.get('latest_included_day')
    if latest_day is None or day > latest_day:
        prices = np.asarray(settled_prices, dtype=float)
        prices = prices[np.isfinite(prices)]
        if len(prices) > 0:
            state['spot_price_sum_eur_per_kwh'] += float(prices.sum())
            state['spot_price_count'] += int(len(prices))
            state['latest_included_day'] = day

    count = state['spot_price_count']
    state['spot_price_mean_eur_per_kwh'] = (
        state['spot_price_sum_eur_per_kwh'] / count if count else None
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(state_path.suffix + '.tmp')
    with temp_path.open('w') as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, state_path)
    return state['spot_price_mean_eur_per_kwh']


def fetch_settled_daily_spot_prices(entity_id: str) -> Optional[np.ndarray]:
    """Return the selected sensor's complete settled raw-energy price day."""
    sources = [(entity_id, False)]
    if entity_id == 'sensor.nordpool_total':
        sources = [
            ('sensor.average_electricity_price_today', False),
            ('sensor.nordpool_total', True),
        ]

    for source, is_inclusive in sources:
        state_data = get_ha_state(source)
        if not state_data:
            continue
        attrs = state_data.get('attributes', {})
        raw_today = attrs.get('raw_today') or attrs.get('today') or attrs.get('prices_today') or attrs.get('prices')
        if not isinstance(raw_today, list) or not raw_today:
            continue
        if isinstance(raw_today[0], dict):
            values = [entry.get('value', entry.get('price')) for entry in raw_today]
        else:
            values = raw_today
        try:
            prices = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        prices = prices[np.isfinite(prices)]
        if len(prices):
            return np.maximum(0.0, prices - get_grid_fees()) if is_inclusive else prices
    return None


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


def _fetch_predicted_spot_prices() -> Optional[pd.Series]:
    """Fetch raw spot-price predictions in EUR/kWh from nordpool-predict-fi."""
    url = os.getenv('NORDPOOL_PREDICTION_URL', NORDPOOL_PREDICTION_URL)
    cache_path = Path(os.getenv('NORDPOOL_PREDICTION_CACHE_FILE', NORDPOOL_PREDICTION_CACHE_FILE))
    cached_payload: Optional[list[Any]] = None
    cache_is_fresh = False
    try:
        with cache_path.open() as f:
            payload = json.load(f)
        if isinstance(payload, list) and payload:
            cached_payload = payload
            cache_is_fresh = time.time() - cache_path.stat().st_mtime <= NORDPOOL_PREDICTION_CACHE_MAX_AGE_SECONDS
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    if cache_is_fresh:
        payload = cached_payload
    else:
        payload = None
    try:
        if payload is None:
            timeout = float(os.getenv('NORDPOOL_PREDICTION_TIMEOUT_SECONDS', '10'))
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list) and payload:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
                with temp_path.open('w') as f:
                    json.dump(payload, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, cache_path)
    except (requests.RequestException, TypeError, ValueError):
        payload = cached_payload
    except OSError:
        pass

    if not isinstance(payload, list):
        return None

    timestamps: list[Any] = []
    prices: list[float] = []
    for entry in payload:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        try:
            timestamp_ms = float(entry[0])
            price_eur_per_kwh = float(entry[1]) / 100.0
        except (TypeError, ValueError):
            continue
        if np.isfinite(timestamp_ms) and np.isfinite(price_eur_per_kwh):
            timestamps.append(timestamp_ms)
            prices.append(price_eur_per_kwh)

    if not timestamps:
        return None

    series = pd.Series(prices, index=pd.to_datetime(timestamps, unit='ms', utc=True))
    return series[~series.index.duplicated(keep='last')].sort_index()


def _align_predicted_spot_prices(
    prices: pd.Series,
    prediction_timestamps: list[str],
    interval_minutes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align hourly predictions without extending values beyond their horizon."""
    target_index = pd.to_datetime(prediction_timestamps, utc=True)
    interval = pd.Timedelta(minutes=interval_minutes)
    aligned = prices.resample(f'{interval_minutes}min').ffill().reindex(target_index, method='ffill')

    if len(prices) > 1:
        source_interval = pd.Timedelta(prices.index.to_series().diff().dropna().median())
    else:
        source_interval = pd.Timedelta(hours=1)
    if pd.isna(source_interval) or source_interval <= pd.Timedelta(0):
        source_interval = pd.Timedelta(hours=1)

    covered = (
        (target_index >= prices.index.min())
        & (target_index < prices.index.max() + source_interval)
        & aligned.notna().to_numpy()
    )
    return aligned.to_numpy(dtype=float), np.asarray(covered, dtype=bool)


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
        source = sensor
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

                if not tomorrow_valid:
                    predicted_prices = _fetch_predicted_spot_prices()
                    if predicted_prices is not None:
                        external_aligned, external_covered = _align_predicted_spot_prices(
                            predicted_prices, prediction_timestamps, interval_minutes,
                        )
                        use_external = np.asarray(is_fallback, dtype=bool) & external_covered
                        if np.any(use_external):
                            aligned = np.array(aligned, dtype=float, copy=True)
                            aligned[use_external] = external_aligned[use_external]
                            is_fallback = np.array(is_fallback, dtype=bool, copy=True)
                            is_fallback[use_external] = False

                            # The feed is a raw energy price, matching export.
                            # Normalize the inclusive source before mixing it.
                            if is_inclusive:
                                aligned = np.maximum(aligned - get_grid_fees(), 0.0)
                                aligned[use_external] = external_aligned[use_external]
                                is_inclusive = False
                                export_prices_base = None

                            source = f'{sensor} + nordpool-predict-fi'
                            tomorrow_valid = not bool(np.any(is_fallback))
                return aligned, is_fallback, source, is_inclusive, tomorrow_valid, export_prices_base

    return None, None, None, False, False, None
