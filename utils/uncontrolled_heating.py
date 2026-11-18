"""Forecast observed autonomous heating without issuing heater intents."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from utils.archive_history import ArchiveHistory


def load_uncontrolled_heating_forecast(
    timestamps: list[Any] | np.ndarray, history_path: str | Path | None = None,
    *, history: ArchiveHistory | None = None, allow_missing: bool = False,
    status: dict[str, str] | None = None,
) -> np.ndarray:
    """Use a 24-hour observed-mean persistence baseline when all dispatch is off.

    This is electrical consumption, not a COP/thermal model or a control request.
    Only source-valid raw observations before the first plan slot are used.
    Coverage is logged; unavailable history leaves legacy zero-load behavior
    with an explicit warning rather than inventing measurements.
    Explicit archive mode uses the fixed cutoff's preceding day and normally
    raises on unavailable history. Opt-in fallback reports an assumption, not
    an observation, through status.
    """
    zeros = np.zeros(len(timestamps))
    if status is not None:
        status['source'] = 'managed'
    if len(timestamps) == 0:
        return zeros
    for key, default in [('GSHP_OPTIMIZE_ENABLED', 'true'),
                         ('RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'false'),
                         ('BULK_HEATER_OPTIMIZE_ENABLED', 'false')]:
        if os.getenv(key, default).strip().lower() in {'1', 'true', 'yes', 'on'}:
            return zeros
    if history is not None:
        if history_path is not None:
            raise ValueError('Choose archive history or compatibility history_path, not both')
        try:
            frame = history.raw(history.cutoff - 86400, history.cutoff, measurements=['gshp_power'])
            power = pd.to_numeric(frame['gshp_power'], errors='coerce')
            power = power.where(history.eligible(frame, 'gshp_power') & np.isfinite(power) & power.ge(0)).dropna()
            print(f'Uncontrolled heating archive coverage: {len(power)}/1440 eligible minute samples '
                  f'(policy={history.legacy_policy}).')
            if power.empty:
                raise ValueError('Archive uncontrolled heating history unavailable; refusing zero-load fallback')
        except (OSError, ValueError, KeyError, sqlite3.Error) as error:
            if not allow_missing:
                raise
            if status is not None:
                status['source'] = 'unavailable'
            print(f'⚠️ Uncontrolled heating forecast unavailable: {error}; zero is a planning assumption, not a measurement.')
            return zeros
        mean_kw = float(power.mean()) / 1000.0
        if status is not None:
            status['source'] = 'observed_mean'
        print(f'Uncontrolled heating forecast: {mean_kw:.3f} kW (24h observed mean; '
              'not weather-adjusted). Heating dispatch remains disabled.')
        return np.full(len(timestamps), mean_kw)
    path = Path(history_path) if history_path is not None else Path(os.getenv('TEST_TMP_DIR', 'state')) / 'raw_data.csv'
    try:
        frame = pd.read_csv(path, index_col=0, low_memory=False)
        frame.index = pd.to_datetime(frame.index, utc=True)
        if frame.index.has_duplicates:
            raise ValueError('duplicate history timestamps')
        frame = frame.sort_index()
        step = frame.index.to_series().diff().dt.total_seconds().median()
        if not np.isfinite(step) or step <= 0 or step > 900:
            raise ValueError('invalid history resolution')
        start = pd.to_datetime(timestamps, utc=True).min()
        window = frame.loc[(frame.index >= start - pd.Timedelta(days=1)) & (frame.index < start)]
        power = pd.to_numeric(window['gshp_power'], errors='coerce')
        valid = window['gshp_power_source_valid'].astype(str).str.lower().eq('true')
        power = power.where(valid & np.isfinite(power) & power.ge(0)).dropna()
        if power.empty:
            raise ValueError('no source-valid heating observations in the preceding 24 hours')
        mean_kw = float(power.mean()) / 1000.0
        if status is not None:
            status['source'] = 'observed_mean'
        print(f'Uncontrolled heating forecast: {mean_kw:.3f} kW (24h observed mean; '
              f'{len(power)}/{int(86400 / step)} valid samples). Heating dispatch remains disabled.')
        return np.full(len(timestamps), mean_kw)
    except (OSError, ValueError, KeyError) as error:
        if status is not None:
            status['source'] = 'unavailable'
        print(f'⚠️ Uncontrolled heating forecast unavailable: {error}; autonomous heating load is not accounted for.')
        return zeros
