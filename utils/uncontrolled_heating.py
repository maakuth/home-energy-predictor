"""Forecast observed autonomous heating without issuing heater intents."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_uncontrolled_heating_forecast(
    timestamps: list[Any] | np.ndarray, history_path: str | Path | None = None,
) -> np.ndarray:
    """Use a 24-hour observed-mean persistence baseline when all dispatch is off.

    This is electrical consumption, not a COP/thermal model or a control request.
    Only source-valid raw observations before the first plan slot are used.
    Coverage is logged; unavailable history leaves legacy zero-load behavior
    with an explicit warning rather than inventing measurements.
    """
    zeros = np.zeros(len(timestamps))
    if len(timestamps) == 0:
        return zeros
    for key, default in [('GSHP_OPTIMIZE_ENABLED', 'true'),
                         ('RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'false'),
                         ('BULK_HEATER_OPTIMIZE_ENABLED', 'false')]:
        if os.getenv(key, default).strip().lower() in {'1', 'true', 'yes', 'on'}:
            return zeros
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
        print(f'Uncontrolled heating forecast: {mean_kw:.3f} kW (24h observed mean; '
              f'{len(power)}/{int(86400 / step)} valid samples). Heating dispatch remains disabled.')
        return np.full(len(timestamps), mean_kw)
    except (OSError, ValueError, KeyError) as error:
        print(f'⚠️ Uncontrolled heating forecast unavailable: {error}; autonomous heating load is not accounted for.')
        return zeros
