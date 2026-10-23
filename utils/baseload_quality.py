"""Baseload training truth, independent of thermal instrumentation.

Legacy extracts may already contain imputation: validity here means finite,
physically signed inputs, not proof of original sensor freshness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELOAD_FEATURES = [
    'outside_temp', 'wind_speed', 'solar_forecast',
    'leaf_power_lag_1h', 'leaf_energy_24h',
    'baseload_lag_1h', 'baseload_lag_24h', 'is_extended_complex',
    'hour', 'minute', 'quarter_hour', 'day_of_week', 'month',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
]


def baseload_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Use aligned, unfiltered meter samples. Absent equipment is NOT inferred.

    All five meters are required. Known pre-installation zeros must be provided
    by a separately audited upstream history, never inferred from missing data.
    """
    frame = raw.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.has_duplicates:
        raise ValueError('Duplicate meter timestamps')
    frame = frame.sort_index()
    index = pd.DatetimeIndex(frame.index)
    meters = ['total_power', 'solar_actual', 'battery_power', 'gshp_power', 'leaf_power']
    numeric = pd.DataFrame(index=frame.index)
    reason = pd.Series('', index=frame.index)
    for name in meters:
        values = pd.to_numeric(frame[name], errors='coerce') if name in frame else pd.Series(np.nan, index=frame.index)
        valid = np.isfinite(values)
        if name in {'solar_actual', 'gshp_power', 'leaf_power'}:
            valid = valid & values.ge(0)
        source_flag = name + '_source_valid'
        if source_flag in frame:
            valid &= frame[source_flag].astype(str).str.lower().eq('true')
        numeric[name] = values.where(valid)
        reason = reason + pd.Series(np.where(valid, '', name + ';'), index=frame.index, dtype=str)
    target = (numeric.total_power + numeric.solar_actual
              - (numeric.battery_power + numeric.gshp_power + numeric.leaf_power) / 1000)
    reason = reason + pd.Series(np.where(target.lt(0), 'negative_baseload;', ''), index=frame.index, dtype=str)
    good = target.notna() & target.ge(0)
    out = pd.DataFrame(index=frame.index)
    out['baseload_power'] = target.where(good)
    out['baseload_target_valid'] = good
    out['baseload_exclusion_reason'] = reason
    out['baseload_quality_provenance'] = ('source_flags' if all(
        name + '_source_valid' in frame for name in meters) else 'legacy_finite_checks_only')
    for hours in (1, 24):
        # Reindex explicitly: gaps must not move an observation to another hour.
        shifted = out.baseload_power.reindex(index - pd.Timedelta(hours=hours))
        out[f'baseload_lag_{hours}h'] = shifted.to_numpy()
    leaf = numeric.leaf_power
    out['leaf_power_lag_1h'] = leaf.reindex(index - pd.Timedelta(hours=1)).to_numpy()
    step_seconds = float(index.to_series().diff().dt.total_seconds().median()) if len(frame) > 1 else 900.0
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        out['leaf_energy_24h'] = np.nan
    else:
        expected = max(1, int(86400 / step_seconds))
        out['leaf_energy_24h'] = leaf.rolling('24h', closed='left', min_periods=expected).sum() / 1000 * (step_seconds / 3600)
    return out


def select_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Require new target provenance; never silently fit a legacy processed CSV."""
    if 'baseload_target_valid' not in frame:
        raise ValueError('Missing baseload quality metadata; regenerate processed data')
    valid = frame.baseload_target_valid.astype(str).str.lower().eq('true')
    valid &= np.isfinite(pd.to_numeric(frame.baseload_power, errors='coerce'))
    valid &= frame.baseload_power.ge(0)
    valid &= frame[['baseload_lag_1h', 'baseload_lag_24h']].notna().all(axis=1)
    return frame.loc[valid].sort_index()
