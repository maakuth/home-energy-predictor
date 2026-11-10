"""Read-only diagnostic aggregation, preserving policy and verification separately."""
from __future__ import annotations

import pandas as pd

from utils.archive_history import ArchiveHistory
from utils.archive_quality import measurement_eligibility
from utils.archive_quality import METERS
from utils.baseload_quality import baseload_targets


def quarter_measurements(history: ArchiveHistory, start: float, end: float,
                         measurements: list[str]) -> pd.DataFrame:
    """Require all fifteen aligned minute bins, never average across outages."""
    raw = history.raw(start, end, measurements=measurements, interval_seconds=60)
    index = pd.date_range(pd.to_datetime(start, unit='s', utc=True),
                          pd.to_datetime(end, unit='s', utc=True), freq='min', inclusive='left')
    raw = raw.reindex(index)
    out = pd.DataFrame()
    for name in measurements:
        eligible = history.eligible(raw, name)
        values = pd.to_numeric(raw[name], errors='coerce').where(eligible)
        full = values.resample('15min').count().eq(15)
        out[name] = values.resample('15min').mean().where(full)
        verified = measurement_eligibility(raw, name, 'verified-only')
        out[name + '_source_verified'] = verified.resample('15min').sum().eq(15)
        out[name + '_eligible'] = full
    if set(METERS).issubset(measurements):
        policy_raw = raw.copy()
        for name in METERS:
            policy_raw[name + '_source_valid'] = history.eligible(raw, name)
        target = baseload_targets(policy_raw).baseload_power
        out['baseload_power'] = target.resample('15min').mean().where(
            target.resample('15min').count().eq(15))
    out.attrs = {**raw.attrs, 'legacy_policy': history.legacy_policy,
                 'start': start, 'end': end, 'interval_seconds': 900,
                 'legacy_is_source_verified': False}
    return out
