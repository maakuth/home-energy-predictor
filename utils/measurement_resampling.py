"""Time-weighted numeric/state integration with explicit coverage and provenance.

No interpolation, nominal-power modulo decomposition or implicit CSV fallback.
Inputs must be in the same units; scale converts the final value if requested.
"""
from __future__ import annotations

from bisect import bisect_right
import json
import math

import numpy as np
import pandas as pd

from utils.measurement_reader import MeasurementReader, interval_index, numeric
from utils.measurement_sources import MeasurementSource, mapped_ranges


def _event_value(state: object, attributes: object, attribute: str | None,
                 state_values: dict[str, float] | None) -> float:
    if str(state).lower() in {'unknown', 'unavailable', 'none'}:
        return math.nan
    if attribute is not None:
        try:
            values = json.loads(str(attributes))
        except (ValueError, TypeError):
            return math.nan
        return numeric(values.get(attribute)) if isinstance(values, dict) else math.nan
    if state_values is not None:
        return numeric(state_values.get(str(state).lower()))
    return numeric(state)


def resample_numeric(
    reader: MeasurementReader, sources: dict[str, list[MeasurementSource]],
    measurement: str, start: float, end: float, interval_seconds: int = 900,
    *, max_age_seconds: float | None = None, allow_legacy: bool = False,
    legacy_interval_seconds: int = 60, preferred_imports: list[int] | None = None,
    state_values: dict[str, float] | None = None, scale: float = 1.0,
    nonnegative: bool = False,
) -> pd.DataFrame:
    """Return complete-bin means; incomplete bins have NaN, not partial means.

    coverage_fraction reports the valid portion even when the mean is withheld.
    Raw states carry only inside successful fetch ranges and their source period.
    max_age_seconds optionally expires unchanged states. Missing/invalid recorder
    states inside queried coverage always block legacy fallback. Legacy values
    require opt-in and never set source_valid, even if they cover an entire bin.
    """
    index = interval_index(start, end, interval_seconds)
    ranges = mapped_ranges(sources, start, end, [measurement])
    if not math.isfinite(scale):
        raise ValueError('Scale must be finite')
    if max_age_seconds is not None and (not math.isfinite(max_age_seconds) or max_age_seconds <= 0):
        raise ValueError('State expiry must be positive and finite')
    legacy = {}
    if allow_legacy:
        if legacy_interval_seconds <= 0 or interval_seconds % legacy_interval_seconds:
            raise ValueError('Legacy resolution must divide the requested interval')
        frame = reader.legacy_numeric(measurement, start, end, legacy_interval_seconds, preferred_imports)
        legacy = {ts.timestamp(): row for ts, row in frame.iterrows() if row.import_ids}
    sums = np.zeros(len(index))
    seconds = np.zeros(len(index))
    kinds: list[set[str]] = [set() for _ in index]
    conflicts = np.zeros(len(index), dtype=bool)
    imports: list[set[int]] = [set() for _ in index]
    entities: list[set[str]] = [set() for _ in index]

    for _, source, left, right in ranges:
        key = (source.entity, source.attribute or '')
        sid = reader.db.execute('SELECT id FROM sources WHERE entity=? AND attribute=?', key).fetchone()
        events: dict[float, float] = {}
        coverage: list[tuple[float, float]] = []
        if sid is not None:
            prefix = 'SELECT timestamp, recorder_id, state, attributes FROM observations WHERE source_id=?'
            rows = reader.db.execute(prefix + ' AND timestamp<? ORDER BY timestamp DESC, recorder_id DESC LIMIT 1',
                                     (sid[0], left)).fetchall()
            rows += reader.db.execute(prefix + ' AND timestamp>=? AND timestamp<? ORDER BY timestamp, recorder_id',
                                      (sid[0], left, right)).fetchall()
            for row in rows:
                if source.start is None or row['timestamp'] >= source.start.timestamp():
                    events[row['timestamp']] = _event_value(row['state'], row['attributes'], source.attribute, state_values)
            coverage = [(row[0], min(right, row[1])) for row in reader.db.execute(
                'SELECT start, end FROM fetch_ranges WHERE source_id=? AND end>? AND start<? ORDER BY start',
                (sid[0], left, right))]
        event_times = sorted(events)
        coverage_starts = [a for a, _ in coverage]
        cuts = {left, right}
        cuts.update(t for t in event_times if left < t < right)
        cuts.update(t for pair in coverage for t in pair if left < t < right)
        if max_age_seconds is not None:
            cuts.update(t + max_age_seconds for t in event_times if left < t + max_age_seconds < right)
        if allow_legacy:
            cuts.update(t for t in legacy if left < t < right)
            cuts.update(t + legacy_interval_seconds for t in legacy if left < t + legacy_interval_seconds < right)
        ordered = sorted(cuts)
        for a, b in zip(ordered, ordered[1:]):
            event_index = bisect_right(event_times, a) - 1
            coverage_index = bisect_right(coverage_starts, a) - 1
            raw = coverage_index >= 0 and a < coverage[coverage_index][1]
            value = math.nan
            provenance = 'missing'
            conflict = False
            import_ids: tuple[int, ...] = ()
            if raw:
                provenance = 'recorder'
                if event_index >= 0:
                    t = event_times[event_index]
                    if t >= coverage[coverage_index][0] and (max_age_seconds is None or a < t + max_age_seconds):
                        value = events[t]
            elif allow_legacy:
                row = legacy.get((a // legacy_interval_seconds) * legacy_interval_seconds)
                if row is not None:
                    value = row.value
                    provenance = 'legacy_resampled'
                    conflict = bool(row.conflict)
                    import_ids = row.import_ids
            if nonnegative and value < 0:
                value = math.nan
            # Split at output boundaries, retaining correct elapsed-time weights.
            position = a
            while position < b:
                bin_index = int((position - start) // interval_seconds)
                stop = min(b, start + (bin_index + 1) * interval_seconds)
                duration = stop - position
                kinds[bin_index].add(provenance)
                entities[bin_index].add(source.entity)
                imports[bin_index].update(import_ids)
                conflicts[bin_index] |= conflict
                if math.isfinite(value):
                    sums[bin_index] += value * duration
                    seconds[bin_index] += duration
                position = stop

    complete = np.isclose(seconds, interval_seconds, rtol=0, atol=1e-7)
    return pd.DataFrame({
        'value': np.where(complete, sums / interval_seconds * scale, np.nan),
        'coverage_fraction': seconds / interval_seconds,
        'source_valid': [bool(ok and kind == {'recorder'}) for ok, kind in zip(complete, kinds)],
        'provenance': ['mixed' if len(kind) > 1 else next(iter(kind), 'missing') for kind in kinds],
        'conflict': conflicts,
        'import_ids': [tuple(sorted(ids)) for ids in imports],
        'entities': [tuple(sorted(ids)) for ids in entities],
    }, index=index)
