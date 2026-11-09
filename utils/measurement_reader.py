"""Read-only measurement snapshots; legacy conflicts require explicit precedence."""
from __future__ import annotations

import math
from pathlib import Path
import sqlite3

import pandas as pd


def numeric(value: object) -> float:
    try:
        number = float(str(value))
        return number if math.isfinite(number) else math.nan
    except (ValueError, TypeError):
        return math.nan


def interval_index(start: float, end: float, seconds: int) -> pd.DatetimeIndex:
    if (not isinstance(seconds, int) or seconds <= 0 or not math.isfinite(start)
            or not math.isfinite(end) or start >= end or start % seconds or end % seconds):
        raise ValueError('Require finite, increasing interval-aligned bounds and a positive integer interval')
    return pd.date_range(pd.to_datetime(start, unit='s', utc=True),
                         pd.to_datetime(end, unit='s', utc=True),
                         freq=pd.Timedelta(seconds=seconds), inclusive='left', name='timestamp')


class MeasurementReader:
    """Never create an archive, migrate its schema, or write measurements.

    The context holds one read transaction, giving all queries a consistent
    snapshot. Close it before expensive model fitting to avoid pinning the WAL.
    """
    def __init__(self, path: str | Path) -> None:
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute('PRAGMA query_only=ON')
            self.db.execute('BEGIN')
            if self.db.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise ValueError('Unsupported measurement archive version')
            self.db.execute('SELECT id FROM legacy_imports LIMIT 0')
        except Exception:
            self.db.close()
            raise

    def __enter__(self) -> MeasurementReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.db.close()

    def legacy_numeric(self, measurement: str, start: float, end: float,
                       interval_seconds: int = 60,
                       preferred_imports: list[int] | None = None) -> pd.DataFrame:
        """Read native-resolution numeric intervals, not original state events.

        Missing-vs-numeric overlaps are conflicts too. An explicit import order
        selects the first present import, even if its value is missing. Metadata
        always retains the conflict and all contributing import IDs. No legacy
        interval is promoted to source-verified truth.
        """
        index = interval_index(start, end, interval_seconds)
        priority = preferred_imports or []
        if len(priority) != len(set(priority)):
            raise ValueError('Duplicate import precedence')
        metadata = self.db.execute('SELECT id, interval_seconds FROM legacy_imports ORDER BY id').fetchall()
        available = {row[0] for row in metadata}
        if any(import_id not in available for import_id in priority):
            raise ValueError('Unknown import precedence ID')
        groups: dict[float, dict[int, float]] = {}
        # Bind import_id explicitly so the primary-key range scan stays bounded
        # on large archives; joining all imports can scan millions of cells.
        for import_id, resolution in metadata:
            if resolution != interval_seconds:
                continue
            rows = self.db.execute('''SELECT v.timestamp, v.import_id, v.value,
                    f.measurement AS flag_present, f.value AS flag_value
                FROM legacy_intervals v
                LEFT JOIN legacy_intervals f ON f.import_id=v.import_id
                    AND f.timestamp=v.timestamp AND f.measurement=?
                WHERE v.measurement=? AND v.import_id=?
                    AND v.timestamp>=? AND v.timestamp<?
                ORDER BY v.timestamp, v.import_id''',
                (measurement + '_source_valid', measurement, import_id, start, end))
            for row in rows:
                if row['timestamp'] % interval_seconds:
                    raise ValueError('Legacy samples are not aligned to their declared interval')
                value = numeric(row['value'])
                if row['flag_present'] is not None and str(row['flag_value']).lower() != 'true':
                    value = math.nan
                groups.setdefault(row['timestamp'], {})[row['import_id']] = value
        result = []
        for timestamp in index:
            candidates = groups.get(timestamp.timestamp(), {})
            signatures = {None if math.isnan(v) else v for v in candidates.values()}
            conflict = len(signatures) > 1
            selected = next((i for i in priority if i in candidates), None)
            value = candidates[selected] if selected is not None else (
                next(iter(candidates.values())) if len(signatures) == 1 else math.nan)
            result.append({
                'value': value,
                'coverage_fraction': float(math.isfinite(value)),
                'source_valid': False,
                'provenance': 'legacy_resampled' if candidates else 'missing',
                'conflict': conflict,
                'import_ids': tuple(sorted(candidates)),
                'selected_import_id': selected,
            })
        return pd.DataFrame(result, index=index)
