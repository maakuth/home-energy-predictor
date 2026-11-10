"""Read-only archive/CSV diagnostics, deliberately not a migration approval gate."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd

from utils.measurement_reader import MeasurementReader, interval_index, numeric
from utils.measurement_resampling import resample_numeric
from utils.measurement_sources import MeasurementSource, mapped_ranges


def inventory_csv(path: str | Path, interval_seconds: int = 60) -> dict[str, Any]:
    """Hash bytes and stream rows with bounded memory; never infer source identity.

    Nonincreasing timestamps catch duplicates in sorted input. Unsorted files
    require a separate exact duplicate audit; no unbounded timestamp set is kept.
    Hashing and parsing are separate streaming passes, with a stat stability check.
    """
    if interval_seconds <= 0:
        raise ValueError('Interval must be positive')
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    report: dict[str, Any] = {
        'path': str(path), 'size_bytes': before.st_size, 'sha256': digest.hexdigest(),
        'declared_interval_seconds': interval_seconds, 'rows': 0,
        'malformed_rows': 0, 'invalid_timestamps': 0, 'unaligned_timestamps': 0,
        'nonincreasing_timestamps': 0, 'first_timestamp': None, 'last_timestamp': None,
        'provenance': 'legacy_resampled_unverified', 'ready_to_import': False,
        'source_unit_associations': 'pending_external_evidence',
    }
    minimum = maximum = previous = None
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        report['header'] = header
        report['header_valid'] = bool(header and header[0] == 'timestamp'
                                      and len(header) > 1 and len(set(header)) == len(header)
                                      and all(header))
        columns = {name: {'nonempty_rows': 0, 'numeric_rows': 0, 'true_rows': 0, 'false_rows': 0,
                          'first_numeric_timestamp': None, 'last_numeric_timestamp': None}
                   for name in header[1:]}
        report['columns'] = columns
        report['source_flags'] = [name for name in columns if name.endswith('_source_valid')]
        for row in reader:
            report['rows'] += 1
            if len(row) != len(header):
                report['malformed_rows'] += 1
            numeric_columns = []
            for name, value in zip(header[1:], row[1:]):
                column = columns[name]
                column['nonempty_rows'] += bool(value.strip())
                is_numeric = math.isfinite(numeric(value))
                column['numeric_rows'] += is_numeric
                if is_numeric:
                    numeric_columns.append(name)
                column['true_rows'] += value.strip().lower() == 'true'
                column['false_rows'] += value.strip().lower() == 'false'
            try:
                timestamp = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError('Timestamp must have timezone')
                seconds = timestamp.timestamp()
                if not math.isfinite(seconds):
                    raise ValueError('Timestamp must be finite')
            except (ValueError, IndexError, OverflowError):
                report['invalid_timestamps'] += 1
                continue
            report['unaligned_timestamps'] += seconds % interval_seconds != 0
            report['nonincreasing_timestamps'] += previous is not None and seconds <= previous
            previous = seconds
            minimum = seconds if minimum is None else min(minimum, seconds)
            maximum = seconds if maximum is None else max(maximum, seconds)
            for name in numeric_columns:
                column = columns[name]
                first, last = column['first_numeric_timestamp'], column['last_numeric_timestamp']
                column['first_numeric_timestamp'] = seconds if first is None else min(first, seconds)
                column['last_numeric_timestamp'] = seconds if last is None else max(last, seconds)
    for key, value in [('first_timestamp', minimum), ('last_timestamp', maximum)]:
        if value is not None:
            report[key] = datetime.fromtimestamp(value, timezone.utc).isoformat()
    after = path.stat()
    report['stable_during_inventory'] = (
        before.st_size, before.st_mtime_ns, before.st_ino
    ) == (after.st_size, after.st_mtime_ns, after.st_ino)
    report['structurally_importable'] = bool(
        report['header_valid'] and report['rows'] and report['stable_during_inventory']
        and not any(report[k] for k in ('malformed_rows', 'invalid_timestamps',
                                       'unaligned_timestamps', 'nonincreasing_timestamps')))
    report['import_plan'] = 'review source/unit associations and overlap precedence before explicit import'
    return report


def audit_training_eligibility(path: str | Path, start: float, end: float, *,
                               legacy_policy: str = 'verified-only',
                               interval_seconds: int = 60, max_bins: int = 10080,
                               **options: Any) -> dict[str, Any]:
    """Evaluate the shared target/lag contract without processing or fitting.

    Uses one separate read snapshot (identified in returned metadata), including
    the preceding day. Unlike feature generation, all-missing histories remain
    reportable; nothing is filled just to satisfy a feature diagnostic.
    """
    from utils.archive_features import BASELOAD_INPUTS, load_archive_raw
    from utils.archive_quality import apply_training_policy
    from utils.baseload_quality import select_training_rows

    if interval_seconds < 60 or interval_seconds % 60 or 86400 % interval_seconds:
        raise ValueError('Training audit resolution must be whole minutes dividing a day')
    if (end - start) / interval_seconds > max_bins:
        raise ValueError('Audit bin limit exceeded')
    interval_index(start, end, interval_seconds)
    raw = load_archive_raw(path, start - 86400, end, interval_seconds=interval_seconds,
                           measurements=list(BASELOAD_INPUTS), allow_legacy=True, **options)
    eligible = apply_training_policy(raw, pd.DataFrame(index=raw.index), legacy_policy)
    eligible = eligible.loc[eligible.index >= pd.Timestamp(start, unit='s', tz='UTC')]
    selected = select_training_rows(eligible)
    return {
        'migration_ready': False, 'legacy_policy': legacy_policy,
        'start': start, 'end': end, 'bins': len(eligible),
        'eligible_targets': int(eligible.baseload_training_eligible.sum()),
        'source_verified_targets': int(eligible.baseload_target_valid.sum()),
        'eligible_training_rows': len(selected),
        'eligible_unverified_training_rows': int((~selected.baseload_target_valid).sum()),
        'inputs': {name: int(eligible[name + '_training_eligible'].sum()) for name in BASELOAD_INPUTS},
        'lag_eligibility': {name: int(eligible[name + '_training_eligible'].sum()) for name in
                            ('baseload_lag_1h', 'baseload_lag_24h', 'leaf_power_lag_1h', 'leaf_energy_24h')},
        'target_exclusion_reasons': eligible.baseload_training_exclusion_reason.value_counts().to_dict(),
        'snapshot_metadata': eligible.attrs['archive_training'],
        'note': 'Eligibility is not coverage-gate approval, source/unit verification or operational readiness',
    }


def archive_inventory(reader: MeasurementReader) -> dict[str, Any]:
    """Cheap metadata only: indexed endpoint lookups, no legacy cell aggregation."""
    imports = []
    for row in reader.db.execute('SELECT * FROM legacy_imports ORDER BY id'):
        item = dict(row)
        for key, order in [('first_timestamp', 'ASC'), ('last_timestamp', 'DESC')]:
            endpoint = reader.db.execute(
                f'SELECT timestamp FROM legacy_intervals WHERE import_id=? ORDER BY timestamp {order} LIMIT 1',
                (row['id'],)).fetchone()
            item[key] = endpoint[0] if endpoint else None
        imports.append(item)
    sources = []
    for row in reader.db.execute('SELECT * FROM sources ORDER BY id'):
        item = dict(row)
        for key, order in [('first_observation', 'ASC'), ('last_observation', 'DESC')]:
            endpoint = reader.db.execute(
                f'SELECT timestamp FROM observations WHERE source_id=? ORDER BY timestamp {order} LIMIT 1',
                (row['id'],)).fetchone()
            item[key] = endpoint[0] if endpoint else None
        sources.append(item)
    return {
        'schema_version': reader.db.execute('PRAGMA user_version').fetchone()[0],
        'imports': imports,
        'sources': sources,
        'fetch_ranges': [dict(row) for row in reader.db.execute(
            'SELECT * FROM fetch_ranges ORDER BY source_id, start')],
        'cell_counts': 'not_scanned', 'migration_ready': False,
        'coverage_semantics': 'fetch_ranges prove successful queries, not measurement availability',
    }


def resolve_precedence(reader: MeasurementReader, manifest: dict[str, Any]) -> list[int]:
    """Resolve portable content hashes to snapshot-local IDs, failing closed."""
    if not isinstance(manifest, dict) or set(manifest) != {'preferred_sha256'}:
        raise ValueError('Manifest requires only preferred_sha256')
    hashes = manifest['preferred_sha256']
    if (not isinstance(hashes, list) or any(not isinstance(h, str) or len(h) != 64
            or any(c not in '0123456789abcdef' for c in h) for h in hashes)
            or len(set(hashes)) != len(hashes)):
        raise ValueError('Expected unique lowercase SHA256 hashes')
    available = {row['sha256']: row['id'] for row in reader.db.execute('SELECT id, sha256 FROM legacy_imports')}
    if any(h not in available for h in hashes):
        raise ValueError('Unknown import content hash in precedence manifest')
    return [available[h] for h in hashes]


def backfill_plan(reader: MeasurementReader, sources: dict[str, list[MeasurementSource]],
                  start: float, end: float, measurements: list[str] | None = None) -> list[dict[str, Any]]:
    """Subtract successful query ranges from mapped source periods. No fetching."""
    result = []
    for name, source, left, right in mapped_ranges(sources, start, end, measurements):
        rows = reader.db.execute('''SELECT r.start, r.end FROM fetch_ranges r
            JOIN sources s ON s.id=r.source_id WHERE s.entity=? AND s.attribute=?
            AND r.end>? AND r.start<? ORDER BY r.start''',
            (source.entity, source.attribute or '', left, right))
        cursor = left
        for a, b in [*rows, (right, right)]:
            if cursor < min(a, right):
                result.append({'measurement': name, 'entity': source.entity,
                               'attribute': source.attribute, 'start': cursor, 'end': min(a, right)})
            cursor = max(cursor, b)
    return result


def audit_measurements(reader: MeasurementReader, sources: dict[str, list[MeasurementSource]],
                       start: float, end: float, measurements: list[str], *,
                       interval_seconds: int = 900, legacy_policy: str = 'verified-only',
                       preferred_imports: list[int] | None = None,
                       max_bins: int = 10080) -> dict[str, Any]:
    """Bounded numeric availability audit, NOT a baseload/lag training gate.

    Both policies inspect legacy availability. verified-only excludes legacy from
    measurement eligibility; complete-legacy accepts complete nonconflicting
    values. Recorder invalid states and explicit false legacy flags stay missing.
    No relay ratings, unit conversions, counter/category policies are inferred.
    """
    if legacy_policy not in {'verified-only', 'complete-legacy'}:
        raise ValueError('Unknown legacy policy')
    if not isinstance(interval_seconds, int) or interval_seconds <= 0:
        raise ValueError('Interval must be a positive integer')
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError('Invalid audit interval')
    if (end - start) / interval_seconds > max_bins:
        raise ValueError('Audit bin limit exceeded; narrow the window or explicitly increase --max-bins')
    interval_index(start, end, interval_seconds)
    mapped_ranges(sources, start, end, measurements)
    if any(n in {'ev_position', 'mummun_energy', 'mummun_power',
                 'resistive_heater_power', 'bulk_heater_power'} for n in measurements):
        raise ValueError('Numeric audit does not infer category/counter/relay policies')
    hashes = {r['id']: r['sha256'] for r in reader.db.execute('SELECT id, sha256 FROM legacy_imports')}
    report: dict[str, Any] = {
        'start': start, 'end': end, 'interval_seconds': interval_seconds,
        'legacy_policy': legacy_policy, 'legacy_availability_included': True,
        'preferred_sha256': [hashes[i] for i in (preferred_imports or [])],
        'measurement_policy_only': True, 'target_and_lag_eligibility': 'not_evaluated',
        'migration_ready': False, 'measurements': {},
        'pending': ['baseload and lag feature eligibility audit', 'source/unit provenance validation',
                    'consumer integration and non-actuating shadow validation', 'operational cutover approval'],
    }
    for name in dict.fromkeys(measurements):
        counts: dict[str, Any] = {key: 0 for key in (
            'bins', 'available_bins', 'full_coverage_bins', 'source_verified_bins',
            'conflict_bins', 'measurement_eligible_bins')}
        counts['conflict_examples'] = []
        counts['provenance_bins'] = {}
        left = start
        # Keep each resampling frame bounded even when the caller raises max_bins.
        while left < end:
            right = min(end, left + interval_seconds * 1440)
            frame = resample_numeric(reader, sources, name, left, right,
                interval_seconds=interval_seconds, allow_legacy=True,
                preferred_imports=preferred_imports,
                nonnegative=name in {'solar_actual', 'gshp_power', 'leaf_power', 'wind_speed'})
            available = frame['value'].notna()
            complete = frame['coverage_fraction'].ge(1 - 1e-9)
            verified = frame['source_valid']
            conflict = frame['conflict']
            eligible = available & complete & ~conflict
            if legacy_policy == 'verified-only':
                eligible &= verified
            counts['bins'] += len(frame)
            for key, mask in [('available_bins', available), ('full_coverage_bins', complete),
                              ('source_verified_bins', verified), ('conflict_bins', conflict),
                              ('measurement_eligible_bins', eligible)]:
                counts[key] += int(mask.sum())
            for kind, number in frame['provenance'].value_counts().items():
                counts['provenance_bins'][kind] = counts['provenance_bins'].get(kind, 0) + int(number)
            for timestamp, row in frame.loc[conflict].iterrows():
                if len(counts['conflict_examples']) >= 10:
                    break
                counts['conflict_examples'].append({
                    'timestamp': str(timestamp), 'import_sha256': [hashes[i] for i in row['import_ids']],
                    'selected_value_available': math.isfinite(numeric(row['value'])),
                })
            left = right
        report['measurements'][name] = counts
    return report
