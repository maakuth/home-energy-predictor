"""Isolated measurement analysis: read-only SQLite, no live inputs or active outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

import yaml

from utils.measurement_audit import audit_measurements, backfill_plan
from utils.measurement_audit_cli import _timestamp
from utils.measurement_reader import MeasurementReader, interval_index
from utils.measurement_sources import _UniqueSafeLoader, mapped_ranges, parse_sources


MEASUREMENTS = ('outside_temp', 'wind_speed', 'total_power', 'solar_actual',
                'battery_power', 'gshp_power', 'leaf_power')


def _check_hash(value: str) -> None:
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Explicit lowercase SHA256 required for sources and policy')


def analyze(*, archive: str | Path, archive_uuid: str, recorder_namespace: str,
            sources: str | Path, sources_sha256: str, policy: str | Path,
            policy_sha256: str, start: float, end: float,
            interval_seconds: int = 900, max_bins: int = 10080) -> dict[str, Any]:
    """Summarize one fixed window in one read transaction, with verified-only eligibility.

    Legacy records inside SQLite are inspected for availability/conflicts, never
    admitted as verified measurements. No CSV file, environment credentials,
    default source aliases, ML processing or network input is consulted.
    """
    uuid.UUID(archive_uuid)
    if not recorder_namespace.strip():
        raise ValueError('Explicit recorder namespace required')
    if type(interval_seconds) is not int or interval_seconds not in (60, 900):
        raise ValueError('Analysis resolution must be 60 or 900 seconds')
    if type(max_bins) is not int or max_bins <= 0 or (end - start) / interval_seconds > max_bins:
        raise ValueError('Analysis bin limit exceeded or invalid; narrow the window')
    interval_index(start, end, interval_seconds)
    _check_hash(sources_sha256)
    _check_hash(policy_sha256)
    contents = Path(sources).read_bytes()
    if hashlib.sha256(contents).hexdigest() != sources_sha256:
        raise ValueError('Source mappings do not match the approved hash')
    mappings = parse_sources(yaml.load(contents.decode('utf-8'), Loader=_UniqueSafeLoader))
    names: list[str] = list(MEASUREMENTS)
    ranges = mapped_ranges(mappings, start, end, names)
    unmapped = []
    for name in names:
        cursor = start
        for measurement, _, left, right in ranges:
            if measurement != name:
                continue
            if left > cursor:
                unmapped.append({'measurement': name, 'start': cursor, 'end': left})
            cursor = max(cursor, right)
        if cursor < end:
            unmapped.append({'measurement': name, 'start': cursor, 'end': end})
    with MeasurementReader(archive) as reader:
        if reader.version != 2:
            raise ValueError('Analysis requires the existing reviewed v2 archive')
        identity = dict(reader.db.execute('SELECT * FROM archive_identity').fetchone())
        if identity['archive_uuid'] != archive_uuid or identity['recorder_namespace'] != recorder_namespace:
            raise ValueError('Archive UUID or recorder namespace mismatch')
        unqueried = backfill_plan(reader, mappings, start, end, names)
        report = audit_measurements(reader, mappings, start, end, names,
            interval_seconds=interval_seconds, legacy_policy='verified-only',
            policy_path=policy, max_bins=max_bins)
        # The audit hashes precisely the policy bytes it parsed. Reject a changed
        # policy before publishing, rather than merely recording a new identity.
        if report['policy_sha256'] != policy_sha256:
            raise ValueError('Measurement policy does not match the approved hash')
    report.update({
        'format_version': 1, 'run_id': str(uuid.uuid4()),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'archive_read_only': True, 'data_only': True,
        'snapshot_semantics': 'one_read_transaction',
        'archive_path': str(Path(archive).resolve()), 'archive_identity': identity,
        'sources_path': str(Path(sources).resolve()), 'sources_sha256': sources_sha256,
        'policy_path': str(Path(policy).resolve()),
        'analysis_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'unqueried_mapped_ranges': unqueried, 'unmapped_ranges': unmapped,
        'coverage_semantics': 'Successful queries are not proof of valid measurement availability',
        'note': 'Measurement prerequisites only; no target/lag eligibility, fitting, promotion or cutover approval',
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-uuid', required=True)
    parser.add_argument('--recorder-namespace', required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--sources-sha256', required=True)
    parser.add_argument('--measurement-policy', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--from', dest='start', type=_timestamp, required=True)
    parser.add_argument('--to', dest='end', type=_timestamp, required=True)
    parser.add_argument('--interval-seconds', type=int, choices=(60, 900), default=900)
    parser.add_argument('--max-bins', type=int, default=10080)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New isolated directory, never an existing active artifact directory')
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError('Output directory must be new; existing paths are never replaced')
        report = analyze(archive=args.archive, archive_uuid=args.archive_uuid,
            recorder_namespace=args.recorder_namespace, sources=args.sources,
            sources_sha256=args.sources_sha256, policy=args.measurement_policy,
            policy_sha256=args.policy_sha256, start=args.start, end=args.end,
            interval_seconds=args.interval_seconds, max_bins=args.max_bins)
        payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
        args.output_dir.mkdir(mode=0o700)
        report_path = args.output_dir / 'report.json'
        with report_path.open('x') as output:
            output.write(payload)
    except (OSError, ValueError, sqlite3.Error, yaml.YAMLError) as error:
        parser.error(str(error))
    print(json.dumps({'report_path': str(report_path), 'run_id': report['run_id'],
                      'archive_read_only': True, 'migration_ready': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
