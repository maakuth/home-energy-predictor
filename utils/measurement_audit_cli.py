"""Read-only JSON inventory and bounded numeric coverage audit; never ingests."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sqlite3

from utils.measurement_audit import (
    archive_inventory, audit_measurements, audit_training_eligibility, backfill_plan, inventory_csv, resolve_precedence,
)
from utils.measurement_reader import MeasurementReader
from utils.measurement_sources import load_sources


def _timestamp(value: str) -> float:
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError('Timezone required')
        return result.timestamp()
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected ISO timestamp with timezone') from error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inventory = commands.add_parser('inventory', help='Stream CSVs; no archive or recorder writes')
    inventory.add_argument('--csv', type=Path, action='append', default=[])
    inventory.add_argument('--root', type=Path, action='append', default=[],
                           help='Recursively inventory raw_data*.csv (may read many GB)')
    inventory.add_argument('--interval-seconds', type=int, default=60)
    inventory.add_argument('--archive', type=Path, help='Optional read-only hash match for import plans')
    audit = commands.add_parser('archive', help='Cheap metadata unless bounded measurements are selected')
    audit.add_argument('--archive', type=Path, required=True)
    audit.add_argument('--sources', type=Path)
    audit.add_argument('--measurement-policy', type=Path)
    audit.add_argument('--measurement', action='append')
    audit.add_argument('--from', dest='start', type=_timestamp)
    audit.add_argument('--to', dest='end', type=_timestamp)
    audit.add_argument('--interval-seconds', type=int, default=900)
    audit.add_argument('--max-bins', type=int, default=10080)
    audit.add_argument('--legacy-policy', choices=['verified-only', 'complete-legacy'], default='verified-only')
    audit.add_argument('--precedence-manifest', type=Path)
    audit.add_argument('--dry-run-backfill', action='store_true', help='Only mapped/query-coverage gap plan; no resampling')
    audit.add_argument('--training-eligibility', action='store_true',
                       help='Also audit all baseload meters, predictor inputs and exact lag eligibility')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inventory':
            paths = set(args.csv)
            for root in args.root:
                if not root.is_dir():
                    raise ValueError(f'Inventory root does not exist: {root}')
                paths.update(root.rglob('raw_data*.csv'))
            if not paths:
                raise ValueError('Select --csv or --root')
            reports = [inventory_csv(path, args.interval_seconds) for path in sorted(paths)]
            if args.archive:
                with MeasurementReader(args.archive) as reader:
                    hashes = {r['sha256'] for r in reader.db.execute('SELECT sha256 FROM legacy_imports')}
                for item in reports:
                    item['already_imported_content'] = item['sha256'] in hashes
                    if item['already_imported_content']:
                        item['import_plan'] = 'skip: content hash already imported'
            report = {'dry_run': True, 'migration_ready': False, 'files': reports}
        else:
            preferred = None
            scoped = any(value is not None for value in (args.start, args.end, args.measurement)) or args.dry_run_backfill or args.training_eligibility
            if scoped and (args.start is None or args.end is None or not args.sources
                           or (not args.measurement and not args.training_eligibility)):
                raise ValueError('Scoped audit requires --sources, --measurement, --from and --to')
            if args.training_eligibility and args.dry_run_backfill:
                raise ValueError('Training eligibility cannot be combined with dry-run backfill only')
            if args.precedence_manifest and not scoped:
                raise ValueError('Precedence manifest requires a scoped audit')
            with MeasurementReader(args.archive) as reader:
                report = archive_inventory(reader)
                report['dry_run'] = True
                if scoped:
                    sources = load_sources(args.sources)
                    if args.precedence_manifest:
                        preferred = resolve_precedence(reader, json.loads(args.precedence_manifest.read_text()))
                    report['backfill_plan'] = backfill_plan(reader, sources, args.start, args.end, args.measurement)
                    if not args.dry_run_backfill and args.measurement:
                        report.update(audit_measurements(reader, sources, args.start, args.end, args.measurement,
                             interval_seconds=args.interval_seconds, legacy_policy=args.legacy_policy,
                             preferred_imports=preferred, policy_path=args.measurement_policy, max_bins=args.max_bins))
            if args.training_eligibility:
                report['training_eligibility'] = audit_training_eligibility(
                    args.archive, args.start, args.end, sources_path=args.sources,
                    interval_seconds=args.interval_seconds, legacy_policy=args.legacy_policy,
                    policy_path=args.measurement_policy, require_policies=args.measurement_policy is not None,
                    preferred_imports=preferred, max_bins=args.max_bins)
                report['target_and_lag_eligibility'] = 'evaluated_in_separate_consistent_snapshot'
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, sqlite3.Error, csv.Error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
