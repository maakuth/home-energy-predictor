"""Explicit, bounded recorder-to-archive ingestion command."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from utils.db_utils import get_db_connection
from utils.measurement_archive import MeasurementArchive
from utils.measurement_update import update_recorder_sources
from utils.measurement_sources import load_sources, mapped_ranges
from utils.measurement_reader import MeasurementReader


def _timestamp(value: str) -> float:
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected ISO timestamp with timezone') from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise argparse.ArgumentTypeError('Timestamp must include a timezone')
    return result.timestamp()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--entity', action='append')
    selection.add_argument('--sources', type=Path, help='Dated measurement source YAML')
    parser.add_argument('--measurement', action='append', help='Measurement name from --sources; default all')
    parser.add_argument('--include-default-sources', action='store_true')
    parser.add_argument('--from', dest='start', type=_timestamp)
    cutoff = parser.add_mutually_exclusive_group(required=True)
    cutoff.add_argument('--to', dest='end', type=_timestamp)
    cutoff.add_argument('--through-now', action='store_true')
    parser.add_argument('--overlap-seconds', type=float, default=900)
    parser.add_argument('--recorder-namespace')
    parser.add_argument('--chunk-seconds', type=float, default=86400)
    parser.add_argument('--retries', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.through_now:
        args.end = math.floor(time.time() / 60) * 60
    if (not math.isfinite(args.overlap_seconds) or args.overlap_seconds < 0
            or not math.isfinite(args.chunk_seconds) or args.chunk_seconds <= 0 or args.retries < 0):
        parser.error('invalid interval or overlap')
    if args.archive.resolve().name == 'hepo.db':
        parser.error('measurement archive must not be hepo.db')
    if args.include_default_sources and not args.sources:
        parser.error('--include-default-sources requires --sources')
    if args.report:
        protected = {Path(str(args.archive.resolve()) + suffix) for suffix in ('', '-wal', '-shm', '-journal')}
        if args.report.resolve() in protected or args.report.resolve().name == 'hepo.db':
            parser.error('--report must not overwrite archive, sidecars or hepo.db')
        if not args.dry_run and args.report.exists():
            parser.error('--report requires a new output path')
    if args.measurement and not args.sources:
        parser.error('--measurement requires --sources')
    inferred_start = args.start is None
    if inferred_start:
        if not args.archive.exists():
            parser.error('Empty archive requires explicit --from bootstrap floor')
        with MeasurementReader(args.archive) as reader:
            row = reader.db.execute('SELECT MIN(start) FROM fetch_ranges').fetchone()
            if row[0] is None:
                parser.error('Empty coverage requires explicit --from bootstrap floor')
            args.start = row[0]
    if args.start >= args.end:
        parser.error('invalid interval')
    if args.sources:
        try:
            sources = load_sources(args.sources)
            if args.include_default_sources:
                from utils.archive_features import default_sources
                sources = {**default_sources(), **sources}
            ranges = mapped_ranges(sources, args.start, args.end, args.measurement)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        requests = [(source.entity, source.attribute, left, right) for _, source, left, right in ranges]
    else:
        requests = [(entity, None, args.start, args.end) for entity in dict.fromkeys(args.entity)]
    if not requests:
        print('No configured sources cover the requested interval; nothing fetched')
        return
    if inferred_start:
        with MeasurementReader(args.archive) as reader:
            for entity, attribute, _, _ in requests:
                covered = reader.db.execute('''SELECT 1 FROM fetch_ranges r JOIN sources s ON s.id=r.source_id
                    WHERE s.entity=? AND s.attribute=? LIMIT 1''', (entity, attribute or '')).fetchone()
                if covered is None:
                    parser.error(f'{entity}: missing coverage requires explicit --from bootstrap floor')
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'cutoff': args.end, 'requests': requests,
                          'chunk_seconds': args.chunk_seconds, 'overlap_seconds': args.overlap_seconds}))
        return
    if not args.recorder_namespace or not args.recorder_namespace.strip():
        parser.error('Writes require explicit --recorder-namespace')
    load_dotenv()
    failures: dict[str, str] = {}
    with MeasurementArchive(args.archive, recorder_namespace=args.recorder_namespace) as archive:
        first_run = archive.db.execute('SELECT COALESCE(MAX(id), 0) FROM ingestion_runs').fetchone()[0]
        connection = get_db_connection()
        try:
            for entity, attribute, left, right in requests:
                failures.update(update_recorder_sources(archive, connection,
                    [(entity, attribute)], left, right, args.overlap_seconds,
                    chunk_seconds=args.chunk_seconds, retries=args.retries, fetch_predecessor=True))
        finally:
            connection.close()
        if args.report:
            report = {'cutoff': args.end, 'recorder_namespace': args.recorder_namespace,
                      'failures': failures, 'chunks': [dict(row) for row in archive.db.execute(
                          'SELECT * FROM ingestion_runs WHERE id>? ORDER BY id', (first_run,))]}
            with args.report.open('x') as output:
                output.write(json.dumps(report, indent=2) + '\n')
    if failures:
        for source, message in failures.items():
            print(f'{source}: {message}')
        raise SystemExit(1)
    print('Measurement ingestion completed')


if __name__ == '__main__':
    main()
