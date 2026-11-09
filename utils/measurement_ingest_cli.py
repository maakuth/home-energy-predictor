"""Explicit, bounded recorder-to-archive ingestion command."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from utils.db_utils import get_db_connection
from utils.measurement_archive import MeasurementArchive
from utils.measurement_update import update_recorder_sources


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
    parser.add_argument('--entity', action='append', required=True)
    parser.add_argument('--from', dest='start', type=_timestamp, required=True)
    parser.add_argument('--to', dest='end', type=_timestamp, required=True)
    parser.add_argument('--overlap-seconds', type=float, default=900)
    args = parser.parse_args(argv)
    if args.start >= args.end or args.overlap_seconds < 0:
        parser.error('invalid interval or overlap')
    if args.archive.name == 'hepo.db':
        parser.error('measurement archive must not be hepo.db')
    load_dotenv()
    with MeasurementArchive(args.archive) as archive:
        connection = get_db_connection()
        try:
            failures = update_recorder_sources(archive, connection,
                [(entity, None) for entity in args.entity], args.start, args.end, args.overlap_seconds)
        finally:
            connection.close()
    if failures:
        for source, message in failures.items():
            print(f'{source}: {message}')
        raise SystemExit(1)
    print('Measurement ingestion completed')


if __name__ == '__main__':
    main()
