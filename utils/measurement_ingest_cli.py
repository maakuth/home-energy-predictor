"""Explicit, bounded recorder-to-archive ingestion command."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path

from dotenv import load_dotenv
from utils.db_utils import get_db_connection
from utils.measurement_archive import MeasurementArchive
from utils.measurement_update import update_recorder_sources
from utils.measurement_sources import load_sources, mapped_ranges


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
    parser.add_argument('--from', dest='start', type=_timestamp, required=True)
    parser.add_argument('--to', dest='end', type=_timestamp, required=True)
    parser.add_argument('--overlap-seconds', type=float, default=900)
    args = parser.parse_args(argv)
    if args.start >= args.end or not math.isfinite(args.overlap_seconds) or args.overlap_seconds < 0:
        parser.error('invalid interval or overlap')
    if args.archive.name == 'hepo.db':
        parser.error('measurement archive must not be hepo.db')
    if args.measurement and not args.sources:
        parser.error('--measurement requires --sources')
    if args.sources:
        try:
            ranges = mapped_ranges(load_sources(args.sources), args.start, args.end, args.measurement)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        requests = [(source.entity, source.attribute, left, right) for _, source, left, right in ranges]
    else:
        requests = [(entity, None, args.start, args.end) for entity in dict.fromkeys(args.entity)]
    if not requests:
        print('No configured sources cover the requested interval; nothing fetched')
        return
    load_dotenv()
    failures: dict[str, str] = {}
    with MeasurementArchive(args.archive) as archive:
        connection = get_db_connection()
        try:
            for entity, attribute, left, right in requests:
                failures.update(update_recorder_sources(archive, connection,
                    [(entity, attribute)], left, right, args.overlap_seconds))
        finally:
            connection.close()
    if failures:
        for source, message in failures.items():
            print(f'{source}: {message}')
        raise SystemExit(1)
    print('Measurement ingestion completed')


if __name__ == '__main__':
    main()
