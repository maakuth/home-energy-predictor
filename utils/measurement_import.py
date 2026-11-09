"""Explicit offline import of legacy resampled measurements."""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.measurement_archive import MeasurementArchive
from utils.measurement_legacy import import_legacy_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=int, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error('input must be an existing file')
    if args.archive.resolve() == args.input.resolve():
        parser.error('archive and input must differ')
    if args.archive.name == 'hepo.db':
        parser.error('measurement archive must not be hepo.db')
    if args.interval_seconds <= 0:
        parser.error('interval-seconds must be positive')
    with MeasurementArchive(args.archive) as archive:
        import_id = import_legacy_csv(archive, args.input, args.interval_seconds)
        print(f'legacy_resampled import id: {import_id}')


if __name__ == '__main__':
    main()
