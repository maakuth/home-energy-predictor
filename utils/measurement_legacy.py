"""Import legacy resampled CSVs without claiming original-event provenance."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import math
from pathlib import Path
import sqlite3

from utils.measurement_archive import MeasurementArchive


def import_legacy_csv(archive: MeasurementArchive, path: str | Path, interval_seconds: int) -> int:
    if not isinstance(interval_seconds, int) or interval_seconds <= 0:
        raise ValueError('Interval must be positive')
    source = Path(path)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    with archive.db:
        existing = archive.db.execute('SELECT id, interval_seconds FROM legacy_imports WHERE sha256=?',
                                      (digest.hexdigest(),)).fetchone()
        if existing is not None:
            if existing['interval_seconds'] != interval_seconds:
                raise ValueError('Import already exists with different interval')
            return int(existing['id'])
        cur = archive.db.execute('INSERT INTO legacy_imports(sha256, source_path, interval_seconds) VALUES (?, ?, ?)',
                                 (digest.hexdigest(), str(source), interval_seconds))
        import_id = int(cur.lastrowid or 0)
        parsed_digest = hashlib.sha256()
        with source.open('rb') as stream:
            def lines():
                for line in stream:
                    parsed_digest.update(line)
                    yield line.decode('utf-8')
            reader = csv.DictReader(lines(), strict=True)
            header = reader.fieldnames
            if (not header or header[0] != 'timestamp' or len(header) < 2
                    or len(set(header)) != len(header) or not all(header)):
                raise ValueError('Expected timestamp followed by unique nonempty measurement columns')
            measurements = header[1:]
            count = 0
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('Malformed legacy CSV row: incorrect number of fields')
                try:
                    timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                except (KeyError, ValueError, AttributeError) as error:
                    raise ValueError('Invalid timestamp in legacy CSV') from error
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError('Legacy timestamp must have a timezone')
                seconds = timestamp.timestamp()
                if not math.isfinite(seconds) or seconds % interval_seconds:
                    raise ValueError('Legacy timestamps must align with the declared interval')
                try:
                    archive.db.executemany('INSERT INTO legacy_intervals VALUES (?, ?, ?, ?, ?)', [
                        (import_id, seconds, name, row[name] or None, 'legacy_resampled')
                        for name in measurements
                    ])
                except sqlite3.IntegrityError as error:
                    raise ValueError('Duplicate timestamp in legacy CSV') from error
                count += 1
            if not count:
                raise ValueError('Legacy CSV contains no measurement rows')
        if parsed_digest.digest() != digest.digest():
            raise ValueError('Legacy CSV changed while importing; transaction rolled back')
        return import_id
