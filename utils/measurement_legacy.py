"""Import legacy resampled CSVs without claiming original-event provenance."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
from pathlib import Path

from utils.measurement_archive import MeasurementArchive


def import_legacy_csv(archive: MeasurementArchive, path: str | Path, interval_seconds: int) -> int:
    if interval_seconds <= 0:
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
        with source.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or reader.fieldnames[0] != 'timestamp':
                raise ValueError('Expected timestamp as first CSV column')
            measurements = reader.fieldnames[1:]
            for row in reader:
                try:
                    timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                except (KeyError, ValueError, AttributeError) as error:
                    raise ValueError('Invalid timestamp in legacy CSV') from error
                if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                    raise ValueError('Legacy timestamp must have a timezone')
                archive.db.executemany('INSERT INTO legacy_intervals VALUES (?, ?, ?, ?, ?)', [
                    (import_id, timestamp.timestamp(), name, row[name] or None, 'legacy_resampled')
                    for name in measurements
                ])
        return import_id
