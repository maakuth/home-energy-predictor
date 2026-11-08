"""Transactional storage of raw recorder events, separate from hepo.db.

Timestamps are UTC epoch seconds; recorder IDs preserve ordering of events
with equal timestamps. Callers must provide an explicit local archive path.
"""
from __future__ import annotations

import math
from pathlib import Path
import sqlite3
from typing import Iterable

Event = tuple[int, float, str | None, str | None]


class MeasurementArchive:
    def __init__(self, path: str | Path) -> None:
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        try:
            version = self.db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f'Unsupported measurement archive version: {version}')
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY, entity TEXT NOT NULL, attribute TEXT NOT NULL,
                    UNIQUE(entity, attribute)
                );
                CREATE TABLE IF NOT EXISTS observations (
                    source_id INTEGER NOT NULL REFERENCES sources(id),
                    recorder_id INTEGER NOT NULL, timestamp REAL NOT NULL,
                    state TEXT, attributes TEXT,
                    PRIMARY KEY(source_id, recorder_id)
                );
                CREATE INDEX IF NOT EXISTS observations_time
                    ON observations(source_id, timestamp, recorder_id);
                CREATE TABLE IF NOT EXISTS fetch_ranges (
                    source_id INTEGER NOT NULL REFERENCES sources(id),
                    start REAL NOT NULL, end REAL NOT NULL,
                    CHECK(start < end), PRIMARY KEY(source_id, start, end)
                );
                CREATE TABLE IF NOT EXISTS legacy_imports (
                    id INTEGER PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
                    source_path TEXT NOT NULL, interval_seconds INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS legacy_intervals (
                    import_id INTEGER NOT NULL REFERENCES legacy_imports(id),
                    timestamp REAL NOT NULL, measurement TEXT NOT NULL,
                    value TEXT, provenance TEXT NOT NULL,
                    PRIMARY KEY(import_id, timestamp, measurement)
                );
                PRAGMA user_version=1;
            ''')
            self.db.execute('PRAGMA foreign_keys=ON')
        except Exception:
            self.db.close()
            raise

    def __enter__(self) -> MeasurementArchive:
        return self

    def __exit__(self, *args: object) -> None:
        self.db.close()

    def ingest(self, entity: str, attribute: str | None, start: float, end: float,
               events: Iterable[Event]) -> None:
        if not entity or not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError('Invalid source or fetch interval')
        with self.db:
            key = (entity, attribute or '')
            self.db.execute('INSERT OR IGNORE INTO sources(entity, attribute) VALUES (?, ?)', key)
            source_id = self.db.execute('SELECT id FROM sources WHERE entity=? AND attribute=?', key).fetchone()[0]
            for recorder_id, timestamp, state, attributes in events:
                if not math.isfinite(timestamp) or not start <= timestamp < end:
                    raise ValueError('Observation outside fetch interval')
                existing = self.db.execute(
                    'SELECT timestamp, state, attributes FROM observations WHERE source_id=? AND recorder_id=?',
                    (source_id, recorder_id),
                ).fetchone()
                if existing is not None and tuple(existing) != (timestamp, state, attributes):
                    raise ValueError('Conflicting observation with existing recorder ID')
                self.db.execute('INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?)',
                                (source_id, recorder_id, timestamp, state, attributes))
            ranges = [(r[0], r[1]) for r in self.db.execute(
                'SELECT start, end FROM fetch_ranges WHERE source_id=? ORDER BY start', (source_id,))]
            merged: list[tuple[float, float]] = []
            for left, right in sorted([*ranges, (start, end)]):
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                else:
                    merged.append((left, right))
            self.db.execute('DELETE FROM fetch_ranges WHERE source_id=?', (source_id,))
            self.db.executemany('INSERT INTO fetch_ranges VALUES (?, ?, ?)',
                                [(source_id, left, right) for left, right in merged])

    def coverage(self, entity: str, attribute: str | None) -> list[tuple[float, float]]:
        rows = self.db.execute('''SELECT r.start, r.end FROM fetch_ranges r
            JOIN sources s ON s.id=r.source_id WHERE s.entity=? AND s.attribute=? ORDER BY r.start''',
            (entity, attribute or ''))
        return [(row[0], row[1]) for row in rows]

    def observations(self, entity: str, attribute: str | None, start: float, end: float,
                     predecessor: bool = False) -> list[sqlite3.Row]:
        key = (entity, attribute or '')
        prefix = '''SELECT o.recorder_id, o.timestamp, o.state, o.attributes FROM observations o
                    JOIN sources s ON s.id=o.source_id WHERE s.entity=? AND s.attribute=?'''
        result: list[sqlite3.Row] = []
        if predecessor:
            result.extend(self.db.execute(prefix + ' AND o.timestamp < ? ORDER BY o.timestamp DESC, o.recorder_id DESC LIMIT 1',
                                          (*key, start)).fetchall())
        result.extend(self.db.execute(prefix + ' AND o.timestamp >= ? AND o.timestamp < ? ORDER BY o.timestamp, o.recorder_id',
                                      (*key, start, end)).fetchall())
        return result
