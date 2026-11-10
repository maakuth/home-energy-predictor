"""Transactional storage of raw recorder events, separate from hepo.db.

Timestamps are UTC epoch seconds; recorder IDs preserve ordering of events
with equal timestamps. Callers must provide an explicit local archive path.
"""
from __future__ import annotations

import math
from pathlib import Path
import sqlite3
import uuid
from typing import Iterable

Event = tuple[int, float, str | None, str | None]
APPLICATION_ID = 0x4845504D
V1_COLUMNS = {
    'sources': ('id', 'entity', 'attribute'),
    'observations': ('source_id', 'recorder_id', 'timestamp', 'state', 'attributes'),
    'fetch_ranges': ('source_id', 'start', 'end'),
    'legacy_imports': ('id', 'sha256', 'source_path', 'interval_seconds'),
    'legacy_intervals': ('import_id', 'timestamp', 'measurement', 'value', 'provenance'),
}


def validate_archive(db: sqlite3.Connection) -> int:
    """Validate identity before any write, including journal-mode changes."""
    version = db.execute('PRAGMA user_version').fetchone()[0]
    app = db.execute('PRAGMA application_id').fetchone()[0]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if version not in (1, 2):
        raise ValueError(f'Unsupported measurement archive version: {version}')
    expected = set(V1_COLUMNS) | ({'archive_identity', 'predecessor_evidence', 'ingestion_runs'} if version == 2 else set())
    if tables != expected or app not in ((0, APPLICATION_ID) if version == 1 else (APPLICATION_ID,)):
        raise ValueError('Not a measurement archive: unexpected application identity or tables')
    for table, columns in V1_COLUMNS.items():
        if tuple(r[1] for r in db.execute(f'PRAGMA table_info({table})')) != columns:
            raise ValueError('Not a measurement archive: incompatible schema')
    if version == 2:
        rows = db.execute('SELECT application, archive_uuid FROM archive_identity').fetchall()
        if len(rows) != 1 or rows[0][0] != 'home-energy-measurements':
            raise ValueError('Invalid measurement archive identity')
        uuid.UUID(rows[0][1])
    return version


class MeasurementArchive:
    def __init__(self, path: str | Path, recorder_namespace: str | None = None) -> None:
        if Path(path).resolve().name == 'hepo.db':
            raise ValueError('measurement archive must not be hepo.db')
        if recorder_namespace is not None and not recorder_namespace.strip():
            raise ValueError('Empty recorder namespace')
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        try:
            version = self.db.execute('PRAGMA user_version').fetchone()[0]
            tables = self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version != 0 or tables:
                validate_archive(self.db)
            elif self.db.execute('PRAGMA application_id').fetchone()[0] != 0:
                raise ValueError('Not a measurement archive')
            self.db.execute('PRAGMA foreign_keys=ON')
            self.db.execute('BEGIN IMMEDIATE')
            schema = '''
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
            '''
            if version == 0:
                for statement in schema.split(';'):
                    if statement.strip():
                        self.db.execute(statement)
            if version < 2:
                self.db.execute('''CREATE TABLE archive_identity (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), application TEXT NOT NULL,
                    archive_uuid TEXT NOT NULL, recorder_namespace TEXT)''')
                self.db.execute('INSERT INTO archive_identity VALUES (1, ?, ?, ?)',
                                ('home-energy-measurements', str(uuid.uuid4()), recorder_namespace))
                self.db.execute('''CREATE TABLE predecessor_evidence (
                    source_id INTEGER NOT NULL REFERENCES sources(id), start REAL NOT NULL,
                    recorder_id INTEGER NOT NULL, timestamp REAL NOT NULL,
                    PRIMARY KEY(source_id, start),
                    FOREIGN KEY(source_id, recorder_id) REFERENCES observations(source_id, recorder_id))''')
                self.db.execute('''CREATE TABLE ingestion_runs (
                    id INTEGER PRIMARY KEY, entity TEXT NOT NULL, attribute TEXT NOT NULL,
                    start REAL NOT NULL, end REAL NOT NULL, attempt INTEGER NOT NULL,
                    status TEXT NOT NULL, error TEXT, event_count INTEGER NOT NULL DEFAULT 0)''')
                self.db.execute(f'PRAGMA application_id={APPLICATION_ID}')
                self.db.execute('PRAGMA user_version=2')
            if recorder_namespace is not None:
                bound = self.db.execute('SELECT recorder_namespace FROM archive_identity').fetchone()[0]
                if bound is not None and bound != recorder_namespace:
                    raise ValueError('Recorder namespace mismatch; use a separate new archive after a rebuild')
                self.db.execute('UPDATE archive_identity SET recorder_namespace=?', (recorder_namespace,))
            self.db.commit()
            self.db.execute('PRAGMA journal_mode=WAL')
        except Exception:
            self.db.rollback()
            self.db.close()
            raise

    def __enter__(self) -> MeasurementArchive:
        return self

    def __exit__(self, *args: object) -> None:
        self.db.close()

    def ingest(self, entity: str, attribute: str | None, start: float, end: float,
               events: Iterable[Event], predecessor: Event | None = None) -> None:
        if not entity or not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError('Invalid source or fetch interval')
        with self.db:
            key = (entity, attribute or '')
            self.db.execute('INSERT OR IGNORE INTO sources(entity, attribute) VALUES (?, ?)', key)
            source_id = self.db.execute('SELECT id FROM sources WHERE entity=? AND attribute=?', key).fetchone()[0]
            def rows() -> Iterable[Event]:
                if predecessor is not None:
                    yield predecessor
                yield from events
            for index, (recorder_id, timestamp, state, attributes) in enumerate(rows()):
                is_previous = predecessor is not None and index == 0
                if not math.isfinite(timestamp) or not (timestamp < start if is_previous else start <= timestamp < end):
                    raise ValueError('Observation outside fetch interval')
                existing = self.db.execute(
                    'SELECT timestamp, state, attributes FROM observations WHERE source_id=? AND recorder_id=?',
                    (source_id, recorder_id),
                ).fetchone()
                if existing is not None and tuple(existing) != (timestamp, state, attributes):
                    raise ValueError('Conflicting observation with existing recorder ID')
                self.db.execute('INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?)',
                                (source_id, recorder_id, timestamp, state, attributes))
            if predecessor is not None:
                self.db.execute('INSERT OR REPLACE INTO predecessor_evidence VALUES (?, ?, ?, ?)',
                                (source_id, start, predecessor[0], predecessor[1]))
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
