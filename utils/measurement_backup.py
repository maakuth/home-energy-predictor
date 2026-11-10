"""Offline SQLite-aware measurement archive backup, verification and restore.

Never contacts recorder/HA and never migrates a source archive. Destinations
must be new explicit local paths. Safe with a concurrent WAL writer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from utils.measurement_archive import validate_archive


def _local(path: str | Path) -> Path:
    if '://' in str(path) or str(path).startswith('file:'):
        raise ValueError('Require an explicit local filesystem path, not a URI')
    result = Path(path).resolve()
    if result.name == 'hepo.db':
        raise ValueError('measurement archive must not be hepo.db')
    return result


def _verify(db: sqlite3.Connection) -> dict[str, object]:
    version = validate_archive(db)
    integrity = [row[0] for row in db.execute('PRAGMA integrity_check')]
    if integrity != ['ok']:
        raise ValueError(f'Archive integrity failure: {integrity}')
    if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
        raise ValueError('Archive foreign key check failed')
    result: dict[str, object] = {'version': version, 'integrity': 'ok'}
    if version == 2:
        row = db.execute('SELECT archive_uuid, recorder_namespace FROM archive_identity').fetchone()
        result.update(archive_uuid=row[0], recorder_namespace=row[1])
    return result


def verify_archive(path: str | Path) -> dict[str, object]:
    db = sqlite3.connect(_local(path).as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        return _verify(db)
    finally:
        db.close()


def backup_archive(source: str | Path, destination: str | Path) -> dict[str, object]:
    source_path, target_path = _local(source), _local(destination)
    if target_path in {Path(str(source_path) + suffix) for suffix in ('-wal', '-shm', '-journal')}:
        raise ValueError('Backup destination must not be a source archive sidecar')
    source_db = sqlite3.connect(source_path.as_uri() + '?mode=ro', uri=True, timeout=10)
    reserved = False
    target_db = None
    try:
        source_db.execute('BEGIN')
        identity = _verify(source_db)
        # Exclusive reservation prevents silently overwriting any existing file.
        with target_path.open('xb'):
            reserved = True
        target_db = sqlite3.connect(target_path)
        source_db.backup(target_db, pages=1024)
        copied_identity = _verify(target_db)
        if copied_identity != identity:
            raise ValueError('Backup archive identity mismatch')
        return copied_identity
    except Exception:
        if target_db is not None:
            target_db.close()
            target_db = None
        if reserved:
            target_path.unlink()
        raise
    finally:
        if target_db is not None:
            target_db.close()
        source_db.close()


def restore_archive(source: str | Path, destination: str | Path) -> dict[str, object]:
    """Restore and verify to a NEW path; never replace an operational archive."""
    return backup_archive(source, destination)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['backup', 'restore', 'verify'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args(argv)
    if args.operation == 'verify':
        if args.destination:
            parser.error('verify takes no destination')
        report = verify_archive(args.source)
    else:
        if not args.destination:
            parser.error('backup/restore requires --destination (new path)')
        report = backup_archive(args.source, args.destination)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
