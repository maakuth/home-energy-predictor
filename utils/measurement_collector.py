"""Data-only scheduled recorder ingestion. Dry-run unless --execute is explicit.

No HA, model, prediction, optimizer, feedback database or publication imports.
The collector only updates an existing identity-pinned measurement archive and
its own per-run reports while holding the same lock as manual staging tools.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import time
from typing import Any
import uuid

from dotenv import dotenv_values
import psycopg2
import yaml

from utils.measurement_archive import MeasurementArchive
from utils.measurement_reader import MeasurementReader
from utils.measurement_sources import _UniqueSafeLoader, mapped_ranges, parse_sources
from utils.measurement_update import update_recorder_sources


FIELDS = {'format_version', 'archive', 'archive_uuid', 'recorder_namespace', 'sources',
          'sources_sha256', 'measurements', 'bootstrap_from', 'env_file', 'lock_file',
          'reports_dir', 'overlap_seconds', 'chunk_seconds', 'retries'}


def _settings(path: Path) -> dict[str, Any]:
    contents = path.read_bytes()
    data = json.loads(contents)
    if not isinstance(data, dict) or set(data) != FIELDS or type(data['format_version']) is not int or data['format_version'] != 1:
        raise ValueError('Collector config requires exactly the documented v1 fields')
    for name in ('archive', 'sources', 'env_file', 'lock_file', 'reports_dir'):
        if not isinstance(data[name], str) or not Path(data[name]).is_absolute():
            raise ValueError('Collector paths must be explicit absolute local paths')
        data[name] = Path(data[name]).resolve()
    if data['archive'].name == 'hepo.db':
        raise ValueError('Collector archive must not be hepo.db')
    if not isinstance(data['archive_uuid'], str):
        raise ValueError('Explicit archive UUID required')
    uuid.UUID(data['archive_uuid'])
    if not isinstance(data['recorder_namespace'], str) or not data['recorder_namespace'].strip():
        raise ValueError('Explicit recorder namespace required')
    names = data['measurements']
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('Explicit unique measurement selection required')
    for name in ('overlap_seconds', 'chunk_seconds'):
        if type(data[name]) not in (int, float) or not math.isfinite(data[name]):
            raise ValueError('Finite collector intervals required')
    if data['overlap_seconds'] < 0 or not 0 < data['chunk_seconds'] <= 86400:
        raise ValueError('Overlap must be nonnegative and chunks bounded to one day')
    if type(data['retries']) is not int or not 0 <= data['retries'] <= 5:
        raise ValueError('Collector retries must be an integer in 0..5')
    digest = data['sources_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Exact source-file SHA256 required')
    if not isinstance(data['bootstrap_from'], str):
        raise ValueError('Explicit timezone-aware bootstrap floor required')
    start = datetime.fromisoformat(data['bootstrap_from'].replace('Z', '+00:00'))
    if start.tzinfo is None or start.utcoffset() is None or start.timestamp() % 60:
        raise ValueError('Bootstrap floor must be timezone-aware and minute-aligned')
    data['start'] = start.timestamp()
    protected = {path.resolve(), data['sources'], data['env_file'],
                 *(Path(str(data['archive'])+suffix) for suffix in ('', '-wal', '-shm', '-journal'))}
    if data['lock_file'] in protected or data['reports_dir'] in protected or data['lock_file'] == data['reports_dir']:
        raise ValueError('Report/lock paths must not replace inputs, archive or sidecars')
    data['config_sha256'] = hashlib.sha256(contents).hexdigest()
    return data


def _preflight(settings: dict[str, Any], cutoff: float) -> list[dict[str, Any]]:
    if settings['start'] >= cutoff:
        raise ValueError('Bootstrap floor must precede the collector cutoff')
    contents = settings['sources'].read_bytes()
    if hashlib.sha256(contents).hexdigest() != settings['sources_sha256']:
        raise ValueError('Source mappings changed; review and update their explicit hash')
    sources = parse_sources(yaml.load(contents.decode('utf-8'), Loader=_UniqueSafeLoader))
    if not settings['env_file'].is_file():
        raise ValueError('Explicit recorder credentials file is missing')
    if not settings['archive'].is_file():
        raise ValueError('Collector requires an existing reviewed v2 archive; it never creates one')
    with MeasurementReader(settings['archive']) as reader:
        if reader.version != 2:
            raise ValueError('Collector requires reviewed v2 storage; no automatic migration')
        row = reader.db.execute('SELECT archive_uuid,recorder_namespace FROM archive_identity').fetchone()
        if row is None or row[0] != settings['archive_uuid'] or row[1] != settings['recorder_namespace']:
            raise ValueError('Collector archive UUID/recorder namespace mismatch')
    ranges = mapped_ranges(sources, settings['start'], cutoff, settings['measurements'])
    if not ranges:
        raise ValueError('No selected sources cover the collector window')
    return [{'measurement': name, 'entity': source.entity, 'attribute': source.attribute,
             'start': left, 'end': right} for name, source, left, right in ranges]


def _connect_readonly(env_file: Path) -> Any:
    # Do not mutate process environment or read HA credentials into other
    # application modules. Only DB fields from this explicit file are used.
    values = dotenv_values(env_file)
    if not all(values.get(name) for name in ('DB_HOST', 'DB_NAME', 'DB_USER')):
        raise ValueError('Explicit DB_HOST, DB_NAME and DB_USER required')
    connection = psycopg2.connect(host=values['DB_HOST'], database=values['DB_NAME'],
        user=values['DB_USER'], password=values.get('DB_PASSWORD'), port=values.get('DB_PORT') or '5432',
        connect_timeout=10,
        options='-c default_transaction_read_only=on -c statement_timeout=60000 -c lock_timeout=3000')
    try:
        connection.set_session(readonly=True, autocommit=False)
        cursor = connection.cursor()
        try:
            cursor.execute('SHOW transaction_read_only')
            row = cursor.fetchone()
            if row is None or row[0] != 'on':
                raise ValueError('PostgreSQL did not confirm read-only access')
        finally:
            cursor.close()
        connection.rollback()
        return connection
    except BaseException:
        connection.close()
        raise


def _replace_owned_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix('.json.tmp')
    owned = False
    try:
        with temporary.open('x') as output:
            owned = True
            output.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def collect(config: str | Path, *, execute: bool = False, now: float | None = None) -> dict[str, Any]:
    settings = _settings(Path(config))
    clock = time.time() if now is None else now
    if not math.isfinite(clock):
        raise ValueError('Finite collector clock required')
    cutoff = math.floor(clock/60)*60
    report: dict[str, Any] = {'run_id': str(uuid.uuid4()), 'status': 'dry_run' if not execute else 'running',
        'exit_code': 0, 'phase': 'preflight', 'cutoff': cutoff,
        'started_at': datetime.now(timezone.utc).isoformat(), 'archive': str(settings['archive']),
        'archive_uuid': settings['archive_uuid'], 'recorder_namespace': settings['recorder_namespace'],
        'config_sha256': settings['config_sha256'], 'sources_sha256': settings['sources_sha256'],
        'measurements': settings['measurements'], 'chunks': [], 'failures': {},
        'data_only': True, 'upstream_read_only': True}
    if not execute:
        report['requests'] = _preflight(settings, cutoff)
        return report
    settings['reports_dir'].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_path = settings['reports_dir']/f'collector-{stamp}-{report["run_id"]}.json'
    report['report_path'] = str(report_path)
    with report_path.open('x') as output:
        output.write(json.dumps(report, indent=2)+'\n')
    lock = None
    try:
        report['phase'] = 'lock'
        lock = settings['lock_file'].open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report.update(status='skipped_locked', exit_code=75)
            return report
        report['phase'] = 'preflight'
        requests = _preflight(settings, cutoff)
        report['requests'] = requests
        report['phase'] = 'connect_readonly'
        connection = _connect_readonly(settings['env_file'])
        try:
            with MeasurementArchive(settings['archive'], recorder_namespace=settings['recorder_namespace']) as archive:
                identity = archive.db.execute('SELECT archive_uuid FROM archive_identity').fetchone()[0]
                if identity != settings['archive_uuid']:
                    raise ValueError('Archive identity changed after preflight')
                first = archive.db.execute('SELECT COALESCE(MAX(id),0) FROM ingestion_runs').fetchone()[0]
                report['phase'] = 'ingest'
                try:
                    for request in requests:
                        report['failures'].update(update_recorder_sources(archive, connection,
                            [(request['entity'], request['attribute'])], request['start'], request['end'],
                            settings['overlap_seconds'], chunk_seconds=settings['chunk_seconds'],
                            retries=settings['retries'], fetch_predecessor=True))
                finally:
                    report['chunks'] = [dict(row) for row in archive.db.execute(
                        'SELECT * FROM ingestion_runs WHERE id>? ORDER BY id', (first,))]
        finally:
            connection.close()
        report.update(status='failed' if report['failures'] else 'success',
                      exit_code=1 if report['failures'] else 0, phase='complete')
    except KeyboardInterrupt:
        report.update(status='interrupted', exit_code=130, error_type='KeyboardInterrupt')
    except Exception as error:
        # Never serialize a connection error's arbitrary message/kwargs: it can
        # contain credentials. Detailed source/chunk failures are kept separately.
        report.update(status='failed', exit_code=1, error_type=type(error).__name__)
    finally:
        if lock is not None:
            lock.close()
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        _replace_owned_report(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Write only to the explicitly pinned archive and reports')
    args = parser.parse_args(argv)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        result = collect(args.config, execute=args.execute)
    except (ValueError, OSError) as error:
        parser.error(f'Collector configuration error ({type(error).__name__}); check explicit paths and fields')
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps({key: value for key, value in result.items() if key not in {'chunks', 'requests'}}, indent=2))
    return int(result['exit_code'])


if __name__ == '__main__':
    raise SystemExit(main())
