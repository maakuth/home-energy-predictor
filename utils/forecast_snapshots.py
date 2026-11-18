"""Isolated as-received forecast capture and conservative as-of selection.

No planner, device control or measurement database is touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable
from uuid import uuid4

import requests


def timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('Timezone-aware timestamp required') from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return result.astimezone(timezone.utc)


APPLICATION_ID = 0x48465053  # HFPS
SCHEMA_VERSION = 1


def _utc(value: Any) -> str:
    return timestamp(value).isoformat(timespec='microseconds')


def _connect(path: str | Path, *, create: bool) -> sqlite3.Connection:
    """Only initialize newly created files; never adopt an existing database."""
    path = Path(path)
    created = False
    if create:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
            created = True
    if not path.exists():
        raise FileNotFoundError(path)
    # mode=ro/rw prevents sqlite from implicitly creating a different database.
    uri = path.resolve().as_uri() + ('?mode=rw' if create else '?mode=ro')
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        if created:
            conn.executescript(f'''
                BEGIN IMMEDIATE;
                PRAGMA application_id={APPLICATION_ID};
                PRAGMA user_version={SCHEMA_VERSION};
                CREATE TABLE payloads (
                    id INTEGER PRIMARY KEY,
                    digest BLOB NOT NULL UNIQUE,
                    json TEXT NOT NULL
                );
                CREATE TABLE attempts (
                    id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    weather_start TEXT NOT NULL,
                    weather_end TEXT NOT NULL,
                    weather_status TEXT NOT NULL,
                    weather_error TEXT,
                    weather_payload INTEGER REFERENCES payloads(id),
                    price_start TEXT NOT NULL,
                    price_end TEXT NOT NULL,
                    price_status TEXT NOT NULL,
                    price_error TEXT,
                    price_payload INTEGER REFERENCES payloads(id)
                ) WITHOUT ROWID;
                CREATE INDEX attempts_origin ON attempts(origin DESC);
                COMMIT;
            ''')
        elif (conn.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
              or conn.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION):
            raise ValueError('Not a supported forecast archive')
        conn.execute('PRAGMA foreign_keys=ON')
        return conn
    except BaseException:
        conn.close()
        raise


def capture(archive: str | Path, origin: Any, weather: Callable[[], Any],
            price: Callable[[], Any], *, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
            max_attempts: int = 100000, max_bytes: int = 2_000_000_000) -> dict[str, Any]:
    """Capture independently; quota exhaustion fails closed, never deletes vintages."""
    if max_attempts <= 0 or max_bytes <= 0:
        raise ValueError('Archive quota must be positive')
    scheduled = timestamp(origin)
    # Validate or initialize before making any source calls.
    conn = _connect(archive, create=True)
    attempt: dict[str, Any] = {'format_version': 1, 'id': uuid4().hex,
                               'scheduled_origin': scheduled.isoformat()}
    try:
        for name, fetch in (('weather', weather), ('price', price)):
            start = timestamp(clock())
            try:
                payload = fetch()
                json.dumps(payload, allow_nan=False)
                result = {'status': 'ok', 'payload': payload}
            except Exception as exc:
                result = {'status': 'error', 'error_type': type(exc).__name__}
            end = timestamp(clock())
            if end < start:
                raise ValueError('Capture clock moved backwards')
            result.update(started_at=start.isoformat(), completed_at=end.isoformat())
            attempt[name] = result

        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT count(*) FROM attempts').fetchone()[0] >= max_attempts:
            raise ValueError('Archive quota exceeded; no snapshots removed')

        def source_columns(source: dict[str, Any]) -> tuple[Any, ...]:
            payload_id = None
            if source['status'] == 'ok':
                text = json.dumps(source['payload'], allow_nan=False, sort_keys=True,
                                  separators=(',', ':'), ensure_ascii=False)
                digest = hashlib.sha256(text.encode('utf-8')).digest()
                conn.execute('INSERT OR IGNORE INTO payloads(digest, json) VALUES (?, ?)', (digest, text))
                payload_id, stored = conn.execute(
                    'SELECT id, json FROM payloads WHERE digest=?', (digest,)).fetchone()
                if stored != text:
                    raise ValueError('Payload hash collision')
            return (_utc(source['started_at']), _utc(source['completed_at']),
                    source['status'], source.get('error_type'), payload_id)

        conn.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                     (attempt['id'], _utc(scheduled), *source_columns(attempt['weather']),
                      *source_columns(attempt['price'])))
        size = conn.execute('PRAGMA page_count').fetchone()[0] * conn.execute('PRAGMA page_size').fetchone()[0]
        if size > max_bytes:
            raise ValueError('Archive quota exceeded; no snapshots removed')
        conn.commit()
        return attempt
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def select(archive: str | Path, cutoff: Any) -> dict[str, Any] | None:
    """Select latest complete same-attempt pair available by cutoff."""
    cutoff = _utc(cutoff)
    conn = _connect(archive, create=False)
    try:
        row = conn.execute('''
            SELECT a.id, a.origin, a.weather_start, a.weather_end, w.json,
                   a.price_start, a.price_end, p.json
            FROM attempts a JOIN payloads w ON w.id=a.weather_payload
            JOIN payloads p ON p.id=a.price_payload
            WHERE a.origin<=? AND a.weather_status='ok' AND a.price_status='ok'
              AND a.weather_end<=? AND a.price_end<=?
            ORDER BY a.origin DESC, max(a.weather_end,a.price_end) DESC, a.id DESC LIMIT 1
        ''', (cutoff, cutoff, cutoff)).fetchone()
        if row is None:
            return None
        id_, origin, ws, we, wp, ps, pe, pp = row
        return {'format_version': 1, 'id': id_, 'scheduled_origin': timestamp(origin).isoformat(),
                'weather': {'status': 'ok', 'started_at': timestamp(ws).isoformat(),
                            'completed_at': timestamp(we).isoformat(), 'payload': json.loads(wp)},
                'price': {'status': 'ok', 'started_at': timestamp(ps).isoformat(),
                          'completed_at': timestamp(pe).isoformat(), 'payload': json.loads(pp)}}
    finally:
        conn.close()


def map_intervals(points: list[dict[str, Any]], targets: list[Any]) -> list[Any]:
    """Map only explicitly covered quarter-hour intervals; never extrapolate."""
    from datetime import timedelta

    intervals = sorted((timestamp(p['start']), timestamp(p['end']), p['value']) for p in points)
    for i, (start, end, _) in enumerate(intervals):
        if start >= end or (i and start < intervals[i - 1][1]):
            raise ValueError('Invalid or overlapping intervals')
    result = []
    for target in targets:
        t = timestamp(target)
        if t.second or t.microsecond or t.minute % 15:
            raise ValueError('Target must be aligned to 15 minutes')
        result.append(next((value for start, end, value in intervals
                            if start <= t and t + timedelta(minutes=15) <= end), None))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, help='Dedicated forecast SQLite archive path')
    parser.add_argument('--origin', required=True, help='Scheduled origin, timezone-aware')
    parser.add_argument('--host', required=True, help='Home Assistant URL')
    parser.add_argument('--weather-entity', required=True)
    parser.add_argument('--price-entity', required=True)
    parser.add_argument('--price-zone', required=True)
    parser.add_argument('--max-attempts', type=int, default=100000)
    parser.add_argument('--max-bytes', type=int, default=2_000_000_000)
    args = parser.parse_args()
    token = os.environ.get('HA_TOKEN')
    if not token or not args.host.startswith(('http://', 'https://')):
        parser.error('HA_TOKEN and an http(s) host are required')
    host = args.host.rstrip('/')
    headers = {'Authorization': 'Bearer ' + token}

    def state(entity: str) -> Any:
        response = requests.get(host + '/api/states/' + entity, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()

    def weather() -> Any:
        metadata = state(args.weather_entity)
        response = requests.post(host + '/api/services/weather/get_forecasts?return_response',
                                 headers=headers, json={'entity_id': args.weather_entity, 'type': 'hourly'},
                                 timeout=20)
        response.raise_for_status()
        data = response.json()
        service = data.get('service_response', data) if isinstance(data, dict) else {}
        if not isinstance(service, dict) or not isinstance(service.get(args.weather_entity), dict) or not isinstance(service[args.weather_entity].get('forecast'), list) or not service[args.weather_entity]['forecast']:
            raise ValueError('Missing hourly weather forecast')
        return {'source': args.weather_entity, 'state': metadata, 'response': data}

    def price() -> Any:
        data = state(args.price_entity)
        if not isinstance(data, dict) or data.get('state') in ('unavailable', 'unknown'):
            raise ValueError('Price entity unavailable')
        return {'source': args.price_entity, 'zone': args.price_zone,
                'classification': 'unverified', 'state': data}

    result = capture(args.archive, args.origin, weather, price,
                     max_attempts=args.max_attempts, max_bytes=args.max_bytes)
    print(json.dumps({'id': result['id'], 'weather': result['weather']['status'],
                      'price': result['price']['status']}))
    if result['weather']['status'] != 'ok' or result['price']['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
