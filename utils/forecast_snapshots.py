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
SCHEMA_VERSION = 2


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
                CREATE TABLE forecast_points (
                    id INTEGER PRIMARY KEY,
                    digest BLOB NOT NULL UNIQUE,
                    target_key TEXT NOT NULL,
                    target TEXT NOT NULL,
                    temperature,
                    wind_speed,
                    wind_bearing,
                    value,
                    extra_json TEXT NOT NULL
                );
                CREATE INDEX forecast_points_target ON forecast_points(target);
                CREATE TABLE forecast_series (
                    id INTEGER PRIMARY KEY,
                    payload_id INTEGER NOT NULL REFERENCES payloads(id),
                    path TEXT NOT NULL,
                    UNIQUE(payload_id, path)
                );
                CREATE TABLE forecast_members (
                    series_id INTEGER NOT NULL REFERENCES forecast_series(id),
                    position INTEGER NOT NULL,
                    point_id INTEGER NOT NULL REFERENCES forecast_points(id),
                    PRIMARY KEY(series_id, position)
                ) WITHOUT ROWID;
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


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


_TARGET_KEYS = ('datetime', 'start', 'period_start', 'timestamp', 'target')
_NUMERIC_FIELDS = ('temperature', 'wind_speed', 'wind_bearing', 'value')


def _point_value(row: tuple[Any, ...]) -> dict[str, Any]:
    target_key, target, temperature, wind_speed, wind_bearing, value, extra = row
    point = json.loads(extra)
    point[target_key] = target
    for key, field in zip(_NUMERIC_FIELDS, (temperature, wind_speed, wind_bearing, value)):
        if field is not None:
            point[key] = field
    return point


def _store_payload(conn: sqlite3.Connection, payload: Any) -> int:
    """Extract timestamped point arrays; preserve all other JSON as an envelope."""
    text = _json(payload)
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    existing = conn.execute('SELECT id FROM payloads WHERE digest=?', (digest,)).fetchone()
    if existing:
        if _json(_load_payload(conn, existing[0])) != text:
            raise ValueError('Payload hash collision')
        return int(existing[0])

    series: list[tuple[list[Any], list[dict[str, Any]]]] = []

    def split(item: Any, path: list[Any]) -> Any:
        if (isinstance(item, list) and item and
                all(isinstance(point, dict) and any(isinstance(point.get(k), str) for k in _TARGET_KEYS)
                    for point in item)):
            series.append((path, item))
            return []
        if isinstance(item, list):
            return [split(v, [*path, i]) for i, v in enumerate(item)]
        if isinstance(item, dict):
            return {k: split(v, [*path, k]) for k, v in item.items()}
        return item

    # JSON round-trip ensures key/value types match what can be reconstructed.
    envelope = split(json.loads(text), [])
    cursor = conn.execute('INSERT INTO payloads(digest, json) VALUES (?, ?)', (digest, _json(envelope)))
    payload_id = cursor.lastrowid
    assert payload_id is not None
    for path, points in series:
        series_id = conn.execute('INSERT INTO forecast_series(payload_id,path) VALUES (?,?)',
                                 (payload_id, _json(path))).lastrowid
        assert series_id is not None
        for position, point in enumerate(points):
            point_text = _json(point)
            point_digest = hashlib.sha256(point_text.encode('utf-8')).digest()
            target_key = next(k for k in _TARGET_KEYS if isinstance(point.get(k), str))
            remaining = dict(point)
            target = remaining.pop(target_key)
            numbers = []
            for key in _NUMERIC_FIELDS:
                value = remaining.get(key)
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and (not isinstance(value, int) or -(2**63) <= value < 2**63)):
                    numbers.append(remaining.pop(key))
                else:
                    numbers.append(None)
            conn.execute('''INSERT OR IGNORE INTO forecast_points
                (digest,target_key,target,temperature,wind_speed,wind_bearing,value,extra_json)
                VALUES (?,?,?,?,?,?,?,?)''',
                         (point_digest, target_key, target, *numbers, _json(remaining)))
            stored = conn.execute('''SELECT id,target_key,target,temperature,wind_speed,
                                     wind_bearing,value,extra_json FROM forecast_points WHERE digest=?''',
                                  (point_digest,)).fetchone()
            if _json(_point_value(stored[1:])) != point_text:
                raise ValueError('Forecast point hash collision')
            conn.execute('INSERT INTO forecast_members VALUES (?,?,?)',
                         (series_id, position, stored[0]))
    return int(payload_id)


def _load_payload(conn: sqlite3.Connection, payload_id: int) -> Any:
    envelope = json.loads(conn.execute('SELECT json FROM payloads WHERE id=?', (payload_id,)).fetchone()[0])
    rows = conn.execute('''SELECT s.path, p.target_key,p.target,p.temperature,p.wind_speed,
                           p.wind_bearing,p.value,p.extra_json
                           FROM forecast_series s JOIN forecast_members m ON m.series_id=s.id
                           JOIN forecast_points p ON p.id=m.point_id
                           WHERE s.payload_id=? ORDER BY s.path,m.position''', (payload_id,))
    for path_json, *point in rows:
        path = json.loads(path_json)
        container = envelope
        for segment in path:
            container = container[segment]
        container.append(_point_value(tuple(point)))
    return envelope


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
                payload_id = _store_payload(conn, source['payload'])
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
            SELECT a.id, a.origin, a.weather_start, a.weather_end, w.id,
                   a.price_start, a.price_end, p.id
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
                            'completed_at': timestamp(we).isoformat(), 'payload': _load_payload(conn, wp)},
                'price': {'status': 'ok', 'started_at': timestamp(ps).isoformat(),
                          'completed_at': timestamp(pe).isoformat(), 'payload': _load_payload(conn, pp)}}
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
