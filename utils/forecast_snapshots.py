"""Isolated as-received forecast capture and conservative as-of selection.

No planner, device control or measurement database is touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable
from uuid import uuid4

import requests
from dotenv import dotenv_values
import fcntl
import signal
import time


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


_TARGET_KEYS = ('datetime', 'start', 'period_start', 'timestamp', 'target', 'time')
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


def select(archive: str | Path, cutoff: Any, *, exact_origin: Any = None,
           max_age: timedelta | None = None) -> dict[str, Any] | None:
    """Select a complete pair, optionally requiring an origin and maximum origin age."""
    cutoff_time = timestamp(cutoff)
    cutoff = _utc(cutoff_time)
    origin = _utc(exact_origin) if exact_origin is not None else None
    if max_age is not None and (not isinstance(max_age, timedelta) or max_age < timedelta(0)):
        raise ValueError('max_age must be a nonnegative timedelta')
    earliest = _utc(cutoff_time - max_age) if max_age is not None else None
    conn = _connect(archive, create=False)
    try:
        row = conn.execute('''
            SELECT a.id, a.origin, a.weather_start, a.weather_end, w.id,
                   a.price_start, a.price_end, p.id
            FROM attempts a JOIN payloads w ON w.id=a.weather_payload
            JOIN payloads p ON p.id=a.price_payload
            WHERE a.origin<=? AND a.weather_status='ok' AND a.price_status='ok'
              AND a.weather_end<=? AND a.price_end<=?
              AND (? IS NULL OR a.origin=?)
              AND (? IS NULL OR a.origin>=?)
             ORDER BY a.origin DESC, max(a.weather_end,a.price_end) DESC, a.id DESC LIMIT 1
        ''', (cutoff, cutoff, cutoff, origin, origin, earliest, earliest)).fetchone()
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


def vintage_coverage(archive: str | Path, origins: list[Any], cutoff: Any) -> list[dict[str, Any]]:
    """Report retrospective attempt coverage for each explicitly expected origin."""
    cutoff_text = _utc(cutoff)
    expected = [_utc(origin) for origin in origins]
    conn = _connect(archive, create=False)
    try:
        result = []
        for origin in expected:
            rows = conn.execute('''SELECT weather_end, price_end, weather_status, price_status
                                   FROM attempts WHERE origin=?''', (origin,)).fetchall()
            visible = [row for row in rows if origin <= cutoff_text
                       and row[0] <= cutoff_text and row[1] <= cutoff_text]
            if any(row[2] == 'ok' and row[3] == 'ok' for row in visible):
                status = 'complete'
            elif visible:
                status = 'failed'
            elif rows:
                status = 'not_available'
            else:
                status = 'missing'
            result.append({'origin': timestamp(origin).isoformat(), 'status': status,
                           'attempts': len(visible)})
        return result
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


def validate_price_window(state: Any, source: str, zone: str, origin: Any) -> dict[str, Any]:
    """Validate supported raw price shapes without inferring price semantics."""
    if not source or not zone or not isinstance(state, dict) or state.get('state') in (None, 'unknown', 'unavailable'):
        raise ValueError('Price entity unavailable')
    attrs = state.get('attributes')
    if not isinstance(attrs, dict):
        raise ValueError('Missing price attributes')
    nordpool = any(k in attrs for k in ('raw_today', 'raw_tomorrow'))
    entsoe = any(k in attrs for k in ('prices_today', 'prices_tomorrow', 'prices'))
    if nordpool == entsoe:
        raise ValueError('Missing or ambiguous price schema')
    schema = 'nordpool' if nordpool else 'entsoe'
    keys = ('raw_today', 'raw_tomorrow') if nordpool else ('prices_today', 'prices_tomorrow', 'prices')
    if nordpool and 'tomorrow_valid' in attrs and not isinstance(attrs['tomorrow_valid'], bool):
        raise ValueError('Invalid tomorrow_valid')
    points: list[tuple[datetime, datetime | None]] = []
    if entsoe:
        price_by_time: dict[datetime, Any] = {}
        for key in keys:
            if key not in attrs:
                continue
            items = attrs[key]
            if not isinstance(items, list):
                raise ValueError('Invalid price array')
            seen_in_array: set[datetime] = set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError('Invalid price point')
                value = item.get('price')
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Invalid price value')
                if not isinstance(item.get('time'), str):
                    raise ValueError('Invalid price timestamp')
                start = timestamp(item['time'])
                if start in seen_in_array:
                    raise ValueError('Duplicate or overlapping prices')
                seen_in_array.add(start)
                if start in price_by_time and price_by_time[start] != value:
                    raise ValueError('Duplicate or overlapping prices')
                price_by_time[start] = value
                points.append((start, None))
    else:
        for key in keys:
            if key not in attrs or (key == 'raw_tomorrow' and attrs.get('tomorrow_valid') is False):
                continue
            items = attrs[key]
            if not isinstance(items, list):
                raise ValueError('Invalid price array')
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError('Invalid price point')
                value = item.get('value')
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('Invalid price value')
                if not isinstance(item.get('start'), str):
                    raise ValueError('Invalid price timestamp')
                start = timestamp(item['start'])
                end = None
                if not isinstance(item.get('end'), str):
                    raise ValueError('Missing interval end')
                end = timestamp(item['end'])
                if end <= start:
                    raise ValueError('Invalid price interval')
                points.append((start, end))
        points.sort(key=lambda point: point[0])
        for previous, current in zip(points, points[1:]):
            if current[0] == previous[0] or (previous[1] is not None and current[0] < previous[1]):
                raise ValueError('Duplicate or overlapping prices')
    if entsoe:
        points.sort(key=lambda point: point[0])
    cutoff = timestamp(origin)
    if not any((end > cutoff if end is not None else start >= cutoff) for start, end in points):
        raise ValueError('No current or future price window')
    return {'source': source, 'zone': zone, 'classification': 'unverified',
            'schema': schema, 'state': state}


_CONFIG_FIELDS = {
    'format_version', 'archive', 'weather_entity', 'price_entity', 'price_zone',
    'host', 'env_file', 'lock_file', 'reports_dir', 'max_attempts', 'max_bytes',
}


def _forecast_settings(path: str | Path) -> dict[str, Any]:
    contents = Path(path).read_bytes()
    data = json.loads(contents)
    if not isinstance(data, dict):
        raise ValueError('Forecast snapshot config must be a JSON object')
    if type(data.get('format_version')) is not int or data.get('format_version') != 1:
        raise ValueError('Forecast snapshot config format_version must be 1')
    for name in ('archive', 'lock_file'):
        value = data.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f'Forecast snapshot {name} must be an absolute path')
        data[name] = Path(value).resolve()
    if Path(data['archive']).name == 'hepo.db':
        raise ValueError('Forecast snapshot archive must not be hepo.db')
    for name in ('weather_entity', 'price_entity', 'price_zone'):
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Forecast snapshot {name} is required')
    env_file = data.get('env_file')
    if not isinstance(env_file, str) or not Path(env_file).is_absolute():
        raise ValueError('Forecast snapshot env_file must be an absolute path')
    data['env_file'] = Path(env_file).resolve()
    reports_dir = data.get('reports_dir')
    if reports_dir is None:
        reports_dir = data['archive'].parent / 'forecast-snapshot-reports'
    if not isinstance(reports_dir, str) or not Path(reports_dir).is_absolute():
        # allow Path objects too
        if isinstance(reports_dir, Path):
            reports_dir = str(reports_dir)
        else:
            raise ValueError('Forecast snapshot reports_dir must be an absolute path')
    data['reports_dir'] = Path(reports_dir).resolve()
    if data['lock_file'] == data['reports_dir'] or data['lock_file'] in {data['archive'], data['env_file']}:
        raise ValueError('Forecast snapshot lock must not replace an input or archive')
    if data['reports_dir'] in {data['archive'], data['env_file'], data['lock_file']}:
        raise ValueError('Forecast snapshot reports must not replace an input or archive')
    for name in ('max_attempts', 'max_bytes'):
        value = data.get(name, 100000 if name == 'max_attempts' else 2_000_000_000)
        if type(value) is not int or value <= 0:
            raise ValueError(f'Forecast snapshot {name} must be a positive integer')
        data[name] = value
    host = data.get('host')
    if host is None:
        host = dotenv_values(data['env_file']).get('HA_HOST')
    if not isinstance(host, str) or not host.strip():
        raise ValueError('Forecast snapshot host or HA_HOST is required')
    host = host.strip()
    if not host.startswith(('http://', 'https://')):
        host = 'http://' + host
    if not host.startswith(('http://', 'https://')):
        raise ValueError('Forecast snapshot host must be an http(s) URL')
    data['host'] = host.rstrip('/')
    data['config_sha256'] = hashlib.sha256(contents).hexdigest()
    return data


def _ha_headers(token: str) -> dict[str, str]:
    return {'Authorization': 'Bearer ' + token}


def collect(config: str | Path, *, execute: bool = False, now: float | None = None) -> dict[str, Any]:
    """Optional, non-actuating forecast snapshot collection.

    Dry-run by default. Independent of optimization; failures are reported but
    never alter planner state.
    """
    settings = _forecast_settings(Path(config))
    clock_now = time.time() if now is None else now
    if not math.isfinite(clock_now):
        raise ValueError('Finite collector clock required')
    origin_dt = datetime.fromtimestamp((clock_now // 900) * 900, timezone.utc)
    report: dict[str, Any] = {
        'format_version': 1,
        'status': 'dry_run' if not execute else 'running',
        'exit_code': 0,
        'phase': 'preflight',
        'origin': origin_dt.isoformat(),
        'archive': str(settings['archive']),
        'host': settings['host'],
        'weather_entity': settings['weather_entity'],
        'price_entity': settings['price_entity'],
        'price_zone': settings['price_zone'],
        'config_sha256': settings['config_sha256'],
        'data_only': True,
        'upstream_read_only': True,
    }
    if not execute:
        return report

    settings['reports_dir'].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_path = settings['reports_dir'] / f'forecast-snapshot-{stamp}-{uuid4().hex}.json'
    report['report_path'] = str(report_path)
    with report_path.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')

    lock = None
    try:
        report['phase'] = 'lock'
        lock = settings['lock_file'].open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report.update(status='skipped_locked', exit_code=75, phase='lock')
            return report

        report['phase'] = 'credentials'
        values = dotenv_values(settings['env_file'])
        token = values.get('HA_TOKEN')
        if not token:
            raise ValueError('HA_TOKEN is required in the configured env file')
        host = settings['host']

        def state(entity: str) -> Any:
            response = requests.get(host + '/api/states/' + entity,
                                   headers=_ha_headers(token), timeout=15)
            response.raise_for_status()
            return response.json()

        def weather() -> Any:
            metadata = state(settings['weather_entity'])
            response = requests.post(host + '/api/services/weather/get_forecasts?return_response',
                                     headers=_ha_headers(token),
                                     json={'entity_id': settings['weather_entity'], 'type': 'hourly'},
                                     timeout=20)
            response.raise_for_status()
            data = response.json()
            service = data.get('service_response', data) if isinstance(data, dict) else {}
            if (not isinstance(service, dict)
                    or not isinstance(service.get(settings['weather_entity']), dict)
                    or not isinstance(service[settings['weather_entity']].get('forecast'), list)
                    or not service[settings['weather_entity']]['forecast']):
                raise ValueError('Missing hourly weather forecast')
            return {'source': settings['weather_entity'], 'state': metadata, 'response': data}

        def price() -> Any:
            data = state(settings['price_entity'])
            return validate_price_window(data, settings['price_entity'],
                                        settings['price_zone'], origin_dt)

        report['phase'] = 'capture'
        result = capture(settings['archive'], origin_dt, weather, price,
                         clock=lambda: datetime.now(timezone.utc),
                         max_attempts=settings['max_attempts'],
                         max_bytes=settings['max_bytes'])
        weather_status = result['weather']['status']
        price_status = result['price']['status']
        report.update(
            status='success' if weather_status == 'ok' and price_status == 'ok' else 'failed',
            exit_code=0 if weather_status == 'ok' and price_status == 'ok' else 1,
            phase='complete',
            attempt_id=result['id'],
            weather_status=weather_status,
            price_status=price_status,
            weather_error=result['weather'].get('error_type'),
            price_error=result['price'].get('error_type'),
        )
        return report
    except KeyboardInterrupt:
        report.update(status='interrupted', exit_code=130, error_type='KeyboardInterrupt')
        return report
    except Exception as error:  # noqa: BLE001 - nonblocking collection must report, not crash caller
        report.update(status='failed', exit_code=1, phase='failed',
                      error_type=type(error).__name__)
        return report
    finally:
        if lock is not None:
            lock.close()
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        try:
            _replace_forecast_report(report_path, report)
        except Exception:  # noqa: BLE001
            pass


def _replace_forecast_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix('.json.tmp')
    owned = False
    try:
        with temporary.open('x') as output:
            owned = True
            output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional explicit forecast snapshot collector config')
    parser.add_argument('--execute', action='store_true', help='Actually capture forecast snapshots')
    parser.add_argument('--archive', help='Dedicated forecast SQLite archive path')
    parser.add_argument('--origin', help='Scheduled origin, timezone-aware')
    parser.add_argument('--host', help='Home Assistant URL')
    parser.add_argument('--weather-entity')
    parser.add_argument('--price-entity')
    parser.add_argument('--price-zone')
    parser.add_argument('--max-attempts', type=int, default=100000)
    parser.add_argument('--max-bytes', type=int, default=2_000_000_000)
    args = parser.parse_args(argv)
    if args.config is not None:
        try:
            result = collect(args.config, execute=args.execute)
        except (OSError, ValueError) as error:
            parser.error(f'Forecast snapshot configuration error ({type(error).__name__}); check explicit paths')
            return 2
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {'chunks', 'requests'}}, indent=2))
        return int(result['exit_code'])

    if not all((args.archive, args.origin, args.host, args.weather_entity,
                args.price_entity, args.price_zone)):
        parser.error('--archive, --origin, --host, --weather-entity, --price-entity and --price-zone are required unless --config is used')
        return 2
    token = os.environ.get('HA_TOKEN')
    if not token or not args.host.startswith(('http://', 'https://')):
        parser.error('HA_TOKEN and an http(s) host are required')
        return 2
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
        return validate_price_window(data, args.price_entity, args.price_zone, args.origin)

    result = capture(args.archive, args.origin, weather, price,
                     max_attempts=args.max_attempts, max_bytes=args.max_bytes)
    print(json.dumps({'id': result['id'], 'weather': result['weather']['status'],
                      'price': result['price']['status']}))
    return 0 if result['weather']['status'] == 'ok' and result['price']['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
