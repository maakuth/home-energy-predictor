"""Isolated as-received forecast capture and conservative as-of selection.

No planner, device control, database or measurement state is touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
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


def capture(directory: str | Path, origin: Any, weather: Callable[[], Any],
            price: Callable[[], Any], *, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
            max_files: int = 100000, max_bytes: int = 2_000_000_000) -> dict[str, Any]:
    """Capture independently; quota exhaustion fails closed, never deletes vintages."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Archive directory must already exist')
    scheduled = timestamp(origin)
    attempt: dict[str, Any] = {'format_version': 1, 'id': uuid4().hex,
                               'scheduled_origin': scheduled.isoformat()}
    for name, fetch in (('weather', weather), ('price', price)):
        start = timestamp(clock())
        try:
            payload = fetch()
            # Check serializability before recording success.
            json.dumps(payload, allow_nan=False)
            result = {'status': 'ok', 'payload': payload}
        except Exception as exc:
            # Do not persist exception text: it may contain credentials or URLs.
            result = {'status': 'error', 'error_type': type(exc).__name__}
        end = timestamp(clock())
        if end < start:
            raise ValueError('Capture clock moved backwards')
        result.update(started_at=start.isoformat(), completed_at=end.isoformat())
        attempt[name] = result
    encoded = (json.dumps(attempt, allow_nan=False, sort_keys=True) + '\n').encode()
    with (directory / '.forecast-lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = list(directory.glob('*.json'))
        if (max_files <= 0 or max_bytes <= 0 or len(files) >= max_files
                or sum(p.stat().st_size for p in files) + len(encoded) > max_bytes):
            raise ValueError('Archive quota exceeded; no snapshots removed')
        fd, temp_name = tempfile.mkstemp(prefix='.forecast-', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temp_name, directory / (attempt['id'] + '.json'))
        finally:
            os.unlink(temp_name)
    return attempt


def select(directory: str | Path, cutoff: Any) -> dict[str, Any] | None:
    """Select latest complete same-attempt pair available by cutoff."""
    cutoff = timestamp(cutoff)
    candidates = []
    for path in Path(directory).glob('*.json'):
        with path.open(encoding='utf-8') as stream:
            item = json.load(stream)
        if item.get('format_version') != 1:
            raise ValueError(f'Unknown snapshot format: {path}')
        origin = timestamp(item['scheduled_origin'])
        w, p = item['weather'], item['price']
        if origin <= cutoff and w['status'] == p['status'] == 'ok':
            completed = max(timestamp(w['completed_at']), timestamp(p['completed_at']))
            if completed <= cutoff:
                candidates.append((origin, completed, item['id'], item))
    return max(candidates, key=lambda x: x[:3])[3] if candidates else None


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
    parser.add_argument('--directory', required=True, help='Existing dedicated snapshot directory')
    parser.add_argument('--origin', required=True, help='Scheduled origin, timezone-aware')
    parser.add_argument('--host', required=True, help='Home Assistant URL')
    parser.add_argument('--weather-entity', required=True)
    parser.add_argument('--price-entity', required=True)
    parser.add_argument('--price-zone', required=True)
    parser.add_argument('--max-files', type=int, default=100000)
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

    result = capture(args.directory, args.origin, weather, price,
                     max_files=args.max_files, max_bytes=args.max_bytes)
    print(json.dumps({'id': result['id'], 'weather': result['weather']['status'],
                      'price': result['price']['status']}))
    if result['weather']['status'] != 'ok' or result['price']['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
