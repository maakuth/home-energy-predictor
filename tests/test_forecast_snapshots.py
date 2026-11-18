from datetime import datetime, timezone
import sqlite3

import pytest

from utils.forecast_snapshots import capture, select, map_intervals


def clock():
    return datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def test_capture_preserves_partial_failure_and_vintages(tmp_path):
    def bad():
        raise ValueError('unavailable')

    weather = lambda: {'forecast': [{'datetime': '2026-01-01T11:00:00Z', 'temperature': 1}]}
    db = tmp_path / 'forecast.sqlite'
    first = capture(db, '2026-01-01T09:45:00Z', weather, bad, clock=clock)
    second = capture(db, '2026-01-01T09:45:00Z', weather, lambda: {'raw': [1]}, clock=clock)
    assert first['weather']['payload']['forecast'][0]['temperature'] == 1
    assert first['price']['status'] == 'error'
    assert first['id'] != second['id']
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0] == 2
        assert conn.execute('SELECT count(*) FROM payloads').fetchone()[0] == 2
    assert select(db, '2026-01-01T09:59:59Z') is None
    assert select(db, '2026-01-01T10:00:00Z') == second


def test_mapping_never_pads_or_bridges_gaps():
    points = [
        {'start': '2026-01-01T10:00:00Z', 'end': '2026-01-01T11:00:00Z', 'value': 5},
        {'start': '2026-01-01T12:00:00Z', 'end': '2026-01-01T13:00:00Z', 'value': 6},
    ]
    targets = ['2026-01-01T10:45:00Z', '2026-01-01T11:00:00Z', '2026-01-01T13:00:00Z']
    assert map_intervals(points, targets) == [5, None, None]


def test_invalid_timestamps_and_overlaps_rejected():
    with pytest.raises(ValueError):
        select('/does/not/exist', '2026-01-01T10:00:00')
    with pytest.raises(ValueError):
        map_intervals([
            {'start': '2026-01-01T10:00:00Z', 'end': '2026-01-01T11:00:00Z', 'value': 1},
            {'start': '2026-01-01T10:30:00Z', 'end': '2026-01-01T11:30:00Z', 'value': 2},
        ], ['2026-01-01T10:45:00Z'])


def test_quota_does_not_delete_prior_vintages(tmp_path):
    db = tmp_path / 'forecast.sqlite'
    capture(db, '2026-01-01T09:45:00Z', lambda: {}, lambda: {}, clock=clock, max_attempts=1)
    with pytest.raises(ValueError, match='quota'):
        capture(db, '2026-01-01T10:00:00Z', lambda: {}, lambda: {}, clock=clock, max_attempts=1)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0] == 1


def test_selection_uses_completion_not_start(tmp_path):
    times = iter(datetime(2026, 1, 1, 10, minute, tzinfo=timezone.utc)
                 for minute in (0, 1, 2, 3))
    db = tmp_path / 'forecast.sqlite'
    capture(db, '2026-01-01T10:00:00Z', lambda: {}, lambda: {}, clock=lambda: next(times))
    assert select(db, '2026-01-01T10:02:00Z') is None
    assert select(db, '2026-01-01T10:03:00Z') is not None


def test_refuses_existing_unrelated_database(tmp_path):
    db = tmp_path / 'other.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE keep_me (value TEXT)')
        conn.execute("INSERT INTO keep_me VALUES ('safe')")
    with pytest.raises(ValueError, match='archive'):
        capture(db, '2026-01-01T10:00:00Z', lambda: {}, lambda: {}, clock=clock)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT * FROM keep_me').fetchall() == [('safe',)]


def test_no_creation_on_read_and_byte_quota_rolls_back(tmp_path):
    db = tmp_path / 'forecast.sqlite'
    with pytest.raises(FileNotFoundError):
        select(db, '2026-01-01T10:00:00Z')
    assert not db.exists()
    with pytest.raises(ValueError, match='quota'):
        capture(db, '2026-01-01T10:00:00Z', lambda: {'large': 'x' * 100000},
                lambda: {}, clock=clock, max_bytes=32768)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM attempts').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM payloads').fetchone()[0] == 0


def test_fractional_cutoff_does_not_allow_future_completion(tmp_path):
    db = tmp_path / 'forecast.sqlite'
    capture(db, '2026-01-01T09:00:00Z', lambda: {}, lambda: {}, clock=clock)
    assert select(db, '2026-01-01T09:59:59.999999Z') is None
    assert select(db, '2026-01-01T10:00:00.000001Z') is not None


def test_revised_forecasts_keep_both_vintages(tmp_path):
    db = tmp_path / 'forecast.sqlite'
    old_clock = lambda: datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    old = capture(db, '2026-01-01T09:00:00Z', lambda: {'value': 1},
                  lambda: {'price': 3}, clock=old_clock)
    new = capture(db, '2026-01-01T10:00:00Z', lambda: {'value': 2},
                  lambda: {'price': 3}, clock=clock)
    assert select(db, '2026-01-01T09:30:00Z') == old
    assert select(db, '2026-01-01T10:00:00Z') == new
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM payloads').fetchone()[0] == 3


def test_existing_archive_quota_failure_preserves_rows(tmp_path):
    db = tmp_path / 'forecast.sqlite'
    old = capture(db, '2026-01-01T09:00:00Z', lambda: {'value': 1},
                  lambda: {'price': 3}, clock=clock)
    with pytest.raises(ValueError, match='quota'):
        capture(db, '2026-01-01T10:00:00Z', lambda: {'value': 'x' * 100000},
                lambda: {'price': 4}, clock=clock, max_bytes=32768)
    assert select(db, '2026-01-01T10:00:00Z') == old
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM payloads').fetchone()[0] == 2


def test_overlapping_forecast_points_are_normalized_and_reconstructed(tmp_path):
    db = tmp_path / 'forecast.sqlite'
    common = {'datetime': '2026-01-01T11:00:00Z', 'temperature': 2,
              'wind_speed': 3.5, 'wind_bearing': 120, 'condition': 'rain'}
    other = {'datetime': '2026-01-01T12:00:00Z', 'temperature': 4, 'wind_speed': None}
    old_weather = {'response': {'weather.home': {'forecast': [common, other]}}, 'extra': ['x', None]}
    new_weather = {'response': {'weather.home': {'forecast': [common, {**other, 'temperature': 5}]}}}
    prices = {'raw_today': [{'start': '2026-01-01T11:00:00Z',
                             'end': '2026-01-01T11:15:00Z', 'value': -1.25}]}
    old = capture(db, '2026-01-01T09:00:00Z', lambda: old_weather,
                  lambda: prices, clock=lambda: datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))
    new = capture(db, '2026-01-01T10:00:00Z', lambda: new_weather,
                  lambda: prices, clock=clock)
    assert select(db, '2026-01-01T10:00:00Z') == new
    assert select(db, '2026-01-01T09:30:00Z') == old
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM forecast_points').fetchone()[0] == 4
        assert conn.execute("SELECT temperature, wind_speed, wind_bearing FROM forecast_points "
                            "WHERE target = ? AND temperature = 2", (common['datetime'],)).fetchone() == (2, 3.5, 120)
        assert conn.execute('SELECT count(*) FROM forecast_members').fetchone()[0] == 5
    assert old['weather']['payload'] == old_weather


def test_point_extra_fields_and_nonforecast_lists_round_trip(tmp_path):
    db = tmp_path / 'forecast.sqlite'
    payload = {'nested': [[{'start': 'time', 'temperature': None, 'wind_speed': True,
                            'value': 'unknown', 'custom': {'x': [1, 2]}}]],
               'untouched': [{'no_time': 3}], 'empty': []}
    result = capture(db, '2026-01-01T10:00:00Z', lambda: payload, lambda: [], clock=clock)
    assert select(db, '2026-01-01T10:00:00Z') == result


def test_older_sqlite_schema_is_not_modified(tmp_path):
    from utils.forecast_snapshots import APPLICATION_ID

    db = tmp_path / 'old.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute(f'PRAGMA application_id={APPLICATION_ID}')
        conn.execute('PRAGMA user_version=1')
        conn.execute('CREATE TABLE payloads (json TEXT)')
        conn.execute("INSERT INTO payloads VALUES ('original')")
    with pytest.raises(ValueError, match='archive'):
        capture(db, '2026-01-01T10:00:00Z', lambda: {}, lambda: {}, clock=clock)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT json FROM payloads').fetchone() == ('original',)
