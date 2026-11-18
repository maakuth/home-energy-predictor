from datetime import datetime, timezone

import pytest

from utils.forecast_snapshots import capture, select, map_intervals


def clock():
    return datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def test_capture_preserves_partial_failure_and_vintages(tmp_path):
    def bad():
        raise ValueError('unavailable')

    weather = lambda: {'forecast': [{'datetime': '2026-01-01T11:00:00Z', 'temperature': 1}]}
    first = capture(tmp_path, '2026-01-01T09:45:00Z', weather, bad, clock=clock)
    second = capture(tmp_path, '2026-01-01T09:45:00Z', weather, lambda: {'raw': [1]}, clock=clock)
    assert first['weather']['payload']['forecast'][0]['temperature'] == 1
    assert first['price']['status'] == 'error'
    assert first['id'] != second['id']
    assert len(list(tmp_path.glob('*.json'))) == 2
    assert select(tmp_path, '2026-01-01T09:59:59Z') is None
    assert select(tmp_path, '2026-01-01T10:00:00Z')['id'] == second['id']


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
    capture(tmp_path, '2026-01-01T09:45:00Z', lambda: {}, lambda: {}, clock=clock, max_files=1)
    with pytest.raises(ValueError, match='quota'):
        capture(tmp_path, '2026-01-01T10:00:00Z', lambda: {}, lambda: {}, clock=clock, max_files=1)
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_selection_uses_completion_not_start(tmp_path):
    times = iter(datetime(2026, 1, 1, 10, minute, tzinfo=timezone.utc)
                 for minute in (0, 1, 2, 3))
    capture(tmp_path, '2026-01-01T10:00:00Z', lambda: {}, lambda: {}, clock=lambda: next(times))
    assert select(tmp_path, '2026-01-01T10:02:00Z') is None
    assert select(tmp_path, '2026-01-01T10:03:00Z') is not None
