from datetime import datetime, timezone

import pytest

from utils.measurement_sources import parse_sources, source_at


def test_open_intervals_and_timezone_aware_half_open_boundary():
    sources = parse_sources({'outside_temp': [
        {'entity': 'sensor.old', 'to': '2026-10-19T00:00:00+03:00'},
        {'entity': 'sensor.new', 'from': '2026-10-19T00:00:00+03:00'},
    ]})['outside_temp']
    before = datetime(2026, 10, 18, 20, 59, 59, tzinfo=timezone.utc)
    boundary = datetime(2026, 10, 18, 21, 0, tzinfo=timezone.utc)
    assert source_at(sources, before).entity == 'sensor.old'
    assert source_at(sources, boundary).entity == 'sensor.new'


@pytest.mark.parametrize('mapping', [
    [{'entity': 'a'}, {'entity': 'b'}],
    [{'entity': 'a', 'to': '2026-10-19T00:00:00'}],
    [{'entity': 'a', 'from': '2026-10-20T00:00:00Z', 'to': '2026-10-19T00:00:00Z'}],
    [{'entity': 'a', 'to': '2026-10-20T00:00:00Z'}, {'entity': 'b', 'from': '2026-10-19T00:00:00Z'}],
    [{'entity': ''}],
    [{'entity': 'a', 'form': '2026-10-20T00:00:00Z'}],
])
def test_invalid_mapping_is_rejected(mapping):
    with pytest.raises(ValueError):
        parse_sources({'outside_temp': mapping})


def test_explicit_gap_remains_unknown():
    sources = parse_sources({'x': [
        {'entity': 'old', 'to': '2026-01-01T00:00:00Z'},
        {'entity': 'new', 'from': '2026-02-01T00:00:00Z'},
    ]})['x']
    assert source_at(sources, datetime(2026, 1, 15, tzinfo=timezone.utc)) is None
