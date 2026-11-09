from datetime import datetime, timezone

import pytest

from utils.measurement_sources import parse_sources, source_at, load_sources, mapped_ranges


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


def test_repository_sources_keep_old_shared_meter_during_resistance_only_period():
    sources = load_sources('config/measurement_sources.yaml')
    before = datetime.fromisoformat('2026-10-20T08:29:59+03:00')
    switch = datetime.fromisoformat('2026-10-20T08:30:00+03:00')
    assert source_at(sources['gshp_power'], before).entity == 'sensor.mlp_teho'
    assert source_at(sources['gshp_power'], switch).entity == 'sensor.instantaneous_used_power_32167'
    assert source_at(sources['gshp_power'], datetime.fromisoformat('2026-10-10T12:00:00+03:00')).entity == 'sensor.mlp_teho'
    assert source_at(sources['outside_temp'], datetime.fromisoformat('2026-10-18T23:59:59+03:00')).entity == 'sensor.ulkona_temperature_2'
    assert source_at(sources['outside_temp'], datetime.fromisoformat('2026-10-19T00:00:00+03:00')).entity == 'sensor.ulkolampotila'


def test_ranges_clip_to_mapping_bounds_and_preserve_gaps():
    sources = parse_sources({'x': [
        {'entity': 'old', 'to': '2026-10-07T08:00:00+03:00'},
        {'entity': 'new', 'from': '2026-10-20T08:30:00+03:00'},
    ]})
    start = datetime.fromisoformat('2026-10-01T00:00:00+03:00').timestamp()
    end = datetime.fromisoformat('2026-11-01T00:00:00+02:00').timestamp()
    ranges = mapped_ranges(sources, start, end)
    assert ranges == [
        ('x', sources['x'][0], start, datetime.fromisoformat('2026-10-07T08:00:00+03:00').timestamp()),
        ('x', sources['x'][1], datetime.fromisoformat('2026-10-20T08:30:00+03:00').timestamp(), end),
    ]


@pytest.mark.parametrize('content', [
    '[]', '{}', 'x: []', 'x: []\nx: []',
    'x:\n  - entity: old\n    entity: new',
    'x:\n  - entity: old\n    to: 2026-10-19',
])
def test_invalid_yaml_configuration_rejected(tmp_path, content):
    path = tmp_path / 'sources.yaml'
    path.write_text(content)
    with pytest.raises(ValueError):
        load_sources(path)
