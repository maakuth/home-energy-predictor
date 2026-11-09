import json

import pandas as pd
import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_legacy import import_legacy_csv
from utils.measurement_reader import MeasurementReader
from utils.measurement_resampling import resample_numeric
from utils.measurement_sources import parse_sources


def config(entries=None):
    return parse_sources({'power': entries or [{'entity': 'sensor.power'}]})


def test_time_weighted_mean_not_event_mean_and_no_future_leakage(tmp_path):
    path = tmp_path / 'm.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, 0, 120, [
            (1, 0, '1000', None), (2, 10, '3000', None), (3, 60, '9000', None)])
    with MeasurementReader(path) as reader:
        result = resample_numeric(reader, config(), 'power', 0, 60, interval_seconds=60)
    assert result.value.iloc[0] == pytest.approx((1000 * 10 + 3000 * 50) / 60)
    assert result.source_valid.iloc[0]
    assert result.coverage_fraction.iloc[0] == 1


def test_predecessor_works_but_explicit_expiry_limits_carry(tmp_path):
    path = tmp_path / 'm.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, 0, 120, [(1, 0, '1000', None)])
    with MeasurementReader(path) as reader:
        full = resample_numeric(reader, config(), 'power', 60, 120, interval_seconds=60)
        stale = resample_numeric(reader, config(), 'power', 60, 120, interval_seconds=60, max_age_seconds=90)
    assert full.value.iloc[0] == 1000
    assert pd.isna(stale.value.iloc[0])
    assert stale.coverage_fraction.iloc[0] == .5


def test_recorder_outage_cannot_be_hidden_with_legacy_history(tmp_path):
    path = tmp_path / 'm.sqlite'
    csv = tmp_path / 'raw.csv'
    csv.write_text('timestamp,power\n1970-01-01T00:00:00Z,1000\n')
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, 0, 60, [(1, 0, '1000', None), (2, 30, 'unavailable', None)])
        import_legacy_csv(archive, csv, 60)
    with MeasurementReader(path) as reader:
        result = resample_numeric(reader, config(), 'power', 0, 60, interval_seconds=60, allow_legacy=True)
    assert pd.isna(result.value.iloc[0])
    assert result.coverage_fraction.iloc[0] == .5
    assert result.provenance.iloc[0] == 'recorder'


def test_no_carry_across_source_boundary(tmp_path):
    path = tmp_path / 'm.sqlite'
    sources = config([{'entity': 'sensor.old', 'to': '1970-01-01T00:01:00Z'},
                      {'entity': 'sensor.new', 'from': '1970-01-01T00:01:00Z'}])
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.old', None, 0, 120, [(1, 0, '1000', None)])
        archive.ingest('sensor.new', None, 0, 120, [(2, 0, '9000', None)])
    with MeasurementReader(path) as reader:
        result = resample_numeric(reader, sources, 'power', 0, 120, interval_seconds=60)
    assert result.value.iloc[0] == 1000
    assert pd.isna(result.value.iloc[1])


def test_fetch_coverage_limits_raw_state_and_legacy_requires_opt_in(tmp_path):
    path = tmp_path / 'm.sqlite'
    csv = tmp_path / 'raw.csv'
    csv.write_text('timestamp,power\n1970-01-01T00:01:00Z,3000\n')
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, 0, 60, [(1, 0, '1000', None)])
        import_legacy_csv(archive, csv, 60)
    with MeasurementReader(path) as reader:
        raw = resample_numeric(reader, config(), 'power', 0, 120, interval_seconds=60)
        recovered = resample_numeric(reader, config(), 'power', 0, 120, interval_seconds=60, allow_legacy=True)
    assert pd.isna(raw.value.iloc[1])
    assert recovered.value.tolist() == [1000, 3000]
    assert recovered.source_valid.tolist() == [True, False]
    assert recovered.provenance.tolist() == ['recorder', 'legacy_resampled']


def test_relay_integration_preserves_short_run_and_ignores_unavailable(tmp_path):
    path = tmp_path / 'm.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, 0, 60,
                       [(1, 0, 'off', None), (2, 10, 'on', None), (3, 20, 'off', None)])
    with MeasurementReader(path) as reader:
        result = resample_numeric(reader, config(), 'power', 0, 60, interval_seconds=60,
                                  state_values={'on': 6000, 'off': 0})
    assert result.value.iloc[0] == 1000


def test_attributes_and_equal_timestamp_last_event_wins(tmp_path):
    path = tmp_path / 'm.sqlite'
    sources = config([{'entity': 'weather.home', 'attribute': 'wind_speed'}])
    with MeasurementArchive(path) as archive:
        archive.ingest('weather.home', 'wind_speed', 0, 60, [
            (1, 0, 'cloudy', json.dumps({'wind_speed': 2})),
            (2, 0, 'cloudy', json.dumps({'wind_speed': 4}))])
    with MeasurementReader(path) as reader:
        result = resample_numeric(reader, sources, 'power', 0, 60, interval_seconds=60)
    assert result.value.iloc[0] == 4


def test_dst_window_uses_elapsed_utc_time(tmp_path):
    path = tmp_path / 'm.sqlite'
    start = pd.Timestamp('2026-10-25T00:00:00+03:00').timestamp()
    end = pd.Timestamp('2026-10-26T00:00:00+02:00').timestamp()
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, start, end, [(1, start, '1000', None)])
    with MeasurementReader(path) as reader:
        result = resample_numeric(reader, config(), 'power', start, end, interval_seconds=900)
    assert len(result) == 100
    assert result.source_valid.all()


def test_unqueried_gap_does_not_justify_carrying_an_old_state(tmp_path):
    path = tmp_path / 'm.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, 0, 30, [(1, 0, '1000', None)])
        archive.ingest('sensor.power', None, 60, 120, [])
    with MeasurementReader(path) as reader:
        result = resample_numeric(reader, config(), 'power', 60, 120, interval_seconds=60)
    assert pd.isna(result.value.iloc[0])
    assert result.coverage_fraction.iloc[0] == 0


def test_legacy_overlap_conflict_remains_unknown_after_aggregation(tmp_path):
    path = tmp_path / 'm.sqlite'
    with MeasurementArchive(path) as archive:
        for i, value in enumerate([1000, 2000]):
            csv = tmp_path / f'{i}.csv'
            csv.write_text(f'timestamp,power\n1970-01-01T00:00:00Z,{value}\n')
            import_legacy_csv(archive, csv, 60)
    with MeasurementReader(path) as reader:
        unknown = resample_numeric(reader, config(), 'power', 0, 60, interval_seconds=60, allow_legacy=True)
        selected = resample_numeric(reader, config(), 'power', 0, 60, interval_seconds=60,
                                    allow_legacy=True, preferred_imports=[2])
    assert pd.isna(unknown.value.iloc[0])
    assert unknown.conflict.iloc[0]
    assert selected.value.iloc[0] == 2000
    assert selected.conflict.iloc[0]
    assert not selected.source_valid.iloc[0]
