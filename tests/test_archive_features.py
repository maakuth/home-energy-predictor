import pandas as pd
import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_sources import parse_sources
from utils.archive_features import load_archive_raw, load_archive_features, default_sources


START = pd.Timestamp('2026-10-21T00:00:00Z').timestamp()


def build(tmp_path):
    path = tmp_path / 'archive.sqlite'
    values = {'total_power': 3, 'solar_actual': 0, 'battery_power': 0,
              'gshp_power': 1000, 'leaf_power': 0, 'outside_temp': 5,
              'wind_speed': 2, 'solar_forecast': 0}
    with MeasurementArchive(path) as archive:
        for i, (name, value) in enumerate(values.items()):
            archive.ingest('sensor.' + name, None, START - 86400, START + 120,
                           [(i, START - 86400, str(value), None)])
    sources = parse_sources({name: [{'entity': 'sensor.' + name}] for name in values})
    return path, sources


def test_features_load_24_hour_context_without_csv_and_trim_it(tmp_path):
    path, sources = build(tmp_path)
    result = load_archive_features(path, START, START + 120, sources=sources)
    assert len(result) == 2
    assert result.index[0] == pd.Timestamp(START, unit='s', tz='UTC')
    assert result.baseload_lag_24h.tolist() == [2, 2]
    assert result.baseload_target_valid.all()
    assert not list(tmp_path.glob('*.csv'))


def test_missing_measurements_stay_unknown_not_zero(tmp_path):
    path, sources = build(tmp_path)
    # Point this logical meter to an entity with no archived history.
    sources['battery_power'] = parse_sources({'battery_power': [{'entity': 'sensor.missing'}]})['battery_power']
    raw = load_archive_raw(path, START, START + 120, sources=sources)
    assert raw.battery_power.isna().all()
    assert not raw.battery_power_source_valid.any()
    with pytest.raises(ValueError, match='total_home_power'):
        load_archive_features(path, START, START + 120, sources=sources)


def test_chunking_does_not_change_resampled_values_or_lags(tmp_path):
    path, sources = build(tmp_path)
    small = load_archive_features(path, START, START + 120, sources=sources, chunk_seconds=3600)
    large = load_archive_features(path, START, START + 120, sources=sources, chunk_seconds=86400)
    pd.testing.assert_frame_equal(small, large)


def test_source_defaults_are_runtime_configurable_and_historical_overrides_win(monkeypatch):
    monkeypatch.setenv('GSHP_POWER_ENTITY', 'sensor.custom_power')
    monkeypatch.setenv('GSHP_ACCUMULATOR_TEMP_ENTITY', '')
    fallback = default_sources()
    assert fallback['gshp_power'][0].entity == 'sensor.custom_power'
    assert fallback['wind_speed'][0].attribute == 'wind_speed'
    assert 'accumulator_temp' not in fallback


def test_legacy_values_are_never_promoted_to_verified_training_truth(tmp_path):
    from utils.measurement_legacy import import_legacy_csv
    path, sources = build(tmp_path)
    sources['battery_power'] = parse_sources({'battery_power': [{'entity': 'sensor.missing'}]})['battery_power']
    csv = tmp_path / 'raw.csv'
    csv.write_text('timestamp,battery_power\n2026-10-21T00:00:00Z,0\n2026-10-21T00:01:00Z,0\n')
    with MeasurementArchive(path) as archive:
        import_legacy_csv(archive, csv, 60)
    result = load_archive_features(path, START, START + 120, sources=sources, allow_legacy=True)
    assert result.battery_power.tolist() == [0, 0]
    assert (result.battery_power_provenance == 'legacy_resampled').all()
    assert not result.baseload_target_valid.any()


@pytest.mark.parametrize('chunk', [0, -60, 61])
def test_invalid_chunk_size_is_rejected(tmp_path, chunk):
    path, sources = build(tmp_path)
    with pytest.raises(ValueError):
        load_archive_raw(path, START, START + 120, sources=sources, chunk_seconds=chunk)
