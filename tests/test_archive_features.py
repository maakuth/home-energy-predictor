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


def test_complete_legacy_policy_preserves_source_flags_and_hash_provenance(tmp_path):
    from utils.measurement_legacy import import_legacy_csv
    from utils.baseload_quality import select_training_rows
    path, sources = build(tmp_path)
    sources['battery_power'] = parse_sources({'battery_power': [{'entity': 'sensor.missing'}]})['battery_power']
    csv = tmp_path / 'raw.csv'
    index = pd.date_range(pd.Timestamp(START - 86400, unit='s', tz='UTC'), periods=1442, freq='min')
    pd.DataFrame({'battery_power': 0}, index=index).to_csv(csv, index_label='timestamp')
    with MeasurementArchive(path) as archive:
        import_legacy_csv(archive, csv, 60)
    out = load_archive_features(path, START, START + 120, sources=sources,
                                allow_legacy=True, legacy_policy='complete-legacy')
    assert out.baseload_power.eq(2).all()
    assert out.baseload_training_eligible.all()
    assert not out.baseload_target_valid.any()
    assert not out.battery_power_source_valid.any()
    assert out.battery_power_import_ids.tolist() == [(1,), (1,)]
    assert len(out.attrs['archive_training']['imports'][0]['sha256']) == 64
    assert len(select_training_rows(out)) == 2


def test_legacy_policy_requires_explicit_legacy_reading(tmp_path):
    path, sources = build(tmp_path)
    with pytest.raises(ValueError, match='allow_legacy'):
        load_archive_features(path, START, START + 120, sources=sources,
                              legacy_policy='complete-legacy')


@pytest.mark.parametrize('selected', [None, 1, 2])
def test_null_overlap_conflict_stays_ineligible_even_with_precedence(tmp_path, selected):
    from utils.measurement_legacy import import_legacy_csv
    path, sources = build(tmp_path)
    sources['battery_power'] = parse_sources({'battery_power': [{'entity': 'sensor.missing'}]})['battery_power']
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.missing', None, START - 86400, START,
                       [(99, START - 86400, '0', None)])
    for i, value in enumerate(['0', '']):
        csv = tmp_path / f'{i}.csv'
        csv.write_text(f'timestamp,battery_power\n2026-10-21T00:00:00Z,{value}\n')
        with MeasurementArchive(path) as archive:
            import_legacy_csv(archive, csv, 60)
    out = load_archive_features(path, START, START + 120, sources=sources,
        allow_legacy=True, legacy_policy='complete-legacy',
        preferred_imports=[selected] if selected else None)
    assert out.battery_power_conflict.iloc[0]
    assert not out.baseload_training_eligible.any()
    assert out.baseload_power.isna().all()


def test_explicit_invalid_legacy_flag_cannot_become_eligible(tmp_path):
    from utils.measurement_legacy import import_legacy_csv
    path, sources = build(tmp_path)
    sources['battery_power'] = parse_sources({'battery_power': [{'entity': 'sensor.missing'}]})['battery_power']
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.missing', None, START - 86400, START,
                       [(99, START - 86400, '0', None)])
    csv = tmp_path / 'invalid.csv'
    csv.write_text('timestamp,battery_power,battery_power_source_valid\n2026-10-21T00:00:00Z,0,False\n')
    with MeasurementArchive(path) as archive:
        import_legacy_csv(archive, csv, 60)
    out = load_archive_features(path, START, START + 120, sources=sources,
                                allow_legacy=True, legacy_policy='complete-legacy')
    assert out.battery_power.isna().all()
    assert not out.baseload_training_eligible.any()


def test_archive_never_substitutes_forecast_for_a_verified_zero_solar_meter(tmp_path):
    path, sources = build(tmp_path)
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.forecast', None, START - 86400, START + 120,
                       [(999, START - 86400, '2', None)])
    sources['solar_forecast'] = parse_sources({'solar_forecast': [{'entity': 'sensor.forecast'}]})['solar_forecast']
    out = load_archive_features(path, START, START + 120, sources=sources)
    assert out.solar_actual.eq(0).all()
    assert out.total_home_power.eq(3).all()
    assert out.baseload_power.eq(2).all()


def test_explicit_presence_features_preserve_category_rule_and_unknown(tmp_path):
    path, sources = build(tmp_path)
    with MeasurementArchive(path) as archive:
        archive.ingest('device_tracker.car', None, START - 86400, START + 120,
                       [(100, START - 86400, 'home', None),
                        (101, START + 60, 'unknown', None)])
    sources.update(parse_sources({'ev_position': [{'entity': 'device_tracker.car'}]}))
    policies = {'ev_position': {'kind': 'presence', 'input_unit': 'state',
        'canonical_unit': '1', 'sign': 1, 'state_values': {'home': 1, 'away': 0},
        'aggregation': 'any_home'}}
    out = load_archive_features(path, START, START + 120, sources=sources,
        measurements=[*sources], policies=policies)
    assert out.ev_position.iloc[0] == 1
    assert pd.isna(out.ev_position.iloc[1])


def test_optional_numeric_feature_has_no_preinstallation_zeros(tmp_path):
    path, sources = build(tmp_path)
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.aahp', None, START + 60, START + 120,
                       [(100, START + 60, '1000', None)])
    sources.update(parse_sources({'aahp_living_power': [{'entity': 'sensor.aahp',
        'from': pd.Timestamp(START + 60, unit='s', tz='UTC').isoformat()}]}))
    out = load_archive_features(path, START, START + 120, sources=sources,
                               measurements=[*sources])
    assert pd.isna(out.aahp_living_power.iloc[0])
    assert pd.notna(out.aahp_living_power.iloc[1])
