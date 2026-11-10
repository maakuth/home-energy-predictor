import json

import pandas as pd
import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_sources import parse_sources
from utils.archive_features import load_archive_raw
from utils.measurement_policy import parse_policies


def policy(**updates):
    return {'kind': 'numeric', 'input_unit': 'W', 'canonical_unit': 'kW',
            'sign': 1, 'require_unit': True, **updates}


def load(tmp_path, events, spec, *, end=120, chunk=60, attribute=None):
    path = tmp_path / 'archive.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.test', attribute, 0, end + 60, events)
    sources = parse_sources({'test': [{'entity': 'sensor.test', **(
        {'attribute': attribute} if attribute else {})}]})
    return load_archive_raw(path, 0, end, sources=sources, measurements=['test'],
                            chunk_seconds=chunk, policies={'test': spec})


@pytest.mark.parametrize('change', [
    {'kind': 'guess'}, {'sign': 0}, {'max_age_seconds': 0},
    {'max_age_seconds': float('nan')}, {'require_unit': 'yes'},
    {'canonical_unit': 'bananas'}, {'extra': 1}, {'input_unit': 'kWh'},
    {'kind': 'presence', 'input_unit': 'state', 'canonical_unit': '1'},
    {'kind': 'presence', 'input_unit': 'state', 'canonical_unit': '1',
     'state_values': {'home': 2, 'away': 0}},
    {'kind': 'numeric', 'input_unit': 'state', 'canonical_unit': 'state'},
])
def test_strict_registry(change):
    with pytest.raises(ValueError):
        parse_policies({'test': policy(**change)})


def test_units_sign_chunks_and_stale(tmp_path):
    attrs = json.dumps({'unit_of_measurement': 'W'})
    out = load(tmp_path, [(1, 0, '1000', attrs)], policy(sign=-1, max_age_seconds=90))
    assert out.test.iloc[0] == -1
    assert pd.isna(out.test.iloc[1])
    assert out.test_coverage_fraction.tolist() == [1, .5]
    assert out.test_source_valid.tolist() == [True, False]
    assert out.test_unit_verified.tolist() == [True, False]


@pytest.mark.parametrize('attrs', [None, '{}', '{broken', '{"unit_of_measurement":"kWh"}'])
def test_missing_or_wrong_actual_units_fail_required_validation(tmp_path, attrs):
    out = load(tmp_path, [(1, 0, '1000', attrs)], policy())
    assert out.test.isna().all()
    assert not out.test_source_valid.any()
    assert not out.test_unit_verified.any()


def test_optional_unit_absence_is_not_verification(tmp_path):
    out = load(tmp_path, [(1, 0, '1000', None)], policy(require_unit=False))
    assert out.test.tolist() == [1, 1]
    assert not out.test_unit_verified.any()


def test_attribute_units_and_conversion(tmp_path):
    out = load(tmp_path, [(1, 0, 'sunny', json.dumps(
        {'wind_speed': 10, 'wind_speed_unit': 'm/s'}))],
        policy(input_unit='m/s', canonical_unit='km/h', attribute='wind_speed',
               unit_attribute='wind_speed_unit'), attribute='wind_speed')
    assert out.test.tolist() == [36, 36]
    assert out.test_unit_verified.all()


def test_presence_any_home_not_category_average_and_unknown_missing(tmp_path):
    spec = policy(kind='presence', input_unit='state', canonical_unit='1',
                  require_unit=False, state_values={'home': 1, 'away': 0},
                  aggregation='any_home')
    out = load(tmp_path, [(1, 0, 'away', None), (2, 10, 'home', None),
                          (3, 20, 'away', None), (4, 60, 'not_home', None)], spec)
    assert out.test.iloc[0] == 1
    assert pd.isna(out.test.iloc[1])


def test_counter_exact_boundaries_and_chunk_invariance(tmp_path):
    attrs = '{"unit_of_measurement":"kWh"}'
    spec = policy(kind='counter', input_unit='kWh', canonical_unit='kW')
    events = [(1, 0, '10', attrs), (2, 60, '10.1', attrs), (3, 120, '10.2', attrs)]
    small = load(tmp_path, events, spec)
    assert small.test.tolist() == pytest.approx([6, 6])
    assert small.test_source_valid.all()
    large = load_archive_raw(tmp_path / 'archive.sqlite', 0, 120,
        sources=parse_sources({'test': [{'entity': 'sensor.test'}]}),
        measurements=['test'], policies={'test': spec}, chunk_seconds=120)
    pd.testing.assert_frame_equal(small, large)


@pytest.mark.parametrize('events', [
    [(1, 0, '10', None), (2, 120, '11', None)],
    [(1, 0, '10', None), (2, 30, '0', None), (3, 60, '10.1', None)],
    [(1, 0, '10', None), (2, 30, 'unavailable', None), (3, 60, '10.1', None)],
])
def test_counter_no_extrapolation_reset_or_invalid_bridge(tmp_path, events):
    out = load(tmp_path, events, policy(kind='counter', input_unit='kWh',
                                      canonical_unit='kW', require_unit=False))
    assert pd.isna(out.test.iloc[0])
    assert not out.test_source_valid.iloc[0]


def test_policy_file_and_duplicate_keys(tmp_path):
    from utils.measurement_policy import load_policies
    registry = tmp_path / 'policies.yaml'
    registry.write_text('test:\n  kind: numeric\n  input_unit: W\n  canonical_unit: kW\n  sign: 1\n')
    assert load_policies(registry)['test'].canonical_unit == 'kW'
    registry.write_text('test: {}\ntest: {}\n')
    with pytest.raises(ValueError, match='duplicate'):
        load_policies(registry)


def test_attribute_mismatch_is_configuration_error(tmp_path):
    with pytest.raises(ValueError, match='attribute'):
        load(tmp_path, [], policy(attribute='power'))


def test_malformed_unit_metadata_is_missing_not_exception(tmp_path):
    out = load(tmp_path, [(1, 0, '10', '{"unit_of_measurement":[]}')], policy())
    assert out.test.isna().all()


def test_counter_rejects_unqueried_gap_and_source_switch_inside_bin(tmp_path):
    path = tmp_path / 'archive.sqlite'
    spec = policy(kind='counter', input_unit='kWh', canonical_unit='kW', require_unit=False)
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.old', None, 0, 20, [(1, 0, '10', None)])
        archive.ingest('sensor.old', None, 40, 120, [(2, 60, '11', None)])
        archive.ingest('sensor.new', None, 0, 120, [(3, 30, '50', None), (4, 60, '51', None)])
    sources = parse_sources({'test': [{'entity': 'sensor.old'}]})
    out = load_archive_raw(path, 0, 60, sources=sources, measurements=['test'], policies={'test': spec})
    assert out.test.isna().all()
    sources = parse_sources({'test': [
        {'entity': 'sensor.old', 'to': '1970-01-01T00:00:30Z'},
        {'entity': 'sensor.new', 'from': '1970-01-01T00:00:30Z'}]})
    out = load_archive_raw(path, 0, 60, sources=sources, measurements=['test'], policies={'test': spec})
    assert out.test.isna().all()
    assert not out.test_source_valid.any()


def test_canonical_contract_prevents_wrong_builtin_output_units(tmp_path):
    path = tmp_path / 'archive.sqlite'
    with MeasurementArchive(path):
        pass
    with pytest.raises(ValueError, match='canonical'):
        load_archive_raw(path, 0, 60,
            sources=parse_sources({'battery_power': [{'entity': 'sensor.battery'}]}),
            measurements=['battery_power'], policies={'battery_power': policy()})


def test_pure_unit_and_counter_transforms():
    from utils.measurement_policy import convert_unit, counter_power
    assert convert_unit(32, '°F', 'C') == pytest.approx(0)
    assert convert_unit(1, 'kW', 'W') == 1000
    spec = parse_policies({'test': policy(kind='counter', input_unit='Wh', canonical_unit='W')})['test']
    assert counter_power(100, 200, 60, spec) == 6000
    assert pd.isna(counter_power(200, 100, 60, spec))


def test_predecessor_reader_evidence_hook_is_used_but_age_still_applies(tmp_path, monkeypatch):
    from utils.measurement_reader import MeasurementReader
    from utils.measurement_resampling import resample_numeric
    path = tmp_path / 'archive.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.test', None, 0, 30, [(1, 0, '10', None)])
        archive.ingest('sensor.test', None, 60, 120, [])
    sources = parse_sources({'test': [{'entity': 'sensor.test'}]})
    calls = []

    def supported(self, source_id, event_timestamp, coverage_start):
        calls.append((source_id, event_timestamp, coverage_start))
        return event_timestamp == 0 and coverage_start == 60

    monkeypatch.setattr(MeasurementReader, 'carry_supported', supported, raising=False)
    with MeasurementReader(path) as reader:
        out = resample_numeric(reader, sources, 'test', 60, 120, 60)
        stale = resample_numeric(reader, sources, 'test', 60, 120, 60, max_age_seconds=90)
    assert out.value.tolist() == [10]
    assert pd.isna(stale.value.iloc[0])
    assert stale.coverage_fraction.iloc[0] == .5
    assert calls and calls[0][1:] == (0, 60)


def test_counter_legacy_resampled_means_never_become_boundary_observations(tmp_path):
    from utils.measurement_legacy import import_legacy_csv
    path = tmp_path / 'archive.sqlite'
    csv = tmp_path / 'counter.csv'
    csv.write_text('timestamp,mummun_energy\n1970-01-01T00:00:00Z,10\n1970-01-01T00:01:00Z,11\n')
    with MeasurementArchive(path) as archive:
        import_legacy_csv(archive, csv, 60)
    out = load_archive_raw(path, 0, 120,
        sources=parse_sources({'mummun_energy': [{'entity': 'sensor.energy'}]}),
        measurements=['mummun_power'], allow_legacy=True,
        policies={'mummun_power': policy(kind='counter', input_unit='kWh',
                                         canonical_unit='kW', require_unit=False)})
    assert out.mummun_power.isna().all()
    assert not out.mummun_power_source_valid.any()


def test_builtin_category_cannot_be_declared_numeric(tmp_path):
    with pytest.raises(ValueError, match='kind'):
        load_archive_raw(tmp_path / 'absent.sqlite', 0, 60,
            sources=parse_sources({'ev_position': [{'entity': 'device_tracker.car'}]}),
            measurements=['ev_position'], policies={'ev_position': policy(
                input_unit='1', canonical_unit='1')})


def test_policy_path_hash_and_complete_registry(tmp_path):
    import hashlib
    from utils.measurement_policy import require_policy_entries, load_policies
    path = tmp_path / 'archive.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.test', None, 0, 120, [(1, 0, '1000', '{"unit_of_measurement":"W"}')])
    registry = tmp_path / 'policy.yaml'
    registry.write_text('test:\n  kind: numeric\n  input_unit: W\n  canonical_unit: kW\n  sign: 1\n  require_unit: true\n')
    require_policy_entries(load_policies(registry), ['test'])
    with pytest.raises(ValueError, match='missing'):
        require_policy_entries(load_policies(registry), ['test', 'absent'])
    result = load_archive_raw(path, 0, 60,
        sources=parse_sources({'test': [{'entity': 'sensor.test'}]}),
        measurements=['test'], policy_path=registry, require_policies=True)
    assert result.attrs['archive']['policy_sha256'] == hashlib.sha256(registry.read_bytes()).hexdigest()
    assert len(result.attrs['archive']['policy_semantic_sha256']) == 64
    assert result.attrs['archive']['policies']['test']['require_unit'] is True
    parsed = load_archive_raw(path, 0, 60,
        sources=parse_sources({'test': [{'entity': 'sensor.test'}]}),
        measurements=['test'], policies=load_policies(registry), require_policies=True)
    assert parsed.test.tolist() == [1]
    assert parsed.attrs['archive']['policy_semantic_sha256'] == result.attrs['archive']['policy_semantic_sha256']
    with pytest.raises(ValueError, match='missing'):
        load_archive_raw(path, 0, 60,
            sources=parse_sources({'other': [{'entity': 'sensor.test'}]}),
            measurements=['other'], policy_path=registry, require_policies=True)


def test_required_units_do_not_accept_unverifiable_legacy_intervals(tmp_path):
    from utils.measurement_legacy import import_legacy_csv
    path = tmp_path / 'archive.sqlite'
    csv = tmp_path / 'legacy.csv'
    csv.write_text('timestamp,test\n1970-01-01T00:00:00Z,1\n')
    with MeasurementArchive(path) as archive:
        import_legacy_csv(archive, csv, 60)
    out = load_archive_raw(path, 0, 60,
        sources=parse_sources({'test': [{'entity': 'sensor.test'}]}),
        measurements=['test'], policies={'test': policy()}, allow_legacy=True)
    assert out.test.isna().all()
    assert not out.test_unit_verified.any()
