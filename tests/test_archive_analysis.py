import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
import yaml

from utils import archive_analysis as analysis
from utils.measurement_archive import MeasurementArchive
from utils.measurement_legacy import import_legacy_csv


@pytest.fixture
def inputs(tmp_path):
    archive = tmp_path / 'measurements.sqlite'
    sources = tmp_path / 'sources.yaml'
    policy = tmp_path / 'policy.yaml'
    definitions = {name: [{'entity': 'sensor.' + name}] for name in analysis.MEASUREMENTS}
    units = dict(outside_temp='C', wind_speed='m/s', total_power='kW',
                 solar_actual='kW', battery_power='W', gshp_power='W', leaf_power='W')
    policies = {name: dict(kind='numeric', input_unit=unit, canonical_unit=unit,
                           sign=1, require_unit=True) for name, unit in units.items()}
    policies['battery_power'].update(min_value=-11000, max_value=11000)
    with MeasurementArchive(archive, recorder_namespace='test-recorder') as db:
        identity = db.db.execute('SELECT archive_uuid FROM archive_identity').fetchone()[0]
        for i, name in enumerate(analysis.MEASUREMENTS):
            db.ingest('sensor.' + name, None, 0, 180,
                      [(i, 0., '1', json.dumps({'unit_of_measurement': units[name]}))])
    sources.write_text(yaml.safe_dump(definitions))
    policy.write_text(yaml.safe_dump(policies))
    return dict(archive=archive, archive_uuid=identity, recorder_namespace='test-recorder',
                sources=sources, sources_sha256=hashlib.sha256(sources.read_bytes()).hexdigest(),
                policy=policy, policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                start=0., end=180., interval_seconds=60)


def test_analysis_is_readonly_bounded_and_preserves_provenance(inputs, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Live service access attempted')
    monkeypatch.setattr('psycopg2.connect', deny)
    monkeypatch.setattr('requests.sessions.Session.request', deny)
    before = inputs['archive'].read_bytes()
    original = analysis.audit_measurements
    def check_reader(reader, *args, **kwargs):
        assert reader.db.execute('PRAGMA query_only').fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            reader.db.execute('CREATE TABLE forbidden (id INTEGER)')
        return original(reader, *args, **kwargs)
    monkeypatch.setattr(analysis, 'audit_measurements', check_reader)
    report = analysis.analyze(**inputs)
    assert report['archive_read_only'] is True
    assert report['migration_ready'] is False
    assert report['snapshot_semantics'] == 'one_read_transaction'
    assert report['archive_identity']['archive_uuid'] == inputs['archive_uuid']
    assert report['sources_sha256'] == inputs['sources_sha256']
    assert report['policy_sha256'] == inputs['policy_sha256']
    assert report['target_and_lag_eligibility'] == 'not_evaluated'
    assert report['unqueried_mapped_ranges'] == []
    assert report['unmapped_ranges'] == []
    for counts in report['measurements'].values():
        assert counts['bins'] == counts['measurement_eligible_bins'] == 3
        assert counts['provenance_bins'] == {'recorder': 3}
    assert inputs['archive'].read_bytes() == before


@pytest.mark.parametrize('field,value', [('archive_uuid', 'wrong-uuid'),
    ('archive_uuid', '11111111-1111-1111-1111-111111111111'),
    ('recorder_namespace', 'another-generation'), ('sources_sha256', '0'*64),
    ('policy_sha256', '0'*64), ('max_bins', 2), ('start', 1.), ('end', 0.)])
def test_invalid_context_fails_closed(inputs, field, value):
    inputs[field] = value
    with pytest.raises(ValueError):
        analysis.analyze(**inputs)


def test_missing_archive_is_never_created(inputs, tmp_path):
    inputs['archive'] = tmp_path / 'missing.sqlite'
    with pytest.raises((ValueError, sqlite3.Error)):
        analysis.analyze(**inputs)
    assert not inputs['archive'].exists()


def test_invalid_and_out_of_range_values_are_not_invented_zeros(inputs):
    with MeasurementArchive(inputs['archive']) as db:
        db.ingest('sensor.battery_power', None, 0, 180, [
            (99, 60., '12000', '{"unit_of_measurement":"W"}'),
            (100, 120., 'unavailable', None)])
    report = analysis.analyze(**inputs)
    assert report['measurements']['battery_power']['available_bins'] == 1
    assert report['measurements']['battery_power']['measurement_eligible_bins'] == 1
    assert report['unqueried_mapped_ranges'] == []  # Query coverage is not validity.


def test_unmapped_periods_and_unqueried_periods_are_distinct(inputs):
    data = yaml.safe_load(inputs['sources'].read_text())
    data['solar_actual'] = [{'entity': 'sensor.old', 'to': '1970-01-01T00:01:00Z'},
                            {'entity': 'sensor.new', 'from': '1970-01-01T00:02:00Z'}]
    inputs['sources'].write_text(yaml.safe_dump(data))
    inputs['sources_sha256'] = hashlib.sha256(inputs['sources'].read_bytes()).hexdigest()
    report = analysis.analyze(**inputs)
    assert report['unmapped_ranges'] == [{'measurement': 'solar_actual', 'start': 60., 'end': 120.}]
    assert [(r['start'], r['end']) for r in report['unqueried_mapped_ranges']] == [(0., 60.), (120., 180.)]


def test_legacy_availability_never_becomes_verified_eligibility(inputs, tmp_path):
    data = yaml.safe_load(inputs['sources'].read_text())
    data['battery_power'] = [{'entity': 'sensor.not_recorded'}]
    inputs['sources'].write_text(yaml.safe_dump(data))
    inputs['sources_sha256'] = hashlib.sha256(inputs['sources'].read_bytes()).hexdigest()
    policies = yaml.safe_load(inputs['policy'].read_text())
    policies['battery_power']['require_unit'] = False
    inputs['policy'].write_text(yaml.safe_dump(policies))
    inputs['policy_sha256'] = hashlib.sha256(inputs['policy'].read_bytes()).hexdigest()
    csv = tmp_path / 'legacy.csv'
    csv.write_text('timestamp,battery_power\n1970-01-01T00:00:00Z,0\n')
    with MeasurementArchive(inputs['archive']) as db:
        import_legacy_csv(db, csv, 60)
    report = analysis.analyze(**inputs)
    counts = report['measurements']['battery_power']
    assert counts['available_bins'] == 1
    assert counts['source_verified_bins'] == counts['measurement_eligible_bins'] == 0
    assert counts['provenance_bins']['legacy_resampled'] == 1


def test_queries_and_measurement_audit_share_one_snapshot(inputs, monkeypatch):
    original = analysis.audit_measurements
    def concurrent_ingest(reader, *args, **kwargs):
        with MeasurementArchive(inputs['archive']) as db:
            db.ingest('sensor.battery_power', None, 0, 180,
                      [(99, 60., 'unavailable', None)])
        return original(reader, *args, **kwargs)
    monkeypatch.setattr(analysis, 'audit_measurements', concurrent_ingest)
    assert analysis.analyze(**inputs)['measurements']['battery_power']['available_bins'] == 3


def cli_args(inputs, output):
    return ['--archive', str(inputs['archive']), '--archive-uuid', inputs['archive_uuid'],
        '--recorder-namespace', inputs['recorder_namespace'], '--sources', str(inputs['sources']),
        '--sources-sha256', inputs['sources_sha256'], '--measurement-policy', str(inputs['policy']),
        '--policy-sha256', inputs['policy_sha256'], '--from', '1970-01-01T00:00:00Z',
        '--to', '1970-01-01T00:03:00Z', '--interval-seconds', '60', '--output-dir', str(output)]


def test_cli_writes_only_a_new_isolated_report_directory(inputs, tmp_path, capsys):
    output = tmp_path / 'analysis-run'
    assert analysis.main(cli_args(inputs, output)) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['archive_read_only']
    assert json.loads(capsys.readouterr().out)['report_path'] == str(output / 'report.json')
    before = (output / 'report.json').read_bytes()
    with pytest.raises(SystemExit):
        analysis.main(cli_args(inputs, output))
    assert (output / 'report.json').read_bytes() == before


def test_failed_analysis_does_not_publish_a_success_report(inputs, tmp_path):
    output = tmp_path / 'failed-run'
    inputs['archive_uuid'] = 'wrong'
    with pytest.raises(SystemExit):
        analysis.main(cli_args(inputs, output))
    assert not output.exists()


@pytest.mark.parametrize('missing_from', ['sources', 'policy'])
def test_all_seven_measurements_require_explicit_mappings_and_policies(inputs, missing_from):
    path = inputs[missing_from]
    data = yaml.safe_load(path.read_text())
    del data['wind_speed']
    path.write_text(yaml.safe_dump(data))
    inputs[missing_from + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Unknown|missing'):
        analysis.analyze(**inputs)


def test_cli_does_not_replace_an_active_artifact_path(inputs, tmp_path):
    active = tmp_path / 'optimization_plan.json'
    active.write_text('operator-owned plan')
    with pytest.raises(SystemExit):
        analysis.main(cli_args(inputs, active))
    assert active.read_text() == 'operator-owned plan'
