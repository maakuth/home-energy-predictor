import hashlib
import json

import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_legacy import import_legacy_csv
from utils.measurement_reader import MeasurementReader
from utils.measurement_sources import parse_sources
from utils.measurement_audit import (
    archive_inventory, audit_measurements, backfill_plan, inventory_csv,
    resolve_precedence,
)
from utils.measurement_audit_cli import main


def test_streaming_inventory_reports_flags_bad_rows_and_import_blockers(tmp_path):
    path = tmp_path / 'raw.csv'
    path.write_text('timestamp,power,power_source_valid\n'
                    '1970-01-01T00:00:00Z,12,False\n'
                    '1970-01-01T00:01:00Z,,True\n'
                    'not-a-date,3,True,extra\n')
    result = inventory_csv(path, interval_seconds=60)
    assert result['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['rows'] == 3
    assert result['malformed_rows'] == 1
    assert result['invalid_timestamps'] == 1
    assert result['columns']['power']['numeric_rows'] == 2
    assert result['columns']['power']['first_numeric_timestamp'] == 0
    assert result['columns']['power']['last_numeric_timestamp'] == 0
    assert result['columns']['power_source_valid']['false_rows'] == 1
    assert result['source_flags'] == ['power_source_valid']
    assert result['ready_to_import'] is False
    assert result['first_timestamp'] == '1970-01-01T00:00:00+00:00'


def test_inventory_rejects_duplicate_unsorted_unaligned_and_naive_times(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('timestamp,power\n1970-01-01T00:01:01Z,1\n'
                    '1970-01-01T00:01:01Z,2\n1970-01-01T00:00:00Z,3\n'
                    '1970-01-01T00:02:00,4\n')
    result = inventory_csv(path, interval_seconds=60)
    assert result['nonincreasing_timestamps'] == 2
    assert result['unaligned_timestamps'] == 2
    assert result['invalid_timestamps'] == 1


def test_backfill_uses_query_coverage_and_source_periods_not_measurement_presence(tmp_path):
    path = tmp_path / 'archive.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.old', None, 0, 60, [])
        archive.ingest('sensor.new', 'power', 180, 240, [])
    sources = parse_sources({'power': [
        {'entity': 'sensor.old', 'to': '1970-01-01T00:02:00Z'},
        {'entity': 'sensor.new', 'attribute': 'power', 'from': '1970-01-01T00:03:00Z'},
    ]})
    with MeasurementReader(path) as reader:
        plan = backfill_plan(reader, sources, 0, 300)
    assert [(r['entity'], r['start'], r['end']) for r in plan] == [
        ('sensor.old', 60, 120), ('sensor.new', 240, 300)]


def make_legacy(tmp_path):
    path = tmp_path / 'archive.sqlite'
    first = tmp_path / 'first.csv'
    second = tmp_path / 'second.csv'
    first.write_text('timestamp,power\n1970-01-01T00:00:00Z,1\n1970-01-01T00:01:00Z,2\n')
    second.write_text('timestamp,power\n1970-01-01T00:00:00Z,3\n1970-01-01T00:01:00Z,2\n')
    with MeasurementArchive(path) as archive:
        import_legacy_csv(archive, first, 60)
        import_legacy_csv(archive, second, 60)
    return path, hashlib.sha256(second.read_bytes()).hexdigest()


def test_audit_keeps_resolved_conflict_and_verification_distinct_from_eligibility(tmp_path):
    path, digest = make_legacy(tmp_path)
    sources = parse_sources({'power': [{'entity': 'sensor.power'}]})
    before = path.read_bytes()
    with MeasurementReader(path) as reader:
        preferred = resolve_precedence(reader, {'preferred_sha256': [digest]})
        assert preferred == [2]
        report = audit_measurements(reader, sources, 0, 120, ['power'],
            interval_seconds=60, preferred_imports=preferred, legacy_policy='complete-legacy')
        counts = report['measurements']['power']
        assert counts['available_bins'] == 2
        assert counts['full_coverage_bins'] == 2
        assert counts['source_verified_bins'] == 0
        assert counts['conflict_bins'] == 1
        assert counts['measurement_eligible_bins'] == 1
        assert counts['conflict_examples'][0]['import_sha256']
        default = audit_measurements(reader, sources, 0, 120, ['power'], interval_seconds=60)
        assert default['measurements']['power']['measurement_eligible_bins'] == 0
    assert path.read_bytes() == before
    assert report['migration_ready'] is False
    assert report['target_and_lag_eligibility'] == 'not_evaluated'


def test_manifest_fails_closed_and_inventory_does_not_scan_cells(tmp_path):
    path, _ = make_legacy(tmp_path)
    with MeasurementReader(path) as reader:
        with pytest.raises(ValueError, match='Unknown'):
            resolve_precedence(reader, {'preferred_sha256': ['a' * 64]})
        with pytest.raises(ValueError):
            resolve_precedence(reader, {'preferred_sha256': [] , 'unknown': 1})
        with pytest.raises(ValueError):
            resolve_precedence(reader, {'preferred_sha256': ['a' * 64, 'a' * 64]})
        report = archive_inventory(reader)
    assert report['imports'][0]['first_timestamp'] == 0
    assert report['imports'][0]['last_timestamp'] == 60
    assert report['schema_version'] == 2
    assert report['cell_counts'] == 'not_scanned'


def test_cli_metadata_and_dry_run_never_connect_to_recorder(tmp_path, capsys, monkeypatch):
    path, _ = make_legacy(tmp_path)
    sources = tmp_path / 'sources.yaml'
    sources.write_text('power:\n  - entity: sensor.power\n')
    def deny(*args, **kwargs):
        raise AssertionError('recorder connection attempted')
    monkeypatch.setattr('utils.db_utils.get_db_connection', deny)
    main(['archive', '--archive', str(path), '--sources', str(sources),
          '--measurement', 'power', '--from', '1970-01-01T00:00:00Z',
          '--to', '1970-01-01T00:02:00Z', '--dry-run-backfill'])
    report = json.loads(capsys.readouterr().out)
    assert report['dry_run'] is True
    assert report['backfill_plan'][0]['end'] == 120
    assert 'measurements' not in report


def test_missing_archive_not_created_and_scans_require_explicit_bounds(tmp_path):
    missing = tmp_path / 'missing.sqlite'
    with pytest.raises(SystemExit):
        main(['archive', '--archive', str(missing)])
    assert not missing.exists()
    path, _ = make_legacy(tmp_path)
    with pytest.raises(SystemExit):
        main(['archive', '--archive', str(path), '--measurement', 'power'])
    with MeasurementReader(path) as reader:
        with pytest.raises(ValueError, match='limit'):
            audit_measurements(reader, parse_sources({'power': [{'entity': 'sensor.power'}]}),
                               0, 600, ['power'], interval_seconds=60, max_bins=2)


def test_invalid_source_flag_and_partial_recorder_coverage_are_not_eligible(tmp_path):
    path = tmp_path / 'archive.sqlite'
    csv = tmp_path / 'flagged.csv'
    csv.write_text('timestamp,power,power_source_valid\n'
                   '1970-01-01T00:00:00Z,1,False\n'
                   '1970-01-01T00:01:00Z,2,True\n'
                   '1970-01-01T00:02:00Z,3,True\n')
    sources = parse_sources({'power': [{'entity': 'sensor.power'}]})
    with MeasurementArchive(path) as archive:
        import_legacy_csv(archive, csv, 60)
        archive.ingest('sensor.power', None, 120, 180,
                       [(1, 120, '10', None), (2, 150, 'unavailable', None)])
    with MeasurementReader(path) as reader:
        report = audit_measurements(reader, sources, 0, 180, ['power'],
                                    interval_seconds=60, legacy_policy='complete-legacy')
    counts = report['measurements']['power']
    assert counts['available_bins'] == 1
    assert counts['full_coverage_bins'] == 1
    assert counts['measurement_eligible_bins'] == 1
    assert counts['source_verified_bins'] == 0


def test_cli_inventory_import_plan_hash_match(tmp_path, capsys):
    path, _ = make_legacy(tmp_path)
    main(['inventory', '--csv', str(tmp_path / 'first.csv'), '--archive', str(path)])
    report = json.loads(capsys.readouterr().out)
    assert report['files'][0]['already_imported_content'] is True
    assert report['files'][0]['import_plan'].startswith('skip:')


def test_numeric_audit_rejects_counter_category_and_relay_inference(tmp_path):
    path, _ = make_legacy(tmp_path)
    with MeasurementReader(path) as reader:
        for name in ('ev_position', 'mummun_energy', 'resistive_heater_power'):
            with pytest.raises(ValueError, match='does not infer'):
                audit_measurements(reader, parse_sources({name: [{'entity': 'sensor.test'}]}),
                                   0, 60, [name], interval_seconds=60)
