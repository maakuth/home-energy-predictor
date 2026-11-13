import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import fcntl
import pytest
import yaml

from utils import measurement_collector as collector
from utils.measurement_archive import MeasurementArchive
from utils.measurement_reader import MeasurementReader


@pytest.fixture
def settings(tmp_path, monkeypatch):
    archive = tmp_path/'measurements.sqlite'
    with MeasurementArchive(archive, recorder_namespace='test-recorder') as db:
        archive_uuid = db.db.execute('SELECT archive_uuid FROM archive_identity').fetchone()[0]
    sources = tmp_path/'sources.yaml'
    sources.write_text(yaml.safe_dump({'total_power': [{'entity': 'sensor.grid'}]}))
    env = tmp_path/'recorder.env'
    env.write_text('DB_HOST=recorder.invalid\nDB_NAME=ha\nDB_USER=reader\nDB_PASSWORD=private-secret\n')
    data = {'format_version': 1, 'archive': str(archive), 'archive_uuid': archive_uuid,
        'recorder_namespace': 'test-recorder', 'sources': str(sources),
        'sources_sha256': hashlib.sha256(sources.read_bytes()).hexdigest(),
        'measurements': ['total_power'], 'bootstrap_from': '1970-01-01T00:00:00Z',
        'env_file': str(env), 'lock_file': str(tmp_path/'historical-backfill.lock'),
        'reports_dir': str(tmp_path/'collector-reports'), 'overlap_seconds': 60,
        'chunk_seconds': 86400, 'retries': 1}
    config = tmp_path/'collector.json'
    config.write_text(json.dumps(data))
    connection = MagicMock()
    connection.cursor.return_value.fetchone.return_value = ('on',)
    connect = Mock(return_value=connection)
    monkeypatch.setattr(collector.psycopg2, 'connect', connect)
    events = [(1, 0., '1', '{"unit_of_measurement":"kW"}'),
              (2, 60., '2', '{"unit_of_measurement":"kW"}'),
              (3, 120., '3', '{"unit_of_measurement":"kW"}')]
    monkeypatch.setattr('utils.measurement_update.recorder_events',
        lambda connection, entity, left, right: (row for row in events if left <= row[1] < right))
    monkeypatch.setattr('utils.measurement_update.recorder_predecessor',
        lambda connection, entity, left: next((row for row in reversed(events) if row[1] < left), None))
    return config, data, archive, connection, connect


def test_dry_run_has_no_connections_locks_reports_or_archive_writes(settings):
    config, data, archive, _, connect = settings
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = collector.collect(config, now=120.9)
    assert result['status'] == 'dry_run'
    assert result['cutoff'] == 120
    connect.assert_not_called()
    assert not Path(data['lock_file']).exists()
    assert not Path(data['reports_dir']).exists()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before


def test_data_only_collection_is_readonly_upstream_and_resumes_idempotently(settings):
    config, data, archive, connection, connect = settings
    first = collector.collect(config, execute=True, now=120.9)
    assert first['status'] == 'success'
    assert first['exit_code'] == 0
    assert len(first['chunks']) == 1
    assert 'private-secret' not in json.dumps(first)
    assert 'default_transaction_read_only=on' in connect.call_args.kwargs['options']
    connection.set_session.assert_called_with(readonly=True, autocommit=False)
    assert connection.cursor.return_value.execute.call_args_list[0].args[0] == 'SHOW transaction_read_only'
    connection.commit.assert_not_called()
    second = collector.collect(config, execute=True, now=180.9)
    assert second['status'] == 'success'
    collector.collect(config, execute=True, now=180.9)
    with MeasurementReader(archive) as reader:
        assert reader.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 3
    reports = list(Path(data['reports_dir']).glob('collector-*.json'))
    assert len(reports) == 3
    assert all(json.loads(p.read_text())['status'] == 'success' for p in reports)
    assert all(p.parent == Path(data['reports_dir']) for p in reports)


def test_shared_shadow_lock_skips_without_connecting(settings):
    config, data, _, _, connect = settings
    with Path(data['lock_file']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = collector.collect(config, execute=True, now=120)
    assert result['status'] == 'skipped_locked'
    assert result['exit_code'] == 75
    connect.assert_not_called()
    assert len(list(Path(data['reports_dir']).glob('collector-*.json'))) == 1


@pytest.mark.parametrize('problem', ['uuid', 'namespace', 'source_hash', 'missing_archive'])
def test_bad_identity_or_mapping_fails_before_upstream_access(settings, problem):
    config, data, archive, _, connect = settings
    if problem == 'uuid':
        data['archive_uuid'] = '11111111-1111-1111-1111-111111111111'
    elif problem == 'namespace':
        data['recorder_namespace'] = 'other-recorder'
    elif problem == 'source_hash':
        data['sources_sha256'] = '0'*64
    else:
        data['archive'] = str(archive.parent/'missing.sqlite')
    config.write_text(json.dumps(data))
    result = collector.collect(config, execute=True, now=120)
    assert result['status'] == 'failed'
    assert result['exit_code'] == 1
    connect.assert_not_called()
    assert not (archive.parent/'missing.sqlite').exists()


def test_unconfirmed_readonly_session_cannot_write_archive(settings):
    config, _, archive, connection, _ = settings
    connection.cursor.return_value.fetchone.return_value = ('off',)
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = collector.collect(config, execute=True, now=120)
    assert result['status'] == 'failed'
    assert result['chunks'] == []
    connection.close.assert_called()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before


def test_connection_failure_is_reported_without_leaking_secret(settings):
    config, _, _, _, connect = settings
    connect.side_effect = RuntimeError('private-secret should never be reported')
    result = collector.collect(config, execute=True, now=120)
    assert result['status'] == 'failed'
    assert result['error_type'] == 'RuntimeError'
    assert 'private-secret' not in json.dumps(result)
    assert result['phase'] == 'connect_readonly'


def test_source_failure_is_nonzero_and_recorded(settings, monkeypatch):
    config, _, _, connection, _ = settings
    monkeypatch.setattr('utils.measurement_update.recorder_events', Mock(side_effect=RuntimeError('source failed')))
    result = collector.collect(config, execute=True, now=120)
    assert result['status'] == 'failed'
    assert result['exit_code'] == 1
    assert result['failures']
    assert len(result['chunks']) == 2  # bounded retry diagnostic attempts
    connection.close.assert_called()


def test_interruption_leaves_a_report_and_releases_lock(settings, monkeypatch):
    config, data, _, connection, _ = settings
    monkeypatch.setattr(collector, 'update_recorder_sources', Mock(side_effect=KeyboardInterrupt))
    result = collector.collect(config, execute=True, now=120)
    assert result['status'] == 'interrupted'
    assert result['exit_code'] == 130
    connection.close.assert_called()
    with Path(data['lock_file']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize('change', [{'extra': True}, {'archive': 'state/hepo.db'},
    {'retries': True}, {'chunk_seconds': 0}, {'measurements': []}])
def test_strict_config_does_not_guess_unsafe_defaults(settings, change):
    config, data, _, _, connect = settings
    data.update(change)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        collector.collect(config, now=120)
    connect.assert_not_called()


def test_atomic_report_update_does_not_remove_an_unowned_temporary_file(tmp_path):
    report = tmp_path/'report.json'
    report.write_text('{"status":"running"}')
    temporary = tmp_path/'report.json.tmp'
    temporary.write_text('operator data')
    with pytest.raises(FileExistsError):
        collector._replace_owned_report(report, {'status': 'success'})
    assert temporary.read_text() == 'operator data'
    assert json.loads(report.read_text())['status'] == 'running'


def test_collector_preserves_raw_invalid_states_for_later_interpretation(settings, monkeypatch):
    config, _, archive, _, _ = settings
    events = [(1, 0., '-1180257', '{"unit_of_measurement":"W"}'),
              (2, 60., 'unavailable', None)]
    monkeypatch.setattr('utils.measurement_update.recorder_events', lambda *args: iter(events))
    result = collector.collect(config, execute=True, now=120)
    assert result['status'] == 'success'
    with MeasurementReader(archive) as reader:
        assert [row[0] for row in reader.db.execute('SELECT state FROM observations ORDER BY recorder_id')] == [
            '-1180257', 'unavailable']


def test_systemd_templates_are_data_only_and_leave_existing_jobs_alone():
    root = Path(__file__).resolve().parents[1]
    service = (root/'deploy/systemd/hepo-measurement-collector.service.in').read_text()
    timer = (root/'deploy/systemd/hepo-measurement-collector.timer').read_text()
    assert '-m utils.measurement_collector' in service
    assert '--execute' in service
    assert 'SuccessExitStatus=75' in service
    assert 'ProtectSystem=strict' in service
    assert 'ReadWritePaths="@STAGE_DIR@"' in service
    assert all(word not in service for word in ['predict_future', 'optimize_plan', 'push_to_ha', 'train_model'])
    assert 'OnCalendar=*-*-* *:02/5:00' in timer
    assert 'Persistent=true' in timer
