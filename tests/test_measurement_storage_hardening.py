import sqlite3
import random
from unittest.mock import MagicMock, patch

import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_reader import MeasurementReader
from utils.measurement_ingestion import update_source
from utils.measurement_ingest_cli import main


def test_identity_namespace_and_unrelated_database(tmp_path):
    path = tmp_path / 'archive.sqlite'
    with MeasurementArchive(path, recorder_namespace='installation-a') as archive:
        identity = archive.db.execute('SELECT archive_uuid FROM archive_identity').fetchone()[0]
        assert archive.db.execute('PRAGMA user_version').fetchone()[0] == 2
    with MeasurementArchive(path, recorder_namespace='installation-a') as archive:
        assert archive.db.execute('SELECT archive_uuid FROM archive_identity').fetchone()[0] == identity
    with pytest.raises(ValueError, match='namespace'):
        MeasurementArchive(path, recorder_namespace='rebuilt')
    other = tmp_path / 'other.sqlite'
    with sqlite3.connect(other) as db:
        db.execute('CREATE TABLE unrelated (id INTEGER)')
    with pytest.raises(ValueError, match='archive'):
        MeasurementArchive(other)
    with pytest.raises(ValueError, match='hepo'):
        MeasurementArchive(tmp_path / 'hepo.db')


def test_predecessor_evidence_only_commits_with_successful_range(tmp_path):
    path = tmp_path / 'archive.sqlite'
    previous = (1, 5., '10', None)
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.x', None, 10, 20, [], predecessor=previous)
        def broken():
            yield (2, 21., '11', None)
            raise RuntimeError('interrupted')
        with pytest.raises(RuntimeError):
            archive.ingest('sensor.x', None, 20, 30, broken(), predecessor=previous)
        assert archive.coverage('sensor.x', None) == [(10, 20)]
    with MeasurementReader(path) as reader:
        assert reader.carry_supported(1, 5, 10)
        assert not reader.carry_supported(1, 5, 20)
        assert not reader.carry_supported(1, 4, 10)
        assert reader.carry_supported(1, 10, 10)


def test_v1_read_only_and_transactional_upgrade(tmp_path):
    path = tmp_path / 'old.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('x', None, 0, 10, [(1, 1., '2', None)])
        archive.db.execute('DROP TABLE predecessor_evidence')
        archive.db.execute('DROP TABLE ingestion_runs')
        archive.db.execute('DROP TABLE archive_identity')
        archive.db.execute('PRAGMA application_id=0')
        archive.db.execute('PRAGMA user_version=1')
        archive.db.commit()
    before = path.read_bytes()
    with MeasurementReader(path) as reader:
        assert not reader.carry_supported(1, -1, 0)
    assert path.read_bytes() == before
    with MeasurementArchive(path, recorder_namespace='old-recorder') as archive:
        assert archive.coverage('x', None) == [(0, 10)]
        assert len(archive.observations('x', None, 0, 10)) == 1


def test_chunk_resume_retry_late_records_and_diagnostics(tmp_path):
    calls = []
    def fetch(left, right):
        calls.append((left, right))
        if left == 10 and calls.count((10, 20)) <= 2:
            raise RuntimeError('temporary')
        return [(int(left + 1), left + 1, '3', None)]
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        with pytest.raises(RuntimeError):
            update_source(archive, 'x', None, 0, 30, 0, fetch, chunk_seconds=10, retries=1)
        assert archive.coverage('x', None) == [(0, 10)]
        update_source(archive, 'x', None, 0, 30, 0, fetch, chunk_seconds=10)
        assert calls == [(0, 10), (10, 20), (10, 20), (10, 20), (20, 30)]
        update_source(archive, 'x', None, 0, 30, 10,
                      lambda l, r: [(99, 25, 'late', None)], chunk_seconds=10)
        assert len(archive.observations('x', None, 20, 30)) == 2
        assert [r[0] for r in archive.db.execute('SELECT status FROM ingestion_runs')] == [
            'success', 'failed', 'failed', 'success', 'success', 'success']


def test_cli_dry_run_never_connects_creates_archive_or_report(tmp_path, capsys):
    path = tmp_path / 'new.sqlite'
    report = tmp_path / 'report.json'
    with patch('utils.measurement_ingest_cli.get_db_connection') as connect:
        main(['--archive', str(path), '--entity', 'x', '--from', '2026-01-01T00:00:00Z',
              '--to', '2026-01-02T00:00:00Z', '--dry-run', '--report', str(report)])
    connect.assert_not_called()
    assert not path.exists() and not report.exists()
    assert 'dry_run' in capsys.readouterr().out


def test_cli_requires_namespace_before_writes(tmp_path):
    with patch('utils.measurement_ingest_cli.get_db_connection') as connect:
        with pytest.raises(SystemExit):
            main(['--archive', str(tmp_path / 'x.sqlite'), '--entity', 'x',
                  '--from', '2026-01-01T00:00:00Z', '--to', '2026-01-02T00:00:00Z'])
    connect.assert_not_called()


def test_predecessor_adapter_is_bounded(tmp_path):
    from utils.measurement_recorder import recorder_predecessor
    cursor = MagicMock()
    cursor.fetchone.side_effect = [(12,), (8, 2., '5', None)]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    assert recorder_predecessor(connection, 'x', 10) == (8, 2., '5', None)
    assert 'LIMIT 1' in cursor.execute.call_args.args[0]
    assert cursor.execute.call_args.args[1] == (12, 10)
    cursor.close.assert_called_once()


def test_migration_failure_rolls_back_all_schema_changes(tmp_path):
    path = tmp_path / 'old.sqlite'
    with MeasurementArchive(path) as archive:
        for name in ('predecessor_evidence', 'ingestion_runs', 'archive_identity'):
            archive.db.execute(f'DROP TABLE {name}')
        archive.db.execute('PRAGMA application_id=0')
        archive.db.execute('PRAGMA user_version=1')
        archive.db.commit()
    with patch('utils.measurement_archive.uuid.uuid4', side_effect=RuntimeError('interrupted')):
        with pytest.raises(RuntimeError):
            MeasurementArchive(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert db.execute('PRAGMA application_id').fetchone()[0] == 0
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='archive_identity'").fetchall()
    with MeasurementArchive(path):
        pass


def test_random_gap_backfill_queries_exactly_uncovered_intervals(tmp_path):
    rng = random.Random(4981)
    for case in range(40):
        covered = set()
        with MeasurementArchive(tmp_path / f'{case}.sqlite') as archive:
            for _ in range(12):
                left = rng.randrange(0, 99)
                right = rng.randrange(left + 1, 101)
                archive.ingest('x', None, left, right, [])
                covered.update(range(left, right))
            fetched = []
            def fetch(left, right):
                fetched.extend(range(int(left), int(right)))
                return []
            update_source(archive, 'x', None, 0, 100, 0, fetch, chunk_seconds=7)
            assert len(fetched) == len(set(fetched))
            assert set(fetched) == set(range(100)) - covered
            assert archive.coverage('x', None) == [(0, 100)]
            update_source(archive, 'x', None, 0, 100, 0, lambda l, r: pytest.fail('already covered'))


def test_cli_fixed_now_bootstrap_and_namespace(tmp_path):
    path = tmp_path / 'archive.sqlite'
    base = ['--archive', str(path), '--entity', 'x', '--through-now', '--recorder-namespace', 'fixture']
    connection = MagicMock()
    with patch('utils.measurement_ingest_cli.time.time', return_value=3601) as clock, \
         patch('utils.measurement_ingest_cli.get_db_connection', return_value=connection) as connect, \
         patch('utils.measurement_ingest_cli.update_recorder_sources', return_value={}) as update:
        with pytest.raises(SystemExit):
            main(base)
        connect.assert_not_called()
        main([*base, '--from', '1970-01-01T00:00:00Z'])
        assert update.call_args.args[3:5] == (0, 3600)
        assert clock.call_count == 2  # once for each invocation, not each source
    with MeasurementArchive(path) as archive:
        archive.ingest('x', None, 0, 1800, [])
    with patch('utils.measurement_ingest_cli.time.time', return_value=3601), \
         patch('utils.measurement_ingest_cli.get_db_connection', return_value=connection), \
         patch('utils.measurement_ingest_cli.update_recorder_sources', return_value={}) as update:
        main(base)
        assert update.call_args.args[3:5] == (0, 3600)


def test_cli_include_defaults_merges_dated_overrides(tmp_path, capsys):
    config = tmp_path / 'sources.yaml'
    config.write_text('outside_temp:\n  - entity: sensor.audited\n')
    from utils.archive_features import default_sources
    name = next(key for key in default_sources() if key != 'outside_temp')
    main(['--archive', str(tmp_path / 'new.sqlite'), '--sources', str(config),
          '--include-default-sources', '--measurement', 'outside_temp', '--measurement', name,
          '--from', '2026-01-01T00:00:00Z', '--to', '2026-01-02T00:00:00Z', '--dry-run'])
    import json
    plan = json.loads(capsys.readouterr().out)
    entities = [request[0] for request in plan['requests']]
    assert 'sensor.audited' in entities
    assert len(entities) >= 2


def test_predecessor_failure_independent_retry_and_resume(tmp_path):
    from utils.measurement_update import update_recorder_sources
    connection = MagicMock()
    with MeasurementArchive(tmp_path / 'x.sqlite') as archive:
        def previous(conn, entity, start):
            if entity == 'bad':
                raise RuntimeError('query failed')
            return (1, -1., '3', None)
        with patch('utils.measurement_update.recorder_predecessor', side_effect=previous) as predecessors, \
             patch('utils.measurement_update.recorder_events', return_value=[]) as events:
            failures = update_recorder_sources(archive, connection, [('bad', None), ('good', None)],
                                               0, 10, 0, retries=1, fetch_predecessor=True)
        assert failures == {'bad': 'query failed'}
        assert predecessors.call_count == 3
        assert events.call_count == 1
        assert archive.coverage('bad', None) == []
        assert archive.coverage('good', None) == [(0, 10)]
        assert archive.db.execute('SELECT COUNT(*) FROM predecessor_evidence').fetchone()[0] == 1


def test_observation_lookup_uses_time_index(tmp_path):
    with MeasurementArchive(tmp_path / 'x.sqlite') as archive:
        archive.ingest('x', None, 0, 10000, ((i, float(i), '1', None) for i in range(10000)))
        details = ' '.join(row[3] for row in archive.db.execute('''EXPLAIN QUERY PLAN
            SELECT recorder_id, timestamp, state, attributes FROM observations
            WHERE source_id=1 AND timestamp>=100 AND timestamp<200
            ORDER BY timestamp, recorder_id'''))
        assert 'observations_time' in details
        assert 'SCAN observations' not in details


def test_report_json_contains_this_invocation_chunks_and_failures(tmp_path):
    import json
    report = tmp_path / 'run.json'
    connection = MagicMock()
    def update(archive, connection, sources, left, right, overlap, **kwargs):
        update_source(archive, 'x', None, left, right, overlap, lambda l, r: [], chunk_seconds=10)
        return {}
    with patch('utils.measurement_ingest_cli.get_db_connection', return_value=connection), \
         patch('utils.measurement_ingest_cli.update_recorder_sources', side_effect=update):
        main(['--archive', str(tmp_path / 'x.sqlite'), '--entity', 'x', '--recorder-namespace', 'fixture',
              '--from', '1970-01-01T00:00:00Z', '--to', '1970-01-01T00:00:30Z', '--report', str(report)])
    result = json.loads(report.read_text())
    assert len(result['chunks']) == 3
    assert result['failures'] == {}
    assert result['cutoff'] == 30


def test_report_refuses_archive_sidecar_before_connect(tmp_path):
    archive = tmp_path / 'x.sqlite'
    with patch('utils.measurement_ingest_cli.get_db_connection') as connect:
        with pytest.raises(SystemExit):
            main(['--archive', str(archive), '--entity', 'x', '--recorder-namespace', 'fixture',
                  '--from', '1970-01-01T00:00:00Z', '--to', '1970-01-01T00:00:30Z',
                  '--report', str(archive) + '-wal'])
    connect.assert_not_called()
    assert not archive.exists()
