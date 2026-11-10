from unittest.mock import MagicMock, patch

import pytest

from utils.measurement_ingest_cli import main


def test_explicit_ingestion_calls_update_and_closes_connection(tmp_path, capsys):
    archive = tmp_path / 'archive.sqlite'
    conn = MagicMock()
    args = ['--recorder-namespace', 'fixture', '--archive', str(archive), '--entity', 'sensor.example',
            '--from', '2026-10-01T00:00:00Z', '--to', '2026-10-02T00:00:00Z']
    with patch('utils.measurement_ingest_cli.get_db_connection', return_value=conn), \
         patch('utils.measurement_ingest_cli.update_recorder_sources', return_value={}) as update:
        main(args)
    assert update.call_args.args[2] == [('sensor.example', None)]
    conn.close.assert_called_once()
    assert 'completed' in capsys.readouterr().out


def test_partial_failure_exits_nonzero_and_closes_connection(tmp_path):
    conn = MagicMock()
    args = ['--recorder-namespace', 'fixture', '--archive', str(tmp_path / 'archive.sqlite'), '--entity', 'sensor.example',
            '--from', '2026-10-01T00:00:00Z', '--to', '2026-10-02T00:00:00Z']
    with patch('utils.measurement_ingest_cli.get_db_connection', return_value=conn), \
         patch('utils.measurement_ingest_cli.update_recorder_sources', return_value={'sensor.example': 'failed'}):
        with pytest.raises(SystemExit) as error:
            main(args)
    assert error.value.code != 0
    conn.close.assert_called_once()


def test_naive_timestamp_rejected_before_connecting(tmp_path):
    args = ['--archive', str(tmp_path / 'archive.sqlite'), '--entity', 'sensor.example',
            '--from', '2026-10-01T00:00:00', '--to', '2026-10-02T00:00:00Z']
    with patch('utils.measurement_ingest_cli.get_db_connection') as connect:
        with pytest.raises(SystemExit):
            main(args)
    connect.assert_not_called()


def test_mapped_ingestion_clips_queries_at_switch_and_selects_measurement(tmp_path):
    config = tmp_path / 'sources.yaml'
    config.write_text('''gshp_power:
  - entity: sensor.old
    to: "2026-10-20T08:30:00+03:00"
  - entity: sensor.new
    from: "2026-10-20T08:30:00+03:00"
outside_temp:
  - entity: sensor.temperature
''')
    args = ['--recorder-namespace', 'fixture', '--archive', str(tmp_path / 'archive.sqlite'), '--sources', str(config),
            '--measurement', 'gshp_power', '--from', '2026-10-01T00:00:00Z', '--to', '2026-11-01T00:00:00Z']
    conn = MagicMock()
    with patch('utils.measurement_ingest_cli.get_db_connection', return_value=conn), \
         patch('utils.measurement_ingest_cli.update_recorder_sources', return_value={}) as update:
        main(args)
    assert len(update.call_args_list) == 2
    old, new = [call.args for call in update.call_args_list]
    assert old[2] == [('sensor.old', None)]
    assert new[2] == [('sensor.new', None)]
    from datetime import datetime
    boundary = datetime.fromisoformat('2026-10-20T08:30:00+03:00').timestamp()
    assert old[4] == new[3] == boundary
    conn.close.assert_called_once()


def test_invalid_mapping_is_rejected_before_connection_or_archive_creation(tmp_path):
    config = tmp_path / 'sources.yaml'
    config.write_text('x:\n  - entity: old\n  - entity: new')
    archive = tmp_path / 'archive.sqlite'
    with patch('utils.measurement_ingest_cli.get_db_connection') as connect:
        with pytest.raises(SystemExit):
            main(['--archive', str(archive), '--sources', str(config),
                  '--from', '2026-10-01T00:00:00Z', '--to', '2026-11-01T00:00:00Z'])
    connect.assert_not_called()
    assert not archive.exists()


def test_mapping_gap_does_not_open_databases(tmp_path, capsys):
    config = tmp_path / 'sources.yaml'
    config.write_text('x:\n  - entity: old\n    to: "2026-10-01T00:00:00Z"')
    archive = tmp_path / 'archive.sqlite'
    with patch('utils.measurement_ingest_cli.get_db_connection') as connect:
        main(['--archive', str(archive), '--sources', str(config),
              '--from', '2026-10-02T00:00:00Z', '--to', '2026-10-03T00:00:00Z'])
    connect.assert_not_called()
    assert not archive.exists()
    assert 'No configured sources cover' in capsys.readouterr().out


def test_unknown_measurement_rejected_before_connecting(tmp_path):
    config = tmp_path / 'sources.yaml'
    config.write_text('x:\n  - entity: old')
    with patch('utils.measurement_ingest_cli.get_db_connection') as connect:
        with pytest.raises(SystemExit):
            main(['--archive', str(tmp_path / 'archive.sqlite'), '--sources', str(config),
                  '--measurement', 'typo', '--from', '2026-10-01T00:00:00Z', '--to', '2026-11-01T00:00:00Z'])
    connect.assert_not_called()
