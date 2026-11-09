from unittest.mock import MagicMock, patch

import pytest

from utils.measurement_ingest_cli import main


def test_explicit_ingestion_calls_update_and_closes_connection(tmp_path, capsys):
    archive = tmp_path / 'archive.sqlite'
    conn = MagicMock()
    args = ['--archive', str(archive), '--entity', 'sensor.example',
            '--from', '2026-10-01T00:00:00Z', '--to', '2026-10-02T00:00:00Z']
    with patch('utils.measurement_ingest_cli.get_db_connection', return_value=conn), \
         patch('utils.measurement_ingest_cli.update_recorder_sources', return_value={}) as update:
        main(args)
    assert update.call_args.args[2] == [('sensor.example', None)]
    conn.close.assert_called_once()
    assert 'completed' in capsys.readouterr().out


def test_partial_failure_exits_nonzero_and_closes_connection(tmp_path):
    conn = MagicMock()
    args = ['--archive', str(tmp_path / 'archive.sqlite'), '--entity', 'sensor.example',
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
