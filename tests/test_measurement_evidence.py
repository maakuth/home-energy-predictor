import json
from unittest.mock import MagicMock, patch

import pytest

from utils.measurement_evidence import collect_home_assistant, collect_postgres, main, summarize_observations


def test_summary_distinguishes_energy_forecast_and_attribute_units_without_relabeling():
    rows = [(10., '20.5', '{"unit_of_measurement":"kWh","state_class":"measurement"}'),
            (20., 'unavailable', '{"unit_of_measurement":"kWh"}'),
            (30., '21', 'invalid-json')]
    report = summarize_observations(rows, attribute=None)
    assert report['sampled_events'] == 3
    assert report['metadata_variants'][0]['metadata']['unit_of_measurement'] == 'kWh'
    assert report['malformed_attributes'] == 1
    assert report['numeric_values']['minimum'] == 20.5
    assert report['source_verification'] == 'not_established'
    wind = summarize_observations([(10., 'sunny', '{"wind_speed":7,"wind_speed_unit":"km/h"}')],
                                   attribute='wind_speed')
    assert wind['numeric_values']['maximum'] == 7
    assert wind['metadata_variants'][0]['metadata']['wind_speed_unit'] == 'km/h'


def test_pg_collection_enforces_readonly_and_bounded_queries_without_writes():
    connection = MagicMock()
    cursor = connection.cursor.return_value
    current = ['']

    def execute(query, params=None):
        current[0] = query

    def one():
        if 'transaction_read_only' in current[0]:
            return ('on',)
        if 'current_database' in current[0]:
            return ('homeassistant', '16', 100, 200)
        if 'pg_control_system' in current[0]:
            return ('cluster-123',)
        if 'metadata_id FROM states_meta' in current[0]:
            return (12,)
        return (1000.,)

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = one
    cursor.fetchall.return_value = [(1000., '3', '{"unit_of_measurement":"kW"}')]
    result = collect_postgres(connection, [('total_power', 'sensor.grid', None)], 0., 2000., sample_limit=5)
    connection.set_session.assert_called_once_with(readonly=True, autocommit=False)
    connection.commit.assert_not_called()
    assert result['read_only'] is True
    queries = [call.args[0] for call in cursor.execute.call_args_list]
    assert all(not any(word in query.upper() for word in ('INSERT ', 'UPDATE ', 'DELETE ', 'ALTER ', 'CREATE '))
               for query in queries)
    sample = next(call for call in cursor.execute.call_args_list if 'shared_attrs' in call.args[0])
    assert 'LIMIT %s' in sample.args[0]
    assert sample.args[1][-1] == 5


def test_cli_reports_to_new_file_and_never_opens_an_archive(tmp_path):
    sources = tmp_path / 'sources.yaml'
    sources.write_text('total_power:\n  - entity: sensor.grid\n')
    output = tmp_path / 'evidence.json'
    connection = MagicMock()
    with patch('utils.measurement_evidence.get_db_connection', return_value=connection), \
         patch('utils.measurement_evidence.collect_postgres', return_value={'read_only': True}) as collect:
        main(['--sources', str(sources), '--from', '2026-11-01T00:00Z', '--to', '2026-11-02T00:00Z',
              '--output', str(output)])
    assert collect.call_args.args[1] == [('total_power', 'sensor.grid', None)]
    assert json.loads(output.read_text())['postgres']['read_only'] is True
    connection.close.assert_called_once()
    original = output.read_bytes()
    with patch('utils.measurement_evidence.get_db_connection') as connect:
        with pytest.raises(SystemExit):
            main(['--sources', str(sources), '--output', str(output)])
    connect.assert_not_called()
    assert output.read_bytes() == original


def test_ha_is_optional_and_never_calls_service_endpoints(tmp_path):
    sources = tmp_path / 'sources.yaml'
    sources.write_text('total_power:\n  - entity: sensor.grid\n')
    with patch('utils.measurement_evidence.get_db_connection', return_value=MagicMock()), \
         patch('utils.measurement_evidence.collect_postgres', return_value={'read_only': True}), \
         patch('utils.measurement_evidence.collect_home_assistant') as ha:
        main(['--sources', str(sources)])
    ha.assert_not_called()


def test_collector_can_run_as_a_standalone_file_without_repository_imports(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'measurement_evidence.py'
    script.write_text((root / 'utils/measurement_evidence.py').read_text())
    result = subprocess.run([str(root / 'venv/bin/python3'), str(script), '--help'],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': ''}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--sample-limit' in result.stdout


def test_server_must_confirm_readonly_before_any_history_query():
    connection = MagicMock()
    connection.cursor.return_value.fetchone.return_value = ('off',)
    with pytest.raises(ValueError, match='read-only'):
        collect_postgres(connection, [('grid', 'sensor.grid', None)], 0., 60.)
    connection.commit.assert_not_called()
    queries = [call.args[0] for call in connection.cursor.return_value.execute.call_args_list]
    assert queries == ['SHOW transaction_read_only']


def test_ha_gets_are_scoped_no_redirects_and_report_no_credentials_or_location(monkeypatch):
    monkeypatch.setenv('HA_HOST', 'https://ha.example')
    monkeypatch.setenv('HA_TOKEN', 'NEVER-PRINT-THIS')
    opener = MagicMock()
    response = opener.open.return_value.__enter__.return_value
    response.read.return_value = json.dumps({'state': '3', 'attributes': {
        'unit_of_measurement': 'kW', 'latitude': 61.99, 'longitude': 22.99}}).encode()
    with patch('utils.measurement_evidence.build_opener', return_value=opener) as build:
        report = collect_home_assistant(['sensor.grid'])
    request = opener.open.call_args.args[0]
    assert request.get_method() == 'GET'
    assert request.full_url == 'https://ha.example/api/states/sensor.grid'
    assert build.call_args.args[0].redirect_request(None, None, 302, None, None, 'https://other.example') is None
    text = json.dumps(report)
    assert 'NEVER-PRINT-THIS' not in text
    assert 'latitude' not in text and 'longitude' not in text


def test_custom_dated_sources_clip_each_sample_window(tmp_path):
    from datetime import datetime
    sources = tmp_path / 'sources.yaml'
    sources.write_text('''power:
  - entity: sensor.old
    to: "2026-11-01T00:00:00Z"
  - entity: sensor.new
    from: "2026-11-01T00:00:00Z"
''')
    with patch('utils.measurement_evidence.get_db_connection', return_value=MagicMock()), \
         patch('utils.measurement_evidence.collect_postgres', return_value={'read_only': True}) as collect:
        main(['--sources', str(sources), '--from', '2026-10-31T00:00Z', '--to', '2026-11-02T00:00Z'])
    windows = collect.call_args.kwargs['source_ranges']
    boundary = datetime.fromisoformat('2026-11-01T00:00:00+00:00').timestamp()
    assert windows[('power', 'sensor.old', None)][0][1] == boundary
    assert windows[('power', 'sensor.new', None)][0][0] == boundary
