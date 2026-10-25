import json
from unittest.mock import patch

import pytest

from utils.battery_diagnostics import emit_battery_diagnostic, parse_battery_diagnostic
from utils.battery_utils import push_battery_control


def test_prefixed_json_is_single_line_strict_and_parseable(capsys):
    emit_battery_diagnostic({'reason': 'a\nb', 'battery_power_w': -1200,
                             'grid_power_w': float('nan'), 'phase_currents_a': [1, float('inf'), None]})
    line = capsys.readouterr().out
    assert len(line.splitlines()) == 1
    assert line.startswith('HEPO_BATTERY_CONTROL {')
    assert '"battery_power_w":-1200' in line
    assert 'NaN' not in line and 'Infinity' not in line
    record = parse_battery_diagnostic('Oct 25 12:00:00 host python[42]: ' + line)
    assert record is not None
    assert record['schema_version'] == 1
    assert record['grid_power_w'] is None
    assert record['phase_currents_a'] == [1, None, None]
    assert record['reason'] == 'a\nb'
    assert json.loads(line.split(' ', 1)[1]) == record


def test_parser_ignores_other_and_malformed_lines():
    for line in ('Battery Power: 1000W', 'HEPO_BATTERY_CONTROL {broken',
                 'HEPO_BATTERY_CONTROL null', 'HEPO_BATTERY_CONTROL []'):
        assert parse_battery_diagnostic(line) is None


@pytest.mark.parametrize('action', ['follow', 'charge_grid'])
def test_command_diagnostics_distinguish_skipped_and_accepted_without_extra_calls(action):
    for available, result, expected in (
        (False, {}, 'skipped_unavailable'), (True, {}, 'accepted'), (True, None, 'failed'),
    ):
        diagnostic = {}
        with patch('utils.battery_utils.is_battery_available', return_value=available), patch(
            'utils.battery_utils.call_ha_service', return_value=result
        ) as service:
            returned = push_battery_control(0, action, command_result=diagnostic)
        assert returned is (not available or result is not None)
        assert diagnostic['command_status'] == expected
        assert service.call_count == int(available)


def test_command_exception_is_logged_without_changing_boolean_contract():
    diagnostic = {}
    with patch('utils.battery_utils.is_battery_available', return_value=True), patch(
        'utils.battery_utils.call_ha_service', side_effect=RuntimeError('offline')
    ):
        assert not push_battery_control(500, 'discharge_load', command_result=diagnostic)
    assert diagnostic == {'command_status': 'failed', 'command_error': 'exception'}
