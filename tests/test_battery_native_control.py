from unittest.mock import patch

import pytest

from utils.battery_utils import push_battery_control
from utils.type_defs import BatteryAction


@pytest.mark.parametrize('power', [0, 2500, -2500])
def test_follow_requests_native_mode_without_power_command(power):
    with patch('utils.battery_utils.is_battery_available', return_value=True), patch(
        'utils.battery_utils.call_ha_service', return_value={}
    ) as service:
        assert push_battery_control(power, 'follow')

    service.assert_called_once_with(
        domain='mqtt', service='publish',
        service_data={'topic': 'hoymiles/battery/set_mode', 'payload': 'self_consumption'},
        return_response=False,
    )


@pytest.mark.parametrize('action,power', [('idle', 0), ('charge_grid', -2000), ('discharge_load', 2000)])
def test_fixed_actions_keep_numeric_power_control(action: BatteryAction, power: int):
    with patch('utils.battery_utils.is_battery_available', return_value=True), patch(
        'utils.battery_utils.call_ha_service', return_value={}
    ) as service:
        assert push_battery_control(power, action)
    assert service.call_args.kwargs['domain'] == 'number'
    assert service.call_args.kwargs['service_data']['value'] == power


def test_follow_safety_override_uses_fixed_power():
    with patch('utils.battery_utils.is_battery_available', return_value=True), patch(
        'utils.battery_utils.call_ha_service', return_value={}
    ) as service:
        assert push_battery_control(0, 'follow', native_load_following=False)
    assert service.call_args.kwargs['domain'] == 'number'


@pytest.mark.parametrize('result', [None, {}])
def test_native_mode_reports_service_failure(result):
    with patch('utils.battery_utils.is_battery_available', return_value=True), patch(
        'utils.battery_utils.call_ha_service', return_value=result
    ):
        assert push_battery_control(0, 'follow') is (result is not None)


def test_native_mode_skips_unavailable_battery():
    with patch('utils.battery_utils.is_battery_available', return_value=False), patch(
        'utils.battery_utils.call_ha_service'
    ) as service:
        assert push_battery_control(0, 'follow')
    service.assert_not_called()


def test_follow_to_fixed_and_back_does_not_reset_requested_power():
    with patch('utils.battery_utils.is_battery_available', return_value=True), patch(
        'utils.battery_utils.call_ha_service', return_value={}
    ) as service:
        assert push_battery_control(0, 'follow')
        assert push_battery_control(-2000, 'charge_grid')
        assert push_battery_control(0, 'idle')
        assert push_battery_control(0, 'follow')
    assert [c.kwargs['domain'] for c in service.call_args_list] == [
        'mqtt', 'number', 'number', 'mqtt',
    ]
    assert service.call_args_list[1].kwargs['service_data']['value'] == -2000
    assert service.call_args_list[2].kwargs['service_data']['value'] == 0


@pytest.mark.parametrize('action', ['charge_solar', 'discharge_load'])
def test_restricted_actions_use_native_only_when_explicitly_selected(action):
    with patch('utils.battery_utils.is_battery_available', return_value=True), patch(
        'utils.battery_utils.call_ha_service', return_value={}
    ) as service:
        assert push_battery_control(1000, action, native_load_following=True)
    service.assert_called_once_with(
        domain='mqtt', service='publish',
        service_data={'topic': 'hoymiles/battery/set_mode', 'payload': 'self_consumption'},
        return_response=False,
    )
