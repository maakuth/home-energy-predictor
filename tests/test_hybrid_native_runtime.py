import json
from datetime import datetime
from unittest.mock import patch

import pytest


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'state').mkdir()
    monkeypatch.setenv('BATTERY_NET_METERING', 'true')
    monkeypatch.setenv('BATTERY_MIN_SOC_PCT', '10')
    monkeypatch.setenv('BATTERY_RESERVE_SOC_PCT', '10')
    # A deliberately slow ramp must not delay a direction stop or discharge cap.
    monkeypatch.setenv('BATTERY_RAMP_RATE_KW_PER_MIN', '.1')
    monkeypatch.setenv('MAIN_FUSE_SIZE_A', '25')
    monkeypatch.setenv('BATTERY_MAX_CHARGE_KW', '10')
    with patch('run_often.get_ha_state') as state, patch(
        'run_often.push_battery_control', return_value=True
    ) as push, patch('run_often.push_ha_state'), patch('run_often.call_ha_service'):
        def run(action, battery=0, grid=0, planned=-3, soc='50', budget=None, phase=0, manual='auto'):
            now = datetime.now().astimezone()
            slot = now.replace(minute=now.minute // 15 * 15, second=0, microsecond=0)
            entry = {'timestamp': slot.isoformat(), 'battery_action': action,
                     'battery_power_kw': planned, 'soc_pct': 50}
            if budget is not None:
                entry['discharge_budget_kwh'] = budget
            (tmp_path / 'state' / 'optimization_plan.json').write_text(json.dumps([entry]))
            values = {'sensor.be_soc': soc, 'sensor.be_stat_batt_power': battery,
                      'sensor.sahkokauppa_20s': grid / 1000,
                      'sensor.current_phase_1': phase,
                      'input_select.hepo_battery_action': manual}
            state.side_effect = lambda entity: {'state': str(values.get(entity, 0))}
            from run_often import main
            main()
            return push.call_args.kwargs
        yield run, push


@pytest.mark.parametrize('action,battery,grid,planned,native,power', [
    ('charge_solar', 0, -2000, 3, True, -2000),
    ('charge_solar', -2000, 3000, 3, False, 0),
    ('charge_solar', -2000, -4000, 3, False, -2000),
    ('discharge_load', 0, 2000, -3, True, 2000),
    ('discharge_load', 2000, -1000, -3, False, 0),
    ('discharge_load', 2000, 4000, -3, False, 2000),
    ('discharge_load', -4000, 1000, -3, False, 3000),
])
def test_hybrid_direction_and_cap(controller, action, battery, grid, planned, native, power):
    run, _ = controller
    result = run(action, battery, grid, planned)
    assert result['battery_action'] == action
    assert result['native_load_following'] is native
    assert result['battery_power_w'] == power


@pytest.mark.parametrize('action,planned,grid', [('charge_solar', 3, -2000), ('discharge_load', -3, 2000)])
@pytest.mark.parametrize('soc,budget,phase', [('10', None, 0), ('unavailable', None, 0), ('50', None, 25)])
def test_hybrid_keeps_safety_guards(controller, action, planned, grid, soc, budget, phase):
    run, _ = controller
    # Charging at the import fuse and discharging at the export fuse require fixed control.
    if action == 'discharge_load':
        phase = -phase
    result = run(action, grid=grid, planned=planned, soc=soc, budget=budget, phase=phase)
    assert not result['native_load_following']
    if action == 'discharge_load':
        assert result['battery_power_w'] == 0


def test_discharge_budget_and_missing_grid_disable_native(controller):
    run, _ = controller
    result = run('discharge_load', grid=2000, budget=0)
    assert not result['native_load_following']
    assert result['battery_power_w'] == 0
    result = run('charge_solar', grid=float('nan'), planned=3)
    assert not result['native_load_following']
    assert result['battery_power_w'] == 0


def test_hysteresis_survives_runtime_invocations(controller):
    run, _ = controller
    results = [run('charge_solar', grid=grid, planned=3) for grid in (-1000, 100, 300, -100, -1000)]
    assert [r['native_load_following'] for r in results] == [True, True, False, False, True]


def test_failed_push_does_not_change_hysteresis_state(controller):
    run, push = controller
    push.return_value = False
    assert run('charge_solar', grid=-1000, planned=3)['native_load_following']
    push.return_value = True
    assert not run('charge_solar', grid=100, planned=3)['native_load_following']


def test_manual_charge_solar_uses_same_direction_guard(controller):
    run, _ = controller
    result = run('idle', grid=2000, planned=0, manual='charge_solar')
    assert result['battery_action'] == 'charge_solar'
    assert not result['native_load_following']
    assert result['battery_power_w'] == 0


def test_discharge_cap_uses_plan_not_smoothed_carry_over(controller):
    run, _ = controller
    with patch('run_often.smooth_planned_setpoint', return_value=-8):
        result = run('discharge_load', grid=3000, planned=-2)
    assert not result['native_load_following']
    assert result['battery_power_w'] == 2000


def test_partial_discharge_budget_forces_fixed_power(controller):
    run, _ = controller
    with patch('run_often.apply_discharge_budget', return_value=(-.5, 'budget cap')):
        result = run('discharge_load', grid=2000, budget=.5)
    assert not result['native_load_following']
    assert result['battery_power_w'] == 500


def test_return_to_native_is_checked_against_new_direction(controller):
    run, _ = controller
    assert run('charge_solar', grid=-2000, planned=3)['native_load_following']
    result = run('discharge_load', grid=-100, planned=-3)
    assert not result['native_load_following']
    assert result['battery_power_w'] == 0


def _diagnostic_lines(capsys):
    from utils.battery_diagnostics import parse_battery_diagnostic
    return [record for line in capsys.readouterr().out.splitlines()
            if (record := parse_battery_diagnostic(line)) is not None]


def test_cycle_diagnostic_has_signed_snapshot_plan_limits_and_no_fake_native_setpoint(controller, capsys):
    run, _ = controller
    run('discharge_load', battery=-1000, grid=1000, planned=-3)
    records = _diagnostic_lines(capsys)
    assert len(records) == 1
    record = records[0]
    assert record['battery_power_w'] == -1000
    assert record['grid_power_w'] == 1000
    assert record['native_estimate_w'] == -2000
    assert record['planned_power_w'] == -3000
    assert record['discharge_cap_w'] == 3000
    assert record['mode_requested'] == 'self_consumption'
    assert record['fixed_setpoint_w'] is None
    assert record['bridge_set_power_w'] is None
    assert record['inverter_mode_verified'] is False
    assert record['model_version']
    assert datetime.fromisoformat(record['timestamp']).tzinfo is not None


def test_final_diagnostic_does_not_claim_native_after_budget_override(controller, capsys):
    run, _ = controller
    run('discharge_load', grid=2000, budget=0)
    record, = _diagnostic_lines(capsys)
    assert record['native_selected_before_guards'] is True
    assert record['mode_requested'] == 'fixed'
    assert record['fixed_setpoint_w'] == 0
    assert 'discharge_budget_exhausted' in record['decision_reasons']
    assert record['discharge_used_kwh'] == 0


def test_diagnostics_keep_unavailable_telemetry_null(controller, capsys):
    run, _ = controller
    run('charge_solar', grid=float('nan'), planned=3)
    record, = _diagnostic_lines(capsys)
    assert record['grid_power_w'] is None
    assert record['native_estimate_w'] is None
    assert 'grid_telemetry_missing' in record['decision_reasons']


def test_exact_fuse_and_soc_reason_codes(controller, capsys):
    run, _ = controller
    run('charge_solar', grid=-2000, planned=3, phase=25)
    record, = _diagnostic_lines(capsys)
    assert 'phase_current_cap' in record['decision_reasons']
    assert record['mode_requested'] == 'fixed'
    run('discharge_load', grid=2000, soc='10')
    record, = _diagnostic_lines(capsys)
    assert 'soc_floor' in record['decision_reasons']


def test_missing_plan_cycle_is_logged_without_adding_a_command(controller, tmp_path, capsys):
    run, push = controller
    run('charge_solar', grid=-2000, planned=3)
    _diagnostic_lines(capsys)
    (tmp_path / 'state' / 'optimization_plan.json').unlink()
    push.reset_mock()
    from run_often import main
    main()
    record, = _diagnostic_lines(capsys)
    assert record['command_status'] == 'not_attempted'
    assert record['mode_requested'] is None
    assert record['decision_reasons'] == ['no_plan']
    push.assert_not_called()


def test_previous_request_and_failed_command_are_not_reported_as_observed_modes(controller, capsys):
    run, push = controller

    def command(**kwargs):
        kwargs['command_result']['command_status'] = 'accepted'
        return True

    push.side_effect = command
    run('charge_solar', grid=-2000, planned=3)
    _diagnostic_lines(capsys)

    def failed_command(**kwargs):
        kwargs['command_result'].update(command_status='failed', command_error='service_failed')
        return False

    push.side_effect = failed_command
    run('charge_solar', grid=1000, planned=3)
    record, = _diagnostic_lines(capsys)
    assert record['previous_requested_mode'] == 'self_consumption'
    assert record['previous_command_status'] == 'accepted'
    assert record['mode_requested'] == 'fixed'
    assert record['command_status'] == 'failed'
    assert record['command_error'] == 'service_failed'
    assert record['inverter_mode_verified'] is False


def test_no_current_plan_cycle_logs_safe_fixed_command(controller, tmp_path, capsys):
    run, push = controller
    run('charge_solar', grid=-2000, planned=3)
    _diagnostic_lines(capsys)
    path = tmp_path / 'state' / 'optimization_plan.json'
    plan = json.loads(path.read_text())
    plan[0]['timestamp'] = '2000-01-01T00:00:00+00:00'
    path.write_text(json.dumps(plan))
    push.reset_mock()
    from run_often import main
    main()
    record, = _diagnostic_lines(capsys)
    assert record['decision_reasons'] == ['no_current_plan_entry']
    assert record['mode_requested'] == 'fixed'
    assert record['fixed_setpoint_w'] == 0
    push.assert_called_once()


def test_manual_override_logging_keeps_original_plan_action(controller, capsys):
    run, _ = controller
    run('idle', grid=-2000, planned=0, manual='charge_solar')
    record, = _diagnostic_lines(capsys)
    assert record['planned_action'] == 'idle'
    assert record['effective_action'] == 'charge_solar'
    assert record['override_mode'] == 'charge_solar'
