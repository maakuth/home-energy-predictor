import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from push_to_ha import push_plan
from run_often import main


def expired_plan(tmp_path):
    state = tmp_path / 'state'
    state.mkdir(exist_ok=True)
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    entry = {'timestamp': old.isoformat(), 'plan_quality': 'degraded',
             'battery_action': 'idle', 'battery_power_kw': 0,
             'effective_cost': None, 'gshp_intent': 'STOP',
             'resistive_heater_intent': 'OFF', 'bulk_heater_intent': 'OFF', 'leaf_intent': 'OFF'}
    (state / 'optimization_plan.json').write_text(json.dumps([entry]))
    return entry


def test_independent_publisher_marks_stale_plan_entities_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entry = expired_plan(tmp_path)
    monkeypatch.setenv('TEST_PLAN_FILE', str(tmp_path / 'state/optimization_plan.json'))
    with patch('push_to_ha.push_ha_state', return_value=True) as push:
        push_plan()
    calls = {c.args[0]: c.args for c in push.call_args_list}
    for name in ('optimization_plan', 'predicted_24h_usage', 'effective_cost',
                 'low_cost_signal', 'gshp_intent', 'resistive_heater_intent',
                 'bulk_heater_intent', 'leaf_charging_intent'):
        assert calls['sensor.hepo_' + name][1] == 'unavailable'
        assert calls['sensor.hepo_' + name][2]['plan_quality'] == 'degraded'
        assert calls['sensor.hepo_' + name][2]['source_plan_timestamp'] == entry['timestamp']
    assert not any('sg_ready' in name for name in calls)


@pytest.mark.parametrize('action,soc,expected_sign', [('charge_grid', '50', -1),
                                                       ('discharge_export', '10', 0),
                                                       ('idle', '50', 0)])
def test_expired_degraded_plan_preserves_manual_override_and_soc_floor(
        tmp_path, monkeypatch, action, soc, expected_sign):
    monkeypatch.chdir(tmp_path)
    expired_plan(tmp_path)
    monkeypatch.setenv('BATTERY_NET_METERING', '0')
    monkeypatch.setenv('BATTERY_RAMP_RATE_KW_PER_MIN', '0')
    def state(entity):
        values = {'input_select.hepo_battery_action': action, 'sensor.be_soc': soc,
                  'sensor.mlp_varaajan_lampotila': '50'}
        return {'state': values.get(entity, '0')}
    with patch('run_often.get_ha_state', side_effect=state), \
         patch('run_often.push_battery_control') as battery, \
         patch('run_often.push_ha_state'), patch('run_often.call_ha_service'), \
         patch('run_often.emit_battery_diagnostic'):
        main()
    watts = battery.call_args.kwargs['battery_power_w']
    assert (watts > 0) - (watts < 0) == expected_sign


def test_expired_degraded_plan_retains_bulk_recovery_and_leaf_manual(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expired_plan(tmp_path)
    monkeypatch.setenv('BULK_HEATER_OPTIMIZE_ENABLED', '1')
    monkeypatch.setenv('RESISTIVE_HEATER_OPTIMIZE_ENABLED', '1')
    monkeypatch.setenv('GSHP_MIN_TEMP', '45')
    def state(entity):
        values = {'sensor.mlp_varaajan_lampotila': '40',
                  'input_boolean.leaf_manuaalinen_lataus': 'on',
                  'input_select.hepo_battery_action': 'idle'}
        return {'state': values.get(entity, '0')}
    with patch('run_often.get_ha_state', side_effect=state), \
         patch('run_often.push_battery_control'), patch('run_often.push_ha_state'), \
         patch('run_often.call_ha_service') as service, patch('run_often.emit_battery_diagnostic'):
        main()
    calls = [(c.args[1], c.args[2]['entity_id']) for c in service.call_args_list]
    assert ('turn_on', 'switch.mlp_vastus_output_1') in calls
    assert ('turn_off', 'switch.mlp_vastus_output_0') in calls
    assert not any(entity == 'switch.tasmota_3' for _, entity in calls)


def test_publisher_serializes_against_pipeline_lock(tmp_path):
    import fcntl
    from push_to_ha import publication_lock
    path = tmp_path / 'pipeline.lock'
    with publication_lock(str(path)):
        with path.open('a') as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_partial_ha_invalidation_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expired_plan(tmp_path)
    monkeypatch.setenv('TEST_PLAN_FILE', str(tmp_path / 'state/optimization_plan.json'))
    with patch('push_to_ha.push_ha_state', side_effect=[False] + [True] * 7) as push:
        assert push_plan() is False
        assert push.call_count == 8


def test_inherited_lock_reuses_pipeline_lock_in_child(tmp_path, monkeypatch):
    import fcntl
    import os
    import subprocess
    import sys
    path = tmp_path / 'pipeline.lock'
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        env = dict(os.environ, HEPO_PUBLISH_LOCK_FD=str(lock.fileno()))
        code = ('from push_to_ha import publication_lock\n'
                f'with publication_lock({str(path)!r}):\n print("ok")\n')
        result = subprocess.run([sys.executable, '-c', code], env=env, pass_fds=(lock.fileno(),),
                                capture_output=True, text=True, timeout=5, check=True)
        assert result.stdout.strip() == 'ok'


def test_expired_degraded_bulk_recovery_obeys_fuse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expired_plan(tmp_path)
    monkeypatch.setenv('BULK_HEATER_OPTIMIZE_ENABLED', '1')
    monkeypatch.setenv('GSHP_MIN_TEMP', '45')
    def state(entity):
        values = {'sensor.mlp_varaajan_lampotila': '40', 'sensor.current_phase_1': '30',
                  'sensor.be_soc': '10', 'input_select.hepo_battery_action': 'idle'}
        return {'state': values.get(entity, '0')}
    with patch('run_often.get_ha_state', side_effect=state), \
         patch('run_often.push_battery_control'), patch('run_often.push_ha_state'), \
         patch('run_often.call_ha_service') as service, patch('run_often.emit_battery_diagnostic'):
        main()
    bulk = [c.args[1] for c in service.call_args_list
            if c.args[2]['entity_id'] == 'switch.mlp_vastus_output_1']
    assert bulk and 'turn_on' not in bulk and bulk[-1] == 'turn_off'


@pytest.mark.parametrize('net_metering', ['0', '1'])
@pytest.mark.parametrize('action,battery_w,phase1,expected_w', [
    ('auto', '0', '0', 0),
    ('auto', '5000', '0', -4000),  # Existing charge ramps down by 1 kW per tick.
    ('auto', '0', '-30', -3450),  # Export-side phase fuse protection requires charge.
    ('charge_grid', '0', '0', -10000),  # Manual override remains authoritative, subject to guards.
])
def test_generated_degraded_idle_end_to_end_with_solar_export(
        tmp_path, monkeypatch, net_metering, action, battery_w, phase1, expected_w):
    from utils.archive_history import ArchiveHistory
    from utils.degraded_plan import write_degraded_plan
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TEST_PLAN_FILE', str(tmp_path / 'state/optimization_plan.json'))
    monkeypatch.setenv('PLAN_INTERVAL_MINUTES', '15')
    monkeypatch.setenv('BATTERY_NET_METERING', net_metering)
    monkeypatch.setenv('BATTERY_RAMP_RATE_KW_PER_MIN', '3')
    monkeypatch.setenv('BATTERY_MAX_CHARGE_KW', '10')
    monkeypatch.setenv('MAIN_FUSE_SIZE_A', '25')
    cutoff = int(datetime.now(timezone.utc).timestamp() // 60) * 60
    write_degraded_plan(ArchiveHistory('/offline/archive.sqlite', cutoff), 'predict', 'no prices')
    def state(entity):
        values = {'input_select.hepo_battery_action': action, 'sensor.be_soc': '50',
                  'sensor.be_stat_batt_power': battery_w, 'sensor.sahkokauppa_20s': '-5',
                  'sensor.solarh_63038_real_power_kw': '7', 'sensor.current_phase_1': phase1,
                  'sensor.mlp_varaajan_lampotila': '50', 'sensor.cumulative_active_import': '100',
                  'sensor.cumulative_active_export': '100'}
        return {'state': values.get(entity, '0')}
    # Ramp is disabled for the explicit manual test so it tests the selected action,
    # rather than its intentionally rate-limited first command.
    if action != 'auto':
        monkeypatch.setenv('BATTERY_RAMP_RATE_KW_PER_MIN', '0')
    with patch('run_often.get_ha_state', side_effect=state), \
         patch('run_often.push_battery_control') as battery, \
         patch('run_often.push_ha_state'), patch('run_often.call_ha_service'), \
         patch('run_often.emit_battery_diagnostic'):
        main()
    assert battery.call_args.kwargs['battery_power_w'] == expected_w
    assert battery.call_args.kwargs['native_load_following'] is False
