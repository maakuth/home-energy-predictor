from unittest.mock import patch

import pytest

from utils.optimizer_shadow import compare_frozen_plans


@pytest.fixture(autouse=True)
def disable_dotenv(monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')


def snapshot():
    return {
        'format_version': 1,
        'origin': '2026-01-01T00:00:00+00:00',
        'interval_minutes': 15,
        'model_sha256': 'a' * 64,
        'output_timezone': 'Europe/Helsinki',
        'config': {'BATTERY_PLANNER_TYPE': 'heuristic', 'GSHP_OPTIMIZE_ENABLED': 'false'},
        'runtime': {
            'ev_soc_state': {'state': '80'}, 'allow_export': True,
            'battery_soc_pct': 50.0, 'battery_enabled': False,
            'model_version': 'fixture', 'gshp_failed': False,
            'current_acc_temp': 50.0, 'compressor_running': False,
            'resistive_running': False, 'fireplace_on': False,
        },
        'horizon': [
            {'timestamp': '2026-01-01T00:15:00+00:00', 'outside_temp': 0.0,
             'is_sauna_active': 0, 'ev_position': 1, 'solar_kw': 0.0,
             'solar_p10_kw': 0.0, 'solar_p90_kw': 0.0, 'market_price': 0.1,
             'import_price': 0.1, 'export_price': 0.0, 'fallback_price': 0,
             'sarima_lower': None, 'sarima_upper': None},
        ],
        'tomorrow_valid': False,
        'routes': {
            'compatibility': {'baseload_kw': [2.0], 'uncontrolled_heating_kw': [1.0]},
            'archive': {'baseload_kw': [3.0], 'uncontrolled_heating_kw': [1.0]},
        },
    }


def test_compare_frozen_plans_does_not_use_live_services(monkeypatch):
    import optimize_plan
    import battery_planners.joint_linprog

    def forbidden(*args, **kwargs):
        raise AssertionError('live dependency called')

    for name in ('get_ha_state', 'get_db_connection', 'write_plan_atomically',
                 'load_uncontrolled_heating_forecast', 'get_model_version',
                 'is_battery_enabled', 'gshp_is_failed', 'publish_optimization'):
        monkeypatch.setattr(optimize_plan, name, forbidden)
    monkeypatch.setattr(battery_planners.joint_linprog, 'gshp_is_failed', forbidden)
    with patch('socket.create_connection', side_effect=forbidden):
        result = compare_frozen_plans(snapshot())
    assert result['shadow_only'] is True
    assert result['plans']['archive'][0]['predicted_baseload_kw'] == 3.0
    assert result['plans']['compatibility'][0]['predicted_baseload_kw'] == 2.0
    assert result['comparison']['matched_intervals'] == 1
    assert result['comparison']['numeric_deltas']['predicted_baseload_kw']['max_abs'] == 1.0


def test_compare_rejects_missing_misaligned_or_nonfinite_inputs():
    data = snapshot()
    data['routes']['archive']['baseload_kw'] = []
    with pytest.raises(ValueError):
        compare_frozen_plans(data)
    data = snapshot()
    data['routes']['archive']['baseload_kw'] = [float('nan')]
    with pytest.raises(ValueError):
        compare_frozen_plans(data)


def test_cli_writes_only_to_new_output_directory(tmp_path):
    import json
    from utils.optimizer_shadow import main

    source = tmp_path / 'frozen-input.json'
    source.write_text(json.dumps(snapshot()))
    destination = tmp_path / 'new-shadow'
    main(['--input', str(source), '--output-dir', str(destination)])
    report = json.loads((destination / 'optimizer-shadow-comparison.json').read_text())
    assert report['shadow_only'] is True
    with pytest.raises(FileExistsError):
        main(['--input', str(source), '--output-dir', str(destination)])


@pytest.mark.parametrize('planner', ['heuristic', 'nemotron-linprog', 'joint-linprog'])
def test_compare_exercises_builtin_planners_without_live_latch(monkeypatch, planner):
    import battery_planners.joint_linprog
    data = snapshot()
    data['runtime']['battery_enabled'] = True
    data['config']['BATTERY_PLANNER_TYPE'] = planner
    monkeypatch.setattr(battery_planners.joint_linprog, 'gshp_is_failed',
                        lambda: pytest.fail('unexpected runtime latch read'))
    result = compare_frozen_plans(data)
    assert result['comparison']['matched_intervals'] == 1


def test_compare_rejects_custom_planner_even_if_registered():
    from battery_planners import BatteryPlannerFactory
    data = snapshot()
    data['runtime']['battery_enabled'] = True
    data['config']['BATTERY_PLANNER_TYPE'] = 'unknown-plugin'
    # Rejection must not depend on the process-global plugin registry.
    with patch.dict(BatteryPlannerFactory._planners, {'unknown-plugin': object}):
        with pytest.raises(ValueError, match='Unsupported frozen planner'):
            compare_frozen_plans(data)


def test_frozen_comparison_ignores_subsequent_environment_changes(monkeypatch):
    data = snapshot()
    first = compare_frozen_plans(data)
    monkeypatch.setenv('LEAF_MAX_POWER_KW', '9.0')
    monkeypatch.setenv('EV_TARGET_SOC_PCT', '10')
    monkeypatch.setenv('PLAN_INTERVAL_MINUTES', '60')
    assert compare_frozen_plans(data) == first


@pytest.mark.parametrize('field', ['baseload_kw', 'uncontrolled_heating_kw'])
def test_compare_rejects_negative_loads(field):
    data = snapshot()
    data['routes']['archive'][field] = [-1.0]
    with pytest.raises(ValueError, match='negative'):
        compare_frozen_plans(data)


@pytest.mark.parametrize('state', [{}, {'state': None}, {'state': 'bad'},
                                   {'state': 'NaN'}, {'state': '101'}, {'state': '-1'}])
def test_compare_rejects_invalid_ev_state(state):
    data = snapshot()
    data['runtime']['ev_soc_state'] = state
    with pytest.raises(ValueError, match='EV'):
        compare_frozen_plans(data)
