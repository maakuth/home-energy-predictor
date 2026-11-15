from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from battery_planners.joint_linprog import JointLinprogPlanner
from optimize_plan import plan_gshp_dispatch


@pytest.mark.parametrize('planner', ['none', 'heuristic', 'nemotron-linprog', 'joint-linprog'])
def test_calculation_uses_frozen_runtime_without_live_reads(monkeypatch, planner):
    import optimize_plan
    import battery_planners.joint_linprog

    def forbidden(*args, **kwargs):
        raise AssertionError('live dependency called')

    for name in ('get_ha_state', 'get_db_connection', 'write_plan_atomically',
                 'load_uncontrolled_heating_forecast', 'get_model_version',
                 'is_battery_enabled', 'gshp_is_failed'):
        monkeypatch.setattr(optimize_plan, name, forbidden)
    monkeypatch.setattr(battery_planners.joint_linprog, 'gshp_is_failed', forbidden)
    monkeypatch.setenv('GSHP_OPTIMIZE_ENABLED', 'true')
    monkeypatch.setenv('RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'false')
    monkeypatch.setenv('GSHP_INITIAL_TEMP', '42')
    monkeypatch.setenv('BATTERY_PLANNER_TYPE', planner)
    monkeypatch.setenv('LEAF_MAX_POWER_KW', '3.4')
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runtime = optimize_plan.FrozenOptimizationRuntime(
        ev_soc_state={'state': '80'}, allow_export=True,
        battery_soc_pct=50.0, battery_enabled=planner != 'none',
        uncontrolled_heating_kw=np.array([1.0]), model_version='fixture',
        gshp_failed=True, output_timezone='Europe/Helsinki', resistive_running=False,
        config={'BATTERY_PLANNER_TYPE': planner, 'LEAF_MAX_POWER_KW': '0.6',
                'GSHP_OPTIMIZE_ENABLED': 'true'},
    )
    plan = optimize_plan.calculate_optimization(
        [{'outside_temp': 0.0, 'is_sauna_active': 0, 'ev_position': 1}],
        np.array([2.0]), [timestamp], pd.Series([np.nan]), pd.Series([np.nan]),
        [0.1], np.array([0.1]), np.array([0.0]), [0], False,
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
        42.0, False, False, history=None, frozen_runtime=runtime,
    )
    assert len(plan) == 1
    assert plan[0]['uncontrolled_heating_kw'] == 1.0
    assert plan[0]['planned_gshp_kw'] == 0.0
    assert plan[0]['timestamp'].endswith('+02:00')
    assert plan[0]['planned_leaf_kw'] == pytest.approx(0.6)


@pytest.mark.parametrize('heating', [np.array([]), np.array([np.nan]), np.array([1.0, 2.0])])
def test_frozen_uncontrolled_heating_requires_finite_aligned_values(monkeypatch, heating):
    import optimize_plan

    monkeypatch.setenv('GSHP_OPTIMIZE_ENABLED', 'false')
    runtime = optimize_plan.FrozenOptimizationRuntime(
        ev_soc_state=None, allow_export=True, battery_soc_pct=50.0,
        battery_enabled=False, uncontrolled_heating_kw=heating,
        model_version='fixture', gshp_failed=False, output_timezone='Europe/Helsinki', resistive_running=False,
        config={'GSHP_OPTIMIZE_ENABLED': 'false'},
    )
    with pytest.raises(ValueError, match='finite and aligned'):
        optimize_plan.calculate_optimization(
            [{'outside_temp': 0.0}], np.array([2.0]),
            [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            pd.Series([np.nan]), pd.Series([np.nan]), [0.1],
            np.array([0.1]), np.array([0.0]), [0], False,
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
            42.0, False, False, history=None, frozen_runtime=runtime,
        )


def test_publication_is_explicit_and_uses_selected_plan_path(monkeypatch, tmp_path):
    import json
    import optimize_plan

    output = tmp_path / 'isolated-plan.json'
    monkeypatch.setenv('TEST_PLAN_FILE', str(output))
    monkeypatch.setattr(optimize_plan, 'get_db_connection', lambda: (_ for _ in ()).throw(RuntimeError('offline')))
    plan = [{'timestamp': '2026-01-01T00:00:00+00:00'}]
    optimize_plan.publish_optimization(plan, history=None)
    assert json.loads(output.read_text()) == plan


@pytest.mark.parametrize('failed', [False, True])
def test_heating_plan_accepts_frozen_failure_latch(monkeypatch, failed):
    monkeypatch.setenv('GSHP_OPTIMIZE_ENABLED', 'true')
    monkeypatch.setenv('GSHP_INITIAL_TEMP', '42')
    with patch('optimize_plan.gshp_is_failed', side_effect=AssertionError('live latch read')):
        result = plan_gshp_dispatch(
            [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            [0], [0.0], np.array([0.1]), gshp_failed=failed,
        )
    assert (result[0]['gshp_electric_kw'] > 0) is (not failed)


def test_heating_plan_uses_explicit_initial_state(monkeypatch):
    monkeypatch.setenv('GSHP_OPTIMIZE_ENABLED', 'true')
    monkeypatch.setenv('GSHP_INITIAL_TEMP', '60')
    monkeypatch.setenv('GSHP_IS_RUNNING', 'false')
    result = plan_gshp_dispatch(
        [datetime(2026, 1, 1, tzinfo=timezone.utc)],
        [0], [0.0], np.array([0.1]), gshp_failed=False,
        initial_temp=42.0, compressor_running=False, resistive_running=False,
    )
    assert result[0]['gshp_electric_kw'] > 0


@pytest.mark.parametrize('failed', [False, True])
def test_joint_planner_accepts_frozen_failure_latch(monkeypatch, failed):
    monkeypatch.setenv('GSHP_OPTIMIZE_ENABLED', 'true')
    with patch('battery_planners.joint_linprog.gshp_is_failed',
               side_effect=AssertionError('live latch read')):
        result = JointLinprogPlanner().plan(
            np.array([0.5]), np.array([0.0]), np.array([0.1]), np.array([0.0]),
            [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            initial_soc_pct=50.0,
            context={'gshp_failed': failed, 'current_acc_temp': 42.0},
        )
    assert len(result) == 1
    if failed:
        assert result[0].planned_gshp_kw == 0
