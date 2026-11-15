from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from battery_planners.joint_linprog import JointLinprogPlanner
from optimize_plan import plan_gshp_dispatch


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
