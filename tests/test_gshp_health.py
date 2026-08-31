from __future__ import annotations

from utils.gshp_health import update_gshp_health


def test_latches_failure_and_recovers_from_electrical_response(tmp_path):
    path = tmp_path / 'health.json'
    plan = {'gshp_intent': 'START', 'planned_gshp_kw': 3.0}

    for _ in range(3):
        state = update_gshp_health(plan, shared_power_kw=0.2, element_power_kw=0.0, path=path)
    assert state['status'] == 'failed'

    for _ in range(2):
        state = update_gshp_health(plan, shared_power_kw=3.2, element_power_kw=0.0, path=path)
    assert state['status'] == 'normal'


def test_subtracts_active_element_power_before_judging_response(tmp_path):
    state = update_gshp_health(
        {'gshp_intent': 'START', 'planned_gshp_kw': 3.0},
        shared_power_kw=6.0, element_power_kw=6.0, path=tmp_path / 'health.json',
    )
    assert state['bad_samples'] == 1
