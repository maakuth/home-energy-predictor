from __future__ import annotations

import os
from unittest.mock import patch

from utils.gshp_health import health_attributes, update_gshp_health


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


def test_health_attributes_are_safe_to_publish():
    assert health_attributes({'status': 'failed', 'compressor_kw': 0.2, 'bad_samples': 3}) == {
        'friendly_name': 'HEPO GSHP Health', 'compressor_kw': 0.2,
        'bad_samples': 3, 'good_samples': 0,
    }


def test_recovers_when_shared_power_is_present_without_elements(tmp_path):
    path = tmp_path / 'health.json'
    path.write_text('{"status": "failed", "bad_samples": 3, "good_samples": 0}')

    state = update_gshp_health(None, shared_power_kw=3.5, element_power_kw=0.0, path=path)

    assert state['status'] == 'normal'
    assert state['bad_samples'] == 0


def test_target_temperature_stop_does_not_latch_failure(tmp_path):
    path = tmp_path / 'health.json'
    path.write_text('{"status": "normal", "bad_samples": 2, "good_samples": 0}')
    plan = {'gshp_intent': 'START', 'planned_gshp_kw': 3.0}

    with patch.dict(os.environ, {'GSHP_MAX_TEMP': '55.0'}):
        for _ in range(3):
            state = update_gshp_health(
                plan,
                shared_power_kw=0.0,
                element_power_kw=0.0,
                accumulator_temp=55.0,
                path=path,
            )

    assert state['status'] == 'normal'
    assert state['bad_samples'] == 0
