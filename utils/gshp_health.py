from __future__ import annotations
import json
import os
from pathlib import Path


def update_gshp_health(plan, shared_power_kw, element_power_kw, path=None):
    path = Path(path or os.getenv('GSHP_HEALTH_STATE_FILE', 'state/gshp_health.json'))
    try:
        with path.open() as f:
            state = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        state = {'status': 'normal', 'bad_samples': 0, 'good_samples': 0}
    minimum_kw = float(os.getenv('GSHP_FAILURE_MIN_COMPRESSOR_KW', '1.0'))
    if not plan or plan.get('gshp_intent') != 'START' or float(plan.get('planned_gshp_kw', 0) or 0) < float(os.getenv('GSHP_FAILURE_MIN_PLANNED_KW', '1.0')):
        if element_power_kw <= 0.0 and shared_power_kw >= minimum_kw:
            state.update({'status': 'normal', 'bad_samples': 0, 'good_samples': 1, 'compressor_kw': shared_power_kw})
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('w') as f:
                    json.dump(state, f)
            except OSError:
                pass
        return state
    compressor_kw = max(0.0, shared_power_kw - max(0.0, element_power_kw))
    if compressor_kw < minimum_kw:
        state['bad_samples'] += 1
        state['good_samples'] = 0
        if state['bad_samples'] >= int(os.getenv('GSHP_FAILURE_REQUIRED_BAD_SAMPLES', '3')):
            state['status'] = 'failed'
    else:
        state['good_samples'] += 1
        state['bad_samples'] = 0
        if state['good_samples'] >= int(os.getenv('GSHP_RECOVERY_REQUIRED_GOOD_SAMPLES', '2')):
            state['status'] = 'normal'
    state['compressor_kw'] = compressor_kw
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(state, f)
    except OSError:
        pass
    return state


def gshp_is_failed() -> bool:
    return update_gshp_health(None, 0.0, 0.0).get('status') == 'failed'


def health_attributes(state):
    return {
        'friendly_name': 'HEPO GSHP Health',
        'compressor_kw': float(state.get('compressor_kw', 0.0)),
        'bad_samples': int(state.get('bad_samples', 0)),
        'good_samples': int(state.get('good_samples', 0)),
    }
