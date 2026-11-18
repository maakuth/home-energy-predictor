import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.archive_pipeline import build_plan, execute_plan
from utils.sg_ready import derive_sg_ready_bits
from test_archive_pipeline import CUTOFF, request


@pytest.mark.parametrize('failure', ['predict', 'optimize'])
def test_failed_planning_installs_local_labelled_idle_plan(tmp_path, monkeypatch, failure):
    plan_file = tmp_path / 'optimization_plan.json'
    monkeypatch.setenv('TEST_PLAN_FILE', str(plan_file))
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if failure + ('.py' if failure == 'predict' else '_plan.py') in argv:
            raise subprocess.CalledProcessError(1, argv)
        if failure == 'predict' and 'predict_future.py' in argv:
            raise subprocess.CalledProcessError(1, argv)
    with patch('utils.archive_pipeline.subprocess.run', side_effect=run), \
         patch('utils.archive_pipeline._verify_output'):
        execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                     writer_lock_path=tmp_path / 'writer.lock')
    entry, = json.loads(plan_file.read_text())
    assert entry['plan_quality'] == 'degraded'
    assert entry['battery_action'] == 'idle' and entry['battery_power_kw'] == 0
    assert entry['effective_cost'] is None
    assert 'predicted_usage_kwh' not in entry and 'export_unit_price' not in entry
    assert derive_sg_ready_bits(entry, [entry]) == (0, 0)
    assert calls[-1][-2:] == ['push_to_ha.py', '--plan-only']
    metadata = json.loads((tmp_path / 'optimization_plan_metadata.json').read_text())
    assert metadata['history_cutoff'] == CUTOFF
    assert metadata['failed_step'] == failure


def test_fallback_does_not_mask_publish_failure(tmp_path):
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    def run(argv, **kwargs):
        if 'push_to_ha.py' in argv:
            raise subprocess.CalledProcessError(1, argv)
    with patch('utils.archive_pipeline.subprocess.run', side_effect=run), \
         patch('utils.archive_pipeline._verify_output'):
        with pytest.raises(subprocess.CalledProcessError):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=tmp_path / 'writer.lock')


def test_failed_atomic_replace_preserves_previous_plan(tmp_path, monkeypatch):
    from utils.degraded_plan import write_degraded_plan
    plan = tmp_path / 'optimization_plan.json'
    plan.write_text('[{"old": true}]')
    monkeypatch.setenv('TEST_PLAN_FILE', str(plan))
    import os
    original_replace = os.replace
    def replace(source, destination):
        if Path(destination) == plan:
            raise OSError('disk error')
        original_replace(source, destination)
    with patch('utils.degraded_plan.os.replace', side_effect=replace):
        with pytest.raises(OSError, match='disk error'):
            write_degraded_plan(request(), 'predict', 'failure')
    assert json.loads(plan.read_text()) == [{'old': True}]
    assert not list(tmp_path.glob('.optimization_plan.json.*'))


def test_fallback_cannot_bypass_configuration_change(tmp_path):
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    with patch('utils.archive_pipeline._configuration_hashes', side_effect=[{'x': 'a'}, {'x': 'b'}]), \
         patch('utils.archive_pipeline.write_degraded_plan') as fallback:
        with pytest.raises(ValueError, match='configuration changed'):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=tmp_path / 'writer.lock')
    fallback.assert_not_called()


def test_entry_is_current_and_preserves_manual_leaf_and_idle_guard(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from run_often import control_leaf_charger
    from utils.battery_utils import get_current_plan_entry, adjust_idle_for_cheap_export
    from utils.degraded_plan import write_degraded_plan
    plan_path = tmp_path / 'optimization_plan.json'
    monkeypatch.setenv('TEST_PLAN_FILE', str(plan_path))
    write_degraded_plan(request(), 'predict', 'unavailable')
    plan = json.loads(plan_path.read_text())
    now = datetime.fromtimestamp(CUTOFF, timezone.utc)
    current = get_current_plan_entry(plan, interval_minutes=15, now=now)
    assert current is not None
    with patch('run_often.call_ha_service') as service:
        control_leaf_charger(current, manual_charging=True, now=now)
    service.assert_not_called()
    power, action = adjust_idle_for_cheap_export(
        0.0, 'idle', current.get('export_unit_price'), -5000, 0, 50, 90)
    assert (power, action) == (0.0, 'idle')


def test_ha_invalidation_failure_is_reported_but_local_plan_remains(tmp_path, monkeypatch):
    plan_file = tmp_path / 'optimization_plan.json'
    monkeypatch.setenv('TEST_PLAN_FILE', str(plan_file))
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    def run(argv, **kwargs):
        if 'predict_future.py' in argv or 'push_to_ha.py' in argv:
            raise subprocess.CalledProcessError(1, argv)
    with patch('utils.archive_pipeline.subprocess.run', side_effect=run), \
         patch('utils.archive_pipeline._verify_output'):
        with pytest.raises(subprocess.CalledProcessError) as error:
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=tmp_path / 'writer.lock')
    assert 'push_to_ha.py' in error.value.cmd
    assert json.loads(plan_file.read_text())[0]['plan_quality'] == 'degraded'
    metadata = json.loads((tmp_path / 'optimization_plan_metadata.json').read_text())
    assert metadata['ha_sensor_status'] == 'stale_until_invalidation_confirmed'


@pytest.mark.parametrize('stage', ['predict', 'optimize'])
@pytest.mark.parametrize('bad_metadata', [
    [], 7, None, 'invalid', {'sources_path': []}, {'allow_legacy': 0},
    {'preferred_imports': 'invalid'}, {'policy_sha256': 123},
])
def test_malformed_metadata_types_trigger_degraded_fallback(tmp_path, monkeypatch, stage, bad_metadata):
    import os
    from utils.archive_pipeline import _verify_context
    monkeypatch.setenv('TEST_PLAN_FILE', str(tmp_path / 'optimization_plan.json'))
    monkeypatch.setenv('TEST_PREDICTIONS_FILE', str(tmp_path / 'future_predictions.json'))
    context = {'history_cutoff': CUTOFF, 'archive_path': str(Path(request().path).resolve()),
               'legacy_policy': request().legacy_policy}
    metadata = {**context, **bad_metadata} if isinstance(bad_metadata, dict) else bad_metadata
    with pytest.raises(ValueError):
        _verify_context(metadata, request())
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        for name, variable, current in [('predict_future.py', 'TEST_PREDICTIONS_FILE', 'predict'),
                                        ('optimize_plan.py', 'TEST_PLAN_FILE', 'optimize')]:
            if name in argv:
                path = Path(os.environ[variable])
                path.write_text('[{"timestamp":"2026-11-10T00:00:00+00:00"}]')
                path.with_name(path.stem + '_metadata.json').write_text(
                    json.dumps(metadata if stage == current else context))
    with patch('utils.archive_pipeline.subprocess.run', side_effect=run):
        execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                     writer_lock_path=tmp_path / 'writer.lock')
    assert json.loads(Path(os.environ['TEST_PLAN_FILE']).read_text())[0]['plan_quality'] == 'degraded'
    assert json.loads((tmp_path / 'optimization_plan_metadata.json').read_text())['failed_step'] == stage
    assert calls[-1][-2:] == ['push_to_ha.py', '--plan-only']


def test_programmer_error_is_not_converted_to_degraded_fallback(tmp_path):
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    with patch('utils.archive_pipeline.subprocess.run'), \
         patch('utils.archive_pipeline._verify_output', side_effect=[None, AttributeError('bug')]), \
         patch('utils.archive_pipeline.write_degraded_plan') as fallback:
        with pytest.raises(AttributeError, match='bug'):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=tmp_path / 'writer.lock')
    fallback.assert_not_called()
