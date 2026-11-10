import json
from unittest.mock import patch

import pytest

from utils.archive_history import ArchiveHistory
from utils.archive_pipeline import build_plan, execute_plan, main


CUTOFF = 1794268800.0


def request():
    from pathlib import Path
    return ArchiveHistory('/offline/history.sqlite', CUTOFF,
                           sources_path=Path(__file__).resolve().parents[1] / 'config/measurement_sources.yaml')


def test_frequent_plan_is_incremental_fixed_cutoff_without_csv_or_implicit_publish():
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 730 * 86400,
                      recorder_namespace='ha-recorder-primary')
    commands = [step['argv'] for step in plan]
    ingest = commands[0]
    assert 'utils.measurement_ingest_cli' in ingest
    assert '--include-default-sources' in ingest
    assert '--recorder-namespace' in ingest
    for command in commands[1:]:
        assert '--cutoff' in command
        assert command[command.index('--cutoff') + 1] == '2026-11-10T00:00:00+00:00'
    assert not any('push_to_ha.py' in command for command in commands)
    assert not any('extract_data.py' in command or 'process_data.py' in command for command in commands)


def test_weekly_training_uses_local_archive_and_evaluation_uses_same_cutoff():
    plan = build_plan('weekly', request(), bootstrap_start=CUTOFF - 730 * 86400,
                      recorder_namespace='ha', approve_publish=True)
    by_name = {step['name']: step['argv'] for step in plan}
    assert 'train_model.py' in by_name['train-xgboost']
    assert '--archive' in by_name['train-xgboost']
    assert 'train_sarima.py' in by_name['train-sarima']
    assert '--to' in by_name['train-sarima']
    assert 'push_to_ha.py' in by_name['publish']
    assert '--cutoff' in by_name['evaluate-performance']


def test_dry_run_does_not_connect_lock_or_write(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.iterdir())
    with patch('utils.archive_pipeline.execute_plan') as execute:
        main(['--mode', 'frequent', '--archive', '/not-created.sqlite',
              '--bootstrap-from', '2026-01-01T00:00:00Z',
              '--cutoff', '2026-11-10T08:00:00Z', '--recorder-namespace', 'ha'])
    execute.assert_not_called()
    report = json.loads(capsys.readouterr().out)
    assert report['dry_run'] is True
    assert list(tmp_path.iterdir()) == before


def test_failed_ingest_stops_before_predict_optimize_or_publish(tmp_path):
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    import subprocess
    with patch('utils.archive_pipeline.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ingest')) as run:
        with pytest.raises(subprocess.CalledProcessError):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock')
    assert run.call_count == 1


def test_missing_prediction_metadata_prevents_optimizer_and_publishing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    with patch('utils.archive_pipeline.subprocess.run') as run:
        with pytest.raises((OSError, ValueError), match='metadata|No such file'):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock')
    assert run.call_count == 2


def test_publish_requires_separate_explicit_approval():
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha')
    assert not any(step['name'] == 'publish' for step in plan)
    with patch('utils.archive_pipeline.subprocess.run') as run:
        with pytest.raises(ValueError, match='rollout approval'):
            execute_plan(plan, request(), mode='frequent')
    run.assert_not_called()


def test_isolated_pipeline_without_any_csv_checks_artifacts_before_explicit_publish(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    executed = []

    def run(argv, **kwargs):
        executed.append(argv)
        for command, variable in [('predict_future.py', 'TEST_PREDICTIONS_FILE'),
                                  ('optimize_plan.py', 'TEST_PLAN_FILE')]:
            if command in argv:
                artifact = Path(os.environ[variable])
                artifact.write_text('[{"timestamp":"2026-11-10T00:00:00Z"}]')
                artifact.with_name(artifact.stem + '_metadata.json').write_text(json.dumps({
                    'history_cutoff': CUTOFF, 'archive_path': str(Path(request().path).resolve()),
                    'legacy_policy': 'verified-only'}))

    with patch('utils.archive_pipeline.subprocess.run', side_effect=run):
        execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock')
    assert len(executed) == 4
    assert 'push_to_ha.py' in executed[-1]
    assert not list(tmp_path.rglob('*.csv'))
    state = Path(os.environ['TEST_TMP_DIR'])
    assert json.loads((state / 'net_metering_state.json').read_text()) == {}


def test_wrong_cutoff_sidecar_is_not_publishable(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    artifact = Path(os.environ['TEST_PREDICTIONS_FILE'])
    artifact.write_text('[{}]')
    artifact.with_name(artifact.stem + '_metadata.json').write_text(json.dumps({
        'history_cutoff': CUTOFF - 60, 'archive_path': str(Path(request().path).resolve()),
        'legacy_policy': 'verified-only'}))
    with patch('utils.archive_pipeline.subprocess.run') as run:
        with pytest.raises(ValueError, match='stale'):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock')
    assert run.call_count == 2
