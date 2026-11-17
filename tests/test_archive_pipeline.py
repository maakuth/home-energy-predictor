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
    assert 'train-sarima' not in by_name
    assert all('train_sarima.py' not in command for command in by_name.values())
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


def test_dry_run_reports_missing_policy_entries_without_executing(tmp_path, capsys):
    policy = tmp_path / 'policy.yaml'
    policy.write_text('outside_temp:\n  kind: numeric\n  input_unit: C\n  canonical_unit: C\n  sign: 1\n')
    with patch('utils.archive_pipeline.execute_plan') as execute:
        main(['--mode', 'frequent', '--archive', '/not-created.sqlite',
              '--bootstrap-from', '2026-01-01T00:00:00Z',
              '--cutoff', '2026-11-10T08:00:00Z', '--recorder-namespace', 'ha',
              '--measurement-policy', str(policy)])
    execute.assert_not_called()
    readiness = json.loads(capsys.readouterr().out)['policy_preflight']
    assert 'outside_temp' not in readiness['missing_measurements']
    assert 'sauna_temp' in readiness['missing_measurements']
    assert readiness['complete'] is False


def test_ingest_uses_shared_writer_lock_only_for_ingestion(tmp_path):
    import fcntl
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    writer = tmp_path / 'writer.lock'
    held = []

    def run(argv, **kwargs):
        with writer.open('a') as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(False)
            except BlockingIOError:
                held.append(True)

    with patch('utils.archive_pipeline.subprocess.run', side_effect=run), \
         patch('utils.archive_pipeline._verify_output'):
        execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                     writer_lock_path=writer)
    assert held == [True, False, False, False]


def test_busy_writer_lock_stops_before_reset_or_ingestion(tmp_path):
    import fcntl
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    writer = tmp_path / 'writer.lock'
    with writer.open('a') as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        with patch('utils.archive_pipeline.subprocess.run') as run, \
             patch('utils.archive_pipeline._reset_net_metering') as reset:
            with pytest.raises(BlockingIOError):
                execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                             writer_lock_path=writer)
        run.assert_not_called()
        reset.assert_not_called()


@pytest.mark.parametrize('writer', [None, 'relative.lock'])
def test_execution_requires_explicit_absolute_writer_lock(tmp_path, writer):
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    with patch('utils.archive_pipeline.subprocess.run') as run:
        with pytest.raises(ValueError, match='writer lock'):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=writer)
    run.assert_not_called()


def test_writer_lock_cannot_alias_archive(tmp_path):
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    with pytest.raises(ValueError, match='Writer lock'):
        execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                     writer_lock_path='/offline/history.sqlite')


def test_failed_ingest_stops_before_predict_optimize_or_publish(tmp_path):
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    import subprocess
    with patch('utils.archive_pipeline.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ingest')) as run:
        with pytest.raises(subprocess.CalledProcessError):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=tmp_path / 'writer.lock')
    assert run.call_count == 1


def test_missing_prediction_metadata_prevents_optimizer_and_publishing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha', approve_publish=True)
    with patch('utils.archive_pipeline.subprocess.run') as run:
        with pytest.raises((OSError, ValueError), match='metadata|No such file'):
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=tmp_path / 'writer.lock')
    assert run.call_count == 2


def test_publish_requires_separate_explicit_approval():
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400,
                      recorder_namespace='ha')
    assert not any(step['name'] == 'publish' for step in plan)
    with patch('utils.archive_pipeline.subprocess.run') as run:
        with pytest.raises(ValueError, match='rollout approval'):
            execute_plan(plan, request(), mode='frequent')
    run.assert_not_called()


def test_pipeline_no_longer_ingests_or_requires_the_daily_energy_forecast():
    plan = build_plan('frequent', request(), bootstrap_start=CUTOFF - 86400, recorder_namespace='ha')
    command = plan[0]['argv']
    selected = [command[index + 1] for index, value in enumerate(command) if value == '--measurement']
    assert 'solar_forecast' not in selected
    assert 'solar_actual' in selected


def test_training_artifacts_need_the_new_feature_contract_before_rollout(tmp_path, monkeypatch):
    from pathlib import Path
    from utils.archive_pipeline import _verify_output
    from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract
    monkeypatch.chdir(tmp_path)
    state = Path('state')
    state.mkdir()
    metadata = {'archive_training': {'end': CUTOFF, 'archive_path': str(Path(request().path).resolve()),
                                      'legacy_policy': 'verified-only'}}
    (state / 'model_training_metadata.json').write_text(json.dumps(metadata))
    (state / 'model_features.json').write_text(json.dumps(BASELOAD_FEATURES))
    (state / 'energy_model.json').write_text('unchanged model')
    with pytest.raises(ValueError, match='retrain'):
        _verify_output('train-xgboost', request())
    metadata['feature_contract'] = baseload_feature_contract()
    (state / 'model_training_metadata.json').write_text(json.dumps(metadata))
    _verify_output('train-xgboost', request())


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
        execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                     writer_lock_path=tmp_path / 'writer.lock')
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
            execute_plan(plan, request(), mode='frequent', lock_path=tmp_path / 'pipeline.lock',
                         writer_lock_path=tmp_path / 'writer.lock')
    assert run.call_count == 2
