from pathlib import Path
import os
import subprocess


def test_weekly_pipeline_uses_dedicated_training_data_files():
    script = Path('run_weekly.sh').read_text()

    assert 'state/raw_data_weekly.csv' in script
    assert 'state/processed_data_weekly.csv' in script
    assert 'extract_data.py --days 730 --output state/raw_data_weekly.csv' in script
    assert 'process_data.py --input state/raw_data_weekly.csv --output state/processed_data_weekly.csv' in script
    assert 'train_model.py --data-path state/processed_data_weekly.csv' in script
    assert 'train_sarima.py' not in script


def test_weekly_csv_executes_training_and_analysis_without_planner(tmp_path):
    script = Path('run_weekly.sh').read_text().replace('/tmp/hepo.lock', str(tmp_path / 'lock'))
    (tmp_path / 'run_weekly.sh').write_text(script)
    activation = tmp_path / '.venv/bin/activate'
    activation.parent.mkdir(parents=True)
    activation.write_text(':\n')
    commands = tmp_path / 'bin'
    commands.mkdir()
    python = commands / 'python'
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
    python.chmod(0o755)
    log = tmp_path / 'commands.log'
    result = subprocess.run(['bash', str(tmp_path / 'run_weekly.sh')], cwd=tmp_path,
        env={**os.environ, 'PATH': str(commands) + ':' + os.environ['PATH'], 'COMMAND_LOG': str(log)},
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert any('train_model.py' in call for call in calls)
    assert any('analyze_performance.py' in call for call in calls)
    assert any('analyze_evolution.py' in call for call in calls)
    assert not any(name in call for call in calls
                   for name in ('predict_future.py', 'optimize_plan.py', 'push_to_ha.py'))
