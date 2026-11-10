"""Execute a copied scheduler script with stub commands, never operational state."""
import os
from pathlib import Path
import subprocess


def run_script(tmp_path, args, python_exit=0):
    root = Path(__file__).resolve().parents[1]
    script = (root / 'run_slow.sh').read_text().replace('/tmp/hepo_slow.lock', str(tmp_path / 'slow.lock'))
    (tmp_path / 'run_slow.sh').write_text(script)
    # Synthetic activation only; no host virtualenv or application commands run.
    activate = tmp_path / '.venv/bin/activate'
    activate.parent.mkdir(parents=True)
    activate.write_text(':\n')
    commands = tmp_path / 'commands'
    commands.mkdir()
    log = tmp_path / 'commands.log'
    for name, body in {
        'flock': 'exit 0',
        'cp': 'echo "cp $*" >> "$COMMAND_LOG"; exit 1',
        'python3': f'echo "python3 $*" >> "$COMMAND_LOG"; exit {python_exit}',
    }.items():
        path = commands / name
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o755)
    env = {**os.environ, 'PATH': str(commands) + ':' + os.environ['PATH'], 'COMMAND_LOG': str(log)}
    result = subprocess.run(['bash', str(tmp_path / 'run_slow.sh'), *args], cwd=tmp_path,
                            env=env, text=True, capture_output=True, check=False)
    return result, log.read_text()


def test_archive_slow_pipeline_uses_explicit_window_with_csv_absent(tmp_path):
    result, log = run_script(tmp_path, ['--archive', '/offline/history.sqlite',
        '--from', '2026-10-21T00:00:00Z', '--to', '2026-10-24T00:00:00Z',
        '--no-archive-results'])
    assert result.returncode == 0
    assert 'cp state/processed_data.csv' not in log
    assert 'cp state/sarima_model_params.pkl' in log
    assert '--archive /offline/history.sqlite' in log
    assert '--from 2026-10-21T00:00:00Z --to 2026-10-24T00:00:00Z' in log
    assert '--no-archive-results' in log
    assert 'push_to_ha' not in log


def test_no_arguments_retains_explicit_csv_rollback_and_parameter_sync(tmp_path):
    result, log = run_script(tmp_path, [])
    assert result.returncode == 0
    assert 'cp state/processed_data.csv' in log
    assert 'cp state/sarima_model_params.pkl' in log
    assert 'python3 sarimax_predictor.py' in log
    assert '--archive' not in log


def test_archive_failure_stops_without_trying_csv(tmp_path):
    result, log = run_script(tmp_path, ['--archive', '/missing.sqlite'], python_exit=2)
    assert result.returncode == 2
    assert 'cp state/processed_data.csv' not in log
    assert log.count('python3 ') == 1
    assert 'Slow Task Complete' not in result.stdout
