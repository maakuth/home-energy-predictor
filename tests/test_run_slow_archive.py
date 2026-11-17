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
    return result, log.read_text() if log.exists() else ''


def test_archive_slow_pipeline_is_disabled(tmp_path):
    result, log = run_script(tmp_path, ['--archive', '/offline/history.sqlite',
        '--from', '2026-10-21T00:00:00Z', '--to', '2026-10-24T00:00:00Z',
        '--no-archive-results'])
    assert result.returncode != 0
    assert 'disabled' in (result.stdout + result.stderr).lower()
    assert log == ''
def test_scheduled_slow_pipeline_does_not_run_sarima(tmp_path):
    result, log = run_script(tmp_path, [])
    assert result.returncode == 0
    assert 'disabled' in result.stdout.lower()
    assert log == ''
