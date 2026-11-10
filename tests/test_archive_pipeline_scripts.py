import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('script,mode', [('run_frequent.sh', 'frequent'), ('run_weekly.sh', 'weekly')])
def test_archive_shell_route_precedes_csv_extraction_and_state_changes(tmp_path, script, mode):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / script).write_text((root / script).read_text().replace('/tmp/hepo.lock', str(tmp_path / 'hepo.lock')))
    activation = tmp_path / '.venv/bin/activate'
    activation.parent.mkdir(parents=True)
    activation.write_text(':\n')
    commands = tmp_path / 'commands'
    commands.mkdir()
    executable = commands / 'python3'
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARG_LOG"\n')
    executable.chmod(0o755)
    (commands / 'python').symlink_to(executable)
    log = tmp_path / 'args'
    result = subprocess.run(['bash', str(tmp_path / script), '--archive', '/offline/archive.sqlite',
                             '--bootstrap-from', '2026-01-01T00:00Z', '--recorder-namespace', 'ha'],
                             cwd=tmp_path, env={**os.environ, 'PATH': str(commands) + ':' + os.environ['PATH'],
                                               'ARG_LOG': str(log)}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = log.read_text().splitlines()
    assert args[:4] == ['-m', 'utils.archive_pipeline', '--mode', mode]
    assert '--archive' in args
    assert not (tmp_path / 'state').exists()
    assert 'extract_data.py' not in args
    assert 'push_to_ha.py' not in args
