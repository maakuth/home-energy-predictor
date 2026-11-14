"""Check operator instructions without installing dependencies or contacting services."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_collector_uses_existing_checkout_and_shared_environment():
    doc = (ROOT / 'docs/MEASUREMENT_COLLECTOR.md').read_text()
    assert 'CODE="$REPO"' in doc
    assert 'PYTHON="$REPO/.venv/bin/python3"' in doc
    assert 'git worktree add' not in doc
    assert '/.local/share/hepo/' not in doc
    assert '.venv-collector' not in doc
    assert 'requirements-collector.txt' not in doc
    assert 'python3 -m venv' not in doc
    assert '"$PYTHON" -m pip install -r "$CODE/requirements.txt"' in doc
    assert 'updating the main checkout also updates collector code' in doc


def test_live_dependency_recovery_targets_the_frequent_jobs_environment():
    doc = (ROOT / 'docs/MEASUREMENT_COLLECTOR.md').read_text()
    assert '.venv/bin/python3 -m pip install -r requirements.txt' in doc
    assert '.venv/bin/python3 -c "import yaml; import predict_future"' in doc
    assert 'PyYAML>=6.0' in (ROOT / 'requirements.txt').read_text().splitlines()


def test_documented_renderer_preserves_audited_sources_and_local_paths(tmp_path):
    doc = (ROOT / 'docs/MEASUREMENT_COLLECTOR.md').read_text()
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", doc, re.S)
    assert len(blocks) == 1
    repo = tmp_path / 'repo'
    stage = repo / 'state' / 'migration'
    stage.mkdir(parents=True)
    sources = b'total_power:\n- entity: sensor.grid\n'
    (stage / 'historical-audited-consumer-sources.yaml').write_bytes(sources)
    for name in ('config/measurement_collector.example.json',
                 'deploy/systemd/hepo-measurement-collector.service.in',
                 'deploy/systemd/hepo-measurement-collector.timer'):
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit',
                    '--allow-empty', '-qm', 'Fixture'], check=True)
    config = stage / 'collector' / 'config.json'
    python = repo / '.venv' / 'bin' / 'python3'
    env = dict(os.environ, REPO=str(repo), CODE=str(repo), STAGE=str(stage),
               PYTHON=str(python), CONFIG=str(config))
    subprocess.run([sys.executable, '-c', blocks[0]], env=env, check=True)
    settings = json.loads(config.read_text())
    assert settings['env_file'] == str(repo / '.env')
    assert settings['archive'] == str(stage / 'measurements.sqlite')
    assert settings['lock_file'] == str(stage / 'historical-backfill.lock')
    assert settings['sources_sha256'] == hashlib.sha256(sources).hexdigest()
    assert Path(settings['sources']).read_bytes() == sources
    service = (config.parent / 'hepo-measurement-collector.service').read_text()
    assert f'WorkingDirectory={repo}' in service
    assert f'ExecStart="{python}"' in service
    assert '@' not in service
    # Exclusive creation must stop a repeated preparation, not replace config.
    before = config.read_bytes()
    result = subprocess.run([sys.executable, '-c', blocks[0]], env=env, capture_output=True)
    assert result.returncode != 0
    assert config.read_bytes() == before
