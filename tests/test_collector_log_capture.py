"""Exercise journal capture using a fake journalctl, never the live journal."""
import os
from pathlib import Path
import subprocess


def test_get_logs_captures_collector_user_units_and_preserves_existing_logs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    journalctl = bin_dir / 'journalctl'
    journalctl.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
    journalctl.chmod(0o755)
    logs = tmp_path / 'logs'
    logs.mkdir()
    subprocess.run(['/bin/sh', str(root / 'get-logs.sh')], cwd=tmp_path,
                   env=dict(os.environ, PATH=str(bin_dir)), check=True)
    collector = (logs / 'recentlog-collector.txt').read_text().splitlines()
    assert collector == ['--user', '-u', 'hepo-measurement-collector.service',
                         '-u', 'hepo-measurement-collector.timer', '-S7 days ago']
    for name, unit in [('recentlog.txt', 'home-energy-predictor-15min.service'),
                       ('recentlog-often.txt', 'home-energy-predictor-minutely.service'),
                       ('recentlog-weekly.txt', 'home-energy-predictor-weekly.service')]:
        captured = (logs / name).read_text()
        assert '--user\n' in captured
        assert unit in captured
        assert '-S7 days ago\n' in captured
