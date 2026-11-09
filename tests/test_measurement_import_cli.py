import subprocess
import sys
from pathlib import Path


def test_offline_import_command_is_explicit_and_idempotent(tmp_path):
    source = tmp_path / 'raw.csv'
    source.write_text('timestamp,gshp_power\n2026-10-01T00:00:00Z,1000\n')
    archive = tmp_path / 'archive.sqlite'
    command = [sys.executable, '-m', 'utils.measurement_import', '--archive', str(archive),
               '--input', str(source), '--interval-seconds', '60']
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    second = subprocess.run(command, capture_output=True, text=True, check=True)
    assert 'legacy_resampled' in first.stdout
    assert first.stdout == second.stdout


def test_missing_explicit_archive_is_rejected(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'utils.measurement_import',
                             '--input', str(tmp_path / 'x'), '--interval-seconds', '60'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert '--archive' in result.stderr
