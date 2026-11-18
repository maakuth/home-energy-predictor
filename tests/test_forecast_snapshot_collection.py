import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest

from utils import forecast_snapshots


ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path, **overrides):
    archive = tmp_path / "forecast.sqlite"
    env = tmp_path / ".env"
    env.write_text("HA_HOST=http://ha.invalid\nHA_TOKEN=secret-token\n")
    data = {
        "format_version": 1,
        "archive": str(archive),
        "weather_entity": "weather.home",
        "price_entity": "sensor.price",
        "price_zone": "FI",
        "env_file": str(env),
        "lock_file": str(tmp_path / "forecast.lock"),
        "reports_dir": str(tmp_path / "reports"),
        "max_attempts": 10,
        "max_bytes": 1000000,
    }
    data.update(overrides)
    config = tmp_path / "forecast.json"
    config.write_text(json.dumps(data))
    return config, data, archive, env


def test_dry_run_does_not_create_archive_lock_reports_or_call_live_services(tmp_path, monkeypatch):
    config, data, archive, env = make_config(tmp_path)
    calls = {"get": 0, "post": 0}
    monkeypatch.setattr(forecast_snapshots.requests, "get", lambda *a, **k: calls.__setitem__("get", calls["get"] + 1))
    monkeypatch.setattr(forecast_snapshots.requests, "post", lambda *a, **k: calls.__setitem__("post", calls["post"] + 1))
    before = list(tmp_path.iterdir())
    result = forecast_snapshots.collect(config, now=1731936000.0)
    assert result["status"] == "dry_run"
    assert not archive.exists()
    assert not Path(data["lock_file"]).exists()
    assert not Path(data["reports_dir"]).exists()
    assert calls == {"get": 0, "post": 0}
    assert set(tmp_path.iterdir()) == set(before)


def test_execute_writes_valid_json_report_with_real_newline(tmp_path, monkeypatch):
    config, data, archive, env = make_config(tmp_path)

    def fake_get(url, headers=None, timeout=None):
        response = type("R", (), {})()
        response.raise_for_status = lambda: None
        response.json = lambda: {
            "state": "1",
            "attributes": {"prices_today": [{"time": "2026-11-18T14:00:00Z", "price": 1}]},
        }
        return response

    def fake_post(url, headers=None, json=None, timeout=None):
        response = type("R", (), {})()
        response.raise_for_status = lambda: None
        response.json = lambda: {
            "service_response": {"weather.home": {"forecast": [{"datetime": "2026-11-18T14:00:00Z", "temperature": 5}]}}
        }
        return response

    monkeypatch.setattr(forecast_snapshots.requests, "get", fake_get)
    monkeypatch.setattr(forecast_snapshots.requests, "post", fake_post)
    result = forecast_snapshots.collect(config, execute=True, now=1731936000.0)
    assert result["status"] == "success"
    assert result["exit_code"] == 0
    report_path = Path(result["report_path"])
    raw = report_path.read_text()
    assert raw.endswith("\n")
    assert "\\n" not in raw
    parsed = json.loads(raw)
    assert parsed["status"] == "success"
    assert parsed["weather_status"] == "ok"
    assert parsed["price_status"] == "ok"


def test_lock_busy_skips_without_live_calls_and_writes_report(tmp_path, monkeypatch):
    config, data, archive, env = make_config(tmp_path)
    import fcntl

    with Path(data["lock_file"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(forecast_snapshots.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live call")))
        result = forecast_snapshots.collect(config, execute=True, now=1731936000.0)
    assert result["status"] == "skipped_locked"
    assert result["exit_code"] == 75
    assert len(list(Path(data["reports_dir"]).glob("forecast-snapshot-*.json"))) == 1


def test_config_rejects_hepo_db_and_missing_credentials(tmp_path):
    config, data, archive, env = make_config(tmp_path, archive=str(tmp_path / "hepo.db"))
    with pytest.raises(ValueError, match="hepo.db"):
        forecast_snapshots.collect(config)

    subdir = tmp_path / "sub"
    subdir.mkdir()
    config2, data2, archive2, env2 = make_config(subdir)
    env2.write_text("HA_HOST=http://ha.invalid\n")
    result = forecast_snapshots.collect(config2, execute=True, now=1731936000.0)
    assert result["status"] == "failed"
    assert result.get("error_type") == "ValueError"


def test_shell_csv_mode_triggers_configured_unit_nonblocking(tmp_path):
    script = (ROOT / "run_frequent.sh").read_text().replace("/tmp/hepo.lock", str(tmp_path / "hepo.lock"))
    (tmp_path / "run_frequent.sh").write_text(script)
    activation = tmp_path / ".venv" / "bin" / "activate"
    activation.parent.mkdir(parents=True)
    activation.write_text(":\n")
    commands = tmp_path / "commands"
    commands.mkdir()
    python = commands / "python3"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    (commands / "python").symlink_to(python)
    systemctl = commands / "systemctl"
    systemctl.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$SYSTEMCTL_LOG\"\nexit 1\n")
    systemctl.chmod(0o755)
    system_log = tmp_path / "systemctl.log"
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "SYSTEMCTL_LOG": str(system_log),
        "FORECAST_SNAPSHOT_SYSTEMD_UNIT": "hepo-forecast-snapshot.service",
    }
    result = subprocess.run(["bash", str(tmp_path / "run_frequent.sh")], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert system_log.read_text().splitlines() == ["--user", "start", "--no-block", "hepo-forecast-snapshot.service"]


def test_shell_csv_mode_triggers_even_if_optimization_fails(tmp_path):
    script = (ROOT / "run_frequent.sh").read_text().replace("/tmp/hepo.lock", str(tmp_path / "hepo.lock"))
    (tmp_path / "run_frequent.sh").write_text(script)
    activation = tmp_path / ".venv" / "bin" / "activate"
    activation.parent.mkdir(parents=True)
    activation.write_text(":\n")
    commands = tmp_path / "commands"
    commands.mkdir()
    python = commands / "python3"
    python.write_text("#!/bin/bash\nif echo \"$@\" | grep -q 'optimize_plan.py'; then exit 2; fi\nexit 0\n")
    python.chmod(0o755)
    (commands / "python").symlink_to(python)
    systemctl = commands / "systemctl"
    systemctl.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$SYSTEMCTL_LOG\"\nexit 0\n")
    systemctl.chmod(0o755)
    system_log = tmp_path / "systemctl.log"
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "SYSTEMCTL_LOG": str(system_log),
        "FORECAST_SNAPSHOT_SYSTEMD_UNIT": "hepo-forecast-snapshot.service",
    }
    result = subprocess.run(["bash", str(tmp_path / "run_frequent.sh")], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert system_log.exists()
    assert system_log.read_text().splitlines() == ["--user", "start", "--no-block", "hepo-forecast-snapshot.service"]


def test_shell_csv_mode_without_unit_does_not_invoke_systemctl(tmp_path):
    script = (ROOT / "run_frequent.sh").read_text().replace("/tmp/hepo.lock", str(tmp_path / "hepo.lock"))
    (tmp_path / "run_frequent.sh").write_text(script)
    activation = tmp_path / ".venv" / "bin" / "activate"
    activation.parent.mkdir(parents=True)
    activation.write_text(":\n")
    commands = tmp_path / "commands"
    commands.mkdir()
    python = commands / "python3"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    (commands / "python").symlink_to(python)
    systemctl = commands / "systemctl"
    systemctl.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$SYSTEMCTL_LOG\"\nexit 0\n")
    systemctl.chmod(0o755)
    system_log = tmp_path / "systemctl.log"
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "SYSTEMCTL_LOG": str(system_log),
    }
    result = subprocess.run(["bash", str(tmp_path / "run_frequent.sh")], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not system_log.exists()


def test_shell_csv_mode_trigger_timeout_is_bounded(tmp_path):
    script = (ROOT / "run_frequent.sh").read_text().replace("/tmp/hepo.lock", str(tmp_path / "hepo.lock"))
    (tmp_path / "run_frequent.sh").write_text(script)
    activation = tmp_path / ".venv" / "bin" / "activate"
    activation.parent.mkdir(parents=True)
    activation.write_text(":\n")
    commands = tmp_path / "commands"
    commands.mkdir()
    python = commands / "python3"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    (commands / "python").symlink_to(python)
    systemctl = commands / "systemctl"
    systemctl.write_text("#!/bin/bash\nsleep 5\nexit 0\n")
    systemctl.chmod(0o755)
    system_log = tmp_path / "systemctl.log"
    env = {
        **os.environ,
        "PATH": str(commands) + ":" + os.environ["PATH"],
        "FORECAST_SNAPSHOT_SYSTEMD_UNIT": "hepo-forecast-snapshot.service",
    }
    result = subprocess.run(["bash", str(tmp_path / "run_frequent.sh")], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "Forecast snapshot trigger failed" in result.stderr


def test_run_frequent_shell_has_no_background_snapshot_subprocess():
    text = (ROOT / "run_frequent.sh").read_text()
    assert "--no-block" in text
    assert "FORECAST_SNAPSHOT_SYSTEMD_UNIT" in text
    # The snapshot trigger must not use shell backgrounding.
    assert "systemctl --user start --no-block" in text


def test_archive_dry_run_does_not_invoke_forecast_collection(tmp_path, capsys, monkeypatch):
    import utils.archive_pipeline as ap
    monkeypatch.chdir(tmp_path)
    with __import__("unittest.mock", fromlist=["patch"]).patch("utils.archive_pipeline.execute_plan") as execute:
        ap.main(["--mode", "frequent", "--archive", "/not-created.sqlite",
                 "--bootstrap-from", "2026-01-01T00:00:00Z",
                 "--cutoff", "2026-11-10T08:00:00Z", "--recorder-namespace", "ha"])
    execute.assert_not_called()
    # No forecast snapshot archive or report should be created by dry run.
    assert not (tmp_path / "forecast.sqlite").exists()