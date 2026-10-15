from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

import analyze_evolution as evolution


FIXED_NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz else FIXED_NOW.replace(tzinfo=None)


def _seed_predictions(db_path: str) -> pd.DatetimeIndex:
    timestamps = pd.date_range(FIXED_NOW - timedelta(hours=2), periods=2, freq="15min", tz="UTC")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE predictions (
            target_timestamp TEXT, generated_at TEXT, predicted_usage_kw REAL,
            battery_action TEXT, version TEXT, grid_import_kwh REAL,
            import_price REAL, grid_export_kwh REAL, export_price REAL
        )"""
    )
    for target, old_prediction, new_prediction in zip(timestamps, (2.0, 3.0), (4.0, 1.0)):
        conn.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (target.isoformat(), (target - timedelta(hours=1)).isoformat(), old_prediction,
                 "charge", "1.0.0", 1.0, 0.10, 0.0, 0.08),
                (target.isoformat(), (target - timedelta(hours=2)).isoformat(), new_prediction,
                 "discharge", "2.0.0", 2.0, 0.20, 0.5, 0.08),
            ],
        )
    conn.commit()
    conn.close()
    return timestamps


def test_evolution_reports_accuracy_cost_stability_and_version_quality(monkeypatch, tmp_path, capsys):
    """The report must expose the quality comparison used for model changes."""
    timestamps = _seed_predictions(str(tmp_path / "hepo.db"))
    actuals = pd.DataFrame(
        {
            "actual_usage": [2.0, 3.0],
            "solar_actual": [0.0, 0.0],
            "gshp_actual_kw": [0.0, 0.0],
        },
        index=timestamps,
    )

    monkeypatch.setattr(evolution, "datetime", FrozenDatetime)
    monkeypatch.setattr(evolution, "db_exists", lambda: True)
    monkeypatch.setattr(evolution, "get_db_connection", lambda: sqlite3.connect(tmp_path / "hepo.db"))
    monkeypatch.setattr(evolution, "fetch_actuals", lambda days: actuals)
    monkeypatch.setattr(evolution, "get_model_version", lambda: "2.0.0")

    evolution.analyze_evolution(days=2)

    output = capsys.readouterr().out
    assert "Accuracy by Lead Time" in output
    assert "Planned Cost Evolution" in output
    assert "Average StdDev of predictions for same timestamp: 1.414 kW" in output
    assert "Timestamps where Battery Action evolved:        100.0%" in output
    assert "Accuracy by Model Version" in output
    # Version 1 is exact (MAE 0); version 2 has a 2 kW MAE on the same actuals.
    assert "2.0.0          2       2.0" in output
    assert "1.0.0          2       0.0" in output


def test_evolution_reports_missing_actuals_without_opening_the_archive(monkeypatch, capsys):
    monkeypatch.setattr(evolution, "fetch_actuals", lambda days: pd.DataFrame())
    monkeypatch.setattr(evolution, "fetch_all_predictions", lambda days: (_ for _ in ()).throw(AssertionError()))

    evolution.analyze_evolution(days=2)

    assert "No actuals found" in capsys.readouterr().out
