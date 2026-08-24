from __future__ import annotations
"""Tests for utils/solar_calibration.py (Solcast p10/p50/p90 calibration monitor).

Uses a synthetic SQLite ``predictions`` table in tmp_path via the DB_PATH
override in utils.sqlite_utils, so production state/hepo.db is never touched.
"""

import sqlite3

import pandas as pd
import pytest

from utils import solar_calibration as sc
from utils import sqlite_utils


def _make_predictions_db(path: str, with_p10: bool = True) -> None:
    """Create a minimal predictions table, optionally with p10/p90 columns."""
    conn = sqlite3.connect(path)
    cols = (
        "target_timestamp TEXT, generated_at TEXT, "
        "predicted_usage_kw REAL, solar_forecast_kw REAL"
    )
    if with_p10:
        cols += ", solar_forecast_p10_kw REAL, solar_forecast_p90_kw REAL"
    conn.execute(f"CREATE TABLE predictions ({cols})")
    base = (
        "2026-08-12T10:00:00+00:00",
        "2026-08-12T09:00:00",
        1.0,
        4.0,
    )
    rows = [base + ((3.0, 5.0) if with_p10 else ())]
    if with_p10:
        conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)", rows[0]
        )
    else:
        conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)", rows[0])
    conn.commit()
    conn.close()


class TestLoadArchivedForecasts:
    def test_includes_p10_p90_when_columns_exist(self, monkeypatch, tmp_path):
        db = str(tmp_path / "test.db")
        _make_predictions_db(db, with_p10=True)
        monkeypatch.setattr(sqlite_utils, "DB_PATH", db)

        df = sc.load_archived_forecasts()

        assert not df.empty
        assert "solar_forecast_kw" in df.columns
        assert "solar_forecast_p10_kw" in df.columns
        assert "solar_forecast_p90_kw" in df.columns
        assert df["solar_forecast_p10_kw"].iloc[0] == pytest.approx(3.0)
        assert df["solar_forecast_p90_kw"].iloc[0] == pytest.approx(5.0)

    def test_works_without_p10_columns(self, monkeypatch, tmp_path):
        db = str(tmp_path / "test.db")
        _make_predictions_db(db, with_p10=False)
        monkeypatch.setattr(sqlite_utils, "DB_PATH", db)

        df = sc.load_archived_forecasts()

        assert not df.empty
        assert "solar_forecast_kw" in df.columns
        assert "solar_forecast_p10_kw" not in df.columns
        assert "solar_forecast_p90_kw" not in df.columns


class TestAnalyze:
    def _frame(self) -> pd.DataFrame:
        # 4 intervals with p10 populated, 4 with NaN (pre-migration rows).
        idx = pd.date_range("2026-08-12 10:00", periods=8, freq="15min", tz="UTC")
        return pd.DataFrame(
            {
                "solar_forecast_kw": [4.0] * 8,
                "solar_forecast_p10_kw": [3.0, 3.0, 3.0, 3.0] + [float("nan")] * 4,
                "solar_forecast_p90_kw": [5.0, 5.0, 5.0, 5.0] + [float("nan")] * 4,
                "solar_actual_kw": [2.0, 3.5, 4.0, 4.5] + [4.0] * 4,
            },
            index=idx,
        )

    def test_p10_fraction_uses_only_populated_rows(self):
        metrics = sc.analyze(self._frame())

        # 1 of 4 populated rows has actual <= p10 (2.0 <= 3.0).
        # Pre-migration NaN rows must not dilute the denominator.
        assert metrics["actual_le_p10_frac"] == pytest.approx(0.25)
        assert metrics["n_p10_intervals"] == pytest.approx(4.0)

    def test_p90_fraction_uses_only_populated_rows(self):
        metrics = sc.analyze(self._frame())

        # 0 of 4 populated rows has actual >= p90 (all <= 4.5 < 5.0).
        assert metrics["actual_ge_p90_frac"] == pytest.approx(0.0)
        assert metrics["n_p90_intervals"] == pytest.approx(4.0)

    def test_p50_fraction_uses_all_valid_rows(self):
        metrics = sc.analyze(self._frame())

        # actual <= p50 on 7 of 8 rows (the 4.5 kW row exceeds the 4.0 forecast).
        assert metrics["actual_le_p50_frac"] == pytest.approx(7.0 / 8.0)
        assert metrics["n_intervals"] == pytest.approx(8.0)


class TestDailyShortfall:
    def test_shortfall_integrates_positive_misses_only(self):
        idx = pd.DatetimeIndex(
            [
                "2026-08-12 10:00",
                "2026-08-12 10:15",
                "2026-08-12 10:30",
                "2026-08-13 10:00",
            ],
            tz="UTC",
        )
        df = pd.DataFrame(
            {
                "solar_forecast_kw": [4.0, 4.0, 2.0, 3.0],
                # 1.0 kW miss, 0.5 kW miss, 0.5 kW *surplus* (clipped), 2.0 kW miss
                "solar_actual_kw": [3.0, 3.5, 2.5, 1.0],
            },
            index=idx,
        )

        shortfall = sc.daily_shortfall_kwh(df)

        import datetime as dt

        # (1.0 + 0.5 + 0) kW * 0.25 h = 0.375 kWh
        assert shortfall[dt.date(2026, 8, 12)] == pytest.approx(0.375)
        # 2.0 kW * 0.25 h = 0.5 kWh
        assert shortfall[dt.date(2026, 8, 13)] == pytest.approx(0.5)

    def test_shortfall_empty_when_no_forecast(self):
        idx = pd.date_range("2026-08-12 10:00", periods=2, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {"solar_forecast_kw": [0.0, 0.0], "solar_actual_kw": [1.0, 1.0]},
            index=idx,
        )
        assert sc.daily_shortfall_kwh(df).empty
