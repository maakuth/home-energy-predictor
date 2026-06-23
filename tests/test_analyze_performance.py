from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock
import sqlite3
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime, timedelta, timezone


def _floor_15min(dt: datetime) -> datetime:
    """Round datetime down to nearest 15-minute boundary."""
    return dt.replace(microsecond=0, second=0, minute=(dt.minute // 15) * 15)


class TestAnalyzePerformance(unittest.TestCase):
    """Test analyze_performance.analyze() with mocked HA and DB."""

    @classmethod
    def setUpClass(cls):
        cls.orig_cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.test_dir, 'state'), exist_ok=True)
        os.chdir(self.test_dir)
        self._seed_db()

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _seed_db(self):
        """Create a synthetic SQLite hepo.db with predictions and actuals."""
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_usage_kw REAL,
                solar_forecast_kw REAL,
                is_fallback_price INTEGER,
                version TEXT,
                battery_action TEXT,
                battery_power_kw REAL,
                battery_soc_pct REAL,
                import_price REAL,
                export_price REAL,
                grid_import_kwh REAL,
                grid_export_kwh REAL,
                charge_from_solar_kwh REAL,
                charge_from_grid_kwh REAL,
                discharge_to_load_kwh REAL,
                discharge_to_export_kwh REAL,
                planned_gshp_kw REAL
            )
        ''')
        now = _floor_15min(datetime.now(timezone.utc))
        base = now - timedelta(hours=24)
        for i in range(96):
            ts = base + timedelta(minutes=15 * i)
            row = (
                ts.isoformat(),
                ts.isoformat(),
                2.0 + 0.5 * np.sin(i * 2 * np.pi / 96),
                1.0,
                0,
                '1.0.0',
                'charge' if i % 4 < 2 else 'discharge',
                1.0,
                50.0,
                0.10 + 0.02 * np.sin(i * 2 * np.pi / 96),
                0.08 + 0.02 * np.sin(i * 2 * np.pi / 96),
                2.0 if i % 4 < 2 else 0.0,
                0.0 if i % 4 < 2 else 1.0,
                0.5,
                1.5,
                1.0,
                0.0,
                0.0
            )
            cur.execute('''
                INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', row)
        conn.commit()
        conn.close()

    @patch('analyze_performance.fetch_states_history')
    @patch('analyze_performance.get_db_connection')
    @patch('analyze_performance.db_exists')
    @patch('analyze_performance.push_ha_state')
    @patch('analyze_performance.get_model_version')
    def test_analyze_stores_results(self, mock_version, mock_push, mock_db_exist,
                                     mock_connect, mock_fetch):
        from analyze_performance import analyze
        mock_version.return_value = '1.0.0'
        mock_db_exist.return_value = True

        # Mock get_db_connection to return a fresh connection each time
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        mock_connect.side_effect = lambda: sqlite3.connect(db_path)

        # Mock fetch_states_history to return synthetic actuals
        now = _floor_15min(datetime.now(timezone.utc))
        base = now - timedelta(hours=24)
        n = 96
        idx = pd.date_range(base, periods=n, freq='15min', tz='UTC')
        df_actual = pd.DataFrame({
            'state': [2.0 + 0.5 * np.sin(i * 2 * np.pi / 96) + 0.05 * np.random.randn() for i in range(n)]
        }, index=idx)
        df_actual.index.name = 'timestamp'
        mock_fetch.return_value = {
            'sensor.sahkokauppa_nyt': df_actual,
            'sensor.solarh_63038_real_power_kw': df_actual * 0,
            'sensor.mlp_teho': df_actual * 0,
            'sensor.be_stat_batt_power': df_actual * 0,
        }

        analyze(days=2, do_backtest=False)

        mock_push.assert_called()

        # Verify results stored in DB
        verify_conn = sqlite3.connect(db_path)
        cur = verify_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM performance_analysis")
        count = cur.fetchone()[0]
        self.assertEqual(count, 1, "analyze() should store one performance_analysis row")

        cur.execute("SELECT mae_kw, bias_kw, model_version FROM performance_analysis ORDER BY analysis_timestamp DESC LIMIT 1")
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertGreater(row[0], 0, "MAE should be positive")
        self.assertEqual(row[2], '1.0.0')
        verify_conn.close()

    @patch('analyze_performance.fetch_states_history')
    @patch('analyze_performance.get_db_connection')
    @patch('analyze_performance.db_exists')
    @patch('analyze_performance.push_ha_state')
    @patch('analyze_performance.get_model_version')
    def test_analyze_handles_empty_actuals(self, mock_version, mock_push,
                                            mock_db_exist, mock_connect, mock_fetch):
        from analyze_performance import analyze
        mock_version.return_value = '1.0.0'
        mock_db_exist.return_value = True
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        mock_connect.return_value = sqlite3.connect(db_path)

        # Empty fetch_states_history
        mock_fetch.return_value = {}

        analyze(days=2, do_backtest=False)
        mock_push.assert_not_called()

    @patch('analyze_performance.fetch_states_history')
    @patch('analyze_performance.get_db_connection')
    @patch('analyze_performance.db_exists')
    @patch('analyze_performance.push_ha_state')
    @patch('analyze_performance.get_model_version')
    def test_analyze_no_archived_predictions(self, mock_version, mock_push,
                                              mock_db_exist, mock_connect, mock_fetch):
        from analyze_performance import analyze, get_archived_predictions
        mock_version.return_value = '2.0.0'  # different version from seeded DB
        mock_db_exist.return_value = True
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        mock_connect.return_value = sqlite3.connect(db_path)

        now = _floor_15min(datetime.now(timezone.utc))
        base = now - timedelta(hours=24)
        n = 96
        idx = pd.date_range(base, periods=n, freq='15min', tz='UTC')
        df_actual = pd.DataFrame({'state': [1.0] * n}, index=idx)
        df_actual.index.name = 'timestamp'
        mock_fetch.return_value = {
            'sensor.sahkokauppa_nyt': df_actual,
            'sensor.solarh_63038_real_power_kw': df_actual * 0,
            'sensor.mlp_teho': df_actual * 0,
            'sensor.be_stat_batt_power': df_actual * 0,
        }

        analyze(days=2, do_backtest=False)
        # Should still run without crashing, just print warning
        mock_push.assert_not_called()


if __name__ == '__main__':
    unittest.main()
