from __future__ import annotations
import unittest
from unittest.mock import patch
import sqlite3
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime, timedelta, timezone


def _floor_15min(dt: datetime) -> datetime:
    return dt.replace(microsecond=0, second=0, minute=(dt.minute // 15) * 15)


class TestAnalyzeEvolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orig_cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.test_dir, 'state'), exist_ok=True)
        os.chdir(self.test_dir)
        self.seeded = False

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _seed_db(self, num_versions: int = 1):
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                target_timestamp TEXT,
                generated_at TEXT,
                predicted_usage_kw REAL,
                battery_action TEXT,
                version TEXT,
                grid_import_kwh REAL,
                import_price REAL,
                grid_export_kwh REAL,
                export_price REAL
            )
        ''')
        now = _floor_15min(datetime.now(timezone.utc))
        base = now - timedelta(hours=12)
        versions = [f'{v}.0.0' for v in range(1, num_versions + 1)]

        for i in range(48):
            target_ts = base + timedelta(minutes=15 * i)
            for vi, version in enumerate(versions):
                lead = (vi + 1) * 1
                generated_ts = target_ts - timedelta(hours=lead)
                cur.execute('''
                    INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)
                ''', (
                    target_ts.isoformat(),
                    generated_ts.isoformat(),
                    2.0 + 0.3 * np.sin(i * 2 * np.pi / 48),
                    'charge' if version == '1.0.0' else 'discharge',
                    version,
                    0.5 + 0.1 * np.sin(i * 2 * np.pi / 48),
                    0.10 + 0.02 * np.sin(i * 2 * np.pi / 48),
                    0.2 + 0.05 * np.sin(i * 2 * np.pi / 48),
                    0.08 + 0.01 * np.sin(i * 2 * np.pi / 48),
                ))
        conn.commit()
        conn.close()
        self.seeded = True

    def _make_actuals(self):
        now = _floor_15min(datetime.now(timezone.utc))
        base = now - timedelta(hours=12)
        idx = pd.date_range(base, periods=48, freq='15min', tz='UTC')
        df = pd.DataFrame({
            'actual_usage': 2.0 + 0.5 * np.sin(np.arange(48) * 2 * np.pi / 48),
            'solar_actual': np.zeros(48),
            'gshp_actual_kw': np.zeros(48),
        }, index=idx)
        df.index.name = 'timestamp'
        return df

    @patch('analyze_evolution.fetch_actuals')
    @patch('analyze_evolution.get_model_version')
    @patch('analyze_evolution.get_db_connection')
    @patch('analyze_evolution.db_exists')
    def test_evolution_happy_path_one_version(self, mock_db_exists,
                                                mock_get_db, mock_version,
                                                mock_fetch_actuals):
        self._seed_db(num_versions=1)
        mock_db_exists.return_value = True
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        mock_get_db.side_effect = lambda: sqlite3.connect(db_path)
        mock_version.return_value = '1.0.0'
        mock_fetch_actuals.return_value = self._make_actuals()

        from analyze_evolution import analyze_evolution
        analyze_evolution(days=2)

    @patch('analyze_evolution.fetch_actuals')
    @patch('analyze_evolution.get_model_version')
    @patch('analyze_evolution.get_db_connection')
    @patch('analyze_evolution.db_exists')
    def test_evolution_happy_path_two_versions(self, mock_db_exists,
                                                mock_get_db, mock_version,
                                                mock_fetch_actuals):
        self._seed_db(num_versions=2)
        mock_db_exists.return_value = True
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        mock_get_db.side_effect = lambda: sqlite3.connect(db_path)
        mock_version.return_value = '2.0.0'
        mock_fetch_actuals.return_value = self._make_actuals()

        from analyze_evolution import analyze_evolution
        analyze_evolution(days=2)

    @patch('analyze_evolution.fetch_actuals')
    @patch('analyze_evolution.get_model_version')
    @patch('analyze_evolution.get_db_connection')
    @patch('analyze_evolution.db_exists')
    def test_evolution_no_actuals(self, mock_db_exists,
                                   mock_get_db, mock_version,
                                   mock_fetch_actuals):
        self._seed_db(num_versions=1)
        mock_db_exists.return_value = True
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        mock_get_db.side_effect = lambda: sqlite3.connect(db_path)
        mock_version.return_value = '1.0.0'
        mock_fetch_actuals.return_value = pd.DataFrame()

        from analyze_evolution import analyze_evolution
        analyze_evolution(days=2)

    @patch('analyze_evolution.fetch_actuals')
    @patch('analyze_evolution.get_model_version')
    @patch('analyze_evolution.get_db_connection')
    @patch('analyze_evolution.db_exists')
    def test_evolution_no_predictions(self, mock_db_exists,
                                       mock_get_db, mock_version,
                                       mock_fetch_actuals):
        mock_db_exists.return_value = True
        db_path = os.path.join(self.test_dir, 'state', 'hepo.db')
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                target_timestamp TEXT, generated_at TEXT,
                predicted_usage_kw REAL, battery_action TEXT,
                version TEXT, grid_import_kwh REAL, import_price REAL,
                grid_export_kwh REAL, export_price REAL
            )
        ''')
        conn.commit()
        conn.close()
        mock_get_db.side_effect = lambda: sqlite3.connect(db_path)
        mock_version.return_value = '1.0.0'
        mock_fetch_actuals.return_value = self._make_actuals()

        from analyze_evolution import analyze_evolution
        analyze_evolution(days=2)

    @patch('analyze_evolution.fetch_actuals')
    @patch('analyze_evolution.get_model_version')
    @patch('analyze_evolution.get_db_connection')
    @patch('analyze_evolution.db_exists')
    def test_evolution_db_not_found(self, mock_db_exists,
                                     mock_get_db, mock_version,
                                     mock_fetch_actuals):
        mock_db_exists.return_value = False
        mock_version.return_value = '1.0.0'
        mock_fetch_actuals.return_value = self._make_actuals()

        from analyze_evolution import analyze_evolution
        analyze_evolution(days=2)


if __name__ == '__main__':
    unittest.main()
