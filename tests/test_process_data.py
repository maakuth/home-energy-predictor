from __future__ import annotations
import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


class TestProcessData(unittest.TestCase):
    """Integration test for process_data() — runs the full function on synthetic CSV data."""

    @classmethod
    def setUpClass(cls):
        cls.orig_cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        state_dir = os.path.join(self.test_dir, 'state')
        os.makedirs(state_dir, exist_ok=True)
        os.chdir(self.test_dir)
        self._create_synthetic_raw_csv(state_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_raw_csv(self, state_dir: str):
        n = 100
        base_ts = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        idx = pd.date_range(base_ts, periods=n, freq='15min', tz='UTC')

        np.random.seed(42)
        df = pd.DataFrame(index=idx)
        df['outside_temp'] = 5.0 + 3.0 * np.sin(np.arange(n) * 2 * np.pi / 96)
        df['gshp_power'] = 1500.0 + 200.0 * np.random.randn(n)
        df['aahp_living_power'] = 300.0 + 50.0 * np.random.randn(n)
        df['aahp_cabin_power'] = 200.0 + 40.0 * np.random.randn(n)
        df['mummun_power'] = 50.0 + 20.0 * np.random.randn(n)
        df['solar_forecast'] = np.maximum(0, 2.0 * np.sin(np.arange(n) * 2 * np.pi / 96 - 2))
        df['solar_actual'] = np.maximum(0, 1.8 * np.sin(np.arange(n) * 2 * np.pi / 96 - 2))
        df['leaf_power'] = np.where(np.arange(n) % 12 == 0, 3000.0, 0.0)
        ev_vals = np.full(n, np.nan)
        for i in range(0, n, 12):
            vals = np.linspace(20, 80, min(6, n - i))
            ev_vals[i:i+len(vals)] = vals
        df['ev_soc'] = ev_vals
        df['ev_position'] = 'home'
        df.iloc[20:40, df.columns.get_loc('ev_position')] = 'not_home'
        df['total_power'] = 2.0 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 12)
        df['battery_power'] = np.where(np.arange(n) % 4 < 2, 2000.0, -1000.0)
        df['accumulator_temp'] = 45.0 + 2.0 * np.sin(np.arange(n) * 2 * np.pi / 96)
        df['sauna_temp'] = np.where(np.arange(n) == 50, 80.0, 22.0)
        df['gshp_pump_temp'] = np.where(df['gshp_power'] >= 100, 35.0, np.nan)
        df['wind_speed'] = 3.0 + 1.0 * np.random.randn(n)

        df.to_csv(os.path.join(state_dir, 'raw_data.csv'))

    def test_output_exists(self):
        from process_data import process_data
        process_data()
        output_path = os.path.join(self.test_dir, 'state', 'processed_data.csv')
        self.assertTrue(os.path.exists(output_path), "process_data() should create processed_data.csv")

    def test_critical_columns_present(self):
        from process_data import process_data
        process_data()
        output_path = os.path.join(self.test_dir, 'state', 'processed_data.csv')
        df = pd.read_csv(output_path, index_col=0)

        expected = {
            'baseload_power', 'total_home_power', 'baseload_lag_1h', 'baseload_lag_24h',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'hour', 'minute', 'quarter_hour', 'day_of_week', 'month',
        }
        missing = expected - set(df.columns)
        self.assertEqual(missing, set(), f"Missing columns in processed_data.csv: {missing}")

    def test_baseload_non_negative(self):
        from process_data import process_data
        process_data()
        output_path = os.path.join(self.test_dir, 'state', 'processed_data.csv')
        df = pd.read_csv(output_path, index_col=0)
        self.assertGreaterEqual(df['baseload_power'].min(), -0.001)

    def test_cyclic_encoding_bounds(self):
        from process_data import process_data
        process_data()
        output_path = os.path.join(self.test_dir, 'state', 'processed_data.csv')
        df = pd.read_csv(output_path, index_col=0)
        for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']:
            self.assertGreaterEqual(df[col].min(), -1.0,
                                    msg=f"{col} min ({df[col].min():.4f}) should be >= -1.0")
            self.assertLessEqual(df[col].max(), 1.0,
                                 msg=f"{col} max ({df[col].max():.4f}) should be <= 1.0")

    def test_fireplace_active_column_present(self):
        from process_data import process_data
        process_data()
        output_path = os.path.join(self.test_dir, 'state', 'processed_data.csv')
        df = pd.read_csv(output_path, index_col=0)
        self.assertIn('is_fireplace_active', df.columns)
        self.assertIn('is_fireplace_lag1', df.columns)

    def test_total_home_power_formula(self):
        from process_data import process_data
        process_data()
        output_path = os.path.join(self.test_dir, 'state', 'processed_data.csv')
        df = pd.read_csv(output_path, index_col=0)
        # After median filtering, the relationship is approximate
        raw = df['total_power'] + df['solar_actual'] - (df['battery_power'] / 1000.0)
        mae = np.abs(df['total_home_power'] - raw).mean()
        self.assertLess(mae, 0.5,
                        f"total_home_power MAE vs expected formula: {mae:.4f}")

    def test_is_extended_complex_present(self):
        from process_data import process_data
        process_data()
        output_path = os.path.join(self.test_dir, 'state', 'processed_data.csv')
        df = pd.read_csv(output_path, index_col=0)
        self.assertIn('is_extended_complex', df.columns)
        self.assertTrue(df['is_extended_complex'].isin([0, 1]).all(),
                        "is_extended_complex should be 0 or 1")


if __name__ == '__main__':
    unittest.main()
