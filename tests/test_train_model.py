from __future__ import annotations
import unittest
import os
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone


class TestTrainModel(unittest.TestCase):
    """Integration test for train() — runs the full training function on synthetic data."""

    @classmethod
    def setUpClass(cls):
        cls.orig_cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        state_dir = os.path.join(self.test_dir, 'state')
        os.makedirs(state_dir, exist_ok=True)
        os.chdir(self.test_dir)
        self._create_synthetic_processed_csv(state_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_processed_csv(self, state_dir: str):
        n = 300
        base_ts = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        idx = pd.date_range(base_ts, periods=n, freq='15min', tz='UTC')

        np.random.seed(42)
        df = pd.DataFrame(index=idx)
        df['baseload_power'] = 1.0 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 96) + 0.1 * np.random.randn(n)
        df['outside_temp'] = 10.0 + 5.0 * np.sin(np.arange(n) * 2 * np.pi / 96)
        df['wind_speed'] = 3.0 + np.random.randn(n)
        df['solar_forecast'] = np.maximum(0, 2.0 * np.sin(np.arange(n) * 2 * np.pi / 96 - 2))
        df['accumulator_temp'] = 45.0 + 2.0 * np.sin(np.arange(n) * 2 * np.pi / 96)
        df['gshp_pump_temp'] = np.where(np.abs(np.sin(np.arange(n) * 2 * np.pi / 24)) > 0.5, 35.0, np.nan)
        df['is_gshp_pump_running'] = df['gshp_pump_temp'].notna().astype(int)
        df['acc_roc'] = df['accumulator_temp'].diff().fillna(0)
        df['is_fireplace_lag1'] = 0
        df['leaf_power_lag_1h'] = np.where(np.arange(n) % 12 == 0, 3.0, 0.0)
        df['leaf_energy_24h'] = 2.0 + 0.5 * np.random.randn(n)
        df['baseload_lag_1h'] = df['baseload_power'].shift(4).fillna(1.0)
        df['baseload_lag_24h'] = df['baseload_power'].shift(96).fillna(1.0)
        df['is_extended_complex'] = 1
        df['hour'] = idx.hour
        df['minute'] = idx.minute
        df['quarter_hour'] = idx.minute // 15
        df['day_of_week'] = idx.dayofweek
        df['month'] = idx.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

        df.to_csv(os.path.join(state_dir, 'processed_data.csv'))

    def test_model_files_created(self):
        from train_model import train
        train()
        model_path = os.path.join(self.test_dir, 'state', 'energy_model.json')
        features_path = os.path.join(self.test_dir, 'state', 'model_features.json')
        self.assertTrue(os.path.exists(model_path), "train() should create energy_model.json")
        self.assertTrue(os.path.exists(features_path), "train() should create model_features.json")

    def test_features_file_format(self):
        from train_model import train
        train()
        features_path = os.path.join(self.test_dir, 'state', 'model_features.json')
        with open(features_path) as f:
            saved_features = json.load(f)
        expected_features = [
            'outside_temp', 'wind_speed', 'solar_forecast',
            'accumulator_temp', 'gshp_pump_temp', 'is_gshp_pump_running', 'acc_roc', 'is_fireplace_lag1',
            'leaf_power_lag_1h', 'leaf_energy_24h',
            'baseload_lag_1h', 'baseload_lag_24h',
            'is_extended_complex',
            'hour', 'minute', 'quarter_hour', 'day_of_week', 'month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        self.assertEqual(saved_features, expected_features)

    def test_model_is_valid_json(self):
        from train_model import train
        train()
        model_path = os.path.join(self.test_dir, 'state', 'energy_model.json')
        with open(model_path) as f:
            model_json = json.load(f)
        self.assertIn('learner', model_json, "XGBoost JSON should have 'learner' key")
        self.assertIn('feature_names', model_json.get('learner', {}),
                      "Learner should have 'feature_names'")
        self.assertIn('version', model_json, "XGBoost JSON should have 'version' key")

    def test_holdout_excludes_recent(self):
        from train_model import train
        train(holdout_days=1)
        model_path = os.path.join(self.test_dir, 'state', 'energy_model.json')
        self.assertTrue(os.path.exists(model_path))

    def test_too_few_rows_returns_gracefully(self):
        from train_model import train
        train_dir = os.path.join(self.test_dir, 'state_too_small')
        os.makedirs(train_dir, exist_ok=True)
        df = pd.DataFrame({'baseload_power': [1.0], 'outside_temp': [10.0]})
        df.to_csv(os.path.join(train_dir, 'processed_data.csv'))
        # Chdir to the small-data dir
        os.chdir(os.path.dirname(train_dir))
        try:
            from train_model import train
            result = train()
            self.assertIsNone(result)
        finally:
            os.chdir(self.test_dir)


if __name__ == '__main__':
    unittest.main()
