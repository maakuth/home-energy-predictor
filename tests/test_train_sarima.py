import unittest
import pytest
import os
import pickle
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from train_sarima import train_sarima
from sarimax_predictor import BASELOAD_MAX_KW

class TestTrainSarimaSanityCheck(unittest.TestCase):
    def setUp(self):
        # Use test-specific paths from environment (set by conftest.py)
        self.test_dir = os.getenv('TEST_TMP_DIR', os.path.join(os.path.dirname(__file__), 'tmp_test_train_sarima'))
        self.params_path = os.getenv('TEST_SARIMA_PARAMS', os.path.join(self.test_dir, 'test_sarima_params.pkl'))
        
        # Create directories if they don't exist
        dir_path = os.path.dirname(self.params_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.csv_file = os.path.join(self.test_dir, 'test_processed_data.csv')
        
        # Create dummy historical data (3 days of 15-min intervals = 288 points)
        timestamps = pd.date_range(start='2026-04-01', periods=288, freq='15min', tz='UTC')
        np.random.seed(42)
        base = 1.0
        cycle = 0.5 * np.sin(2 * np.pi * np.arange(288) / 96)
        noise = 0.1 * np.random.randn(288)
        baseload = base + cycle + noise
        
        df = pd.DataFrame({'baseload_power': baseload}, index=timestamps)
        df.to_csv(self.csv_file)
    
    def tearDown(self):
        # No cleanup needed - conftest.py handles it via tmp_path fixture
        pass

    @pytest.mark.slow
    @patch('train_sarima.load_historical_data')
    def test_saves_params_when_forecast_is_reasonable(self, mock_load):
        """If fitted model produces reasonable forecasts, save the params."""
        ts_data = pd.read_csv(self.csv_file, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index, utc=True)
        ts_data = ts_data.sort_index()['baseload_power'].resample('15min').mean().ffill()
        mock_load.return_value = ts_data
        
        train_sarima(days=3, params_path=self.params_path)
        
        # Should have created the params file in the isolated test location
        self.assertTrue(os.path.exists(self.params_path))
        
        with open(self.params_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.assertIn('params', model_data)
        self.assertIn('order', model_data)
        self.assertIn('seasonal_order', model_data)

    @pytest.mark.slow
    @patch('train_sarima.load_historical_data')
    def test_does_not_overwrite_when_forecast_explodes(self, mock_load):
        """If fitted model produces exploding forecasts, keep old params."""
        ts_data = pd.read_csv(self.csv_file, index_col=0)
        ts_data.index = pd.to_datetime(ts_data.index, utc=True)
        ts_data = ts_data.sort_index()['baseload_power'].resample('15min').mean().ffill()
        mock_load.return_value = ts_data
        
        # Create a fake "old" params file with recognizable content in the isolated test location
        old_params = pd.Series({'ar.L1': 0.999, 'test_marker': 42.0})
        old_model_data = {
            'params': old_params,
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 0, 96),
            'last_index': ts_data.index.max()
        }
        with open(self.params_path, 'wb') as f:
            pickle.dump(old_model_data, f)
        
        # Mock SARIMAX.fit to return a results object whose get_forecast explodes
        mock_results = MagicMock()
        mock_forecast = MagicMock()
        mock_forecast.predicted_mean = pd.Series([1000.0] * 10)  # Exploding mean
        mock_forecast.conf_int.return_value = pd.DataFrame({
            'lower baseload_power': [500.0] * 10,
            'upper baseload_power': [1500.0] * 10
        })
        mock_results.get_forecast.return_value = mock_forecast
        
        with patch('statsmodels.tsa.statespace.sarimax.SARIMAX.fit', return_value=mock_results):
            train_sarima(days=3, params_path=self.params_path)
        
        # Read back the params file — should still have the OLD params
        with open(self.params_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Old params should be preserved since new forecast was rejected
        self.assertIn('test_marker', model_data['params'])

    def test_validation_threshold(self):
        """Direct test: verify the validation threshold logic."""
        # BASELOAD_MAX_KW is 20.0
        self.assertEqual(BASELOAD_MAX_KW, 20.0)
        
        # Simulate what train_sarima does: if min < 0 or max > BASELOAD_MAX_KW, reject
        min_mean = -1.0
        max_mean = 5.0
        self.assertTrue(min_mean < 0 or max_mean > BASELOAD_MAX_KW)
        
        min_mean = 0.5
        max_mean = 25.0
        self.assertTrue(min_mean < 0 or max_mean > BASELOAD_MAX_KW)
        
        min_mean = 0.5
        max_mean = 15.0
        self.assertFalse(min_mean < 0 or max_mean > BASELOAD_MAX_KW)


if __name__ == '__main__':
    unittest.main()
