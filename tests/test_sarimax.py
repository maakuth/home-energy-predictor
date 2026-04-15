import unittest
import pandas as pd
import numpy as np
import os
import json
from sarimax_predictor import load_historical_data, predict_sarimax, save_benchmark_results

class TestSARIMAPredictor(unittest.TestCase):
    def setUp(self):
        self.test_csv = 'test_processed_data.csv'
        self.test_json = 'test_sarimax_predictions.json'
        
        # Create dummy historical data (3 days of 15-min intervals)
        # 3 * 24 * 4 = 288 points
        timestamps = pd.date_range(start='2026-04-01', periods=288, freq='15min', tz='UTC')
        # Create a daily cycle + some noise
        # 96 points per day
        base = 1.0
        cycle = 0.5 * np.sin(2 * np.pi * np.arange(288) / 96)
        noise = 0.1 * np.random.randn(288)
        baseload = base + cycle + noise
        
        df = pd.DataFrame({'baseload_power': baseload}, index=timestamps)
        df.to_csv(self.test_csv)

    def tearDown(self):
        for f in [self.test_csv, self.test_json]:
            if os.path.exists(f):
                os.remove(f)

    def test_load_historical_data(self):
        ts_data = load_historical_data(self.test_csv, last_n_days=2)
        # Should be roughly 2 days of 15-min points (96 * 2 = 192)
        self.assertIsNotNone(ts_data)
        # The first point of last_n_days might depend on exact timing, but should be > 190
        self.assertGreaterEqual(len(ts_data), 190)
        self.assertIsInstance(ts_data.index, pd.DatetimeIndex)

    def test_predict_sarimax(self):
        # Load all 3 days
        ts_data = load_historical_data(self.test_csv, last_n_days=3)
        # Forecast for 1 hour (4 steps)
        # Note: fitting might be slow, but 288 points should be fine.
        forecast = predict_sarimax(ts_data, forecast_steps=4)
        
        self.assertIsNotNone(forecast)
        self.assertEqual(len(forecast), 4)
        self.assertTrue(all(forecast >= 0))

    def test_save_benchmark_results(self):
        # Mock forecast
        ts = pd.date_range(start='2026-04-04', periods=2, freq='15min', tz='UTC')
        mock_forecast = pd.Series([1.2, 1.3], index=ts)
        
        save_benchmark_results(mock_forecast, filename=self.test_json)
        
        self.assertTrue(os.path.exists(self.test_json))
        with open(self.test_json, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]['predicted_baseload'], 1.2)
            self.assertEqual(data[0]['model'], 'SARIMA')

if __name__ == '__main__':
    unittest.main()
