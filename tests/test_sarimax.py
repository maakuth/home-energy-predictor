import unittest
import pandas as pd
import numpy as np
import os
import json
from sarimax_predictor import load_historical_data, predict_sarimax, save_benchmark_results

class TestSARIMAPredictor(unittest.TestCase):
    def setUp(self):
        # Use test-specific paths from environment (set by conftest.py)
        self.test_csv = os.getenv('TEST_SARIMA_CSV', 'test_processed_data.csv')
        self.test_json = os.getenv('TEST_SARIMA_FILE', 'test_sarimax_predictions.json')
        
        # Create directories if they don't exist (only if not in current directory)
        csv_dir = os.path.dirname(self.test_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        json_dir = os.path.dirname(self.test_json)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        
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
        # No cleanup needed - conftest.py handles it via tmp_path fixture
        pass

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
        # Use a non-existent params_path to force default model fit (isolated from production)
        fake_params = os.path.join(os.path.dirname(self.test_json), 'nonexistent_params.pkl')
        forecast_mean, forecast_ci = predict_sarimax(ts_data, forecast_steps=4, params_path=fake_params)
        
        self.assertIsNotNone(forecast_mean)
        self.assertEqual(len(forecast_mean), 4)
        self.assertTrue(all(forecast_mean >= 0))
        self.assertIsNotNone(forecast_ci)

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

    def test_predict_sarimax_fallback_from_bad_params(self):
        """If saved params produce exploding CIs, predictor should refit."""
        import pickle
        import tempfile
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Create a pickle with the bad params observed in production
        bad_params = pd.Series({
            'ar.L1': 0.562383,
            'ma.L1': -0.776564,
            'ar.S.L96': -0.621604,
            'sigma2': 0.205888
        })
        model_data = {
            'params': bad_params,
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 0, 96),
            'last_index': pd.Timestamp('2026-05-31 05:45:00+0000')
        }

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model_data, f)
            bad_pickle = f.name

        try:
            # Use real production data if available, otherwise synthetic
            ts_data = load_historical_data('processed_data.csv', last_n_days=3)
            if ts_data is None or len(ts_data) < 100:
                ts_data = load_historical_data(self.test_csv, last_n_days=3)

            # Demonstrate that smooth() with bad params gives exploding CIs on real data
            model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 96),
                           enforce_stationarity=False, enforce_invertibility=False)
            results_smooth = model.smooth(bad_params)
            ci_smooth = results_smooth.get_forecast(steps=10).conf_int(alpha=0.05)
            max_smooth_upper = ci_smooth.iloc[:, 1].max()
            # This confirms the numerical bug exists on real data (>1000 kW upper CI)
            # On synthetic data it may not reproduce; skip assertion if so
            if max_smooth_upper > 1000:
                # predict_sarimax should handle this gracefully
                forecast_mean, forecast_ci = predict_sarimax(ts_data, forecast_steps=10, params_path=bad_pickle)

                # Mean should be reasonable (home baseload < 50 kW)
                self.assertTrue(all(forecast_mean < 50))
                self.assertTrue(all(forecast_mean >= 0))

                # CI should also be reasonable after fallback
                max_upper = forecast_ci.iloc[:, 1].max()
                self.assertLess(max_upper, 100, f"Upper CI {max_upper:.1f} kW is unreasonably high for home baseload")
            else:
                # Synthetic data doesn't trigger the bug; just verify predict_sarimax returns reasonable values
                forecast_mean, forecast_ci = predict_sarimax(ts_data, forecast_steps=10, params_path=bad_pickle)
                self.assertTrue(all(forecast_mean < 50))
                max_upper = forecast_ci.iloc[:, 1].max()
                self.assertLess(max_upper, 100)
        finally:
            os.unlink(bad_pickle)

    def test_save_benchmark_results_clamps_exploding_ci(self):
        """If forecast CIs are unreasonably large, they should be clamped."""
        ts = pd.date_range(start='2026-04-04', periods=3, freq='15min', tz='UTC')
        mock_forecast = pd.Series([1.2, 1.3, 1.4], index=ts)
        # Mock exploding CI: upper bound in thousands
        mock_ci = pd.DataFrame({
            'lower baseload_power': [0.5, 0.6, 0.7],
            'upper baseload_power': [1.5, 1500.0, 2000.0]
        }, index=ts)

        save_benchmark_results(mock_forecast, mock_ci, filename=self.test_json)

        with open(self.test_json, 'r') as f:
            data = json.load(f)

        # All upper bounds should be clamped to reasonable values
        for row in data:
            self.assertLess(row['upper_95'], 100,
                           f"Upper CI {row['upper_95']} should be clamped for baseload")
            self.assertGreaterEqual(row['upper_95'], row['predicted_baseload'])

if __name__ == '__main__':
    unittest.main()
