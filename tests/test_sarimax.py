from __future__ import annotations

import os
import pickle
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from sarimax_predictor import _get_safe_forecast, load_historical_data, predict_sarimax, save_benchmark_results


def _series(days: int = 3) -> pd.Series:
    index = pd.date_range('2026-04-01', periods=days * 96, freq='15min', tz='UTC')
    n = np.arange(len(index))
    values = 1.0 + 0.5 * np.sin(2 * np.pi * n / 96) + 0.15 * np.sin(4 * np.pi * n / 96)
    return pd.Series(values, index=index, name='baseload_power')


def _forecast(mean: float = 1.0, upper: float = 2.0) -> MagicMock:
    forecast = MagicMock()
    forecast.predicted_mean = pd.Series([mean, mean])
    forecast.conf_int.return_value = pd.DataFrame({'lower': [0.0, 0.0], 'upper': [upper, upper]})
    return forecast


class TestSARIMAPredictor:
    def test_load_historical_data(self):
        path = os.environ['TEST_SARIMA_CSV']
        _series().to_frame().to_csv(path)
        loaded = load_historical_data(path, last_n_days=2)
        assert loaded is not None
        assert len(loaded) >= 190
        assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_warm_start_explosion_retries_clean_fit(self):
        model = MagicMock()
        bad = _forecast(1.0, 101.0)
        good = _forecast(1.0, 2.0)
        model.fit.side_effect = [MagicMock(get_forecast=MagicMock(return_value=bad)),
                                 MagicMock(get_forecast=MagicMock(return_value=good))]

        assert _get_safe_forecast(model, 2, start_params=np.array([0.1])) is good
        assert model.fit.call_args_list == [call(start_params=np.array([0.1]), disp=False), call(disp=False)]

    def test_warm_start_exception_retries_clean_fit(self):
        model = MagicMock()
        good = _forecast()
        model.fit.side_effect = [RuntimeError('bad params'), MagicMock(get_forecast=MagicMock(return_value=good))]

        assert _get_safe_forecast(model, 2, start_params=np.array([0.1])) is good
        assert model.fit.call_args_list == [call(start_params=np.array([0.1]), disp=False), call(disp=False)]

    def test_unreasonable_clean_fit_returns_none(self):
        model = MagicMock()
        model.fit.return_value.get_forecast.return_value = _forecast(21.0, 22.0)
        assert _get_safe_forecast(model, 2) is None

    def test_predict_uses_flat_fallback_after_all_fits_fail(self):
        params_path = os.environ['TEST_SARIMA_PARAMS']
        with open(params_path, 'wb') as f:
            pickle.dump({'params': np.array([0.1]), 'order': (1, 1, 1), 'seasonal_order': (1, 1, 0, 96)}, f)
        with patch('sarimax_predictor.SARIMAX') as sarimax:
            sarimax.return_value.fit.side_effect = RuntimeError('cannot fit')
            mean, ci = predict_sarimax(_series(), forecast_steps=4, params_path=params_path)
        assert len(mean) == len(ci) == 4
        assert (mean == _series().mean()).all()
        assert mean.index.freqstr == '15min'
        assert np.isfinite(ci.to_numpy()).all()
        assert sarimax.return_value.fit.call_count == 2

    @pytest.mark.slow
    def test_real_sarima_forecast_beats_flat_baseline_on_high_signal_period(self):
        """A fitted seasonal model must beat the historical mean on an unseen day."""
        all_data = _series(days=4)
        train, actual = all_data.iloc[:288], all_data.iloc[288:]
        safe_results = []

        def record_safe_forecast(*args, **kwargs):
            result = _get_safe_forecast(*args, **kwargs)
            safe_results.append(result)
            return result

        with patch('sarimax_predictor._get_safe_forecast', side_effect=record_safe_forecast) as safe:
            predicted, ci = predict_sarimax(train, forecast_steps=96,
                                            params_path=os.path.join(os.environ['TEST_TMP_DIR'], 'missing.pkl'))
        assert safe.call_count == 1
        assert len(safe_results) == 1
        assert safe_results[0] is not None, "real fit unexpectedly used flat safety fallback"
        assert len(predicted) == len(actual) == len(ci) == 96
        mae = float(np.mean(np.abs(predicted.to_numpy() - actual.to_numpy())))
        baseline_mae = float(np.mean(np.abs(train.mean() - actual.to_numpy())))
        assert mae < 0.20
        assert mae < baseline_mae * 0.5
        assert abs(float((predicted.to_numpy() - actual.to_numpy()).mean())) < 0.10

    def test_save_benchmark_results_clamps_exploding_ci(self):
        path = os.environ['TEST_SARIMA_FILE']
        ts = pd.date_range('2026-04-04', periods=3, freq='15min', tz='UTC')
        mean = pd.Series([1.2, 1.3, 1.4], index=ts)
        ci = pd.DataFrame({'lower baseload_power': [0.5, 0.6, 0.7],
                           'upper baseload_power': [1.5, 1500.0, 2000.0]}, index=ts)
        save_benchmark_results(mean, ci, filename=path)
        data = pd.read_json(path)
        assert (data['upper_95'] < 100).all()
        assert (data['upper_95'] >= data['predicted_baseload']).all()
