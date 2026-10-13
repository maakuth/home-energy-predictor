from __future__ import annotations

import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sarimax_predictor import BASELOAD_MAX_KW
from train_sarima import train_sarima


def _history() -> pd.Series:
    index = pd.date_range('2026-04-01', periods=288, freq='15min', tz='UTC')
    values = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(len(index)) / 96)
    return pd.Series(values, index=index, name='baseload_power')


def _results(mean: list[float]) -> MagicMock:
    results = MagicMock()
    forecast = MagicMock()
    forecast.predicted_mean = pd.Series(mean)
    results.get_forecast.return_value = forecast
    results.params = pd.Series({'ar.L1': 0.5, 'ma.L1': 0.1, 'ar.S.L96': 0.3, 'sigma2': 1.0})
    return results


class TestTrainSarimaSanityCheck:
    def test_saves_params_when_forecast_is_reasonable(self):
        params_path = os.environ['TEST_SARIMA_PARAMS']
        with patch('train_sarima.load_historical_data', return_value=_history()), \
             patch('train_sarima.SARIMAX') as sarimax:
            sarimax.return_value.fit.return_value = _results([0.2, 0.5, 1.2])
            train_sarima(days=3, params_path=params_path)

        with open(params_path, 'rb') as f:
            model_data = pickle.load(f)
        assert model_data['order'] == (1, 1, 1)
        assert model_data['seasonal_order'] == (1, 1, 0, 96)
        assert model_data['last_index'] == _history().index.max()
        sarimax.return_value.fit.assert_called_once_with(start_params=None, disp=False)

    def test_rejects_exploding_forecast_without_overwriting_old_params(self):
        params_path = os.environ['TEST_SARIMA_PARAMS']
        old_data = {'params': pd.Series({'test_marker': 42.0}), 'order': (1, 1, 1)}
        with open(params_path, 'wb') as f:
            pickle.dump(old_data, f)

        with patch('train_sarima.load_historical_data', return_value=_history()), \
             patch('train_sarima.SARIMAX') as sarimax:
            sarimax.return_value.fit.return_value = _results([BASELOAD_MAX_KW + 1])
            train_sarima(days=3, params_path=params_path)

        with open(params_path, 'rb') as f:
            retained = pickle.load(f)
        assert retained['order'] == old_data['order']
        assert retained['params']['test_marker'] == 42.0
        assert sarimax.return_value.fit.call_count == 1
        kwargs = sarimax.return_value.fit.call_args.kwargs
        assert kwargs['disp'] is False
        assert kwargs['start_params']['test_marker'] == 42.0

    def test_accepts_small_negative_forecast_that_inference_clips(self):
        params_path = os.environ['TEST_SARIMA_PARAMS']
        with patch('train_sarima.load_historical_data', return_value=_history()), \
             patch('train_sarima.SARIMAX') as sarimax:
            sarimax.return_value.fit.return_value = _results([-0.3, 0.5])
            train_sarima(days=3, params_path=params_path)

        assert os.path.exists(params_path)

    def test_rejects_forecast_below_negative_tolerance(self):
        params_path = os.environ['TEST_SARIMA_PARAMS']
        with patch('train_sarima.load_historical_data', return_value=_history()), \
             patch('train_sarima.SARIMAX') as sarimax:
            sarimax.return_value.fit.return_value = _results([-0.5001])
            train_sarima(days=3, params_path=params_path)
        assert not os.path.exists(params_path)
