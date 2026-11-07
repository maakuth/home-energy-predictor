from unittest.mock import patch, MagicMock
import json

import numpy as np
import pandas as pd

from analyze_performance import fetch_actuals, backtest_current_model, backtest_sarima_model


def test_observed_baseload_excludes_shared_heating_and_leaf():
    index = pd.date_range('2026-11-01', periods=2, freq='15min', tz='UTC')
    history = {
        name: pd.DataFrame({'state': [value] * 2}, index=index)
        for name, value in [('sensor.sahkokauppa_nyt', 5.0),
                            ('sensor.solarh_63038_real_power_kw', 1.0),
                            ('sensor.mlp_teho', 2000.0),
                            ('sensor.be_stat_batt_power', 1000.0),
                            ('sensor.tasmota_energy_power_3', 1500.0)]
    }
    with patch('analyze_performance.fetch_states_history', return_value=history):
        actual = fetch_actuals(days=1)
    np.testing.assert_allclose(actual.baseload_actual_kw, 1.5)


def test_sarima_does_not_score_total_usage_or_unknown_baseload():
    index = pd.date_range('2026-11-01', periods=2, freq='15min', tz='UTC')
    observed = pd.DataFrame({'actual_usage': [7.0, 8.0],
                             'baseload_actual_kw': [2.0, np.nan]}, index=index)
    forecast = pd.DataFrame({'predicted_baseload': [2.0, 2.0]}, index=index)
    with patch('compare_models.get_archived_sarima_predictions', return_value=forecast):
        result = backtest_sarima_model(observed)
    assert len(result) == 1
    assert result.actual_usage.iloc[0] == 2.0


def test_xgboost_hindsight_averages_complete_quarters_without_adding_heating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'energy_model.json').write_text('{}')
    (state / 'model_features.json').write_text(json.dumps(['outside_temp']))
    index = pd.date_range('2026-11-01', periods=30, freq='min', tz='UTC')
    frame = pd.DataFrame({'outside_temp': np.arange(30),
                          'gshp_power': 6000.0,
                          'baseload_power': 2.0,
                          'baseload_target_valid': True,
                          'baseload_lag_1h': 2.0,
                          'baseload_lag_24h': 2.0}, index=index)
    frame.loc[index[-1], 'baseload_target_valid'] = False
    frame.to_csv(state / 'processed_data.csv')
    actuals = pd.DataFrame({'baseload_actual_kw': [2.0, 2.0]}, index=index[::15])
    model = MagicMock()
    model.predict.side_effect = lambda features: features.outside_temp.to_numpy()
    with patch('analyze_performance.xgb.XGBRegressor', return_value=model):
        result = backtest_current_model(actuals)
    assert len(result) == 1
    assert result.hindsight_usage.iloc[0] == 7.0
