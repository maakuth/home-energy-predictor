"""Evaluation must use the configured meter and preserve unknown readings."""
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest
from analyze_performance import fetch_actuals


def test_actuals_use_configured_shared_heating_meter_without_double_counting(monkeypatch):
    monkeypatch.setenv('GSHP_POWER_ENTITY', 'sensor.new_heat_meter')
    index = pd.date_range('2026-11-01', periods=2, freq='15min', tz='UTC')
    history = {
        'sensor.sahkokauppa_nyt': pd.DataFrame({'state': [5.0, 5.0]}, index=index),
        'sensor.solarh_63038_real_power_kw': pd.DataFrame({'state': [0.0, 0.0]}, index=index),
        'sensor.be_stat_batt_power': pd.DataFrame({'state': [0.0, 0.0]}, index=index),
        'sensor.new_heat_meter': pd.DataFrame({'state': [4000.0, 4000.0]}, index=index),
        'switch.mlp_vastus_output_0': pd.DataFrame({'state': ['on', 'on']}, index=index),
    }
    with patch('analyze_performance.fetch_states_history', return_value=history) as fetch:
        actuals = fetch_actuals(days=1)

    assert 'sensor.new_heat_meter' in fetch.call_args.args[0]
    assert 'sensor.mlp_teho' not in fetch.call_args.args[0]
    assert actuals.gshp_actual_kw.tolist() == [4.0, 4.0]


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf, 'unavailable'])
def test_invalid_reading_invalidates_its_quarter_not_just_the_sample(bad_value):
    index = pd.date_range('2026-11-01', periods=30, freq='min', tz='UTC')
    history = {
        entity: pd.DataFrame({'state': [value] * 30}, index=index)
        for entity, value in [('sensor.sahkokauppa_nyt', 2.0),
                              ('sensor.solarh_63038_real_power_kw', 0.0),
                              ('sensor.mlp_teho', 1000.0),
                              ('sensor.be_stat_batt_power', 0.0)]
    }
    history['sensor.be_stat_batt_power'] = history['sensor.be_stat_batt_power'].astype(object)
    history['sensor.be_stat_batt_power'].iloc[4, 0] = bad_value
    with patch('analyze_performance.fetch_states_history', return_value=history) as fetch:
        actuals = fetch_actuals(days=1)

    assert fetch.call_args.kwargs['preserve_invalid_states'] is True
    assert pd.isna(actuals.actual_usage.iloc[0])
    assert actuals.actual_usage.iloc[1] == 2.0


def test_history_reader_can_preserve_invalid_source_states():
    from utils.db_utils import fetch_states_history
    from unittest.mock import MagicMock

    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.fetchall.side_effect = [
        [(1, 'sensor.meter')],
        [(1, 1793491200.0, '3.0'), (1, 1793491260.0, 'unavailable')],
    ]
    with patch('utils.db_utils.get_db_connection', return_value=connection):
        result = fetch_states_history('sensor.meter', preserve_invalid_states=True)
    assert len(result['sensor.meter']) == 2
    assert pd.isna(result['sensor.meter'].state.iloc[1])
