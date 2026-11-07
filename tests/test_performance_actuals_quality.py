"""Evaluation must use the configured meter and preserve unknown readings."""
from unittest.mock import patch

import pandas as pd
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
