"""Autonomous heating is forecast load, never a request to start a heater."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.uncontrolled_heating import load_uncontrolled_heating_forecast


@pytest.fixture
def uncontrolled(monkeypatch, tmp_path):
    for key in ('GSHP_OPTIMIZE_ENABLED', 'RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'BULK_HEATER_OPTIMIZE_ENABLED'):
        monkeypatch.setenv(key, 'false')
    path = tmp_path / 'raw.csv'
    index = pd.date_range('2026-11-01', periods=2 * 1440, freq='min', tz='UTC')
    pd.DataFrame({'gshp_power': [1000.0] * 1440 + [3000.0] * 1440,
                  'gshp_power_source_valid': True}, index=index).to_csv(path)
    return path, index[-1] + pd.Timedelta(minutes=1)


def test_forecast_uses_last_24_hours_and_converts_shared_watts(uncontrolled):
    path, start = uncontrolled
    result = load_uncontrolled_heating_forecast([start, start + pd.Timedelta(days=1)], path)
    np.testing.assert_allclose(result, [3.0, 3.0])


@pytest.mark.parametrize('control', ['GSHP_OPTIMIZE_ENABLED', 'RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'BULK_HEATER_OPTIMIZE_ENABLED'])
def test_managed_heating_is_not_counted_again(uncontrolled, monkeypatch, control):
    path, start = uncontrolled
    monkeypatch.setenv(control, 'true')
    with patch('utils.uncontrolled_heating.pd.read_csv', side_effect=AssertionError('must not read history')):
        np.testing.assert_allclose(load_uncontrolled_heating_forecast([start], path), [0.0])


def test_invalid_sources_and_future_observations_are_not_used(uncontrolled):
    path, start = uncontrolled
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.loc[start - pd.Timedelta(minutes=1), ['gshp_power', 'gshp_power_source_valid']] = [50000.0, False]
    frame.loc[start] = [50000.0, True]
    frame.to_csv(path)
    np.testing.assert_allclose(load_uncontrolled_heating_forecast([start], path), [3.0])


def test_missing_history_is_explicit_not_a_control_command(uncontrolled, capsys):
    path, start = uncontrolled
    result = load_uncontrolled_heating_forecast([start], path.with_name('absent.csv'))
    assert result.tolist() == [0.0]
    assert 'unavailable' in capsys.readouterr().out.lower()


@pytest.mark.parametrize('problem', ['all_invalid', 'missing_provenance', 'duplicate_timestamps'])
def test_unusable_history_is_not_treated_as_observed_heating(uncontrolled, capsys, problem):
    path, start = uncontrolled
    frame = pd.read_csv(path, index_col=0)
    if problem == 'all_invalid':
        frame['gshp_power_source_valid'] = False
    elif problem == 'missing_provenance':
        frame = frame.drop(columns='gshp_power_source_valid')
    else:
        frame = pd.concat([frame, frame.iloc[[-1]]])
    frame.to_csv(path)
    assert load_uncontrolled_heating_forecast([start], path).tolist() == [0.0]
    assert 'unavailable' in capsys.readouterr().out.lower()


@pytest.mark.parametrize('planner', ['none', 'nemotron-linprog', 'joint-linprog'])
def test_optimizer_includes_autonomous_heat_without_enabling_dispatch(uncontrolled, monkeypatch, tmp_path, planner):
    import optimize_plan

    path, start = uncontrolled
    monkeypatch.setenv('TEST_TMP_DIR', str(tmp_path))
    path.rename(tmp_path / 'raw_data.csv')
    monkeypatch.setenv('BATTERY_PLANNER_TYPE', planner)
    monkeypatch.setenv('HEPO_DISABLE_BATTERY', 'true' if planner == 'none' else 'false')
    monkeypatch.setenv('LEAF_OPTIMIZE_ENABLED', 'false')
    monkeypatch.setenv('LEAF_MAX_POWER_KW', '0')
    monkeypatch.setenv('PLAN_INTERVAL_MINUTES', '15')
    predictions = [{'timestamp': (start + pd.Timedelta(minutes=15 * i)).isoformat(),
                    'predicted_baseload': 2.0, 'solar_forecast': 0.0,
                    'outside_temp': 5.0, 'is_fallback_price': 0, 'ev_position': 0} for i in range(4)]
    monkeypatch.setattr(optimize_plan, 'fetch_market_prices', lambda *args: ([0.1] * 4, [0] * 4, 'test', False, False, None))
    monkeypatch.setattr(optimize_plan, 'get_ha_state', lambda entity: {'state': '50'})
    monkeypatch.setattr(optimize_plan, 'fetch_states_history', lambda *args, **kwargs: {})
    from pathlib import Path
    import os
    import sqlite3
    monkeypatch.setattr(optimize_plan, 'get_db_connection', lambda: sqlite3.connect(os.environ['TEST_DB_PATH']))
    Path(os.environ['TEST_PREDICTIONS_FILE']).write_text(json.dumps(predictions))

    optimize_plan.optimize()

    plan = json.loads(Path(os.environ['TEST_PLAN_FILE']).read_text())
    for entry in plan:
        assert entry['uncontrolled_heating_kw'] == 3.0
        assert entry['predicted_usage_kw'] == 5.0
        assert entry['planned_gshp_kw'] == 0.0
        assert entry['gshp_intent'] == 'STOP'
        assert entry['resistive_heater_intent'] == 'OFF'
        assert entry['bulk_heater_intent'] == 'OFF'
        net_kwh = entry['grid_import_kwh'] - entry['grid_export_kwh']
        battery_kwh = entry['charge_from_grid_kwh'] + entry['charge_from_solar_kwh'] - entry['discharge_to_load_kwh'] - entry['discharge_to_export_kwh']
        assert net_kwh == pytest.approx(5.0 * 0.25 + battery_kwh)
