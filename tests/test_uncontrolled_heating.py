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


def test_archive_missing_history_can_be_labelled_for_planning(uncontrolled):
    from unittest.mock import Mock
    history = Mock()
    history.cutoff = 100000
    history.raw.return_value = pd.DataFrame({
        'gshp_power': [float('nan')], 'gshp_power_source_valid': [False],
        'gshp_power_coverage_fraction': [0.], 'gshp_power_provenance': ['missing'],
        'gshp_power_conflict': [False]})
    history.eligible.return_value = pd.Series([False])
    status = {}
    result = load_uncontrolled_heating_forecast([uncontrolled[1]], history=history,
                                                allow_missing=True, status=status)
    assert result.tolist() == [0.0]
    assert status['source'] == 'unavailable'


@pytest.mark.parametrize('error', [ValueError('invalid policy'), KeyError('corrupt schema'),
                                   OSError('missing configuration'), __import__('sqlite3').DatabaseError('corrupt database')])
def test_archive_read_errors_are_not_missing_observations(uncontrolled, error):
    from unittest.mock import Mock
    history = Mock()
    history.cutoff = 100000
    history.raw.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        load_uncontrolled_heating_forecast([uncontrolled[1]], history=history,
                                            allow_missing=True, status={})


def test_archive_missing_quality_columns_is_not_missing_observation(uncontrolled):
    from unittest.mock import Mock
    history = Mock()
    history.cutoff = 100000
    history.raw.return_value = pd.DataFrame({'gshp_power': [float('nan')]})
    with pytest.raises(ValueError, match='quality columns'):
        load_uncontrolled_heating_forecast([uncontrolled[1]], history=history,
                                            allow_missing=True, status={})


def test_missing_telemetry_and_history_publish_labelled_unmanaged_plan(uncontrolled, monkeypatch, tmp_path, capsys):
    import optimize_plan
    import os
    from pathlib import Path
    import sqlite3
    _, start = uncontrolled
    monkeypatch.setenv('TEST_TMP_DIR', str(tmp_path))
    monkeypatch.setenv('BATTERY_PLANNER_TYPE', 'none')
    monkeypatch.setenv('HEPO_DISABLE_BATTERY', 'true')
    monkeypatch.setenv('LEAF_OPTIMIZE_ENABLED', 'false')
    predictions = [{'timestamp': (start + pd.Timedelta(minutes=15 * i)).isoformat(),
                    'predicted_baseload': 2.0, 'solar_forecast': 0.0,
                    'outside_temp': 5.0, 'is_fallback_price': 0, 'ev_position': 0} for i in range(4)]
    monkeypatch.setattr(optimize_plan, 'fetch_market_prices', lambda *args: ([0.1] * 4, [0] * 4, 'test', False, False, None))
    monkeypatch.setattr(optimize_plan, 'get_ha_state', lambda entity: None)
    monkeypatch.setattr(optimize_plan, 'fetch_states_history', lambda *args, **kwargs: {})
    monkeypatch.setattr(optimize_plan, 'get_db_connection', lambda: sqlite3.connect(os.environ['TEST_DB_PATH']))
    Path(os.environ['TEST_PREDICTIONS_FILE']).write_text(json.dumps(predictions))
    optimize_plan.optimize()
    plan = json.loads(Path(os.environ['TEST_PLAN_FILE']).read_text())
    assert all(p['heating_forecast_source'] == 'unavailable' for p in plan)
    assert all(p['gshp_compressor_state'] == 'unknown' for p in plan)
    output = capsys.readouterr().out
    assert 'UNKNOWN' in output
    assert '45.0°C, STOPPED' not in output
    assert all(p['gshp_intent'] == 'STOP' and p['resistive_heater_intent'] == 'OFF'
               and p['bulk_heater_intent'] == 'OFF' for p in plan)


@pytest.mark.parametrize('control', ['GSHP_OPTIMIZE_ENABLED', 'RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'BULK_HEATER_OPTIMIZE_ENABLED'])
def test_unknown_compressor_remains_forbidden_with_any_managed_heater(uncontrolled, monkeypatch, control):
    from optimize_plan import _resolve_gshp_compressor_state
    monkeypatch.setenv(control, 'true')
    with pytest.raises(RuntimeError, match='telemetry unavailable'):
        _resolve_gshp_compressor_state(None)


@pytest.mark.parametrize('control', ['GSHP_OPTIMIZE_ENABLED', 'RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'BULK_HEATER_OPTIMIZE_ENABLED'])
def test_unknown_compressor_never_publishes_with_managed_heater(uncontrolled, monkeypatch, control):
    import optimize_plan
    from unittest.mock import Mock
    from datetime import datetime, timezone
    monkeypatch.setenv(control, 'true')
    values = np.array([0.1])
    monkeypatch.setattr(optimize_plan, 'load_predictions', lambda: (
        [], values, [datetime.now(timezone.utc)], values, values, values, values, values))
    monkeypatch.setattr(optimize_plan, 'fetch_market_prices', lambda *args: (values, [False], None, False, False, values))
    monkeypatch.setattr(optimize_plan, 'build_tariff_prices', lambda *args, **kwargs: (values, values))
    monkeypatch.setattr(optimize_plan, 'get_gshp_initial_temp', lambda: 45.)
    monkeypatch.setattr(optimize_plan, 'get_ha_state', lambda entity: None)
    publish = Mock()
    monkeypatch.setattr(optimize_plan, 'publish_optimization', publish)
    with pytest.raises(RuntimeError, match='telemetry unavailable'):
        optimize_plan.optimize()
    publish.assert_not_called()


def test_unknown_compressor_does_not_latch_old_heater_on(uncontrolled, monkeypatch):
    import optimize_plan
    from datetime import datetime, timezone
    from pathlib import Path
    import os
    from utils.plan_time import slot_start
    now = slot_start(datetime.now(timezone.utc), 15).isoformat()
    path = Path(os.environ['TEST_PLAN_FILE'])
    path.write_text(json.dumps([{'timestamp': now, 'resistive_heater_intent': 'ON',
                                 'planned_resistive_kw': 6., 'bulk_heater_intent': 'ON',
                                 'planned_bulk_heater_kw': 6.}]))
    candidate = [{'timestamp': now, 'gshp_compressor_state': 'unknown',
                  'resistive_heater_intent': 'OFF', 'planned_resistive_kw': 0.,
                  'bulk_heater_intent': 'OFF', 'planned_bulk_heater_kw': 0.}]
    monkeypatch.setattr(optimize_plan, 'get_db_connection', lambda: (_ for _ in ()).throw(OSError('isolated')))
    optimize_plan.publish_optimization(candidate, history=None)
    row = json.loads(path.read_text())[0]
    assert row['resistive_heater_intent'] == row['bulk_heater_intent'] == 'OFF'
    assert row['planned_resistive_kw'] == row['planned_bulk_heater_kw'] == 0.


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
