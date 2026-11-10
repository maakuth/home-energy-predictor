from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from analyze_performance import fetch_actuals
from utils.archive_history import ArchiveHistory
from utils.archive_quality import METERS, measurement_eligibility


def history_frame():
    index = pd.date_range('2026-10-08', periods=31, freq='min', tz='UTC')
    raw = pd.DataFrame(index=index)
    for name, value in zip(METERS, [5, 2, 1000, 2000, 500]):
        raw[name] = value
        raw[name + '_source_valid'] = True
        raw[name + '_coverage_fraction'] = 1.0
        raw[name + '_provenance'] = 'recorder'
        raw[name + '_conflict'] = False
    history = Mock(spec=ArchiveHistory)
    history.cutoff = index[-1].timestamp() + 60
    history.legacy_policy = 'verified-only'
    history.raw.return_value = raw
    history.eligible.side_effect = lambda frame, name: measurement_eligibility(frame, name, history.legacy_policy)
    return history, raw


def test_actuals_full_aligned_bins_and_original_meters(monkeypatch):
    history, raw = history_frame()
    raw.loc[raw.index[16], 'battery_power_coverage_fraction'] = 0
    monkeypatch.setattr('analyze_performance.fetch_states_history', Mock(side_effect=AssertionError('PG forbidden')))
    result = fetch_actuals(1, history=history).loc[raw.index[0]:]
    assert result.iloc[0].actual_usage == 6
    assert result.iloc[0].baseload_actual_kw == 3.5
    assert np.isnan(result.iloc[1].actual_usage)
    assert np.isnan(result.iloc[2].actual_usage)
    assert result.iloc[0].actuals_source_verified


def test_legacy_eligible_is_not_verified_and_archive_failure_propagates():
    history, raw = history_frame()
    raw['leaf_power_source_valid'] = False
    raw['leaf_power_provenance'] = 'legacy_resampled'
    history.legacy_policy = 'complete-legacy'
    result = fetch_actuals(1, history=history).loc[raw.index[0]:]
    assert result.iloc[0].baseload_actual_kw == 3.5
    assert not result.iloc[0].baseload_source_verified
    history.raw.side_effect = ValueError('bad archive')
    with pytest.raises(ValueError, match='bad archive'):
        fetch_actuals(1, history=history)


def test_removal_day_is_uncertain_not_october_six():
    from utils.heating_benchmark import heating_regimes
    index = pd.DatetimeIndex(['2026-10-06T12:00Z', '2026-10-07T04:00Z',
                              '2026-10-07T06:00Z', '2026-10-08T00:00Z'])
    assert heating_regimes(index).tolist() == [
        'old_intermittent', 'removal_transition', 'removal_transition', 'resistance_only']


def test_hindsight_archive_errors_are_not_swallowed_or_csv_fallback(monkeypatch):
    from analyze_performance import backtest_current_model
    history, raw = history_frame()
    history.features.side_effect = ValueError('archive unavailable')
    monkeypatch.setattr(pd, 'read_csv', Mock(side_effect=AssertionError('CSV forbidden')))
    with pytest.raises(ValueError, match='archive unavailable'):
        backtest_current_model(raw, history=history)


def test_solar_and_battery_diagnostics_use_explicit_archive(monkeypatch):
    from utils.solar_calibration import fetch_actual_solar
    from dump_battery_data import fetch_ha_measurements
    history, raw = history_frame()
    raw['outside_temp'] = 5.0
    for suffix, value in [('_source_valid', True), ('_coverage_fraction', 1.0),
                          ('_provenance', 'recorder'), ('_conflict', False)]:
        raw['outside_temp' + suffix] = value
    monkeypatch.setattr('utils.solar_calibration.fetch_states_history', Mock(side_effect=AssertionError('PG forbidden')))
    actual = fetch_actual_solar(1, history=history).loc[raw.index[0]:]
    assert actual.iloc[0].solar_actual_kw == 2
    assert pd.isna(actual.iloc[-1].solar_actual_kw)
    records = fetch_ha_measurements(raw.index[0].to_pydatetime(),
        pd.to_datetime(history.cutoff, unit='s', utc=True).to_pydatetime(), history=history)
    assert records[0]['gshp_power_kw'] == 2
    assert records[0]['source_verified']['gshp_power']
    assert 'gshp_power_kw' not in records[-1]


def test_fireplace_requires_explicit_archive_window():
    from estimate_fireplace_savings import load_diagnostic_features
    history, _ = history_frame()
    with pytest.raises(ValueError, match='positive'):
        load_diagnostic_features(None, history=history)
    load_diagnostic_features(2, history=history)
    history.features.assert_called_once_with(history.cutoff - 2 * 86400, history.cutoff)


def test_negative_minute_target_does_not_become_valid_after_averaging():
    history, raw = history_frame()
    raw.loc[raw.index[0], 'total_power'] = -5
    actuals = fetch_actuals(1, history=history)
    assert pd.isna(actuals.loc[raw.index[0], 'baseload_actual_kw'])


def test_sqlite_predictions_share_archive_cutoff(isolated_test_env, monkeypatch):
    import sqlite3
    from analyze_performance import get_archived_predictions
    from analyze_evolution import fetch_all_predictions
    from compare_models import get_archived_xgboost_predictions
    monkeypatch.setattr('utils.sqlite_utils.DB_PATH', isolated_test_env['db_path'])
    history, raw = history_frame()
    target = raw.index[0].isoformat()
    before = raw.index[0].isoformat()
    after = pd.to_datetime(history.cutoff + 60, unit='s', utc=True).isoformat()
    with sqlite3.connect(isolated_test_env['db_path']) as db:
        db.execute('''CREATE TABLE predictions(target_timestamp TEXT, generated_at TEXT,
            predicted_usage_kw REAL, solar_forecast_kw REAL, is_fallback_price INTEGER,
            version TEXT, battery_action TEXT, grid_import_kwh REAL, import_price REAL,
            grid_export_kwh REAL, export_price REAL)''')
        for generated, prediction in [(before, 6), ('2026-10-08T01:00:00+03:00', 0), (after, 99)]:
            db.execute('INSERT INTO predictions VALUES (?, ?, ?, 2, 0, ?, ?, 0, .1, 0, .05)',
                       (target, generated, prediction, 'test', 'idle'))
    assert get_archived_predictions(history=history).iloc[0].predicted_usage == 6
    assert fetch_all_predictions(1, history=history).predicted_usage_kw.tolist() == [6, 0]
    assert get_archived_xgboost_predictions(1, history=history).iloc[0].predicted_baseload == 6


def test_solar_unknown_actuals_are_not_zero_shortfall_or_calibration_samples():
    from utils.solar_calibration import analyze, daily_shortfall_kwh
    index = pd.date_range('2026-10-08', periods=2, freq='D', tz='UTC')
    frame = pd.DataFrame({'solar_forecast_kw': [2.0, 2.0],
                          'solar_actual_kw': [1.0, np.nan]}, index=index)
    assert analyze(frame)['n_intervals'] == 1
    shortfall = daily_shortfall_kwh(frame)
    assert len(shortfall) == 1
    assert shortfall.iloc[0] == .25


def test_archive_hindsight_uses_training_policy_without_processed_csv(tmp_path, monkeypatch):
    import json
    from analyze_performance import backtest_current_model
    monkeypatch.chdir(tmp_path)
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'energy_model.json').write_text('{}')
    (state / 'model_features.json').write_text(json.dumps(['outside_temp']))
    history, raw = history_frame()
    frame = pd.DataFrame({'outside_temp': 5.0, 'baseload_power': 3.5}, index=raw.index)
    required = ['baseload', 'baseload_lag_1h', 'baseload_lag_24h', 'leaf_power_lag_1h',
                'leaf_energy_24h', 'outside_temp', 'wind_speed', 'solar_forecast']
    for name in required:
        frame[name + '_training_eligible'] = True
    frame.attrs['archive_training'] = {'contract_version': 1, 'legacy_policy': 'complete-legacy'}
    frame.loc[frame.index[16], 'outside_temp_training_eligible'] = False
    history.features.return_value = frame
    model = Mock()
    model.predict.side_effect = lambda features: np.full(len(features), 3.5)
    monkeypatch.setattr('analyze_performance.xgb.XGBRegressor', lambda: model)
    monkeypatch.setattr(pd, 'read_csv', Mock(side_effect=AssertionError('CSV forbidden')))
    actuals = pd.DataFrame({'baseload_actual_kw': 3.5}, index=raw.index[::15])
    result = backtest_current_model(actuals, history=history)
    assert result.hindsight_usage.tolist() == [3.5]
    history.features.assert_called_once_with(raw.index[0].timestamp(), history.cutoff)


def test_archive_export_never_contacts_ha_or_synthesizes(tmp_path, monkeypatch):
    import pickle
    import dump_battery_data as dump
    history, raw = history_frame()
    raw['outside_temp'] = 5
    for suffix, value in [('_source_valid', True), ('_coverage_fraction', 1),
                          ('_provenance', 'recorder'), ('_conflict', False)]:
        raw['outside_temp' + suffix] = value
    monkeypatch.setattr(dump, 'fetch_ha_snapshot', Mock(side_effect=AssertionError('HA forbidden')))
    monkeypatch.setattr(dump, 'fetch_predictions', Mock(side_effect=AssertionError('current file forbidden')))
    monkeypatch.setattr(dump, 'fetch_nordpool_prices_from_ha', Mock(side_effect=AssertionError('PG forbidden')))
    monkeypatch.setattr(dump, 'synthesize_predictions', Mock(side_effect=AssertionError('synthetic data forbidden')))
    monkeypatch.setattr(dump, 'fetch_sqlite_predictions', lambda *args, **kwargs: [])
    output = tmp_path / 'export.pkl'
    assert dump.dump_battery_data(days=1, output=str(output), history=history)
    with output.open('rb') as stream:
        exported = pickle.load(stream)
    assert exported['ha_states'] == {}
    assert exported['predictions'] == []
    assert exported['market_prices'] == []
    assert exported['metadata']['measurement_archive_cutoff'] == history.cutoff
    assert 'predictions_archive' not in exported['history']


def test_archive_comparison_keeps_total_and_baseload_targets_distinct(tmp_path, monkeypatch):
    import csv
    import compare_models
    history, raw = history_frame()
    index = raw.index[:30:15]
    monkeypatch.setattr(compare_models, 'load_actual_baseload', Mock(side_effect=AssertionError('CSV forbidden')))
    monkeypatch.setattr(compare_models, 'get_archived_xgboost_predictions',
        lambda **kwargs: pd.DataFrame({'predicted_baseload': [6.0, 6.0]}, index=index))
    monkeypatch.setattr(compare_models, 'get_archived_sarima_predictions',
        lambda **kwargs: pd.DataFrame({'predicted_baseload': [3.5, 3.5]}, index=index))
    output = tmp_path / 'comparison.csv'
    compare_models.compare(1, output_csv=str(output), show_blend=True, history=history)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['model'] for row in rows] == ['xgb', 'sarima']
    assert [float(row['mae']) for row in rows] == [0.0, 0.0]
    assert [row['target'] for row in rows] == ['total_load', 'baseload']
