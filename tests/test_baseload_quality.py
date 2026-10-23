import pandas as pd
import numpy as np

from utils.baseload_quality import baseload_targets, BASELOAD_FEATURES, select_training_rows


def data():
    return pd.DataFrame({'total_power': 4.0, 'solar_actual': 1.0,
                         'battery_power': 1000.0, 'gshp_power': 1000.0,
                         'leaf_power': 1000.0},
                        index=pd.date_range('2026-10-07', periods=2000, freq='min', tz='UTC'))


def test_shared_meter_and_signed_battery_accounting():
    raw = data()
    raw['resistive_heater_power'] = 6000.0
    result = baseload_targets(raw)
    assert result.baseload_power.eq(2.0).all()
    assert result.baseload_target_valid.all()


def test_missing_not_zero_and_invalid_lags_not_forward_filled():
    raw = data()
    raw.loc[raw.index[100], 'gshp_power'] = np.nan
    result = baseload_targets(raw)
    assert not result.iloc[100].baseload_target_valid
    assert 'gshp_power' in result.iloc[100].baseload_exclusion_reason
    assert pd.isna(result.iloc[160].baseload_lag_1h)
    assert pd.isna(result.iloc[1540].baseload_lag_24h)
    assert pd.notna(result.iloc[161].baseload_lag_1h)


def test_negative_baseload_is_invalid_not_clipped_into_truth():
    raw = data()
    raw['total_power'] = -10.0
    result = baseload_targets(raw)
    assert not result.baseload_target_valid.any()
    assert result.baseload_power.isna().all()


def test_short_heater_pulse_cancels_without_median_filter():
    raw = data()
    raw.iloc[50:55, raw.columns.get_loc('gshp_power')] += 6000
    raw.iloc[50:55, raw.columns.get_loc('total_power')] += 6
    assert baseload_targets(raw).baseload_power.eq(2.0).all()


def test_missing_meter_column_is_unknown_and_no_thermal_feature_required():
    result = baseload_targets(data().drop(columns='battery_power'))
    assert not result.baseload_target_valid.any()
    assert not any('temp' in c and c != 'outside_temp' for c in BASELOAD_FEATURES)
    assert 'is_fireplace_lag1' not in BASELOAD_FEATURES


def test_lags_follow_clock_time_not_number_of_surviving_rows():
    raw = data().drop(data().index[100:120])
    result = baseload_targets(raw)
    assert pd.isna(result.loc[raw.index[0] + pd.Timedelta(minutes=160), 'baseload_lag_1h'])


def test_selection_rejects_invalid_targets_and_legacy_metadata():
    import pytest
    result = baseload_targets(data())
    result.iloc[-1, result.columns.get_loc('baseload_target_valid')] = False
    selected = select_training_rows(result)
    assert result.index[-1] not in selected.index
    assert selected.index.min() == result.index[1440]
    with pytest.raises(ValueError, match='metadata'):
        select_training_rows(result.drop(columns='baseload_target_valid'))


def test_processing_keeps_quality_reasons_and_does_not_require_tank(tmp_path):
    from process_data import process_data
    raw = data()
    raw['outside_temp'] = 5.0
    raw['solar_forecast'] = 5.0
    raw.iloc[100, raw.columns.get_loc('gshp_power')] = np.nan
    source, output = tmp_path / 'raw.csv', tmp_path / 'processed.csv'
    raw.to_csv(source)
    process_data(str(source), str(output))
    result = pd.read_csv(output)
    assert len(result) == len(raw)
    assert not result.iloc[100].baseload_target_valid
    assert pd.isna(result.iloc[100].baseload_power)
    assert pd.isna(result.iloc[160].baseload_lag_1h)


def test_training_schema_is_available_at_inference():
    from predict_future import generate_inference_data
    from datetime import datetime, timezone
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    solar = pd.DataFrame({'pv_estimate': [0.0]}, index=pd.DatetimeIndex([now]))
    rows, _ = generate_inference_data(now, now, 15, solar, pd.DataFrame(), {}, {})
    assert set(BASELOAD_FEATURES).issubset(rows[0])


def test_invalid_source_state_cannot_be_hidden_by_extraction_fill():
    raw = data()
    raw['gshp_power_source_valid'] = True
    raw.loc[raw.index[100], 'gshp_power_source_valid'] = False
    assert not baseload_targets(raw).iloc[100].baseload_target_valid


def test_sarima_does_not_fill_invalid_or_partial_quarters(tmp_path):
    from sarimax_predictor import load_historical_data
    raw = data()
    result = baseload_targets(raw)
    result.loc[result.index[15:30], 'baseload_target_valid'] = False
    result.loc[result.index[40], 'baseload_power'] = np.nan
    path = tmp_path / 'processed.csv'
    result.to_csv(path)
    loaded = load_historical_data(str(path), last_n_days=3)
    assert loaded is not None
    assert pd.isna(loaded.iloc[1])
    assert pd.isna(loaded.iloc[2])
