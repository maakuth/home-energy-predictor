import pandas as pd

from process_data import process_data, process_frame


def raw_frame():
    index = pd.date_range('2026-10-21', periods=100, freq='15min', tz='UTC', name='timestamp')
    return pd.DataFrame({'outside_temp': 5.0, 'wind_speed': 2.0,
                         'total_power': 3.0, 'gshp_power': 1000.0,
                         'leaf_power': 0.0, 'battery_power': 0.0,
                         'solar_actual': 0.0, 'solar_forecast': 0.0}, index=index)


def test_frame_processing_matches_csv_path_and_does_not_mutate_input(tmp_path):
    raw = raw_frame()
    original = raw.copy(deep=True)
    input_path, output_path = tmp_path / 'raw.csv', tmp_path / 'processed.csv'
    raw.to_csv(input_path)
    expected = process_frame(raw)
    process_data(str(input_path), str(output_path))
    actual = pd.read_csv(output_path, index_col=0)
    actual.index = pd.to_datetime(actual.index, utc=True)
    # CSV interprets empty exclusion-reason strings as NaN.
    actual['baseload_exclusion_reason'] = actual.baseload_exclusion_reason.fillna('')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_freq=False)
    pd.testing.assert_frame_equal(raw, original)


def test_frame_processing_keeps_invalid_meter_truth_and_time_aware_lags():
    raw = raw_frame()
    raw.loc[raw.index[0], 'battery_power'] = float('nan')
    result = process_frame(raw)
    assert not result.baseload_target_valid.iloc[0]
    assert 'battery_power;' in result.baseload_exclusion_reason.iloc[0]
    assert pd.isna(result.baseload_lag_24h.iloc[96])
    assert result.baseload_lag_24h.iloc[97] == 2.0
