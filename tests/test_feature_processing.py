"""Characterization/regression tests for isolated feature processing."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.feature_processing import (
    process_frame,
    diagnose_feature_coverage,
    check_feature_coverage,
    feature_coverage,
)


def raw_frame():
    index = pd.date_range('2026-10-21', periods=100, freq='15min', tz='UTC', name='timestamp')
    return pd.DataFrame({'outside_temp': 5.0, 'wind_speed': 2.0,
                         'total_power': 3.0, 'gshp_power': 1000.0,
                         'leaf_power': 0.0, 'battery_power': 0.0,
                         'solar_actual': 0.0, 'solar_forecast': 0.0}, index=index)


def test_imports_are_single_implementation():
    import process_data
    import utils.feature_processing as fp
    assert process_data.process_frame is fp.process_frame
    assert process_data.diagnose_feature_coverage is fp.diagnose_feature_coverage
    assert process_data.check_feature_coverage is fp.check_feature_coverage
    assert process_data.feature_coverage is fp.feature_coverage
    assert check_feature_coverage is diagnose_feature_coverage
    assert feature_coverage is diagnose_feature_coverage


def test_process_frame_does_not_use_csv_io(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('CSV I/O should not be used by feature processing')

    monkeypatch.setattr(pd, 'read_csv', boom)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', boom)
    result = process_frame(raw_frame())
    assert not result.empty


def test_process_frame_does_not_mutate_input():
    raw = raw_frame()
    original_index = raw.index.copy()
    original_values = raw.copy(deep=True)
    result = process_frame(raw)
    pd.testing.assert_frame_equal(raw, original_values)
    assert raw.index.equals(original_index)
    assert result is not raw


def test_process_frame_normalizes_timestamps_to_utc():
    naive = raw_frame().copy()
    naive.index = naive.index.tz_localize(None)
    result_naive = process_frame(naive, validate_coverage=False)

    aware = raw_frame().copy()
    aware.index = aware.index.tz_convert('Europe/Helsinki')
    result_aware = process_frame(aware, validate_coverage=False)

    assert str(result_naive.index.tz) == 'UTC'
    assert str(result_aware.index.tz) == 'UTC'
    # Naive values are interpreted as UTC, preserving wall clock.
    assert result_naive.index[0] == pd.Timestamp('2026-10-21 00:00:00+00:00')
    assert result_aware.index[0] == pd.Timestamp('2026-10-21 00:00:00+00:00')


def test_invalid_timestamp_raises():
    frame = raw_frame().copy()
    frame.index = pd.Index(['not-a-timestamp'] + list(frame.index[1:]), name='timestamp')
    with pytest.raises((ValueError, TypeError)):
        process_frame(frame)


def test_invalid_intervals_are_preserved():
    raw = raw_frame()
    raw.loc[raw.index[0], 'battery_power'] = float('nan')
    result = process_frame(raw)
    assert not result.baseload_target_valid.iloc[0]
    assert 'battery_power;' in result.baseload_exclusion_reason.iloc[0]
    assert pd.isna(result.baseload_lag_24h.iloc[96])
    assert result.baseload_lag_24h.iloc[97] == 2.0


def test_coverage_helper_reports_and_raises():
    frame = process_frame(raw_frame())
    report = diagnose_feature_coverage(frame, required_cols=['total_home_power', 'outside_temp'])
    assert report['outside_temp']['present'] is True
    # missing column
    missing_frame = process_frame(raw_frame()).drop(columns=['total_home_power'])
    with pytest.raises(ValueError, match='missing'):
        diagnose_feature_coverage(missing_frame, required_cols=['total_home_power'])
    # all null
    null_frame = process_frame(raw_frame()).copy()
    null_frame['outside_temp'] = float('nan')
    with pytest.raises(ValueError, match='no valid'):
        diagnose_feature_coverage(null_frame, required_cols=['outside_temp'])


def test_coverage_aliases_are_callable():
    frame = process_frame(raw_frame())
    assert check_feature_coverage(frame) == feature_coverage(frame) == diagnose_feature_coverage(frame)
