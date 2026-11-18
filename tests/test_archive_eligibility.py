import pandas as pd
import pytest

from utils.archive_quality import apply_training_policy
from utils.baseload_quality import select_training_rows


def raw_frame():
    index = pd.date_range('2026-10-21', periods=100, freq='15min', tz='UTC')
    values = dict(total_power=4., solar_actual=1., battery_power=500.,
                  gshp_power=2000., leaf_power=500., outside_temp=5.,
                  wind_speed=2., solar_forecast=0.)
    frame = pd.DataFrame(values, index=index)
    for name in values:
        frame[name + '_source_valid'] = False
        frame[name + '_coverage_fraction'] = 1.
        frame[name + '_provenance'] = 'legacy_resampled'
        frame[name + '_conflict'] = False
    return frame


def test_explicit_legacy_policy_separates_verification_target_and_eligibility():
    raw = raw_frame()
    features = pd.DataFrame({'baseload_target_valid': False}, index=raw.index)
    strict = apply_training_policy(raw, features, 'verified-only')
    assert not strict.baseload_training_eligible.any()
    out = apply_training_policy(raw, features, 'complete-legacy')
    assert not out.baseload_target_valid.any()
    assert out.baseload_training_eligible.all()
    assert out.baseload_power.eq(2).all()
    assert not raw.total_power_source_valid.any()
    assert len(select_training_rows(out)) == 4
    assert out.attrs['archive_training']['legacy_policy'] == 'complete-legacy'


@pytest.mark.parametrize('name', ['total_power', 'solar_actual', 'battery_power', 'gshp_power', 'leaf_power'])
@pytest.mark.parametrize('problem', ['missing', 'partial', 'conflict', 'invalid'])
def test_each_mandatory_meter_must_be_complete_unambiguous_and_valid(name, problem):
    raw = raw_frame()
    timestamp = raw.index[-1]
    if problem == 'missing':
        raw.loc[timestamp, name] = float('nan')
    elif problem == 'partial':
        raw.loc[timestamp, name + '_coverage_fraction'] = .5
    elif problem == 'conflict':
        raw.loc[timestamp, name + '_conflict'] = True
    else:
        raw.loc[timestamp, name + '_provenance'] = 'recorder'
        raw.loc[timestamp, name + '_source_valid'] = False
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.loc[timestamp, 'baseload_training_eligible']
    assert pd.isna(out.loc[timestamp, 'baseload_power'])


def test_exact_lag_and_full_leaf_window_eligibility_not_forward_filled():
    raw = raw_frame()
    raw.loc[raw.index[0], 'leaf_power'] = float('nan')
    raw.loc[raw.index[-5], 'total_power_conflict'] = True
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.iloc[96].baseload_lag_24h_training_eligible
    assert not out.iloc[96].leaf_energy_24h_training_eligible
    assert not out.iloc[-1].baseload_lag_1h_training_eligible
    assert select_training_rows(out).index.tolist() == raw.index[97:99].tolist()


def test_mixed_and_verified_bins_have_distinct_policy_eligibility():
    raw = raw_frame()
    raw['total_power_provenance'] = 'mixed'
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert out.baseload_training_eligible.all()
    strict = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'verified-only')
    assert not strict.baseload_training_eligible.any()


def test_unknown_policy_rejected():
    raw = raw_frame()
    with pytest.raises(ValueError, match='policy'):
        apply_training_policy(raw, pd.DataFrame(index=raw.index), 'anything')


def minute_raw():
    frame = raw_frame().iloc[:1].copy()
    frame = pd.concat([frame] * 1443, ignore_index=True)
    frame.index = pd.date_range('2026-10-21', periods=len(frame), freq='min', tz='UTC')
    frame['leaf_power_provenance'] = 'recorder'
    frame['leaf_power_source_valid'] = True
    return frame


def test_leaf_energy_estimates_two_missing_minutes_causally_but_not_meter_truth():
    raw = minute_raw()
    raw.loc[raw.index[100:102], 'leaf_power'] = float('nan')
    raw.loc[raw.index[102], 'leaf_power'] = 1800.
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    row = out.iloc[1440]
    assert row.leaf_energy_24h_training_eligible
    assert row.leaf_energy_24h_estimated_minutes == 2
    assert row.leaf_energy_24h == pytest.approx((1439 * 500 + 1800) / 60000)
    assert not out.iloc[100].baseload_training_eligible
    assert pd.isna(out.iloc[100].baseload_power)
    assert pd.isna(raw.iloc[100].leaf_power)


def test_leaf_energy_rejects_more_than_two_minutes_and_unanchored_gap():
    raw = minute_raw()
    raw.loc[raw.index[100:103], 'leaf_power'] = float('nan')
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.iloc[1440].leaf_energy_24h_training_eligible
    assert pd.isna(out.iloc[1440].leaf_energy_24h)
    raw = minute_raw()
    raw.loc[raw.index[0], 'leaf_power'] = float('nan')
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.iloc[1440].leaf_energy_24h_training_eligible


def test_leaf_energy_never_estimates_conflicts():
    raw = minute_raw()
    raw.loc[raw.index[100], 'leaf_power_conflict'] = True
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.iloc[1440].leaf_energy_24h_training_eligible


def test_three_partial_bins_with_under_two_minutes_missing_are_estimated():
    raw = minute_raw()
    for offset, coverage in enumerate((0.743066, 0.170751, 0.257376)):
        t = raw.index[100 + offset]
        raw.loc[t, 'leaf_power'] = float('nan')
        raw.loc[t, 'leaf_power_source_valid'] = False
        raw.loc[t, 'leaf_power_coverage_fraction'] = coverage
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    row = out.iloc[1440]
    assert row.leaf_energy_24h_training_eligible
    assert row.leaf_energy_24h_estimated_minutes == 3
    assert row.leaf_energy_24h_missing_seconds == pytest.approx(109.72842)
    assert row.leaf_energy_24h == pytest.approx(12.)
    assert not out.iloc[100:103].baseload_training_eligible.any()
    assert out.iloc[100:103].baseload_power.isna().all()


def test_three_partial_bins_over_two_minutes_missing_remain_ineligible():
    raw = minute_raw()
    for offset, coverage in enumerate((0.5, 0.1, 0.2)):
        t = raw.index[100 + offset]
        raw.loc[t, 'leaf_power'] = float('nan')
        raw.loc[t, 'leaf_power_source_valid'] = False
        raw.loc[t, 'leaf_power_coverage_fraction'] = coverage
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.iloc[1440].leaf_energy_24h_training_eligible


def test_bad_or_inconsistent_coverage_cannot_bypass_gap_limit():
    raw = minute_raw()
    for offset in range(3):
        t = raw.index[100 + offset]
        raw.loc[t, 'leaf_power'] = float('nan')
        raw.loc[t, 'leaf_power_coverage_fraction'] = 1.0
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.iloc[1440].leaf_energy_24h_training_eligible


def test_four_partial_bins_remain_ineligible_even_when_missing_duration_is_short():
    raw = minute_raw()
    for offset in range(4):
        t = raw.index[100 + offset]
        raw.loc[t, 'leaf_power'] = float('nan')
        raw.loc[t, 'leaf_power_coverage_fraction'] = 0.9
    out = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert not out.iloc[1440].leaf_energy_24h_training_eligible
