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
