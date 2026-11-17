"""Declared archive training eligibility, distinct from source verification.

complete-legacy accepts complete finite conflict-free legacy/mixed bins. It
does not certify original recorder fidelity or infer pre-installation zeros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.baseload_quality import baseload_targets, baseload_feature_contract

LEGACY_POLICIES = ('verified-only', 'complete-legacy')
METERS = ('total_power', 'solar_actual', 'battery_power', 'gshp_power', 'leaf_power')
PREDICTOR_INPUTS = ('outside_temp', 'wind_speed')
LEAF_ENERGY_ESTIMATE_POLICY = {
    'version': 1, 'resolution_seconds': 60, 'max_missing_minutes': 2,
    'method': 'previous_eligible_recorder_value',
}


def _true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq('true')


def measurement_eligibility(raw: pd.DataFrame, name: str, policy: str) -> pd.Series:
    required = [name, *(name + suffix for suffix in
                       ('_source_valid', '_coverage_fraction', '_provenance', '_conflict'))]
    if any(column not in raw for column in required):
        return pd.Series(False, index=raw.index)
    values = pd.to_numeric(raw[name], errors='coerce')
    good = pd.Series(np.isfinite(values), index=raw.index) & raw[name + '_coverage_fraction'].eq(1)
    good &= raw[name + '_conflict'].astype(str).str.lower().eq('false')
    if name in {'solar_actual', 'gshp_power', 'leaf_power', 'wind_speed', 'solar_forecast'}:
        good &= values.ge(0)
    provenance = raw[name + '_provenance']
    verified = _true(raw[name + '_source_valid']) & provenance.eq('recorder')
    if policy == 'complete-legacy':
        verified |= provenance.isin(['legacy_resampled', 'mixed'])
    return good & verified


def apply_training_policy(raw: pd.DataFrame, features: pd.DataFrame,
                          policy: str = 'verified-only') -> pd.DataFrame:
    """Use original aligned meters, never smoothed/filled values, for targets.

    Public source and target verification flags are never relaxed. A separate
    eligibility timeline supplies exact lags and complete previous-day Leaf
    energy. Resampling has already removed explicit invalid source flags.
    """
    if policy not in LEGACY_POLICIES:
        raise ValueError(f'Unknown legacy training policy: {policy}')
    out = features.copy(deep=True)
    eligible_raw = raw.copy(deep=True)
    for name in (*METERS, *PREDICTOR_INPUTS):
        good = measurement_eligibility(raw, name, policy)
        out[name + '_training_eligible'] = good
        # These private flags govern the declared policy, not source verification.
        eligible_raw[name + '_source_valid'] = good
        if name in PREDICTOR_INPUTS:
            out[name] = raw[name].where(good) if name in raw else np.nan
    targets = baseload_targets(eligible_raw)
    out['baseload_training_eligible'] = targets.baseload_target_valid
    for name in ('baseload_power', 'baseload_lag_1h', 'baseload_lag_24h',
                 'leaf_power_lag_1h', 'leaf_energy_24h'):
        out[name] = targets[name]
        if name != 'baseload_power':
            out[name + '_training_eligible'] = targets[name].notna()
    # Estimate only this derived feature, never the meter or baseload target.
    # A previous eligible observation is the only permitted source, so each
    # historical training row uses no data from after its own origin.
    index = pd.DatetimeIndex(raw.index)
    if len(index) > 1 and index.to_series().diff().dropna().eq(pd.Timedelta(minutes=1)).all():
        valid = measurement_eligibility(raw, 'leaf_power', policy)
        values = pd.to_numeric(raw['leaf_power'], errors='coerce').where(valid)
        recorder = raw['leaf_power_provenance'].eq('recorder')
        conflict_free = raw['leaf_power_conflict'].astype(str).str.lower().eq('false')
        estimate = values.ffill(limit=2).where(~valid & recorder & conflict_free)
        combined = values.combine_first(estimate)
        missing = (~valid).astype(int).rolling('24h', closed='left', min_periods=1440).sum()
        estimated = estimate.notna().astype(int).rolling('24h', closed='left', min_periods=1440).sum()
        energy = combined.rolling('24h', closed='left', min_periods=1440).sum() / 60000
        acceptable = missing.le(2) & estimated.eq(missing)
        out['leaf_energy_24h'] = energy.where(acceptable)
        out['leaf_energy_24h_training_eligible'] = out['leaf_energy_24h'].notna()
        out['leaf_energy_24h_estimated_minutes'] = estimated.where(acceptable)
    else:
        out['leaf_energy_24h_estimated_minutes'] = targets['leaf_energy_24h'].notna().map(
            {True: 0., False: np.nan})
    out['baseload_training_exclusion_reason'] = targets.baseload_exclusion_reason
    if 'baseload_target_valid' not in out:
        out['baseload_target_valid'] = baseload_targets(raw).baseload_target_valid
    out.attrs['archive_training'] = {
        **raw.attrs.get('archive', {}),
        'legacy_policy': policy,
        'contract_version': 1,
        'legacy_is_source_verified': False,
        'feature_contract': baseload_feature_contract(),
        'leaf_energy_estimate_policy': LEAF_ENERGY_ESTIMATE_POLICY,
    }
    return out
