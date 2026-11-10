"""Declared archive training eligibility, distinct from source verification.

complete-legacy accepts complete finite conflict-free legacy/mixed bins. It
does not certify original recorder fidelity or infer pre-installation zeros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.baseload_quality import baseload_targets

LEGACY_POLICIES = ('verified-only', 'complete-legacy')
METERS = ('total_power', 'solar_actual', 'battery_power', 'gshp_power', 'leaf_power')
PREDICTOR_INPUTS = ('outside_temp', 'wind_speed', 'solar_forecast')


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
    out['baseload_training_exclusion_reason'] = targets.baseload_exclusion_reason
    if 'baseload_target_valid' not in out:
        out['baseload_target_valid'] = baseload_targets(raw).baseload_target_valid
    out.attrs['archive_training'] = {
        **raw.attrs.get('archive', {}),
        'legacy_policy': policy,
        'contract_version': 1,
        'legacy_is_source_verified': False,
    }
    return out
