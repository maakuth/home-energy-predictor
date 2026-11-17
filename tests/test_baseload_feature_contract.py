import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract, select_training_rows
from utils.baseload_model import validate_baseload_model
from utils.archive_features import BASELOAD_INPUTS, load_archive_features
from utils.archive_quality import LEAF_ENERGY_ESTIMATE_POLICY, apply_training_policy


def test_solar_forecast_is_not_a_model_or_required_history_input():
    assert 'solar_forecast' not in BASELOAD_FEATURES
    assert 'solar_forecast' not in BASELOAD_INPUTS
    assert 'solar_actual' in BASELOAD_INPUTS


def test_missing_energy_forecast_cannot_exclude_otherwise_eligible_rows():
    from tests.test_archive_eligibility import raw_frame
    raw = raw_frame().drop(columns=[name for name in raw_frame() if name.startswith('solar_forecast')])
    frame = apply_training_policy(raw, pd.DataFrame(index=raw.index), 'complete-legacy')
    assert len(select_training_rows(frame)) == 4
    assert 'solar_forecast_training_eligible' not in frame


def test_archive_features_and_quality_work_without_any_forecast_source(tmp_path):
    from tests.test_archive_features import build, START
    path, sources = build(tmp_path)
    sources.pop('solar_forecast')
    result = load_archive_features(path, START, START + 120, sources=sources)
    assert len(select_training_rows(result)) == 2
    assert 'solar_forecast' not in result
    assert 'solar_forecast' not in result.attrs['archive_training']['measurements']
    assert result.baseload_power.eq(2).all()


def test_optional_raw_forecast_keeps_daily_energy_units_without_becoming_a_predictor(tmp_path):
    from utils.archive_features import load_archive_raw
    from utils.measurement_archive import MeasurementArchive
    from utils.measurement_sources import parse_sources
    path = tmp_path / 'daily.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.daily', None, 0, 120,
                       [(1, 0., '12', '{"unit_of_measurement":"kWh"}')])
    sources = parse_sources({'solar_forecast': [{'entity': 'sensor.daily'}]})
    policy = {'solar_forecast': dict(kind='numeric', input_unit='kWh', canonical_unit='kWh',
                                     sign=1, require_unit=True)}
    raw = load_archive_raw(path, 0, 120, sources=sources, measurements=['solar_forecast'], policies=policy)
    assert raw.solar_forecast.eq(12).all()
    assert raw.solar_forecast_unit_verified.all()


@pytest.mark.parametrize('problem', ['metadata', 'embedded', 'old_feature', 'model_names'])
def test_archive_contract_rejects_legacy_or_mixed_artifacts(problem):
    contract = baseload_feature_contract()
    features = list(BASELOAD_FEATURES)
    metadata = {'feature_contract': contract}
    embedded = json.dumps(contract)
    model_names = list(features)
    if problem == 'metadata':
        metadata = {}
    elif problem == 'embedded':
        embedded = None
    elif problem == 'old_feature':
        features.insert(2, 'solar_forecast')
    else:
        model_names = ['solar_forecast', *features]
    with pytest.raises(ValueError, match='retrain'):
        validate_baseload_model(features, metadata, embedded, model_names)


def test_contract_accepts_only_matching_declared_and_embedded_features():
    contract = baseload_feature_contract()
    validate_baseload_model(list(BASELOAD_FEATURES), {'feature_contract': contract,
                            'archive_training': {'leaf_energy_estimate_policy': LEAF_ENERGY_ESTIMATE_POLICY}},
                            json.dumps(contract), list(BASELOAD_FEATURES))


def test_archive_rejects_model_without_matching_leaf_estimate_policy():
    contract = baseload_feature_contract()
    with pytest.raises(ValueError, match='Leaf|leaf'):
        validate_baseload_model(list(BASELOAD_FEATURES), {'feature_contract': contract},
                                json.dumps(contract), list(BASELOAD_FEATURES))


def test_training_omits_forecast_and_persists_explicit_contract(tmp_path, monkeypatch):
    from process_data import process_frame
    from train_model import train
    monkeypatch.chdir(tmp_path)
    index = pd.date_range('2026-10-21', periods=160, freq='15min', tz='UTC')
    raw = pd.DataFrame(dict(outside_temp=5., wind_speed=2., total_power=3., solar_actual=0.,
                            battery_power=0., gshp_power=1000., leaf_power=0.), index=index)
    frame = process_frame(raw)
    model = MagicMock()
    model.predict.side_effect = lambda features: np.full(len(features), 2.)
    with patch('train_model.xgb.XGBRegressor', return_value=model):
        train(frame=frame, min_coverage_days=.1)
    assert 'solar_forecast' not in model.fit.call_args.args[0]
    saved = json.loads((tmp_path / 'state/model_training_metadata.json').read_text())
    assert saved['feature_contract'] == baseload_feature_contract()
    embedded = model.get_booster.return_value.set_attr.call_args.kwargs['baseload_feature_contract']
    assert json.loads(embedded) == saved['feature_contract']


def test_small_real_model_roundtrips_the_embedded_contract(tmp_path):
    import xgboost as xgb
    contract = baseload_feature_contract()
    frame = pd.DataFrame({name: np.arange(12, dtype=float) for name in BASELOAD_FEATURES})
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1, n_jobs=1)
    model.fit(frame, np.ones(12))
    model.get_booster().set_attr(baseload_feature_contract=json.dumps(contract))
    path = tmp_path / 'model.json'
    model.save_model(path)
    loaded = xgb.XGBRegressor()
    loaded.load_model(path)
    validate_baseload_model(list(BASELOAD_FEATURES), {'feature_contract': contract,
        'archive_training': {'leaf_energy_estimate_policy': LEAF_ENERGY_ESTIMATE_POLICY}},
        loaded.get_booster().attr('baseload_feature_contract'), loaded.get_booster().feature_names)
