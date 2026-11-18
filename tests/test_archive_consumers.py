"""Archive consumers never substitute compatibility history or filled truth."""
from __future__ import annotations

import os
import json
import sqlite3
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import predict_future as prediction
import optimize_plan as optimizer
from utils.archive_history import ArchiveHistory
from utils.archive_quality import METERS
from utils.uncontrolled_heating import load_uncontrolled_heating_forecast


CUTOFF = pd.Timestamp('2026-11-01T12:07:00Z')


def raw_frame(index, values):
    frame = pd.DataFrame(values, index=index)
    for name in values:
        frame[name + '_source_valid'] = True
        frame[name + '_coverage_fraction'] = 1.0
        frame[name + '_provenance'] = 'recorder'
        frame[name + '_conflict'] = False
    return frame


@pytest.fixture
def history(monkeypatch):
    context = ArchiveHistory('unused.sqlite', CUTOFF.timestamp())
    index = pd.date_range(CUTOFF - pd.Timedelta(days=1), CUTOFF, freq='min', inclusive='left')
    frame = raw_frame(index, dict(zip(METERS, [10., 2., -1000., 6000., 1000.])))
    reader = Mock(return_value=frame)
    monkeypatch.setattr(ArchiveHistory, 'raw', reader)
    return context, frame, reader


def test_anchors_exact_unfilled_shared_meter_and_complete_day(history):
    context, frame, reader = history
    assert prediction.archive_inference_anchors(context) == pytest.approx((6., 6., 1000., 24.))
    reader.assert_called_once_with(context.cutoff - 86400, context.cutoff, measurements=list(METERS))


def test_archive_forecast_uses_labelled_short_leaf_estimate(history, capsys):
    context, frame, _ = history
    frame.loc[frame.index[200:202], 'leaf_power'] = np.nan
    assert prediction.archive_inference_anchors(context) == pytest.approx((6., 6., 1000., 24.))
    assert 'estimated 2 missing minute' in capsys.readouterr().out


def test_archive_forecast_accepts_three_partial_bins_within_two_minutes(history):
    context, frame, _ = history
    for offset, coverage in enumerate((0.743066, 0.170751, 0.257376)):
        t = frame.index[200 + offset]
        frame.loc[t, 'leaf_power'] = np.nan
        frame.loc[t, 'leaf_power_source_valid'] = False
        frame.loc[t, 'leaf_power_coverage_fraction'] = coverage
    diagnostics = {}
    assert prediction.archive_inference_anchors(context, diagnostics=diagnostics) == pytest.approx((6., 6., 1000., 24.))
    assert diagnostics['estimated_bins'] == 3
    assert diagnostics['missing_seconds'] == pytest.approx(109.72842)


def test_archive_forecast_still_rejects_long_leaf_gaps(history):
    context, frame, _ = history
    frame.loc[frame.index[200:203], 'leaf_power'] = np.nan
    with pytest.raises(ValueError, match='Leaf'):
        prediction.archive_inference_anchors(context)


@pytest.mark.parametrize('defect', ['gap', 'conflict', 'coverage', 'missing_meter', 'negative'])
def test_anchor_defects_fail_closed(history, defect):
    context, frame, _ = history
    target = CUTOFF - pd.Timedelta(hours=1)
    if defect == 'gap':
        frame.loc[frame.index[0], 'leaf_power'] = np.nan
    elif defect == 'conflict':
        frame.loc[target, 'battery_power_conflict'] = True
    elif defect == 'coverage':
        frame.loc[target, 'solar_actual_coverage_fraction'] = .5
    elif defect == 'missing_meter':
        frame.drop(columns='battery_power', inplace=True)
    else:
        frame.loc[target, 'total_power'] = -100
    with pytest.raises(ValueError, match='anchor|Leaf'):
        prediction.archive_inference_anchors(context)


def test_anchor_legacy_policy_is_explicit(history):
    context, frame, _ = history
    for name in METERS:
        frame[name + '_source_valid'] = False
        frame[name + '_provenance'] = 'legacy_resampled'
    with pytest.raises(ValueError):
        prediction.archive_inference_anchors(context)
    permissive = ArchiveHistory(context.path, context.cutoff, allow_legacy=True, legacy_policy='complete-legacy')
    assert prediction.archive_inference_anchors(permissive) == pytest.approx((6., 6., 1000., 24.))


def test_sauna_uses_helsinki_full_date_not_day_number(history):
    context, _, reader = history
    index = pd.to_datetime(['2026-10-01T17:00Z', '2026-10-31T16:00Z', '2026-11-01T11:37Z'])
    reader.return_value = raw_frame(index, {'sauna_temp': [90., 55., 35.]})
    assert prediction.archive_sauna_history(context, 40.) == (True, True)
    reader.return_value.loc[index[1], 'sauna_temp_conflict'] = True
    assert prediction.archive_sauna_history(context, 40.) == (True, False)
    assert reader.call_args.args[1] == context.cutoff


def test_uncontrolled_shared_meter_and_cutoff_coverage(history, monkeypatch, capsys):
    context, frame, reader = history
    for name in ('GSHP', 'RESISTIVE_HEATER', 'BULK_HEATER'):
        monkeypatch.setenv(name + '_OPTIMIZE_ENABLED', 'false')
    result = load_uncontrolled_heating_forecast([CUTOFF.floor('15min')], history=context)
    assert result.tolist() == [6.]
    assert '1440/1440' in capsys.readouterr().out
    reader.assert_called_once_with(context.cutoff - 86400, context.cutoff, measurements=['gshp_power'])
    frame['gshp_power_conflict'] = True
    with pytest.raises(ValueError, match='heating'):
        load_uncontrolled_heating_forecast([CUTOFF], history=context)


def test_fireplace_archive_observation_not_live_initial_state(history, monkeypatch):
    context, _, reader = history
    index = pd.date_range(CUTOFF - pd.Timedelta(minutes=2), periods=2, freq='min')
    reader.return_value = raw_frame(index, {'accumulator_temp': [45., 46.], 'gshp_power': [0., 0.]})
    assert optimizer.archive_fireplace_detected(context)
    reader.return_value.loc[index[-1], 'gshp_power'] = 1000.
    assert not optimizer.archive_fireplace_detected(context)
    monkeypatch.setenv('GSHP_ACCUMULATOR_TEMP_ENTITY', '')
    reader.reset_mock()
    assert not optimizer.archive_fireplace_detected(context)
    reader.assert_not_called()


@pytest.fixture
def prediction_runtime(history, monkeypatch, tmp_path):
    context, frame, reader = history
    def read(start, end, *, measurements, **kwargs):
        if measurements == ['sauna_temp']:
            return raw_frame(pd.date_range(CUTOFF - pd.Timedelta(minutes=30), periods=30, freq='min'), {'sauna_temp': 20.})
        return frame
    reader.side_effect = read
    monkeypatch.chdir(tmp_path)
    Path('state').mkdir()
    from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract
    Path('state/model_features.json').write_text(json.dumps(BASELOAD_FEATURES))
    from utils.archive_quality import LEAF_ENERGY_ESTIMATE_POLICY
    Path('state/model_training_metadata.json').write_text(json.dumps({
        'feature_contract': baseload_feature_contract(),
        'archive_training': {'leaf_energy_estimate_policy': LEAF_ENERGY_ESTIMATE_POLICY}}))
    model = Mock()
    model.get_booster.return_value.feature_names = list(BASELOAD_FEATURES)
    model.get_booster.return_value.attr.return_value = json.dumps(baseload_feature_contract())
    model.predict.side_effect = lambda features: np.ones(len(features))
    monkeypatch.setattr(prediction.xgb, 'XGBRegressor', Mock(return_value=model))
    solar = {'attributes': {'detailedHourly': [{'period_start': CUTOFF.isoformat(), 'pv_estimate': 1.}]}}
    monkeypatch.setattr(prediction, 'get_ha_state', lambda entity: (
        solar if entity.endswith('forecast_today') else
        {'attributes': {'detailedHourly': []}} if entity.endswith('forecast_tomorrow') else {'state': '40'}))
    monkeypatch.setattr(prediction, 'call_ha_service', Mock(return_value=None))
    monkeypatch.setattr(prediction, 'fetch_states_history', Mock(side_effect=AssertionError('PG history forbidden')))
    monkeypatch.setattr(prediction, 'fetch_market_prices', Mock(return_value=([.1], [0], None, False, False, None)))
    monkeypatch.setattr(prediction, 'get_model_version', lambda: 'test-version')
    monkeypatch.setattr(prediction, 'get_db_connection', lambda: sqlite3.connect(os.environ['TEST_DB_PATH']))
    return context, model


def test_prediction_fixed_cutoff_and_success_metadata(prediction_runtime):
    context, model = prediction_runtime
    prediction.predict(history=context)
    output = Path(os.environ['TEST_PREDICTIONS_FILE'])
    rows = json.loads(output.read_text())
    assert pd.Timestamp(rows[0]['timestamp']) == CUTOFF.floor('15min')
    metadata = json.loads(output.with_name('future_predictions_metadata.json').read_text())
    assert metadata['history_cutoff'] == context.cutoff
    assert metadata['archive_path'] == str(Path(context.path).resolve())
    assert metadata['legacy_policy'] == 'verified-only'
    assert metadata['model_version'] == 'test-version'
    inputs = model.predict.call_args.args[0]
    assert inputs.iloc[0][['baseload_lag_1h', 'leaf_energy_24h']].tolist() == [6., 24.]
    assert 'solar_forecast' not in inputs
    assert rows[0]['solar_forecast'] == 1.  # Planner forecast survives predictor removal.


def test_archive_prediction_can_use_separate_model_directory(prediction_runtime, tmp_path):
    context, model = prediction_runtime
    import shutil
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    for name in ('model_features.json', 'model_training_metadata.json'):
        shutil.copyfile(Path('state') / name, candidate / name)
    (candidate / 'energy_model.json').write_text('synthetic model')
    Path('state/model_features.json').write_text('[]')
    prediction.predict(history=context, model_dir=candidate)
    model.load_model.assert_called_once_with(str(candidate / 'energy_model.json'))
    output = Path(os.environ['TEST_PREDICTIONS_FILE'])
    metadata = json.loads(output.with_name('future_predictions_metadata.json').read_text())
    assert metadata['model_directory'] == str(candidate.resolve())
    assert len(metadata['model_files_sha256']['energy_model.json']) == 64


def test_archive_prediction_rejects_model_changed_during_inference(prediction_runtime, tmp_path):
    context, model = prediction_runtime
    import shutil
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    for name in ('model_features.json', 'model_training_metadata.json'):
        shutil.copyfile(Path('state') / name, candidate / name)
    (candidate / 'energy_model.json').write_text('initial model')
    def mutate(features):
        (candidate / 'energy_model.json').write_text('changed model')
        return np.ones(len(features))
    model.predict.side_effect = mutate
    with pytest.raises(ValueError, match='model.*changed'):
        prediction.predict(history=context, model_dir=candidate)
    assert not Path(os.environ['TEST_PREDICTIONS_FILE']).exists()


@pytest.mark.parametrize('problem', ['old_features', 'missing_contract', 'old_booster'])
def test_archive_prediction_rejects_incompatible_models_before_writing_outputs(prediction_runtime, problem):
    context, model = prediction_runtime
    from utils.baseload_quality import BASELOAD_FEATURES
    if problem == 'old_features':
        Path('state/model_features.json').write_text(json.dumps([*BASELOAD_FEATURES, 'solar_forecast']))
    elif problem == 'missing_contract':
        Path('state/model_training_metadata.json').write_text('{}')
    else:
        model.get_booster.return_value.attr.return_value = None
    with pytest.raises(ValueError, match='retrain'):
        prediction.predict(history=context)
    model.predict.assert_not_called()
    assert not Path(os.environ['TEST_PREDICTIONS_FILE']).exists()


@pytest.mark.parametrize('failure', ['solar', 'model', 'features', 'anchors', 'empty_horizon', 'empty_model'])
def test_prediction_archive_failure_does_not_publish(prediction_runtime, monkeypatch, failure):
    context, model = prediction_runtime
    if failure == 'solar':
        monkeypatch.setattr(prediction, 'get_ha_state', lambda entity: None)
    elif failure == 'model':
        model.load_model.side_effect = FileNotFoundError('missing model')
    elif failure == 'features':
        Path('state/model_features.json').unlink()
    elif failure == 'anchors':
        monkeypatch.setattr(prediction, 'archive_inference_anchors', Mock(side_effect=ValueError('missing anchor')))
    elif failure == 'empty_horizon':
        monkeypatch.setattr(prediction, 'get_ha_state', lambda entity: {'attributes': {'detailedHourly': [
            {'period_start': (CUTOFF - pd.Timedelta(days=1)).isoformat(), 'pv_estimate': 1.}]}})
    else:
        model.predict.return_value = []
        model.predict.side_effect = None
    with pytest.raises((ValueError, FileNotFoundError)):
        prediction.predict(history=context)
    output = Path(os.environ['TEST_PREDICTIONS_FILE'])
    assert not output.exists()
    assert not output.with_name('future_predictions_metadata.json').exists()


@pytest.mark.parametrize('failure', ['predictions', 'prices', 'empty'])
def test_optimizer_archive_failure_raises(history, monkeypatch, failure):
    context, _, _ = history
    if failure != 'predictions':
        Path(os.environ['TEST_PREDICTIONS_FILE']).write_text(json.dumps([] if failure == 'empty' else [{
            'timestamp': CUTOFF.isoformat(), 'predicted_baseload': 1., 'solar_forecast': 0.}]))
    monkeypatch.setattr(optimizer, 'fetch_market_prices', Mock(return_value=(None, None, None, False, False, None)))
    with pytest.raises((FileNotFoundError, ValueError)):
        optimizer.optimize(history=context)
    assert not Path(os.environ['TEST_PLAN_FILE']).exists()


def test_optimizer_success_keeps_live_thermal_state_and_metadata(history, monkeypatch):
    context, _, reader = history
    monkeypatch.setenv('GSHP_ACCUMULATOR_TEMP_ENTITY', '')
    monkeypatch.setenv('GSHP_OPTIMIZE_ENABLED', 'false')
    monkeypatch.setenv('RESISTIVE_HEATER_OPTIMIZE_ENABLED', 'false')
    monkeypatch.setenv('BULK_HEATER_OPTIMIZE_ENABLED', 'false')
    monkeypatch.setenv('BATTERY_CAPACITY_KWH', '0')
    monkeypatch.setenv('GSHP_INITIAL_TEMP', '50')
    monkeypatch.setenv('GSHP_IS_RUNNING', 'false')
    monkeypatch.setenv('RESISTIVE_HEATER_IS_RUNNING', 'false')
    Path(os.environ['TEST_PREDICTIONS_FILE']).write_text(json.dumps([{
        'timestamp': CUTOFF.floor('15min').isoformat(), 'predicted_baseload': 1., 'solar_forecast': 0.}]))
    live_temp = Mock(return_value=49.)
    monkeypatch.setattr(optimizer, 'get_gshp_initial_temp', live_temp)
    monkeypatch.setattr(optimizer, 'get_ha_state', Mock(return_value={'state': '0'}))
    monkeypatch.setattr(optimizer, 'fetch_states_history', Mock(side_effect=AssertionError('PG forbidden')))
    monkeypatch.setattr(optimizer, 'fetch_market_prices', Mock(return_value=([.1], [0], 'test', False, False, None)))
    monkeypatch.setattr(optimizer, 'get_db_connection', lambda: sqlite3.connect(os.environ['TEST_DB_PATH']))
    optimizer.optimize(history=context)
    live_temp.assert_called_once()
    assert os.environ['GSHP_INITIAL_TEMP'] == '49.0'
    output = Path(os.environ['TEST_PLAN_FILE'])
    rows = json.loads(output.read_text())
    assert rows[0]['uncontrolled_heating_kw'] == 6.
    metadata = json.loads(output.with_name('optimization_plan_metadata.json').read_text())
    assert metadata['history_cutoff'] == context.cutoff
    assert metadata['archive_path'] == str(Path(context.path).resolve())


@pytest.mark.parametrize('script', ['predict_future.py', 'optimize_plan.py'])
def test_archive_cli_valid_flags_propagate_failure(script, monkeypatch):
    import utils.ha_utils
    import utils.db_utils
    monkeypatch.setattr(utils.ha_utils, 'get_ha_state', Mock(return_value=None))
    monkeypatch.setattr(utils.ha_utils, 'call_ha_service', Mock(return_value=None))
    monkeypatch.setattr(utils.db_utils, 'fetch_states_history', Mock(side_effect=AssertionError('PG forbidden')))
    monkeypatch.setattr(sys, 'argv', [script, '--archive', 'unused.sqlite', '--cutoff', CUTOFF.isoformat()])
    with pytest.raises((ValueError, FileNotFoundError)):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / script), run_name='__main__')


def test_failed_prediction_does_not_refresh_stale_success_metadata(prediction_runtime):
    context, model = prediction_runtime
    model.load_model.side_effect = FileNotFoundError('missing')
    output = Path(os.environ['TEST_PREDICTIONS_FILE'])
    sidecar = output.with_name('future_predictions_metadata.json')
    output.write_text('old predictions')
    sidecar.write_text('old metadata')
    with pytest.raises(FileNotFoundError):
        prediction.predict(history=context)
    assert output.read_text() == 'old predictions'
    assert sidecar.read_text() == 'old metadata'


def test_archive_errors_never_retry_csv(history, monkeypatch):
    context, _, reader = history
    for name in ('GSHP', 'RESISTIVE_HEATER', 'BULK_HEATER'):
        monkeypatch.setenv(name + '_OPTIMIZE_ENABLED', 'false')
    reader.side_effect = OSError('archive unavailable')
    csv = Mock(side_effect=AssertionError('CSV forbidden'))
    monkeypatch.setattr(pd, 'read_csv', csv)
    with pytest.raises(OSError, match='archive unavailable'):
        load_uncontrolled_heating_forecast([CUTOFF], history=context)
    csv.assert_not_called()


def test_real_archive_defaults_and_missing_source_fail_closed(tmp_path, monkeypatch):
    from utils.archive_features import default_sources
    from utils.measurement_archive import MeasurementArchive
    from utils.measurement_sources import load_sources, source_at
    path = tmp_path / 'measurements.sqlite'
    sources = default_sources()
    sources.update(load_sources(Path(__file__).resolve().parents[1] / 'config/measurement_sources.yaml'))
    values = dict(zip(METERS, [10., 2., -1000., 6000., 1000.]))
    values['sauna_temp'] = 20.
    with MeasurementArchive(path) as archive:
        for state_id, (name, value) in enumerate(values.items(), 1):
            source = source_at(sources[name], CUTOFF.to_pydatetime())
            assert source is not None
            archive.ingest(source.entity, None, CUTOFF.timestamp() - 90000,
                           CUTOFF.timestamp(), [(state_id, CUTOFF.timestamp() - 90000, str(value), None)])
    context = ArchiveHistory(path, CUTOFF.timestamp())
    assert prediction.archive_inference_anchors(context) == pytest.approx((6., 6., 1000., 24.))
    assert prediction.archive_sauna_history(context, 20.) == (False, False)
    for name in ('GSHP', 'RESISTIVE_HEATER', 'BULK_HEATER'):
        monkeypatch.setenv(name + '_OPTIMIZE_ENABLED', 'false')
    assert load_uncontrolled_heating_forecast([CUTOFF], history=context).tolist() == [6.]
    override = tmp_path / 'sources.yaml'
    override.write_text(json.dumps({'battery_power': [{'entity': 'sensor.missing'}]}))
    with pytest.raises(ValueError, match='anchor'):
        prediction.archive_inference_anchors(ArchiveHistory(path, context.cutoff, sources_path=override))
