import json
import hashlib
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from utils.archive_history import ArchiveHistory
from utils.archive_quality import LEAF_ENERGY_ESTIMATE_POLICY
from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract
from utils.measurement_archive import MeasurementArchive
from utils import forward_shadow as shadow

CUTOFF = pd.Timestamp('2026-11-10T12:00:00Z')


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    units = {'total_power': 'kW', 'solar_actual': 'kW', 'battery_power': 'W',
             'gshp_power': 'W', 'leaf_power': 'W'}
    values = {'total_power': 3, 'solar_actual': 0, 'battery_power': 0,
              'gshp_power': 1000, 'leaf_power': 0}
    archive = tmp_path / 'archive.sqlite'
    with MeasurementArchive(archive, recorder_namespace='fixture') as db:
        for i, (name, value) in enumerate(values.items()):
            db.ingest('sensor.' + name, None, CUTOFF.timestamp()-172800, CUTOFF.timestamp()+259200,
                      [(i, CUTOFF.timestamp()-172800, str(value), json.dumps({'unit_of_measurement': units[name]}))])
    sources = tmp_path / 'sources.yaml'
    sources.write_text(yaml.safe_dump({name: [{'entity': 'sensor.' + name}] for name in units}))
    policy = tmp_path / 'policy.yaml'
    policy.write_text(yaml.safe_dump({name: dict(kind='numeric', input_unit=unit,
        canonical_unit=unit, sign=1, require_unit=True) for name, unit in units.items()}))
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'energy_model.json').write_text('{}')
    (model_dir / 'model_features.json').write_text(json.dumps(BASELOAD_FEATURES))
    (model_dir / 'model_training_metadata.json').write_text(json.dumps({
        'feature_contract': baseload_feature_contract(), 'model_version': '1.47.1',
        'archive_training': {'leaf_energy_estimate_policy': LEAF_ENERGY_ESTIMATE_POLICY},
        'last_eligible': str(CUTOFF-pd.Timedelta(days=2))}))
    model = Mock()
    model.get_booster.return_value.feature_names = list(BASELOAD_FEATURES)
    model.get_booster.return_value.attr.return_value = json.dumps(baseload_feature_contract())
    model.predict.side_effect = lambda rows: np.full(len(rows), 2.1)
    monkeypatch.setattr(shadow.xgb, 'XGBRegressor', lambda: model)
    history = ArchiveHistory(archive, CUTOFF.timestamp(), sources_path=sources, policy_path=policy)
    periods = pd.date_range(CUTOFF, periods=49, freq='h')
    snapshot = {
        'format_version': 1, 'history_cutoff': CUTOFF.timestamp(),
        'started_at': CUTOFF.isoformat(), 'completed_at': (CUTOFF+pd.Timedelta(seconds=30)).isoformat(),
        'solar': [{'period_start': str(t), 'pv_estimate': 1., 'pv_estimate10': .5, 'pv_estimate90': 1.5} for t in periods],
        'weather': [{'datetime': str(t), 'temperature': 5., 'wind_speed': 2.} for t in periods],
        'current_temperature': {'state': '5', 'attributes': {'unit_of_measurement': '°C'}},
        'current_weather': {'state': 'cloudy', 'attributes': {'wind_speed': 2., 'wind_speed_unit': 'km/h', 'temperature_unit': '°C'}},
    }
    return history, model_dir, snapshot, model


def test_capture_only_reads_states_and_weather_forecast():
    session = MagicMock()
    session.get.return_value.json.return_value = {'state': '1', 'attributes': {'detailedHourly': []}}
    session.post.return_value.json.return_value = {'weather.home': {'forecast': []}}
    data = shadow.capture_inputs(session, 'http://ha', 'private-token')
    assert len(session.get.call_args_list) == 4
    assert all('/api/states/' in c.args[0] for c in session.get.call_args_list)
    session.post.assert_called_once()
    assert session.post.call_args.args[0] == 'http://ha/api/services/weather/get_forecasts?return_response'
    assert session.post.call_args.kwargs['json'] == {'entity_id': 'weather.home', 'type': 'hourly'}
    assert 'private-token' not in json.dumps(data)


def test_capture_understands_ha_service_response_envelope():
    session = MagicMock()
    session.get.return_value.json.return_value = {'attributes': {'detailedHourly': []}}
    session.post.return_value.json.return_value = {'changed_states': [], 'service_response': {
        'weather.home': {'forecast': [{'datetime': str(CUTOFF), 'temperature': 5., 'wind_speed': 2.}]}}}
    data = shadow.capture_inputs(session, 'http://ha', 'token')
    assert data['weather'][0]['temperature'] == 5.


def test_generation_uses_only_explicit_readonly_history_and_future_targets(runtime, monkeypatch, tmp_path):
    history, model_dir, snapshot, model = runtime
    import predict_future
    import requests
    from utils import sqlite_utils, db_utils
    forbidden = Mock(side_effect=AssertionError('live pipeline/network/feedback DB forbidden'))
    monkeypatch.setattr(predict_future, 'predict', forbidden)
    monkeypatch.setattr(sqlite_utils, 'get_db_connection', forbidden)
    monkeypatch.setattr(db_utils, 'get_db_connection', forbidden)
    monkeypatch.setattr(requests, 'get', forbidden)
    monkeypatch.setattr(requests, 'post', forbidden)
    before = set(tmp_path.rglob('*'))
    archive_hash = hashlib.sha256(history.path.read_bytes()).hexdigest()
    result = shadow.generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    # SQLite read-only WAL readers may create shared-memory/empty WAL sidecars.
    assert set(tmp_path.rglob('*')) - before <= {tmp_path / 'archive.sqlite-wal', tmp_path / 'archive.sqlite-shm'}
    assert hashlib.sha256(history.path.read_bytes()).hexdigest() == archive_hash
    assert result['shadow_only'] is True
    assert result['history_cutoff'] == history.cutoff
    rows = result['forecasts']
    assert rows and all(pd.Timestamp(row['timestamp']) > pd.Timestamp(result['generated_at']) for row in rows)
    assert rows[0]['one_hour_origin_baseline_kw'] == pytest.approx(2.)
    assert rows[0]['previous_day_profile_kw'] == pytest.approx(2.)
    assert rows[-1]['previous_day_profile_kw'] is None  # No future actuals used as a baseline.
    assert rows[0]['solar_forecast_kw'] == 1.
    assert 'solar_forecast' not in model.predict.call_args.args[0]
    assert result['calendar_feature_timezone'] == 'UTC'
    first_target = pd.Timestamp(rows[0]['timestamp'])
    assert model.predict.call_args.args[0].iloc[0]['hour'] == first_target.hour
    assert model.predict.call_args.args[0].iloc[0]['hour_sin'] == pytest.approx(np.sin(2*np.pi*first_target.hour/24))
    forbidden.assert_not_called()


@pytest.mark.parametrize('problem', ['empty_solar', 'wind_units', 'bad_model', 'future_capture',
                                     'wrong_cutoff', 'future_model', 'weather_unavailable', 'solar_gap', 'stale_capture'])
def test_generation_fails_closed_on_incompatible_or_missing_inputs(runtime, problem):
    history, model_dir, snapshot, model = runtime
    if problem == 'empty_solar':
        snapshot['solar'] = []
    elif problem == 'wind_units':
        snapshot['current_weather']['attributes']['wind_speed_unit'] = 'm/s'
    elif problem == 'bad_model':
        model.get_booster.return_value.attr.return_value = None
    elif problem == 'future_capture':
        snapshot['completed_at'] = (CUTOFF+pd.Timedelta(days=1)).isoformat()
    elif problem == 'wrong_cutoff':
        snapshot['history_cutoff'] += 60
    elif problem == 'future_model':
        path = model_dir / 'model_training_metadata.json'
        data = json.loads(path.read_text())
        data['last_eligible'] = str(CUTOFF+pd.Timedelta(days=1))
        path.write_text(json.dumps(data))
    elif problem == 'weather_unavailable':
        snapshot['current_weather']['state'] = 'unavailable'
    elif problem == 'solar_gap':
        snapshot['solar'] = [snapshot['solar'][0], snapshot['solar'][-1]]
    else:
        snapshot['completed_at'] = str(CUTOFF-pd.Timedelta(hours=1))
    with pytest.raises(ValueError):
        shadow.generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))


def test_scoring_only_uses_completed_valid_actual_quarters(runtime):
    history, model_dir, snapshot, _ = runtime
    forecast = shadow.generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    later = ArchiveHistory(history.path, (CUTOFF+pd.Timedelta(hours=1)).timestamp(),
                           sources_path=history.sources_path, policy_path=history.policy_path)
    report = shadow.score_shadow(later, forecast)
    assert report['matched_actual_quarters'] == 3
    assert report['candidate']['mae_kw'] == pytest.approx(.1)
    assert report['comparisons']['one_hour_origin_baseline_kw']['baseline']['mae_kw'] == 0
    assert report['not_yet_settled_quarters'] > 0
    empty = shadow.score_shadow(history, forecast)
    assert empty['matched_actual_quarters'] == 0
    assert empty['candidate']['mae_kw'] is None


def test_scoring_rejects_changed_policy_or_nonfuture_vintage(runtime):
    history, model_dir, snapshot, _ = runtime
    forecast = shadow.generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    forecast['forecasts'][0]['timestamp'] = str(CUTOFF)
    with pytest.raises(ValueError, match='future'):
        shadow.score_shadow(history, forecast)
    forecast = shadow.generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    history.policy_path.write_text(history.policy_path.read_text() + '# changed\n')
    with pytest.raises(ValueError, match='provenance'):
        shadow.score_shadow(history, forecast)


def test_invalid_future_actuals_are_excluded_not_replaced_with_zero(runtime):
    history, model_dir, snapshot, _ = runtime
    forecast = shadow.generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    with MeasurementArchive(history.path) as db:
        db.ingest('sensor.total_power', None, CUTOFF.timestamp(), CUTOFF.timestamp()+3600,
            [(999, CUTOFF.timestamp()+1200, 'unavailable', '{"unit_of_measurement":"kW"}'),
             (1000, CUTOFF.timestamp()+1800, '3', '{"unit_of_measurement":"kW"}')])
    later = ArchiveHistory(history.path, CUTOFF.timestamp()+3600,
                           sources_path=history.sources_path, policy_path=history.policy_path)
    report = shadow.score_shadow(later, forecast)
    assert report['matched_actual_quarters'] == 2
    assert report['invalid_actual_quarters'] == 1
    assert report['candidate']['mae_kw'] == pytest.approx(.1)


def test_shadow_output_never_overwrites_existing_files(tmp_path):
    output = tmp_path / 'existing.json'
    output.write_text('operator data')
    with pytest.raises(FileExistsError):
        shadow.write_new(output, {'shadow_only': True})
    assert output.read_text() == 'operator data'


@pytest.mark.parametrize('name', ['hepo.db', 'future_predictions.json', 'optimization_plan.json'])
def test_shadow_cannot_create_an_active_artifact_filename(tmp_path, name):
    with pytest.raises(ValueError, match='active'):
        shadow.write_new(tmp_path / name, {})
    assert not (tmp_path / name).exists()
