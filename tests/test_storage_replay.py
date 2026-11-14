import hashlib
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_forward_shadow import CUTOFF, runtime
from utils.forward_shadow import generate_shadow
from utils import storage_replay
from utils.shadow_csv import freeze_csv


@pytest.fixture
def replay(runtime, tmp_path):
    history, model_dir, snapshot, model = runtime
    policies = yaml.safe_load(history.policy_path.read_text())
    policies['battery_power'].update(min_value=-11000, max_value=11000)
    history.policy_path.write_text(yaml.safe_dump(policies))
    model.predict.side_effect = lambda matrix: matrix.baseload_lag_1h.to_numpy()
    vintage = generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    frame = pd.DataFrame({'total_power': 3., 'solar_actual': 0., 'battery_power': 0.,
        'gshp_power': 1000., 'leaf_power': 0., 'outside_temp': 999.},
        index=pd.date_range(CUTOFF-pd.Timedelta(days=1), periods=1440, freq='min'))
    for name in ('total_power', 'solar_actual', 'battery_power', 'gshp_power', 'leaf_power'):
        frame[name+'_source_valid'] = True
    csv = tmp_path / 'raw.csv'
    frame.to_csv(csv, index_label='timestamp')
    return history, model_dir, snapshot, vintage, model, csv, frame


def run(replay, **kwargs):
    history, model_dir, snapshot, vintage, _, csv, _ = replay
    return storage_replay.compare(csv=csv, history=history, model_dir=model_dir,
        snapshot=snapshot, vintage=vintage, **kwargs)


def test_equal_backends_replay_same_model_and_weather_without_live_calls(replay, monkeypatch, tmp_path):
    import predict_future
    import requests
    from utils import sqlite_utils, db_utils
    forbidden = Mock(side_effect=AssertionError('No operational pipeline, network or feedback DB'))
    monkeypatch.setattr(predict_future, 'predict', forbidden)
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    monkeypatch.setattr(sqlite_utils, 'get_db_connection', forbidden)
    monkeypatch.setattr(db_utils, 'get_db_connection', forbidden)
    before = set(tmp_path.rglob('*'))
    report = run(replay, actuals_cutoff=CUTOFF.timestamp()+3600)
    assert report['status'] == 'compared'
    assert report['predictions']['max_absolute_difference_kw'] == 0
    assert report['archive_saved_vintage_replay']['max_absolute_difference_kw'] == 0
    assert report['history']['csv']['heating_observed_mean_kw'] == 1
    assert report['history']['archive']['heating_observed_mean_kw'] == 1
    assert report['retrospective_actuals']['matched_quarters'] == 3
    assert report['retrospective_actuals']['csv']['mae_kw'] == 0
    assert report['retrospective_actuals']['archive']['mae_kw'] == 0
    assert report['optimizer_replay']['status'] == 'not_evaluated'
    assert report['legacy_serving_bit_replay'] is False
    assert report['feature_differences']['leaf_energy_24h']['unit'] == 'kWh'
    assert 'max_absolute_difference_kw' not in report['feature_differences']['leaf_energy_24h']
    for call in replay[4].predict.call_args_list:
        assert call.args[0].outside_temp.eq(5).all()  # Never CSV's 999 weather.
    added = set(tmp_path.rglob('*'))-before
    assert added <= {tmp_path/'archive.sqlite-shm', tmp_path/'archive.sqlite-wal'}
    forbidden.assert_not_called()


def test_anchor_difference_is_the_only_model_input_difference(replay):
    _, _, _, _, _, csv, frame = replay
    frame.loc[CUTOFF-pd.Timedelta(hours=1), 'total_power'] = 4
    frame.to_csv(csv, index_label='timestamp')
    report = run(replay)
    assert report['anchor_differences']['baseload_lag_1h']['archive_minus_csv'] == -1
    assert report['predictions']['max_absolute_difference_kw'] == 1
    assert report['feature_differences']['baseload_lag_1h']['different_rows'] > 0
    assert report['feature_differences']['outside_temp']['different_rows'] == 0
    assert report['raw_differences']['total_power']['different_rows'] == 1


@pytest.mark.parametrize('problem', ['missing_flag', 'partial_window', 'duplicates'])
def test_bad_csv_is_not_silently_shortened_or_certified(replay, problem):
    _, _, _, _, _, csv, frame = replay
    if problem == 'missing_flag':
        frame = frame.drop(columns=['battery_power_source_valid'])
    elif problem == 'partial_window':
        frame = frame.iloc[2:]
    else:
        frame = pd.concat([frame, frame.iloc[-1:]])
    frame.to_csv(csv, index_label='timestamp')
    with pytest.raises(ValueError):
        run(replay)


@pytest.mark.parametrize('problem', ['invalid_leaf', 'battery_spike'])
def test_missing_anchor_blocks_route_instead_of_filling_zero(replay, problem):
    _, _, _, _, _, csv, frame = replay
    if problem == 'invalid_leaf':
        frame.iloc[10, frame.columns.get_loc('leaf_power_source_valid')] = False
    else:
        frame.loc[CUTOFF-pd.Timedelta(hours=1), 'battery_power'] = 1200000
    frame.to_csv(csv, index_label='timestamp')
    report = run(replay)
    assert report['status'] == 'blocked_inputs'
    assert report['history']['csv']['missing_anchors']
    assert report['history']['archive']['missing_anchors'] == []
    assert report['predictions'] is None


@pytest.mark.parametrize('problem', ['model', 'snapshot', 'policy'])
def test_changed_controls_cannot_be_presented_as_storage_only(replay, problem):
    history, model_dir, snapshot, _, _, _, _ = replay
    if problem == 'model':
        (model_dir/'energy_model.json').write_text('{"changed":true}')
    elif problem == 'snapshot':
        snapshot['weather'][0]['temperature'] = -30
    else:
        history.policy_path.write_text(history.policy_path.read_text()+'# changed\n')
    with pytest.raises(ValueError, match='control|provenance|model|snapshot'):
        run(replay)


def test_origin_frozen_sidecar_is_bound_to_generation_and_replay(replay, tmp_path):
    history, model_dir, snapshot, _, model, csv, _ = replay
    frozen = tmp_path/'shadow-history.csv'
    metadata = freeze_csv(csv, frozen, history.cutoff)
    metadata['frozen_at'] = (CUTOFF-pd.Timedelta(seconds=1)).isoformat()
    snapshot['csv_sidecar'] = metadata
    vintage = generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    bound = [history, model_dir, snapshot, vintage, model, frozen, replay[-1]]
    assert run(bound)['status'] == 'compared'
    frozen.write_text(frozen.read_text()+'\n')
    with pytest.raises(ValueError, match='hash'): run(bound)
    model.predict.reset_mock()
    with pytest.raises(ValueError, match='hash'):
        generate_shadow(history, model_dir, snapshot, now=CUTOFF+pd.Timedelta(minutes=1))
    model.predict.assert_not_called()
