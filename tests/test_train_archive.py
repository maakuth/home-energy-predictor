from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from process_data import process_frame
from train_model import train, main


def test_frame_training_uses_existing_quality_gate_and_never_reads_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    index = pd.date_range('2026-10-21', periods=160, freq='15min', tz='UTC')
    raw = pd.DataFrame({'outside_temp': 5.0, 'wind_speed': 2.0, 'solar_forecast': 0.0,
                         'total_power': 3.0, 'solar_actual': 0.0, 'gshp_power': 1000.0,
                         'leaf_power': 0.0, 'battery_power': 0.0}, index=index)
    frame = process_frame(raw)
    model = MagicMock()
    model.predict.side_effect = lambda features: np.full(len(features), 2.0)
    with patch('train_model.pd.read_csv', side_effect=AssertionError('CSV must not be used')), \
         patch('train_model.xgb.XGBRegressor', return_value=model):
        train(frame=frame, min_coverage_days=.1)
    model.fit.assert_called_once()
    assert model.fit.call_args.args[1].eq(2.0).all()
    assert (tmp_path / 'state/model_features.json').exists()


def test_archive_cli_passes_explicit_window_without_csv_fallback(tmp_path):
    fake = pd.DataFrame({'baseload_power': [2.0]})
    with patch('train_model.load_archive_features', return_value=fake) as load, \
         patch('train_model.train') as fit:
        main(['--archive', str(tmp_path / 'measurements.sqlite'),
              '--from', '2026-10-21T00:00:00+03:00', '--to', '2026-10-22T00:00:00+03:00'])
    assert load.call_args.args[1] == pd.Timestamp('2026-10-21T00:00:00+03:00').timestamp()
    assert fit.call_args.kwargs['frame'] is fake


@pytest.mark.parametrize('extra', [[], ['--from', '2026-10-21T00:00:00'],
                                  ['--from', '2026-10-22T00:00:00Z', '--to', '2026-10-21T00:00:00Z']])
def test_archive_cli_rejects_missing_naive_or_reversed_bounds(tmp_path, extra):
    with patch('train_model.load_archive_features') as load:
        with pytest.raises(SystemExit):
            main(['--archive', str(tmp_path / 'measurements.sqlite'), *extra])
    load.assert_not_called()


def test_archive_failure_never_falls_back_to_csv(tmp_path):
    with patch('train_model.load_archive_features', side_effect=FileNotFoundError('missing archive')), \
         patch('train_model.pd.read_csv', side_effect=AssertionError('CSV fallback is forbidden')) as read:
        with pytest.raises(FileNotFoundError, match='missing archive'):
            main(['--archive', str(tmp_path / 'absent.sqlite'),
                  '--from', '2026-10-21T00:00:00Z', '--to', '2026-10-22T00:00:00Z'])
    read.assert_not_called()


def test_archive_cli_declares_legacy_policy_and_requires_opt_in():
    window = ['--archive', 'archive.sqlite', '--from', '2026-10-21T00:00:00Z',
              '--to', '2026-10-22T00:00:00Z']
    with patch('train_model.load_archive_features', return_value=pd.DataFrame()) as load, \
         patch('train_model.train'):
        with pytest.raises(SystemExit):
            main([*window, '--legacy-policy', 'complete-legacy'])
        load.assert_not_called()
        main([*window, '--legacy-policy', 'complete-legacy', '--allow-legacy'])
        assert load.call_args.kwargs['legacy_policy'] == 'complete-legacy'


def test_legacy_policy_saved_with_model_without_claiming_verification(tmp_path, monkeypatch):
    import json
    from tests.test_archive_eligibility import raw_frame
    from utils.archive_quality import apply_training_policy
    monkeypatch.chdir(tmp_path)
    raw = raw_frame()
    values = raw[[name for name in raw if not name.endswith(('_provenance', '_coverage_fraction', '_conflict'))]]
    frame = apply_training_policy(raw, process_frame(values), 'complete-legacy')
    model = MagicMock()
    model.predict.side_effect = lambda features: np.full(len(features), 2.)
    # More than ten eligible samples after the 24h lag.
    frame = pd.concat([frame.iloc[-4:]] * 4, ignore_index=True)
    frame.index = pd.date_range('2026-10-22', periods=len(frame), freq='15min', tz='UTC')
    with patch('train_model.xgb.XGBRegressor', return_value=model):
        train(frame=frame, min_coverage_days=.1)
    model.fit.assert_called_once()
    metadata = json.loads((tmp_path / 'state/model_training_metadata.json').read_text())
    assert metadata['archive_training']['legacy_policy'] == 'complete-legacy'
    assert metadata['archive_training']['legacy_is_source_verified'] is False
    assert metadata['eligible_rows'] == 16
