from __future__ import annotations

import pickle
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sarimax_predictor as predictor
import train_sarima as training


def features(periods=288, freq='15min'):
    frame = pd.DataFrame({'baseload_power': 2.0, 'baseload_target_valid': False,
                          'baseload_training_eligible': True},
                         index=pd.date_range('2026-04-01', periods=periods, freq=freq, tz='UTC'))
    frame.attrs['archive_eligibility'] = {'legacy_policy': 'complete-legacy'}
    return frame


def test_dataframe_uses_distinct_eligibility_without_changing_source_truth():
    frame = features()
    frame.loc[frame.index[1], 'baseload_training_eligible'] = False
    result = predictor.load_historical_data(frame)
    assert result is not None
    assert result.iloc[0] == 2
    assert np.isnan(result.iloc[1])
    assert not frame.baseload_target_valid.any()
    assert result.index.equals(frame.index)


def test_archive_dataframe_requires_eligibility_no_csv_fallback():
    with pytest.raises(ValueError, match='eligib'):
        predictor.load_historical_data(features().drop(columns='baseload_training_eligible'))


def test_archive_partial_quarters_and_nonfinite_targets_are_missing():
    frame = features(45, '1min')
    frame.loc[frame.index[16], 'baseload_training_eligible'] = False
    frame.loc[frame.index[31], 'baseload_power'] = np.inf
    result = predictor.load_historical_data(frame)
    assert result is not None
    assert result.iloc[0] == 2
    assert result.iloc[1:].isna().all()


def test_archive_nondivisor_resolution_is_rejected():
    with pytest.raises(ValueError, match='resolution'):
        predictor.load_historical_data(features(30, '10min'))


def test_training_persists_archive_metadata_after_materializing(tmp_path):
    frame = features()
    result = MagicMock(params=np.array([0.5]))
    result.get_forecast.return_value.predicted_mean = pd.Series([2.0])
    with patch('train_sarima.SARIMAX') as constructor:
        constructor.return_value.fit.return_value = result
        training.train_sarima(data_path=frame, params_path=str(tmp_path / 'params.pkl'))
    with (tmp_path / 'params.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    assert saved['training_metadata']['input'] == 'measurement_archive'
    assert saved['training_metadata']['archive_eligibility']['legacy_policy'] == 'complete-legacy'
    assert saved['last_index'] == frame.index[-1]


@pytest.mark.parametrize('module', [training, predictor])
def test_cli_archive_materializes_explicit_window_before_fit(module, tmp_path):
    params = tmp_path / 'params.pkl'
    with params.open('wb') as handle:
        pickle.dump({'params': np.array([.5]), 'order': (1, 1, 1),
                     'seasonal_order': (1, 1, 0, 96)}, handle)
    argv = ['--archive', 'history.sqlite', '--from', '2026-04-01T00:00:00+03:00',
            '--to', '2026-04-04T00:00:00+03:00', '--sources', 'sources.yaml',
            '--allow-legacy', '--legacy-policy', 'complete-legacy',
            '--preferred-import', '2', '--params-path', str(params), '--interval-seconds', '900']
    if module is predictor:
        argv += ['--no-archive-results', '--output', str(tmp_path / 'forecast.json')]
    with patch('sarimax_predictor.load_archive_features', return_value=features()) as loader, \
         patch.object(module, 'SARIMAX') as constructor:
        constructor.return_value.fit.return_value.get_forecast.return_value.predicted_mean = pd.Series([2.0])
        if module is predictor:
            with patch.object(module, 'predict_sarimax', return_value=(pd.Series([2.0], index=features().index[:1]), None)):
                module.main(argv)
        else:
            module.main(argv)
    args, kwargs = loader.call_args
    assert args == ('history.sqlite', pd.Timestamp('2026-03-31T21:00Z').timestamp(),
                    pd.Timestamp('2026-04-03T21:00Z').timestamp())
    assert kwargs['legacy_policy'] == 'complete-legacy'
    assert kwargs['allow_legacy'] is True
    assert kwargs['preferred_imports'] == [2]
    assert kwargs['sources_path'] == 'sources.yaml'


@pytest.mark.parametrize('module', [training, predictor])
@pytest.mark.parametrize('options', [
    ['--archive', 'a'],
    ['--archive', 'a', '--from', '2026-04-01', '--to', '2026-04-02'],
    ['--archive', 'a', '--from', '2026-04-02T00:00Z', '--to', '2026-04-01T00:00Z'],
    ['--legacy-policy', 'complete-legacy'],
    ['--archive', 'a', '--from', '2026-04-01T00:00Z', '--to', '2026-04-02T00:00Z',
     '--legacy-policy', 'complete-legacy'],
])
def test_cli_rejects_ambiguous_archive_options(module, options):
    with pytest.raises(SystemExit):
        module.main(options)


def test_archive_failure_never_reads_csv():
    with patch('sarimax_predictor.load_archive_features', side_effect=ValueError('archive broken')), \
         patch('pandas.read_csv') as csv:
        with pytest.raises(ValueError, match='archive broken'):
            training.main(['--archive', 'missing', '--from', '2026-04-01T00:00Z',
                           '--to', '2026-04-04T00:00Z'])
        csv.assert_not_called()


def test_full_bins_keep_gaps_and_reject_partial_edges():
    frame = features(45, '1min').iloc[1:].drop(pd.Timestamp('2026-04-01T00:20Z'))
    frame.attrs['sarima_interval_seconds'] = 60
    result = predictor.load_historical_data(frame)
    assert result is not None
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 2


def test_archive_rejects_duplicates_and_off_grid_timestamps():
    frame = features()
    with pytest.raises(ValueError, match='Duplicate'):
        predictor.load_historical_data(pd.concat([frame, frame.iloc[:1]]))
    frame.index += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='align'):
        predictor.load_historical_data(frame)


def test_all_ineligible_archive_does_not_fit_or_overwrite(tmp_path):
    frame = features()
    frame['baseload_training_eligible'] = False
    params = tmp_path / 'params.pkl'
    params.write_bytes(b'keep previous model')
    with patch('train_sarima.SARIMAX') as constructor:
        training.train_sarima(data_path=frame, params_path=str(params))
        constructor.assert_not_called()
    assert params.read_bytes() == b'keep previous model'


def test_real_archive_snapshot_is_closed_before_fitting(tmp_path):
    from utils.archive_features import load_archive_features
    from utils.measurement_archive import MeasurementArchive
    from utils.measurement_reader import MeasurementReader
    from utils.measurement_sources import parse_sources

    start = pd.Timestamp('2026-04-01T00:00Z').timestamp()
    path = tmp_path / 'history.sqlite'
    values = {'total_power': 3, 'solar_actual': 0, 'battery_power': 0,
              'gshp_power': 1000, 'leaf_power': 0, 'outside_temp': 5,
              'wind_speed': 2, 'solar_forecast': 0}
    with MeasurementArchive(path) as archive:
        for i, (name, value) in enumerate(values.items()):
            archive.ingest('sensor.' + name, None, start - 86400, start + 3 * 86400,
                           [(i, start - 86400, str(value), None)])
    sources = parse_sources({name: [{'entity': 'sensor.' + name}] for name in values})
    readers = []

    class TrackedReader(MeasurementReader):
        def __enter__(self):
            readers.append(self)
            return super().__enter__()

    with patch('utils.archive_features.MeasurementReader', TrackedReader):
        frame = load_archive_features(path, start, start + 3 * 86400, sources=sources)
    # This branch predates the shared policy patch. Only supply its conservative
    # verified-only eligibility when absent; integration uses the real column.
    if 'baseload_training_eligible' not in frame:
        frame['baseload_training_eligible'] = frame.baseload_target_valid

    def fitted(*args, **kwargs):
        assert len(readers) == 1
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            readers[0].db.execute('SELECT 1')
        result = MagicMock(params=np.array([.5]))
        result.get_forecast.return_value.predicted_mean = pd.Series([2.0])
        return result

    with patch('train_sarima.SARIMAX') as constructor, patch('pandas.read_csv') as csv:
        constructor.return_value.fit.side_effect = fitted
        training.train_sarima(data_path=frame, params_path=str(tmp_path / 'params.pkl'))
        constructor.return_value.fit.assert_called_once()
        csv.assert_not_called()
