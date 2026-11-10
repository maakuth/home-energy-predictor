import json

import numpy as np
import pandas as pd
import pytest

from utils import measurement_shadow as shadow


START = pd.Timestamp('2026-10-21T00:00:00Z').timestamp()


def frames():
    index = pd.date_range('2026-10-20', periods=26, freq='h', tz='UTC', name='timestamp')
    raw = pd.DataFrame({'total_power': 3., 'solar_actual': 0., 'battery_power': 0.,
                        'gshp_power': 1000., 'leaf_power': 0., 'outside_temp': 5.,
                        'wind_speed': 2., 'solar_forecast': 0.}, index=index)
    window = index[-2:]
    archive = raw.loc[window].copy()
    for name in raw:
        archive[name + '_source_valid'] = True
        archive[name + '_coverage_fraction'] = 1.
        archive[name + '_conflict'] = False
        archive[name + '_provenance'] = 'recorder'
    features = pd.DataFrame({'baseload_power': [2., 2.], 'baseload_lag_24h': [2., 2.],
                             'baseload_target_valid': True, 'baseload_training_eligible': True,
                             'baseload_lag_24h_training_eligible': True}, index=window)
    features.attrs['archive_training'] = {'legacy_policy': 'verified-only', 'imports': []}
    return raw, archive, features


def setup_inputs(tmp_path, monkeypatch):
    raw, archive, features = frames()
    csv = tmp_path / 'explicit.csv'
    raw.to_csv(csv)
    db = tmp_path / 'archive.sqlite'
    db.touch()
    sources = tmp_path / 'sources.yaml'
    sources.write_text('outside_temp:\n  - entity: sensor.outside\n')
    calls = []

    def load_raw(*args, **kwargs):
        calls.append(('raw', args, kwargs))
        return archive.copy()

    def load_features(*args, **kwargs):
        calls.append(('features', args, kwargs))
        result = features.copy()
        result.attrs['archive_training']['legacy_policy'] = kwargs['legacy_policy']
        return result

    monkeypatch.setattr(shadow, 'load_archive_raw', load_raw)
    monkeypatch.setattr(shadow, 'load_archive_features', load_features)
    options = dict(csv=csv, csv_mode='raw', archive=db, sources=sources,
                   start=START, end=START + 7200, interval_seconds=3600)
    return options, calls, raw, archive, features


def test_explicit_loader_contract_and_matched_baseline(tmp_path, monkeypatch):
    options, calls, _, _, _ = setup_inputs(tmp_path, monkeypatch)
    report = shadow.compare(**options)
    assert [call[0] for call in calls] == ['raw', 'features']
    assert calls[1][2]['legacy_policy'] == 'verified-only'
    assert calls[1][2]['sources_path'] == options['sources']
    assert report['baseline']['verified']['matched_rows'] == 2
    assert report['baseline']['verified']['archive']['mae_kw'] == 0
    assert report['baseline']['legacy_eligible']['matched_rows'] == 0
    assert report['archive_training']['legacy_policy'] == 'verified-only'
    json.dumps(report, allow_nan=False)


def test_legacy_policy_requires_explicit_opt_in(tmp_path, monkeypatch):
    options, calls, _, _, _ = setup_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='allow_legacy'):
        shadow.compare(**options, legacy_policy='complete-legacy')
    assert not calls
    shadow.compare(**options, legacy_policy='complete-legacy', allow_legacy=True)
    assert calls[1][2]['legacy_policy'] == 'complete-legacy'
    assert calls[1][2]['allow_legacy'] is True


@pytest.mark.parametrize('missing', ['csv', 'archive', 'sources'])
def test_absent_explicit_files_fail_without_fallback(tmp_path, monkeypatch, missing):
    options, calls, _, _, _ = setup_inputs(tmp_path, monkeypatch)
    options[missing] = tmp_path / 'absent'
    with pytest.raises(FileNotFoundError):
        shadow.compare(**options)
    assert not calls


def test_mismatched_csv_window_is_not_silently_intersected(tmp_path, monkeypatch):
    options, calls, raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    raw.iloc[:-1].to_csv(options['csv'])
    with pytest.raises(ValueError, match='window'):
        shadow.compare(**options)
    assert not calls


def test_naive_or_duplicate_csv_timestamps_rejected(tmp_path, monkeypatch):
    options, _, raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    raw.index = raw.index.tz_localize(None)
    raw.to_csv(options['csv'])
    with pytest.raises(ValueError, match='timezone'):
        shadow.compare(**options)
    raw.index = raw.index.tz_localize('UTC')
    pd.concat([raw, raw.iloc[-1:]]).to_csv(options['csv'])
    with pytest.raises(ValueError, match='Duplicate'):
        shadow.compare(**options)


def test_coverage_conflict_missing_actuals_and_legacy_are_honest(tmp_path, monkeypatch):
    options, _, raw, archive, features = setup_inputs(tmp_path, monkeypatch)
    archive.loc[archive.index[0], 'total_power'] = np.nan
    archive.loc[archive.index[0], 'total_power_coverage_fraction'] = .5
    archive.loc[archive.index[0], 'total_power_conflict'] = True
    features.loc[features.index[0], 'baseload_power'] = np.nan
    features.loc[features.index[0], 'baseload_training_eligible'] = False
    features['baseload_target_valid'] = False
    raw.loc[raw.index[-2], 'solar_actual'] = np.nan
    raw.to_csv(options['csv'])
    report = shadow.compare(**options, allow_legacy=True, legacy_policy='complete-legacy')
    assert report['raw']['total_power']['archive_missing_rows'] == 1
    assert report['raw']['total_power']['archive_conflict_rows'] == 1
    assert report['raw']['total_power']['archive_mean_coverage_fraction'] == .75
    assert report['eligibility']['archive']['verified'] == 0
    assert report['eligibility']['archive']['legacy_eligible'] == 1
    assert report['baseline']['legacy_eligible']['matched_rows'] == 1
    assert report['baseline']['verified']['matched_rows'] == 0


def test_processed_mode_without_truth_metadata_does_not_score_filled_targets(tmp_path, monkeypatch):
    options, _, _, _, features = setup_inputs(tmp_path, monkeypatch)
    options['csv_mode'] = 'processed'
    features.drop(columns=['baseload_target_valid']).to_csv(options['csv'])
    report = shadow.compare(**options)
    assert report['raw'] is None
    assert report['baseline']['verified']['matched_rows'] == 0
    assert 'quality metadata' in report['csv_actuals_note']


def test_cli_requires_explicit_inputs_and_timezone_bounds():
    with pytest.raises(SystemExit):
        shadow.main([])
    with pytest.raises(SystemExit):
        shadow.main(['--csv', 'x', '--csv-mode', 'raw', '--archive', 'y', '--sources', 'z',
                     '--from', '2026-10-21', '--to', '2026-10-22'])


def test_archive_window_mismatch_fails(tmp_path, monkeypatch):
    options, _, _, archive, _ = setup_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(shadow, 'load_archive_raw', lambda *a, **kw: archive.iloc[:1])
    with pytest.raises(ValueError, match='mismatched window'):
        shadow.compare(**options)


def test_missing_csv_lag_is_not_filled_or_scored(tmp_path, monkeypatch):
    options, _, raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    raw.loc[raw.index[0], 'solar_actual'] = np.nan
    raw.to_csv(options['csv'])
    report = shadow.compare(**options)
    assert report['baseline']['verified']['matched_rows'] == 1
    assert report['eligibility']['csv_honest_actual_rows'] == 2


def test_processed_mode_uses_exact_actual_lags_not_filled_lag_column(tmp_path, monkeypatch):
    options, _, raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    options['csv_mode'] = 'processed'
    processed = pd.DataFrame({'baseload_power': 2., 'baseload_target_valid': True,
                              'baseload_quality_provenance': 'legacy_finite_checks_only',
                              'baseload_lag_24h': 999.}, index=raw.index)
    processed.loc[processed.index[0], 'baseload_target_valid'] = False
    processed.to_csv(options['csv'])
    report = shadow.compare(**options)
    assert report['baseline']['verified']['matched_rows'] == 1
    assert report['baseline']['verified']['csv']['mae_kw'] == 0


def test_offset_timestamps_are_aligned_in_utc(tmp_path, monkeypatch):
    options, _, raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    raw.index = raw.index.tz_convert('Europe/Helsinki')
    raw.to_csv(options['csv'])
    assert shadow.compare(**options)['baseline']['verified']['matched_rows'] == 2


def test_cli_stdout_is_json(tmp_path, monkeypatch, capsys):
    options, _, _, _, _ = setup_inputs(tmp_path, monkeypatch)
    assert shadow.main(['--csv', str(options['csv']), '--csv-mode', 'raw',
                        '--archive', str(options['archive']), '--sources', str(options['sources']),
                        '--from', '2026-10-21T03:00:00+03:00', '--to', '2026-10-21T02:00:00Z',
                        '--interval-seconds', '3600']) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)['window']['expected_rows'] == 2
    assert 'Denoising' in captured.err


def test_forecast_differences_and_bias_sign_are_explicit(tmp_path, monkeypatch):
    options, _, _, _, features = setup_inputs(tmp_path, monkeypatch)
    features['baseload_lag_24h'] = 3.
    report = shadow.compare(**options)
    summary = report['baseline']['verified']
    assert summary['archive']['bias_kw'] == 1.
    assert summary['csv']['bias_kw'] == 0.
    assert summary['prediction_difference']['bias_kw'] == 1.
    assert summary['actual_difference']['mae_kw'] == 0.


def test_no_eligible_lag_produces_null_not_zero_errors(tmp_path, monkeypatch):
    options, _, _, _, features = setup_inputs(tmp_path, monkeypatch)
    features['baseload_lag_24h_training_eligible'] = False
    report = shadow.compare(**options)
    assert report['baseline']['verified']['matched_rows'] == 0
    assert report['baseline']['verified']['archive']['mae_kw'] is None
    assert report['baseline']['verified']['csv']['mae_kw'] is None


def test_interior_csv_timestamp_gap_is_reported_and_not_filled(tmp_path, monkeypatch):
    options, _, raw, archive, features = setup_inputs(tmp_path, monkeypatch)
    # Extend the fixture to three rows so the missing row is interior, not a shorter window.
    third = pd.Timestamp(START + 7200, unit='s', tz='UTC')
    raw.loc[third] = raw.iloc[-1]
    raw = raw.drop(pd.Timestamp(START + 3600, unit='s', tz='UTC'))
    raw.to_csv(options['csv'])
    archive.loc[third] = archive.iloc[-1]
    features.loc[third] = features.iloc[-1]
    options['end'] = START + 10800
    report = shadow.compare(**options)
    assert report['window']['csv_missing_timestamps'] == 1
    assert report['raw']['total_power']['csv_missing_rows'] == 1
    assert report['eligibility']['csv_honest_actual_rows'] == 2


def test_csv_explicit_false_source_flag_excludes_actual(tmp_path, monkeypatch):
    options, _, raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    raw['solar_actual_source_valid'] = True
    raw.loc[raw.index[-1], 'solar_actual_source_valid'] = False
    raw.to_csv(options['csv'])
    report = shadow.compare(**options)
    assert report['eligibility']['csv_honest_actual_rows'] == 1
    assert report['baseline']['verified']['matched_rows'] == 1


def test_real_read_only_archive_time_weighted_comparison(tmp_path):
    from utils.measurement_archive import MeasurementArchive

    raw, _, _ = frames()
    csv = tmp_path / 'raw.csv'
    raw.to_csv(csv)
    archive_path = tmp_path / 'archive.sqlite'
    sources = tmp_path / 'sources.yaml'
    sources.write_text(''.join(f'{name}:\n  - entity: sensor.{name}\n' for name in raw))
    with MeasurementArchive(archive_path) as archive:
        for i, name in enumerate(raw):
            events = [(i * 10, START - 86400, str(raw[name].iloc[0]), None)]
            if name == 'total_power':
                events.append((i * 10 + 1, START + 1800, '5', None))
            archive.ingest('sensor.' + name, None, START - 86400, START + 7200, events)
    before = archive_path.read_bytes()
    report = shadow.compare(csv=csv, csv_mode='raw', archive=archive_path, sources=sources,
                            start=START, end=START + 7200, interval_seconds=3600)
    assert report['raw']['total_power']['archive_minus_csv_mean'] == 1.5
    assert report['raw']['total_power']['archive_mean_coverage_fraction'] == 1.
    assert report['raw']['total_power']['archive_source_verified_rows'] == 2
    assert report['baseline']['verified']['matched_rows'] == 2
    assert report['baseline']['verified']['archive']['mae_kw'] == 1.5
    assert report['features']['baseload_power']['mean_absolute_difference'] == 1.5
    assert archive_path.read_bytes() == before
