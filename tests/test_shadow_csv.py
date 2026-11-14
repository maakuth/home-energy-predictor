import hashlib
import json
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from utils import shadow_csv
from utils import forward_shadow as shadow


CUTOFF = pd.Timestamp('2026-11-14T12:00:00Z')


@pytest.fixture
def csv_input(tmp_path):
    frame = pd.DataFrame(index=pd.date_range(CUTOFF-pd.Timedelta(days=1), periods=1440, freq='min'))
    for name in ('total_power', 'solar_actual', 'battery_power', 'gshp_power', 'leaf_power'):
        frame[name] = 1.
        frame[name+'_source_valid'] = True
    source = tmp_path/'raw.csv'
    frame.to_csv(source, index_label='timestamp')
    return source, tmp_path/'shadow-history.csv', frame


def test_freeze_preserves_bytes_and_declared_validity_not_recorder_verification(csv_input):
    source, output, _ = csv_input
    before = source.read_bytes()
    metadata = shadow_csv.freeze_csv(source, output, CUTOFF.timestamp())
    assert output.read_bytes() == source.read_bytes() == before
    assert metadata['sha256'] == hashlib.sha256(before).hexdigest()
    assert metadata['history_cutoff'] == CUTOFF.timestamp()
    assert metadata['required_minutes'] == 1440
    assert metadata['source_verification'] == 'not_established'
    assert metadata['available_declared_valid_minutes']['total_power'] == 1440
    assert metadata['frozen_at']


@pytest.mark.parametrize('problem', ['stale', 'interior_gap', 'duplicate', 'unordered',
                                     'naive', 'unaligned', 'missing_flag'])
def test_incomplete_or_malformed_csv_is_not_fabricated(csv_input, problem):
    source, output, frame = csv_input
    if problem == 'stale': frame = frame.iloc[:-1]
    elif problem == 'interior_gap': frame = frame.drop(frame.index[500])
    elif problem == 'duplicate': frame = pd.concat([frame.iloc[:1], frame])
    elif problem == 'unordered': frame = frame.iloc[::-1]
    elif problem == 'naive': frame.index = frame.index.tz_localize(None)
    elif problem == 'unaligned': frame.index += pd.Timedelta(seconds=1)
    else: frame = frame.drop(columns=['solar_actual_source_valid'])
    frame.to_csv(source, index_label='timestamp')
    with pytest.raises(ValueError): shadow_csv.freeze_csv(source, output, CUTOFF.timestamp())
    assert not output.exists()


def test_missing_values_are_retained_and_existing_outputs_not_overwritten(csv_input):
    source, output, frame = csv_input
    frame.loc[frame.index[-1], 'solar_actual'] = float('nan')
    frame.loc[frame.index[-1], 'solar_actual_source_valid'] = False
    frame.to_csv(source, index_label='timestamp')
    metadata = shadow_csv.freeze_csv(source, output, CUTOFF.timestamp())
    assert metadata['available_declared_valid_minutes']['solar_actual'] == 1439
    before = output.read_bytes()
    with pytest.raises(FileExistsError): shadow_csv.freeze_csv(source, output, CUTOFF.timestamp())
    assert output.read_bytes() == before


def test_size_limit_and_active_names_fail_before_writes(csv_input, monkeypatch):
    source, output, _ = csv_input
    monkeypatch.setattr(shadow_csv, 'MAX_CSV_BYTES', 16)
    with pytest.raises(ValueError, match='size'): shadow_csv.freeze_csv(source, output, CUTOFF.timestamp())
    assert not output.exists()
    with pytest.raises(ValueError): shadow_csv.freeze_csv(source, output.with_name('optimization_plan.json'), CUTOFF.timestamp())


def test_sidecar_hash_cutoff_and_observation_time_are_bound_to_snapshot(csv_input):
    source, output, _ = csv_input
    metadata = shadow_csv.freeze_csv(source, output, CUTOFF.timestamp())
    snapshot = {'history_cutoff': CUTOFF.timestamp(), 'started_at': metadata['frozen_at'], 'csv_sidecar': metadata}
    shadow_csv.validate_sidecar(snapshot)
    snapshot['history_cutoff'] += 60
    with pytest.raises(ValueError, match='cutoff'): shadow_csv.validate_sidecar(snapshot)
    snapshot['history_cutoff'] = CUTOFF.timestamp()
    snapshot['started_at'] = '2000-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='before'): shadow_csv.validate_sidecar(snapshot)
    snapshot['started_at'] = metadata['frozen_at']
    output.write_text('changed history')
    with pytest.raises(ValueError, match='hash'): shadow_csv.validate_sidecar(snapshot)


def test_capture_freezes_before_ha_and_does_not_change_the_capture_clock(csv_input, monkeypatch):
    source, output, _ = csv_input
    clock = Mock(wraps=shadow.datetime)
    clock.now.return_value = CUTOFF.to_pydatetime()
    monkeypatch.setattr(shadow, 'datetime', clock)
    def captured(*args):
        assert output.exists()
        return {'history_cutoff': CUTOFF.timestamp(),
                'started_at': pd.Timestamp.now(tz='UTC').isoformat(), 'completed_at': pd.Timestamp.now(tz='UTC').isoformat()}
    capture = Mock(side_effect=captured)
    monkeypatch.setattr(shadow, 'capture_inputs', capture)
    result = shadow.capture_with_csv(Mock(), 'http://ha', 'secret', source, output)
    assert result['history_cutoff'] == CUTOFF.timestamp()
    assert 'secret' not in json.dumps(result)
    capture.assert_called_once()


def test_stale_csv_blocks_before_any_ha_access(csv_input, monkeypatch):
    source, output, frame = csv_input
    frame.iloc[:-1].to_csv(source, index_label='timestamp')
    clock = Mock(wraps=shadow.datetime)
    clock.now.return_value = CUTOFF.to_pydatetime()
    monkeypatch.setattr(shadow, 'datetime', clock)
    capture = Mock()
    monkeypatch.setattr(shadow, 'capture_inputs', capture)
    with pytest.raises(ValueError): shadow.capture_with_csv(Mock(), 'ha', 'secret', source, output)
    capture.assert_not_called()


def test_capture_minute_rollover_is_rejected_not_backdated(csv_input, monkeypatch):
    source, output, _ = csv_input
    clock = Mock(wraps=shadow.datetime)
    clock.now.return_value = CUTOFF.to_pydatetime()
    monkeypatch.setattr(shadow, 'datetime', clock)
    monkeypatch.setattr(shadow, 'capture_inputs', Mock(return_value={
        'history_cutoff': CUTOFF.timestamp()+60, 'started_at': pd.Timestamp.now(tz='UTC').isoformat()}))
    with pytest.raises(ValueError, match='cutoff'):
        shadow.capture_with_csv(Mock(), 'ha', 'secret', source, output)


def test_source_replacement_during_copy_is_rejected(csv_input, monkeypatch):
    source, output, _ = csv_input
    original = Path.stat
    calls = 0
    def stat(path, *args, **kwargs):
        nonlocal calls
        result = original(path, *args, **kwargs)
        if path == source:
            calls += 1
            if calls == 2:
                return SimpleNamespace(st_ino=result.st_ino+1, st_size=result.st_size,
                                       st_mtime_ns=result.st_mtime_ns)
        return result
    monkeypatch.setattr(Path, 'stat', stat)
    with pytest.raises(ValueError, match='changed'):
        shadow_csv.freeze_csv(source, output, CUTOFF.timestamp())
    assert not output.exists()


@pytest.mark.parametrize('extra', [['--csv', 'raw.csv'], ['--csv-output', 'sidecar.csv'],
    ['--csv', 'raw.csv', '--csv-output', 'snapshot.json'], ['--wait-for-csv-seconds', '900']])
def test_cli_rejects_incomplete_or_colliding_outputs_before_ha(tmp_path, extra, monkeypatch):
    capture = Mock()
    monkeypatch.setattr(shadow, 'capture_inputs', capture)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        shadow.main(['capture', '--env-file', 'unused.env', '--output', 'snapshot.json', *extra])
    capture.assert_not_called()


def test_waits_for_normal_csv_refresh_before_freezing_and_capturing(csv_input, monkeypatch):
    source, output, frame = csv_input
    frame.iloc[:-1].to_csv(source, index_label='timestamp')
    ready = CUTOFF+pd.Timedelta(minutes=15)
    clock = Mock(wraps=shadow.datetime)
    clock.now.side_effect = [CUTOFF.to_pydatetime(), ready.to_pydatetime()]
    monkeypatch.setattr(shadow, 'datetime', clock)
    monkeypatch.setattr(shadow.time, 'monotonic', Mock(side_effect=[0., 0.]))
    def refresh(seconds):
        assert not output.exists()
        frame.index += pd.Timedelta(minutes=15)
        frame.to_csv(source, index_label='timestamp')
    sleep = Mock(side_effect=refresh)
    monkeypatch.setattr(shadow.time, 'sleep', sleep)
    capture = Mock(side_effect=lambda *args: {'history_cutoff': ready.timestamp(),
                                              'started_at': pd.Timestamp.now(tz='UTC').isoformat()})
    monkeypatch.setattr(shadow, 'capture_inputs', capture)
    snapshot = shadow.capture_with_csv(Mock(), 'ha', 'secret', source, output, wait_seconds=900)
    assert snapshot['history_cutoff'] == snapshot['csv_sidecar']['history_cutoff'] == ready.timestamp()
    sleep.assert_called_once_with(2.)
    capture.assert_called_once()


def test_wait_timeout_keeps_no_sidecar_and_does_not_contact_ha(csv_input, monkeypatch):
    source, output, frame = csv_input
    frame.iloc[:-1].to_csv(source, index_label='timestamp')
    clock = Mock(wraps=shadow.datetime)
    clock.now.return_value = CUTOFF.to_pydatetime()
    monkeypatch.setattr(shadow, 'datetime', clock)
    monkeypatch.setattr(shadow.time, 'monotonic', Mock(side_effect=[0., 901.]))
    capture, sleep = Mock(), Mock()
    monkeypatch.setattr(shadow, 'capture_inputs', capture)
    monkeypatch.setattr(shadow.time, 'sleep', sleep)
    with pytest.raises(ValueError, match='missing'):
        shadow.capture_with_csv(Mock(), 'ha', 'secret', source, output, wait_seconds=900)
    capture.assert_not_called()
    sleep.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize('wait', [-1, float('nan'), float('inf'), 1801])
def test_invalid_wait_budget_does_not_freeze_or_contact_ha(csv_input, wait, monkeypatch):
    source, output, _ = csv_input
    capture = Mock()
    monkeypatch.setattr(shadow, 'capture_inputs', capture)
    with pytest.raises(ValueError, match='wait'):
        shadow.capture_with_csv(Mock(), 'ha', 'secret', source, output, wait_seconds=wait)
    assert not output.exists()
    capture.assert_not_called()


def test_wait_does_not_retry_malformed_csv(csv_input, monkeypatch):
    source, output, frame = csv_input
    frame.drop(columns=['solar_actual_source_valid']).to_csv(source, index_label='timestamp')
    clock = Mock(wraps=shadow.datetime)
    clock.now.return_value = CUTOFF.to_pydatetime()
    monkeypatch.setattr(shadow, 'datetime', clock)
    sleep, capture = Mock(), Mock()
    monkeypatch.setattr(shadow.time, 'sleep', sleep)
    monkeypatch.setattr(shadow, 'capture_inputs', capture)
    with pytest.raises(ValueError, match='flags'):
        shadow.capture_with_csv(Mock(), 'ha', 'secret', source, output, wait_seconds=900)
    sleep.assert_not_called()
    capture.assert_not_called()


def test_cli_forwards_wait_budget_to_sidecar_capture(csv_input, tmp_path, monkeypatch):
    source, output, _ = csv_input
    env = tmp_path/'capture.env'
    env.write_text('HA_HOST=http://invalid\nHA_TOKEN=private-secret\n')
    snapshot = tmp_path/'shadow-inputs.json'
    capture = Mock(return_value={'shadow_only': True})
    monkeypatch.setattr(shadow, 'capture_with_csv', capture)
    shadow.main(['capture', '--env-file', str(env), '--csv', str(source),
        '--csv-output', str(output), '--wait-for-csv-seconds', '900', '--output', str(snapshot)])
    assert capture.call_args.kwargs['wait_seconds'] == 900
    assert 'private-secret' not in snapshot.read_text()
