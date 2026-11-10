import pandas as pd
import pytest

from utils.archive_features import iter_archive_features, load_archive_features, load_archive_training_features
from utils.baseload_quality import BASELOAD_FEATURES
from utils.measurement_archive import MeasurementArchive
from utils.measurement_sources import parse_sources


def history_fixture(tmp_path):
    start = pd.Timestamp('2026-10-21T00:00Z').timestamp()
    end = start + 3 * 86400
    path = tmp_path / 'history.sqlite'
    values = dict(total_power=4, solar_actual=0, battery_power=0, gshp_power=1000,
                  leaf_power=0, outside_temp=5, wind_speed=2, solar_forecast=2)
    with MeasurementArchive(path) as archive:
        for i, (name, value) in enumerate(values.items()):
            events = [(10 * i, start - 86400, str(value), None)]
            if name == 'total_power':
                events.append((10 * i + 1, start + 3660, '5', None))
            archive.ingest('sensor.' + name, None, start - 86400, end, events)
    sources = parse_sources({name: [{'entity': 'sensor.' + name}] for name in values})
    return path, sources, start, end


def test_feature_stream_uses_one_snapshot_and_matches_whole_window(tmp_path, monkeypatch):
    import utils.archive_features as module
    path, sources, start, end = history_fixture(tmp_path)
    whole = load_archive_features(path, start, end, sources=sources, interval_seconds=900)
    original = module.MeasurementReader
    readers = []

    class Tracked(original):
        def __enter__(self):
            readers.append(self)
            return super().__enter__()

    monkeypatch.setattr(module, 'MeasurementReader', Tracked)
    pieces = list(iter_archive_features(path, start, end, sources=sources,
                                        interval_seconds=900, chunk_seconds=86400))
    out = pd.concat(pieces)
    assert len(readers) == 1
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        readers[0].db.execute('SELECT 1')
    pd.testing.assert_frame_equal(out, whole)
    assert max(map(len, pieces)) == 96


def test_projected_training_frames_do_not_retain_measurement_metadata_columns(tmp_path):
    path, sources, start, end = history_fixture(tmp_path)
    frame = load_archive_training_features(path, start, end, sources=sources,
                                           interval_seconds=900, chunk_seconds=86400)
    assert set(BASELOAD_FEATURES) <= set(frame)
    assert 'gshp_power_import_ids' not in frame
    assert frame.attrs['archive_training']['start'] == start
    assert frame.attrs['archive_training']['end'] == end
    assert frame.attrs['archive_training']['projected_training_frame'] is True


def test_feature_stream_snapshot_is_stable_while_a_wal_writer_commits(tmp_path):
    path, sources, start, end = history_fixture(tmp_path)
    stream = iter_archive_features(path, start, end, sources=sources,
                                    interval_seconds=900, chunk_seconds=86400)
    first = next(stream)
    with MeasurementArchive(path) as writer:
        writer.ingest('sensor.total_power', None, start + 86400, end,
                       [(9999, start + 86400, '9', None)])
    second = next(stream)
    assert second.baseload_power.eq(4).all()
    list(stream)
    assert len(first) == 96
    current = load_archive_features(path, start + 86400, end, sources=sources, interval_seconds=900)
    assert current.baseload_power.eq(8).all()


def test_wholly_invalid_chunks_are_retained_as_ineligible_not_filled_or_dropped(tmp_path):
    path, sources, start, end = history_fixture(tmp_path)
    with MeasurementArchive(path) as writer:
        writer.ingest('sensor.battery_power', None, start + 86400, end,
                       [(9999, start + 86400, 'unavailable', None)])
    out = load_archive_training_features(path, start, end, sources=sources,
                                         interval_seconds=900, chunk_seconds=86400)
    assert len(out) == 3 * 96
    assert not out.loc[out.index >= pd.Timestamp(start + 86400, unit='s', tz='UTC'),
                        'baseload_training_eligible'].any()
