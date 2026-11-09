import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_ingestion import update_source


def test_first_update_then_incremental_overlap_and_late_data(tmp_path):
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        calls = []
        def fetch(start, end):
            calls.append((start, end))
            return [(1, 95.0, '1', None)] if start <= 95 < end else []
        update_source(archive, 'sensor.x', None, 0, 100, 10, fetch)
        update_source(archive, 'sensor.x', None, 0, 120, 10, fetch)
        assert calls == [(0, 100), (90, 120)]
        assert archive.coverage('sensor.x', None) == [(0, 120)]
        assert len(archive.observations('sensor.x', None, 0, 120)) == 1


def test_failed_fetch_does_not_advance_range(tmp_path):
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        def broken(start, end):
            yield (1, 1.0, '1', None)
            raise RuntimeError('disconnected')
        with pytest.raises(RuntimeError):
            update_source(archive, 'sensor.x', None, 0, 100, 10, broken)
        assert archive.coverage('sensor.x', None) == []
        assert archive.observations('sensor.x', None, 0, 100) == []


def test_older_backfill_does_not_skip_gap_or_new_history(tmp_path):
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        archive.ingest('sensor.x', None, 20, 30, [])
        calls = []
        update_source(archive, 'sensor.x', None, 0, 10, 2, lambda a, b: calls.append((a, b)) or [])
        assert calls == [(0, 10)]
        assert archive.coverage('sensor.x', None) == [(0, 10), (20, 30)]
