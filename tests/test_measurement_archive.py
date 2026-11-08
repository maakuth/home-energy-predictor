"""The archive is separate from HEPO's prediction database."""
import sqlite3

import pytest

from utils.measurement_archive import MeasurementArchive


def test_archive_records_raw_events_and_invalid_states_idempotently(tmp_path):
    path = tmp_path / 'measurements.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.outdoor', None, 100, 200, [
            (1, 101.5, '3.2', '{"unit_of_measurement":"°C"}'),
            (2, 102.5, 'unavailable', None),
            (3, 102.5, '4.0', None),
        ])
        archive.ingest('sensor.outdoor', None, 100, 200, [
            (1, 101.5, '3.2', '{"unit_of_measurement":"°C"}'),
        ])
        rows = archive.observations('sensor.outdoor', None, 100, 200)
        assert [(r['recorder_id'], r['state']) for r in rows] == [(1, '3.2'), (2, 'unavailable'), (3, '4.0')]
        assert rows[0]['attributes'] == '{"unit_of_measurement":"°C"}'
        assert archive.coverage('sensor.outdoor', None) == [(100, 200)]


def test_empty_success_and_failed_write_do_not_lose_checkpoint(tmp_path):
    with MeasurementArchive(tmp_path / 'm.sqlite') as archive:
        archive.ingest('sensor.x', None, 0, 10, [])
        assert archive.coverage('sensor.x', None) == [(0, 10)]
        with pytest.raises(ValueError):
            archive.ingest('sensor.x', None, 10, 20, [(1, 25, '1', None)])
        assert archive.coverage('sensor.x', None) == [(0, 10)]
        assert archive.observations('sensor.x', None, 0, 30) == []


def test_ranges_are_per_source_and_backfills_merge_without_losing_gaps(tmp_path):
    with MeasurementArchive(tmp_path / 'm.sqlite') as archive:
        archive.ingest('sensor.x', None, 20, 30, [])
        archive.ingest('sensor.x', None, 0, 10, [])
        archive.ingest('sensor.y', None, 5, 6, [])
        assert archive.coverage('sensor.x', None) == [(0, 10), (20, 30)]
        archive.ingest('sensor.x', None, 10, 20, [])
        assert archive.coverage('sensor.x', None) == [(0, 30)]
        assert archive.coverage('sensor.y', None) == [(5, 6)]


def test_reader_includes_predecessor_without_future_values(tmp_path):
    with MeasurementArchive(tmp_path / 'm.sqlite') as archive:
        archive.ingest('sensor.x', None, 0, 30, [(1, 2, '1', None), (2, 10, '2', None), (3, 20, '3', None)])
        rows = archive.observations('sensor.x', None, 10, 20, predecessor=True)
        assert [r['state'] for r in rows] == ['1', '2']


def test_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / 'm.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=42')
    with pytest.raises(ValueError, match='version'):
        MeasurementArchive(path)
