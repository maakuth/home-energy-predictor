from unittest.mock import MagicMock, patch

import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_update import update_recorder_sources


def test_update_sources_uses_independent_coverage_and_fixed_cutoff(tmp_path):
    connection = MagicMock()
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        archive.ingest('sensor.a', None, 0, 80, [])
        calls = []
        def read(conn, entity, start, end):
            calls.append((entity, start, end))
            return [(1, start + 1, 'unavailable', None)]
        with patch('utils.measurement_update.recorder_events', side_effect=read):
            failures = update_recorder_sources(archive, connection, [('sensor.a', None), ('sensor.b', None)], 0, 100, 10)
        assert failures == {}
        assert calls == [('sensor.a', 70, 100), ('sensor.b', 0, 100)]
        assert archive.coverage('sensor.a', None) == [(0, 100)]
        assert archive.coverage('sensor.b', None) == [(0, 100)]


def test_one_source_failure_does_not_block_others_or_advance_failed_source(tmp_path):
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        def read(conn, entity, start, end):
            if entity == 'sensor.a':
                raise RuntimeError('recorder unavailable')
            return []
        with patch('utils.measurement_update.recorder_events', side_effect=read):
            failures = update_recorder_sources(archive, MagicMock(), [('sensor.a', None), ('sensor.b', None)], 0, 100, 10)
        assert 'sensor.a' in failures
        assert archive.coverage('sensor.a', None) == []
        assert archive.coverage('sensor.b', None) == [(0, 100)]
