import sqlite3

import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_legacy import import_legacy_csv


def test_import_records_interval_provenance_and_is_idempotent(tmp_path):
    source = tmp_path / 'raw.csv'
    source.write_text('timestamp,gshp_power,outside_temp\n2026-10-01T00:00:00Z,1000,2\n2026-10-01T00:01:00Z,,3\n')
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        first = import_legacy_csv(archive, source, interval_seconds=60)
        second = import_legacy_csv(archive, source, interval_seconds=60)
        assert first == second
        rows = archive.db.execute('SELECT measurement, value, provenance FROM legacy_intervals ORDER BY timestamp, measurement').fetchall()
        assert [tuple(r) for r in rows] == [
            ('gshp_power', '1000', 'legacy_resampled'), ('outside_temp', '2', 'legacy_resampled'),
            ('gshp_power', None, 'legacy_resampled'), ('outside_temp', '3', 'legacy_resampled')]


def test_overlaps_are_retained_with_distinct_import_provenance(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text('timestamp,gshp_power\n2026-10-01T00:00:00Z,1000\n')
    b.write_text('timestamp,gshp_power\n2026-10-01T00:00:00Z,2000\n')
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        import_legacy_csv(archive, a, interval_seconds=60)
        import_legacy_csv(archive, b, interval_seconds=60)
        values = archive.db.execute('SELECT value FROM legacy_intervals ORDER BY value').fetchall()
        assert [r[0] for r in values] == ['1000', '2000']


def test_bad_timestamp_rolls_back_entire_import(tmp_path):
    source = tmp_path / 'bad.csv'
    source.write_text('timestamp,gshp_power\n2026-10-01T00:00:00Z,1000\ninvalid,2000\n')
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        with pytest.raises(ValueError):
            import_legacy_csv(archive, source, interval_seconds=60)
        assert archive.db.execute('SELECT COUNT(*) FROM legacy_imports').fetchone()[0] == 0
        assert archive.db.execute('SELECT COUNT(*) FROM legacy_intervals').fetchone()[0] == 0
