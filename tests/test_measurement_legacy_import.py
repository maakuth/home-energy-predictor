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


@pytest.mark.parametrize('text', [
    'timestamp\n2026-10-01T00:00:00Z\n',
    'timestamp,power,power\n2026-10-01T00:00:00Z,1,2\n',
    'timestamp,power,\n2026-10-01T00:00:00Z,1,2\n',
    'timestamp,power\n2026-10-01T00:00:01Z,1\n',
    'timestamp,power\n2026-10-01T00:00:00Z,1,2\n',
    'timestamp,power,other\n2026-10-01T00:00:00Z,1\n',
    'timestamp,power\n2026-10-01T00:00:00Z,1\n2026-10-01T00:00:00Z,2\n',
    'timestamp,power\n',
])
def test_malformed_or_unaligned_import_rolls_back_without_partial_metadata(tmp_path, text):
    source = tmp_path / 'bad.csv'
    source.write_text(text)
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        with pytest.raises(ValueError):
            import_legacy_csv(archive, source, 60)
        assert archive.db.execute('SELECT COUNT(*) FROM legacy_imports').fetchone()[0] == 0
        assert archive.db.execute('SELECT COUNT(*) FROM legacy_intervals').fetchone()[0] == 0


def test_invalid_source_flags_preserved_as_metadata_not_promoted(tmp_path):
    source = tmp_path / 'flags.csv'
    source.write_text('timestamp,power,power_source_valid\n2026-10-01T00:00:00Z,123,False\n')
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        import_legacy_csv(archive, source, 60)
        rows = archive.db.execute('SELECT measurement,value FROM legacy_intervals ORDER BY measurement').fetchall()
        assert [tuple(row) for row in rows] == [('power', '123'), ('power_source_valid', 'False')]


def test_content_hash_matches_bytes_parsed_or_rolls_back(tmp_path, monkeypatch):
    from pathlib import Path
    source = tmp_path / 'changing.csv'
    source.write_text('timestamp,power\n2026-10-01T00:00:00Z,1\n')
    original = Path.open
    calls = 0

    def changed(path, mode='r', *args, **kwargs):
        nonlocal calls
        if path == source and mode == 'rb':
            calls += 1
            if calls == 2:
                source.write_text('timestamp,power\n2026-10-01T00:00:00Z,2\n')
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', changed)
    with MeasurementArchive(tmp_path / 'archive.sqlite') as archive:
        with pytest.raises(ValueError, match='changed'):
            import_legacy_csv(archive, source, 60)
        assert archive.db.execute('SELECT COUNT(*) FROM legacy_imports').fetchone()[0] == 0
        assert archive.db.execute('SELECT COUNT(*) FROM legacy_intervals').fetchone()[0] == 0
