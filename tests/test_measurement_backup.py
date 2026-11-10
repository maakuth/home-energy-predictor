import sqlite3

import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_backup import backup_archive, restore_archive, verify_archive


def test_backup_includes_uncheckpointed_wal_and_restores_new_path(tmp_path):
    source, backup, restored = [tmp_path / name for name in ('source.sqlite', 'backup.sqlite', 'restored.sqlite')]
    with MeasurementArchive(source, recorder_namespace='fixture') as archive:
        archive.db.execute('PRAGMA wal_autocheckpoint=0')
        archive.ingest('x', None, 0, 10, [(1, 1., '2', None)])
        identity = backup_archive(source, backup)
        assert identity['integrity'] == 'ok'
        assert restore_archive(backup, restored)['archive_uuid'] == identity['archive_uuid']
        with MeasurementArchive(restored, recorder_namespace='fixture') as copy:
            assert copy.observations('x', None, 0, 10)[0]['state'] == '2'
    assert verify_archive(source)['archive_uuid'] == identity['archive_uuid']


def test_no_overwrite_and_reject_unrelated_or_corrupt_source(tmp_path):
    destination = tmp_path / 'existing'
    destination.write_text('preserve')
    source = tmp_path / 'source.sqlite'
    with MeasurementArchive(source):
        pass
    with pytest.raises(FileExistsError):
        restore_archive(source, destination)
    assert destination.read_text() == 'preserve'
    invalid = tmp_path / 'invalid.sqlite'
    with sqlite3.connect(invalid) as db:
        db.execute('CREATE TABLE unrelated(id INTEGER)')
    with pytest.raises(ValueError):
        backup_archive(invalid, tmp_path / 'new.sqlite')
    assert not (tmp_path / 'new.sqlite').exists()
    with pytest.raises(ValueError, match='sidecar'):
        backup_archive(source, str(source) + '-wal')


def test_foreign_key_violation_rejected(tmp_path):
    path = tmp_path / 'archive.sqlite'
    with MeasurementArchive(path) as archive:
        archive.db.execute('PRAGMA foreign_keys=OFF')
        archive.db.execute('INSERT INTO observations VALUES (999, 1, 1, NULL, NULL)')
        archive.db.commit()
    with pytest.raises(ValueError, match='foreign'):
        verify_archive(path)
