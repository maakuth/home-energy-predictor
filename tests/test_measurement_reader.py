import sqlite3

import pandas as pd
import pytest

from utils.measurement_archive import MeasurementArchive
from utils.measurement_legacy import import_legacy_csv
from utils.measurement_reader import MeasurementReader


def build_archive(tmp_path, contents):
    path = tmp_path / 'measurements.sqlite'
    with MeasurementArchive(path) as archive:
        for i, text in enumerate(contents):
            csv = tmp_path / f'{i}.csv'
            csv.write_text(text)
            import_legacy_csv(archive, csv, 60)
    return path


START = pd.Timestamp('2026-10-01T00:00:00Z').timestamp()


def test_reader_cannot_write_or_create_an_archive(tmp_path):
    missing = tmp_path / 'missing.sqlite'
    with pytest.raises(sqlite3.OperationalError):
        MeasurementReader(missing)
    assert not missing.exists()
    path = build_archive(tmp_path, [])
    with MeasurementReader(path) as reader:
        with pytest.raises(sqlite3.OperationalError):
            reader.db.execute('DELETE FROM legacy_imports')


def test_numeric_agreement_is_unverified_and_gaps_are_not_filled(tmp_path):
    path = build_archive(tmp_path, [
        'timestamp,power\n2026-10-01T00:00:00Z,1000\n',
        'timestamp,power\n2026-10-01T00:00:00Z,1000.0\n',
    ])
    with MeasurementReader(path) as reader:
        result = reader.legacy_numeric('power', START, START + 120)
    assert result.value.iloc[0] == 1000
    assert result.provenance.iloc[0] == 'legacy_resampled'
    assert not result.source_valid.any()
    assert result.import_ids.iloc[0] == (1, 2)
    assert pd.isna(result.value.iloc[1])


@pytest.mark.parametrize('second', ['2000', ''])
def test_conflict_is_unknown_unless_precedence_is_explicit(tmp_path, second):
    path = build_archive(tmp_path, [
        'timestamp,power\n2026-10-01T00:00:00Z,1000\n',
        f'timestamp,power\n2026-10-01T00:00:00Z,{second}\n',
    ])
    with MeasurementReader(path) as reader:
        result = reader.legacy_numeric('power', START, START + 60)
        chosen = reader.legacy_numeric('power', START, START + 60, preferred_imports=[1])
        missing = reader.legacy_numeric('power', START, START + 60, preferred_imports=[2])
    assert pd.isna(result.value.iloc[0])
    assert result.conflict.iloc[0]
    assert chosen.value.iloc[0] == 1000
    assert chosen.selected_import_id.iloc[0] == 1
    assert chosen.conflict.iloc[0]  # Selection must not erase evidence.
    assert missing.value.iloc[0] == 2000 if second else pd.isna(missing.value.iloc[0])


def test_invalid_source_flag_cannot_be_hidden_by_a_numeric_legacy_value(tmp_path):
    path = build_archive(tmp_path, [
        'timestamp,power,power_source_valid\n2026-10-01T00:00:00Z,1000,False\n',
    ])
    with MeasurementReader(path) as reader:
        result = reader.legacy_numeric('power', START, START + 60)
    assert pd.isna(result.value.iloc[0])
    assert result.coverage_fraction.iloc[0] == 0


def test_partial_bin_boundaries_are_rejected(tmp_path):
    path = build_archive(tmp_path, [])
    with MeasurementReader(path) as reader:
        with pytest.raises(ValueError, match='aligned'):
            reader.legacy_numeric('power', START + 1, START + 61)
