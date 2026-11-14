"""Freeze a pre-capture CSV sidecar; no network, ingestion or active writes."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.archive_quality import METERS


MAX_CSV_BYTES = 16 * 1024 * 1024
_ACTIVE_NAMES = {'hepo.db', 'future_predictions.json', 'future_predictions_metadata.json',
                 'optimization_plan.json', 'energy_model.json', 'model_features.json',
                 'model_training_metadata.json', 'net_metering_state.json',
                 'raw_data.csv', 'raw_data_weekly.csv', 'processed_data.csv', 'processed_data_weekly.csv'}


class CsvNotReady(ValueError):
    """Normal extraction has not yet supplied stable, complete timestamp context."""


def freeze_csv(source: str | Path, output: str | Path, cutoff: float) -> dict[str, Any]:
    source, output = Path(source), Path(output)
    if output.name in _ACTIVE_NAMES or output.resolve() == source.resolve():
        raise ValueError('CSV sidecar must not use an active artifact path/name')
    if output.exists() or output.is_symlink():
        raise FileExistsError('CSV sidecar output must be a new explicit path')
    if not math.isfinite(cutoff) or cutoff % 60:
        raise ValueError('CSV cutoff must be a whole UTC minute')
    before = source.stat()
    if before.st_size > MAX_CSV_BYTES:
        raise ValueError('CSV size limit exceeded; use the bounded recent extract')
    with source.open('rb') as stream:
        contents = stream.read(MAX_CSV_BYTES + 1)
    after = source.stat()
    if len(contents) > MAX_CSV_BYTES:
        raise ValueError('CSV size limit exceeded while reading')
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise CsvNotReady('CSV changed while freezing; retry after normal extraction finishes')
    header = next(csv.reader(io.StringIO(contents.decode('utf-8-sig'))), [])
    required = {'timestamp', *METERS, *(name+'_source_valid' for name in METERS)}
    if len(set(header)) != len(header) or not required <= set(header):
        raise ValueError('CSV requires five meters, explicit validity flags and unique columns')
    frame = pd.read_csv(io.BytesIO(contents))
    stamps = [pd.Timestamp(value) for value in frame.pop('timestamp')]
    if not stamps or any(pd.isna(t) or t.tzinfo is None or t.timestamp() % 60 for t in stamps):
        raise ValueError('CSV timestamps must be timezone-aware and minute-aligned')
    frame.index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('CSV timestamps must be sorted and unique')
    expected = pd.date_range(pd.Timestamp(cutoff-86400, unit='s', tz='UTC'), periods=1440, freq='min')
    missing = expected[~expected.isin(frame.index)]
    if len(missing):
        raise CsvNotReady(f'CSV missing {len(missing)} of the preceding 1440 minutes; '
            f'required through {expected[-1].isoformat()}, available through {frame.index[-1].isoformat()}, '
            f'first missing {missing[0].isoformat()}. Use --wait-for-csv-seconds 900 or capture after the normal refresh.')
    window = frame.reindex(expected)
    valid = {name: int((np.isfinite(pd.to_numeric(window[name], errors='coerce')) &
                       window[name+'_source_valid'].astype(str).str.lower().eq('true')).sum()) for name in METERS}
    # Exclusive output: no replacement of operational CSVs or previous captures.
    with output.open('xb') as stream:
        stream.write(contents)
    return {'path': str(output.resolve()), 'source_path': str(source.resolve()),
            'source_mtime_ns': after.st_mtime_ns, 'size_bytes': len(contents),
            'sha256': hashlib.sha256(contents).hexdigest(),
            'frozen_at': datetime.now(timezone.utc).isoformat(), 'history_cutoff': cutoff,
            'first_timestamp': frame.index[0].isoformat(), 'last_timestamp': frame.index[-1].isoformat(),
            'rows': len(frame), 'required_minutes': len(expected),
            'available_declared_valid_minutes': valid, 'source_verification': 'not_established',
            'unit_contract': 'Declared canonical CSV bin means; original event units not independently verified'}


def validate_sidecar(snapshot: dict[str, Any], *, csv_path: str | Path | None = None) -> None:
    """Reject switched history or a post-capture freeze; original snapshots stay valid."""
    if 'csv_sidecar' not in snapshot:
        return
    metadata = snapshot['csv_sidecar']
    if not isinstance(metadata, dict) or metadata.get('history_cutoff') != snapshot.get('history_cutoff'):
        raise ValueError('CSV sidecar cutoff differs from the actual captured minute; do not backdate')
    frozen, started = pd.Timestamp(metadata['frozen_at']), pd.Timestamp(snapshot['started_at'])
    if pd.isna(frozen) or pd.isna(started) or frozen.tzinfo is None or started.tzinfo is None or frozen > started:
        raise ValueError('CSV must be frozen before forecast-input capture starts')
    path = Path(csv_path if csv_path is not None else metadata['path'])
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != metadata.get('sha256'):
        raise ValueError('Frozen CSV hash changed or a different CSV was supplied')
