"""Explicit, non-actuating CSV/archive shadow comparison (no model fitting)."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from process_data import process_frame
from utils.archive_features import BASELOAD_INPUTS, load_archive_features, load_archive_raw
from utils.baseload_quality import baseload_targets
from utils.measurement_reader import MeasurementReader, interval_index


_METADATA = ('_source_valid', '_coverage_fraction', '_provenance', '_conflict',
             '_training_eligible', '_import_ids', '_entities', '_exclusion_reason')


def _true(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(False, index=frame.index)
    return frame[name].astype(str).str.lower().eq('true')


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[name], errors='coerce')
    return values.where(np.isfinite(values))


def _timestamp(value: str) -> float:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('Explicit timezone required for every timestamp')
    if pd.isna(stamp):
        raise ValueError('Invalid timestamp')
    return stamp.timestamp()


def _read_csv(path: Path, index: pd.DatetimeIndex, seconds: int) -> pd.DataFrame:
    context_start = index[0] - pd.Timedelta(days=1)
    pieces = []
    first = previous = None
    columns = []
    with pd.read_csv(path, chunksize=100000) as chunks:
        for frame in chunks:
            if 'timestamp' not in frame:
                raise ValueError('CSV requires a timestamp column with explicit timezone')
            stamps = [_timestamp(str(value)) for value in frame.pop('timestamp')]
            if any(stamp % seconds for stamp in stamps):
                raise ValueError('CSV timestamps must align to the declared interval')
            frame.index = pd.DatetimeIndex(pd.to_datetime(stamps, unit='s', utc=True), name='timestamp')
            if (frame.index.has_duplicates or not frame.index.is_monotonic_increasing
                    or (len(frame) and previous is not None and frame.index[0] <= previous)):
                raise ValueError('Duplicate or unordered CSV timestamps; supply a sorted audited file')
            columns = list(frame.columns)
            if len(frame):
                first = frame.index[0] if first is None else first
                previous = frame.index[-1]
                selected = frame.loc[(frame.index >= context_start) & (frame.index <= index[-1])]
                if len(selected):
                    pieces.append(selected.copy())
    if first is None or first > index[0] or previous is None or previous < index[-1]:
        raise ValueError('CSV does not span the full requested window; refusing a shorter intersection')
    frame = pd.concat(pieces) if pieces else pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz='UTC'))
    # Preserve interior gaps as missing rows, never shift observations or fill them.
    context = pd.date_range(index[0] - pd.Timedelta(days=1), index[-1],
                            freq=pd.Timedelta(seconds=seconds), name='timestamp')
    result = frame.reindex(context)
    result.attrs['window_missing_timestamps'] = int((~index.isin(frame.index)).sum())
    return result


def _differences(csv: pd.DataFrame, archive: pd.DataFrame, names) -> dict:
    result = {}
    for name in names:
        left, right = _numeric(csv, name), _numeric(archive, name)
        paired = left.notna() & right.notna()
        delta = (right - left).loc[paired]
        row = {
            'csv_present': name in csv, 'archive_present': name in archive,
            'csv_missing_rows': int(left.isna().sum()),
            'archive_missing_rows': int(right.isna().sum()),
            'matched_nonmissing_rows': int(paired.sum()),
            'different_rows': int((~np.isclose(left[paired], right[paired], rtol=1e-7, atol=1e-9)).sum()),
            'archive_minus_csv_mean': float(delta.mean()) if len(delta) else None,
            'mean_absolute_difference': float(delta.abs().mean()) if len(delta) else None,
            'max_absolute_difference': float(delta.abs().max()) if len(delta) else None,
        }
        if name + '_coverage_fraction' in archive:
            coverage = _numeric(archive, name + '_coverage_fraction')
            row.update({
                'archive_mean_coverage_fraction': float(coverage.mean()) if coverage.notna().any() else None,
                'archive_source_verified_rows': int(_true(archive, name + '_source_valid').sum()),
                'archive_conflict_rows': int(_true(archive, name + '_conflict').sum()),
                'archive_provenance_rows': {str(k): int(v) for k, v in
                    archive[name + '_provenance'].fillna('missing').value_counts().items()},
            })
        result[name] = row
    return result


def _errors(prediction: pd.Series, actual: pd.Series, mask: pd.Series) -> dict:
    error = (prediction - actual).loc[mask]
    return {'mae_kw': float(error.abs().mean()) if len(error) else None,
            'rmse_kw': float(np.sqrt((error ** 2).mean())) if len(error) else None,
            'bias_kw': float(error.mean()) if len(error) else None}


def compare(*, csv: str | Path, csv_mode: str, archive: str | Path,
            sources: str | Path, start: float, end: float,
             interval_seconds: int = 60, allow_legacy: bool = False,
             legacy_policy: str = 'verified-only', preferred_imports: list[int] | None = None,
             policy_path: str | Path | None = None) -> dict:
    """Compare the same [start, end) grid; never silently shorten either input.

    Raw CSVs are already legacy-resampled measurements, not recorder events.
    Processed CSV baseloads are trusted only with pre-fill quality metadata.
    """
    if csv_mode not in {'raw', 'processed'}:
        raise ValueError('csv_mode must explicitly be raw or processed')
    if legacy_policy not in {'verified-only', 'complete-legacy'}:
        raise ValueError('Unknown legacy policy')
    if legacy_policy == 'complete-legacy' and not allow_legacy:
        raise ValueError('complete-legacy requires allow_legacy=True')
    if interval_seconds < 60 or interval_seconds % 60 or 86400 % interval_seconds:
        raise ValueError('Interval must be whole minutes dividing a day')
    index = interval_index(start, end, interval_seconds)
    for path in (csv, archive, sources):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    csv_frame = _read_csv(Path(csv), index, interval_seconds)
    missing_stamps = csv_frame.attrs['window_missing_timestamps']
    options = {'policy_path': policy_path, 'require_policies': True} if policy_path is not None else {}
    with MeasurementReader(archive) as reader:
        archive_raw = load_archive_raw(archive, start, end, sources_path=sources,
                                   interval_seconds=interval_seconds, allow_legacy=allow_legacy,
                                    preferred_imports=preferred_imports, _reader=reader, **options)
        archive_features = load_archive_features(archive, start, end,
                                            legacy_policy=legacy_policy, sources_path=sources,
                                            interval_seconds=interval_seconds, allow_legacy=allow_legacy,
                                            preferred_imports=preferred_imports, _reader=reader, **options)
    for key in ('source_semantic_sha256', 'policy_semantic_sha256'):
        if (key in archive_raw.attrs.get('archive', {}) and
                archive_raw.attrs['archive'][key] != archive_features.attrs.get('archive_training', {}).get(key)):
            raise ValueError('Measurement source/policy configuration changed during shadow materialization')
    if not archive_raw.index.equals(index) or not archive_features.index.equals(index):
        raise ValueError('Archive loader returned a mismatched window')
    policy_metadata = archive_features.attrs.get('archive_training')
    if not policy_metadata or policy_metadata.get('legacy_policy') != legacy_policy:
        raise ValueError('Archive loader must provide matching archive_training policy/provenance metadata')
    required = ('baseload_target_valid', 'baseload_training_eligible',
                'baseload_lag_24h_training_eligible')
    if any(name not in archive_features for name in required):
        raise ValueError('Archive loader is missing training eligibility metadata')

    if csv_mode == 'raw':
        truth = baseload_targets(csv_frame)
        csv_features = process_frame(csv_frame)
        note = 'Raw CSV actuals use five original CSV meter columns before fills/medians; CSV fidelity is unverified.'
    else:
        csv_features = csv_frame
        truth = csv_frame.copy()
        if 'baseload_target_valid' not in truth or 'baseload_quality_provenance' not in truth:
            truth['baseload_target_valid'] = False
            note = 'Processed CSV lacks pre-fill quality metadata; actuals and baseline scores are unavailable.'
        else:
            provenance = truth.baseload_quality_provenance.isin(['source_flags', 'legacy_finite_checks_only'])
            truth['baseload_target_valid'] = _true(truth, 'baseload_target_valid') & provenance
            note = 'Processed CSV actuals rely on declared pre-fill quality metadata; no verification is inferred.'
    csv_actual = _numeric(truth, 'baseload_power')
    csv_valid = _true(truth, 'baseload_target_valid') & csv_actual.ge(0) & csv_actual.notna()
    csv_actual = csv_actual.where(csv_valid)
    # Compute exact prior-day values from honest actuals, not potentially filled CSV lag features.
    csv_prediction = csv_actual.reindex(index - pd.Timedelta(days=1))
    csv_prediction.index = index
    csv_actual = csv_actual.reindex(index)
    csv_features = csv_features.reindex(index)
    actual = _numeric(archive_features, 'baseload_power')
    prediction = _numeric(archive_features, 'baseload_lag_24h')
    eligible = _true(archive_features, 'baseload_training_eligible') & actual.notna() & actual.ge(0)
    verified = _true(archive_features, 'baseload_target_valid') & eligible
    legacy = eligible & ~verified
    matched = (eligible & _true(archive_features, 'baseload_lag_24h_training_eligible') &
               prediction.notna() & csv_actual.notna() & csv_prediction.notna())
    baseline = {}
    for label, group in (('verified', verified), ('legacy_eligible', legacy)):
        mask = matched & group
        baseline[label] = {'matched_rows': int(mask.sum()),
                           'csv': _errors(csv_prediction, csv_actual, mask),
                           'archive': _errors(prediction, actual, mask),
                           'prediction_difference': _errors(prediction, csv_prediction, mask),
                           'actual_difference': _errors(actual, csv_actual, mask)}
    feature_names = sorted({name for frame in (csv_features, archive_features)
                            for name in frame.select_dtypes(include='number').columns
                            if not name.endswith(_METADATA)})
    return {
        'window': {'from': index[0].isoformat(), 'to': pd.Timestamp(end, unit='s', tz='UTC').isoformat(),
                   'interval_seconds': interval_seconds, 'expected_rows': len(index),
                   'csv_missing_timestamps': missing_stamps},
        'inputs': {'csv': str(Path(csv).resolve()), 'csv_mode': csv_mode,
                   'archive': str(Path(archive).resolve()), 'sources': str(Path(sources).resolve())},
        'archive_training': policy_metadata,
        'aggregation_note': 'Recorder archive values are time-weighted complete-bin means; legacy CSV extraction used event means and may contain prior filling. Differences are not automatically regressions.',
         'snapshot_note': 'Raw and feature loaders share one consistent read-only archive snapshot, closed before comparison/scoring. Source and policy semantic hashes must agree.',
        'csv_actuals_note': note,
        'raw': _differences(csv_frame.reindex(index), archive_raw, BASELOAD_INPUTS) if csv_mode == 'raw' else None,
        'features': _differences(csv_features, archive_features, feature_names),
        'eligibility': {'archive': {'verified': int(verified.sum()), 'legacy_eligible': int(legacy.sum()),
                                    'excluded': int((~eligible).sum())},
                        'csv_honest_actual_rows': int(csv_actual.notna().sum())},
        'baseline': baseline,
        'baseline_note': 'Repeat yesterday means exactly 24 elapsed hours (UTC), not previous local calendar day. Groups classify archive targets; lag eligibility follows the declared policy. Both routes score only identical matched actual/lag timestamps, against their own actuals. Missing values are never zero. No fitting, SARIMA ranking, prediction publication, or control calls.',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--csv-mode', choices=('raw', 'processed'), required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--measurement-policy', dest='policy_path', type=Path)
    parser.add_argument('--from', dest='start', required=True)
    parser.add_argument('--to', dest='end', required=True)
    parser.add_argument('--interval-seconds', type=int, default=60)
    parser.add_argument('--allow-legacy', action='store_true')
    parser.add_argument('--legacy-policy', choices=('verified-only', 'complete-legacy'), default='verified-only')
    parser.add_argument('--preferred-import', type=int, action='append', dest='preferred_imports')
    args = vars(parser.parse_args(argv))
    try:
        args['start'], args['end'] = _timestamp(args['start']), _timestamp(args['end'])
        # Generic feature processing emits progress; keep stdout machine-readable JSON.
        with redirect_stdout(sys.stderr):
            report = compare(**args)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
