"""Offline common-contract CSV/SQLite history replay; not a new forecast vintage.

The model, captured forecasts, UTC calendar, issue time, targets and physical
rules are held fixed. CSV declared validity is NOT recorder verification. This
does not execute the legacy online PostgreSQL anchor reader or any optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.archive_features import load_archive_raw
from utils.archive_history import ArchiveHistory, add_history_arguments, aware_epoch, history_from_arguments
from utils.archive_quality import METERS
from utils.baseload_model import validate_baseload_model
from utils.baseload_quality import baseload_targets
from utils.forward_shadow import _finite, _forecast_table, _metrics, _provenance, _timestamp, write_new
from utils.measurement_policy import MeasurementPolicy, load_policies, require_policy_entries
from utils.measurement_reader import MeasurementReader
from utils.measurement_shadow import _differences, _read_csv

ANCHORS = ('baseload_lag_1h', 'baseload_lag_24h', 'leaf_power_lag_1h', 'leaf_energy_24h')


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _normalise(frame: pd.DataFrame, policies: dict[str, MeasurementPolicy], *, csv: bool) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for name in METERS:
        flag = name + '_source_valid'
        if name not in frame or flag not in frame:
            raise ValueError(f'Missing CSV meter/explicit validity flag: {name}')
        spec = policies[name]
        values = pd.to_numeric(frame[name], errors='coerce')
        valid = frame[flag].astype(str).str.lower().eq('true') & np.isfinite(values)
        if not csv:
            valid &= frame[name+'_coverage_fraction'].eq(1) & frame[name+'_unit_verified']
            valid &= ~frame[name+'_conflict'] & frame[name+'_provenance'].eq('recorder')
        # CSV measurements are declared canonical bin means. Original event
        # units/expiry/outlier durations cannot be recovered from a CSV value.
        if spec.nonnegative:
            valid &= values.ge(0)
        if spec.min_value is not None:
            valid &= values.ge(spec.min_value)
        if spec.max_value is not None:
            valid &= values.le(spec.max_value)
        result[name] = values.where(valid)
        result[flag] = valid
    return result


def _history_summary(raw: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    # The extra row is only a lag lookup position; never a measured future row.
    query = raw.reindex(raw.index.append(pd.DatetimeIndex([cutoff])))
    targets = baseload_targets(query)
    row = targets.iloc[-1]
    anchors = {name: float(row[name]) if pd.notna(row[name]) and np.isfinite(row[name]) else None
               for name in ANCHORS}
    heating = raw.gshp_power.dropna()
    return {'anchors': anchors, 'missing_anchors': [name for name, value in anchors.items() if value is None],
            'heating_observed_mean_kw': float(heating.mean()/1000) if len(heating) else None,
            'heating_valid_minutes': len(heating), 'expected_minutes': len(raw),
            'valid_input_minutes': {name: int(raw[name].notna().sum()) for name in METERS}}


def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    delta = right-left
    return {'rows': len(delta), 'different_rows': int((~np.isclose(left, right, atol=1e-9, rtol=1e-7)).sum()),
            'archive_minus_csv_mean': float(delta.mean()) if len(delta) else None,
            'mean_absolute_difference_kw': float(np.abs(delta).mean()) if len(delta) else None,
            'max_absolute_difference_kw': float(np.abs(delta).max()) if len(delta) else None}


def compare(*, csv: str | Path, history: ArchiveHistory, model_dir: str | Path,
            snapshot: dict[str, Any], vintage: dict[str, Any],
            actuals_cutoff: float | None = None) -> dict[str, Any]:
    from predict_future import generate_inference_data
    provenance = _provenance(history)
    if (vintage.get('provenance') != provenance or vintage.get('calendar_feature_timezone') != 'UTC'
            or vintage.get('history_cutoff') != history.cutoff or snapshot.get('history_cutoff') != history.cutoff):
        raise ValueError('Common-control provenance/cutoff mismatch')
    snapshot_hash = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    if snapshot_hash != vintage.get('input_snapshot_sha256'):
        raise ValueError('Captured snapshot changed; cannot isolate history backend')
    model_dir, csv = Path(model_dir), Path(csv)
    artifact_hashes = {name: _digest(model_dir/name) for name in
                       ('energy_model.json', 'model_features.json', 'model_training_metadata.json')}
    if artifact_hashes['energy_model.json'] != vintage.get('model_sha256'):
        raise ValueError('Different model artifact from the captured vintage')
    origin = _timestamp(vintage['generated_at'])
    stamps = pd.DatetimeIndex([_timestamp(row['timestamp']) for row in vintage['forecasts']])
    if (len(stamps) == 0 or stamps.has_duplicates or not stamps.is_monotonic_increasing
            or any(t <= origin or t.timestamp() % 900 for t in stamps)):
        raise ValueError('Expected strictly future, aligned, ordered vintage targets')
    expected = pd.date_range(stamps[0], stamps[-1], freq='15min')
    if not stamps.equals(expected):
        raise ValueError('Vintage target grid contains gaps')
    cutoff = pd.Timestamp(history.cutoff, unit='s', tz='UTC')
    index = pd.date_range(cutoff-pd.Timedelta(days=1), cutoff, freq='min', inclusive='left')
    if history.policy_path is None:
        raise ValueError('Explicit common policy required')
    policies = load_policies(history.policy_path)
    require_policy_entries(policies, list(METERS))
    if any(policies[name].kind != 'numeric' for name in METERS):
        raise ValueError('Canonical numeric meter policies required')
    stat = csv.stat()
    csv_hash = _digest(csv)
    csv_frame = _read_csv(csv, index, 60).reindex(index)
    after = csv.stat()
    if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('CSV changed while reading; freeze a static snapshot')
    actual_raw = None
    settled = pd.DatetimeIndex([t for t in stamps if t.timestamp()+900 <= actuals_cutoff]) if actuals_cutoff is not None else stamps[:0]
    # All recorder history and optional actuals use one read-only snapshot.
    with MeasurementReader(history.path) as reader:
        archive_raw = load_archive_raw(history.path, history.cutoff-86400, history.cutoff,
            measurements=list(METERS), interval_seconds=60, _reader=reader, **history._options())
        if len(settled):
            actual_raw = load_archive_raw(history.path, settled[0].timestamp(), settled[-1].timestamp()+900,
                measurements=list(METERS), interval_seconds=900, _reader=reader, **history._options())
    if _provenance(history) != provenance:
        raise ValueError('Source/policy provenance changed during replay')
    normal = {'csv': _normalise(csv_frame, policies, csv=True),
              'archive': _normalise(archive_raw, policies, csv=False)}
    summaries = {name: _history_summary(raw, cutoff) for name, raw in normal.items()}
    anchor_differences = {}
    for name in ANCHORS:
        left, right = summaries['csv']['anchors'][name], summaries['archive']['anchors'][name]
        anchor_differences[name] = {'csv': left, 'archive': right,
            'archive_minus_csv': None if left is None or right is None else right-left}
    report: dict[str, Any] = {
        'status': 'blocked_inputs' if any(s['missing_anchors'] for s in summaries.values()) else 'compared',
        'scope': 'common-contract history backend replay, not a newly issued forecast or readiness certificate',
        'legacy_serving_bit_replay': False, 'generated_at_original': vintage['generated_at'],
        'history_cutoff': history.cutoff, 'calendar_feature_timezone': 'UTC',
        'controls': {'model_files_sha256': artifact_hashes, 'snapshot_sha256': snapshot_hash,
                     'targets': [str(t) for t in stamps], 'provenance': provenance},
        'csv_input': {'path': str(csv.resolve()), 'sha256': csv_hash, 'mtime_ns': stat.st_mtime_ns,
                      'unit_contract': 'declared canonical CSV values; original event units are not verifiable',
                      'source_verification': 'not_established; explicit CSV validity flags only'},
        'history': summaries, 'anchor_differences': anchor_differences,
        'raw_differences': _differences(normal['csv'], normal['archive'], METERS),
        'predictions': None, 'feature_differences': None, 'archive_saved_vintage_replay': None,
        'retrospective_actuals': None,
        'optimizer_replay': {'status': 'not_evaluated',
            'missing_captures': ['import/export prices and price metadata', 'battery SOC and constraints',
                                 'thermal/relay/EV state', 'dispatch settings and model parameters',
                                 'current-slot latches and other planner state']},
        'limitations': [
            'Legacy online inference uses a PostgreSQL event reader, not CSV anchors; this is not its exact behavioral replay.',
            'Both routes use the same exact cutoff-relative minute/quality/physical contract; original legacy fallback algorithms are not substituted.',
            'CSV event means/interpolation differ from recorder duration means; old event timestamps and expiry cannot be recovered from CSV.',
            'History is the currently retained data at the event-time cutoff, not a guaranteed as-of-issue snapshot. Late arrivals may change replay.',
            'Retrospective replay outputs are diagnostics, never new forward vintages or independent model evaluation.',
            'Minute-model versus quarter-target and frozen-origin lag semantics are held fixed, not repaired or tuned here.',
            'Heating component is a common autonomous 24h observed-mean assumption, not captured dispatch or a full optimizer plan.',
        ]}
    if report['status'] != 'compared':
        return report
    attrs = snapshot['current_weather']['attributes']
    if (attrs.get('wind_speed_unit') != 'km/h' or attrs.get('temperature_unit') not in {'C', '°C'}
            or snapshot['current_temperature']['attributes'].get('unit_of_measurement') not in {'C', '°C'}):
        raise ValueError('Captured weather units do not match common model controls')
    solar = _forecast_table(snapshot['solar'], 'period_start', ['pv_estimate'])
    weather = _forecast_table(snapshot['weather'], 'datetime', ['temperature', 'wind_speed'])
    if solar.pv_estimate.lt(0).any() or weather.wind_speed.lt(0).any():
        raise ValueError('Invalid captured forecast values')
    for table, maximum in ((solar, pd.Timedelta(hours=1)), (weather, pd.Timedelta(hours=2))):
        nearest = table.index.get_indexer(stamps, method='nearest')
        if any(abs(t-table.index[int(i)]) >= maximum for t, i in zip(stamps, nearest)):
            raise ValueError('Captured forecast does not support the common target grid')
    model = xgb.XGBRegressor()
    model.load_model(model_dir/'energy_model.json')
    features = json.loads((model_dir/'model_features.json').read_text())
    metadata = json.loads((model_dir/'model_training_metadata.json').read_text())
    booster = model.get_booster()
    validate_baseload_model(features, metadata, booster.attr('baseload_feature_contract'), booster.feature_names)
    if _timestamp(metadata['last_eligible']).timestamp() >= history.cutoff:
        raise ValueError('Model training does not precede original origin')
    matrices, predictions = {}, {}
    for name in ('csv', 'archive'):
        anchors = summaries[name]['anchors']
        current = {'temp_val': _finite(snapshot['current_temperature']['state']),
                   'wind_val': _finite(attrs['wind_speed']), 'lag1h_val': anchors['baseload_lag_1h'],
                   'lag24h_val': anchors['baseload_lag_24h'], 'leaf_lag_val': anchors['leaf_power_lag_1h'],
                   'leaf_energy_val': anchors['leaf_energy_24h']}
        rows, timestamps = generate_inference_data(stamps[0].to_pydatetime(), stamps[-1].to_pydatetime(),
            15, solar, weather, current, {'now': origin.to_pydatetime()})
        if list(pd.to_datetime(timestamps, utc=True)) != list(stamps):
            raise ValueError('Generated replay grid differs from captured targets')
        matrix = pd.DataFrame(rows)[features]
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError('Nonfinite replay feature inputs')
        matrices[name] = matrix
        prediction = np.asarray(model.predict(matrix), dtype=float)
        if len(prediction) != len(stamps) or not np.isfinite(prediction).all():
            raise ValueError('Nonfinite or incomplete replay predictions')
        predictions[name] = prediction
    if any(_digest(model_dir/name) != digest for name, digest in artifact_hashes.items()):
        raise ValueError('Model artifacts changed during comparison')
    feature_differences = {}
    units = {'outside_temp': 'C', 'wind_speed': 'km/h', 'leaf_power_lag_1h': 'W',
             'leaf_energy_24h': 'kWh', 'baseload_lag_1h': 'kW', 'baseload_lag_24h': 'kW'}
    for name in features:
        difference = _difference(matrices['csv'][name].to_numpy(), matrices['archive'][name].to_numpy())
        difference['mean_absolute_difference'] = difference.pop('mean_absolute_difference_kw')
        difference['max_absolute_difference'] = difference.pop('max_absolute_difference_kw')
        difference['unit'] = units.get(name, 'calendar/index or dimensionless')
        feature_differences[name] = difference
    report['feature_differences'] = feature_differences
    report['predictions'] = {**_difference(predictions['csv'], predictions['archive']),
        'rows_detail': [{'timestamp': str(t), 'csv_kw': float(left), 'archive_kw': float(right)}
                       for t, left, right in zip(stamps, predictions['csv'], predictions['archive'])]}
    saved = np.asarray([_finite(row['candidate_kw']) for row in vintage['forecasts']])
    report['archive_saved_vintage_replay'] = _difference(saved, predictions['archive'])
    heating = {name: summaries[name]['heating_observed_mean_kw'] for name in summaries}
    report['autonomous_total_load_proxy'] = (_difference(predictions['csv']+heating['csv'],
        predictions['archive']+heating['archive']) if all(value is not None for value in heating.values()) else None)
    if actual_raw is not None:
        actual_frame = _normalise(actual_raw, policies, csv=False)
        actual_targets = baseload_targets(actual_frame)
        actual = actual_targets.baseload_power.where(actual_targets.baseload_target_valid).reindex(stamps)
        valid = actual.notna()
        report['retrospective_actuals'] = {'matched_quarters': int(valid.sum()),
            'csv': _metrics(actual[valid].tolist(), predictions['csv'][valid.to_numpy()].tolist()),
            'archive': _metrics(actual[valid].tolist(), predictions['archive'][valid.to_numpy()].tolist()),
            'note': 'Same recorder actuals/timestamps, retrospective data-source diagnostic only; no causal model ranking.'}
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_history_arguments(parser)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--vintage', type=Path, required=True)
    parser.add_argument('--actuals-cutoff', type=aware_epoch)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must be a new explicit path')
    history = history_from_arguments(args)
    if history is None:
        parser.error('Explicit archive/cutoff required')
    result = compare(csv=args.csv, history=history, model_dir=args.model_dir,
        snapshot=json.loads(args.snapshot.read_text()), vintage=json.loads(args.vintage.read_text()),
        actuals_cutoff=args.actuals_cutoff)
    write_new(args.output, result)
    print(f'Offline storage replay saved to {args.output}')


if __name__ == '__main__':
    main()
