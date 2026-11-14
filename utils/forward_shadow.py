"""Explicit forward shadow capture, forecast and later scoring; never actuates.

Only capture contacts HA: four state GETs and weather.get_forecasts (read-only
despite its POST transport). Generation/scoring read an explicit archive/model
and return data. They never call predict(), prices, feedback SQLite, or planners.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.archive_history import ArchiveHistory, add_history_arguments, history_from_arguments
from utils.archive_quality import METERS, apply_training_policy
from utils.baseload_model import validate_baseload_model
from utils.measurement_reader import MeasurementReader
from utils.shadow_csv import CsvNotReady, freeze_csv, validate_sidecar


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None or pd.isna(result):
        raise ValueError('Timezone-aware timestamp required')
    return result.tz_convert('UTC')


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Finite forecast value required')
    return result


def capture_inputs(session: Any, host: str, token: str) -> dict[str, Any]:
    """The only allowed HA service is the non-actuating hourly weather read."""
    if not token or not host:
        raise ValueError('HA host and token are required')
    host = host if host.startswith(('http://', 'https://')) else 'http://' + host
    host = host.rstrip('/')
    headers = {'Authorization': 'Bearer ' + token, 'content-type': 'application/json'}
    started = datetime.now(timezone.utc)

    def state(entity: str) -> dict[str, Any]:
        response = session.get(host + '/api/states/' + entity, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError('Malformed HA state response')
        return data

    today = state('sensor.solcast_pv_forecast_forecast_today')
    tomorrow = state('sensor.solcast_pv_forecast_forecast_tomorrow')
    temperature = state('sensor.ulkolampotila')
    weather = state('weather.home')
    response = session.post(host + '/api/services/weather/get_forecasts?return_response',
        headers=headers, json={'entity_id': 'weather.home', 'type': 'hourly'}, timeout=20)
    response.raise_for_status()
    forecasts = response.json()
    if not isinstance(forecasts, dict):
        raise ValueError('Malformed weather forecast response')
    forecasts = forecasts.get('service_response', forecasts)
    return {'format_version': 1, 'history_cutoff': math.floor(started.timestamp() / 60) * 60,
            'started_at': started.isoformat(), 'completed_at': datetime.now(timezone.utc).isoformat(),
            'solar': [*today.get('attributes', {}).get('detailedHourly', []),
                      *tomorrow.get('attributes', {}).get('detailedHourly', [])],
            'weather': forecasts.get('weather.home', {}).get('forecast', []),
            'current_temperature': temperature, 'current_weather': weather}


def capture_with_csv(session: Any, host: str, token: str, csv: str | Path,
                     csv_output: str | Path, *, wait_seconds: float = 0) -> dict[str, Any]:
    """Freeze before HA reads, retaining the real capture clock and cutoff."""
    if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 1800:
        raise ValueError('CSV wait must be finite and between 0 and 1800 seconds')
    deadline = time.monotonic() + wait_seconds
    announced = False
    while True:
        cutoff = math.floor(datetime.now(timezone.utc).timestamp() / 60) * 60
        try:
            metadata = freeze_csv(csv, csv_output, cutoff)
            break
        except CsvNotReady:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            if not announced:
                print('Waiting for the normal CSV refresh before any HA capture...', file=sys.stderr, flush=True)
                announced = True
            time.sleep(min(2., remaining))
    snapshot = capture_inputs(session, host, token)
    snapshot['csv_sidecar'] = metadata
    validate_sidecar(snapshot)
    return snapshot


def _provenance(history: ArchiveHistory) -> dict[str, Any]:
    if history.allow_legacy or history.legacy_policy != 'verified-only':
        raise ValueError('Forward shadow requires recorder-only verified history')
    if history.sources_path is None or history.policy_path is None:
        raise ValueError('Explicit audited source and policy files are required')
    with MeasurementReader(history.path) as reader:
        identity = reader.db.execute('SELECT archive_uuid, recorder_namespace FROM archive_identity').fetchone()
    return {'archive_uuid': identity[0], 'recorder_namespace': identity[1],
            'sources_sha256': hashlib.sha256(Path(history.sources_path).read_bytes()).hexdigest(),
            'policy_sha256': hashlib.sha256(Path(history.policy_path).read_bytes()).hexdigest()}


def _forecast_table(entries: Any, date: str, values: list[str]) -> pd.DataFrame:
    if not isinstance(entries, list) or not entries:
        raise ValueError('Missing forecast horizon')
    frame = pd.DataFrame(entries)
    if not set([date, *values]).issubset(frame):
        raise ValueError('Incomplete forecast fields')
    frame[date] = [_timestamp(t) for t in frame[date]]
    for name in values:
        frame[name] = [_finite(value) for value in frame[name]]
    # Overlapping identical vintages are harmless; conflicting same-time values
    # must not silently select whichever array happened to be concatenated last.
    for _, group in frame.groupby(date):
        if any(group[name].nunique() > 1 for name in values):
            raise ValueError('Conflicting duplicate forecast timestamps')
    return frame.drop_duplicates(date).sort_values(date).set_index(date)


def generate_shadow(history: ArchiveHistory, model_dir: str | Path, snapshot: dict[str, Any],
                    *, reference: list[dict[str, Any]] | None = None,
                    now: pd.Timestamp | None = None) -> dict[str, Any]:
    # Reuse only pure helpers, never the operational predictor or its writers.
    from predict_future import archive_inference_anchors, generate_inference_data
    origin = _timestamp(now if now is not None else datetime.now(timezone.utc))
    started, completed = _timestamp(snapshot['started_at']), _timestamp(snapshot['completed_at'])
    if (snapshot.get('format_version') != 1 or snapshot.get('history_cutoff') != history.cutoff
            or not started <= completed <= origin or origin-completed > pd.Timedelta(minutes=15)
            or history.cutoff != started.floor('min').timestamp()):
        raise ValueError('Snapshot/cutoff/forecast availability mismatch')
    validate_sidecar(snapshot)
    provenance = _provenance(history)
    weather_state = snapshot['current_weather']
    if str(weather_state.get('state')).lower() in {'unknown', 'unavailable', 'none'}:
        raise ValueError('Current weather state unavailable')
    attrs = weather_state.get('attributes', {})
    temp_attrs = snapshot['current_temperature'].get('attributes', {})
    if (attrs.get('wind_speed_unit') != 'km/h' or attrs.get('temperature_unit') not in {'C', '°C'}
            or temp_attrs.get('unit_of_measurement') not in {'C', '°C'}):
        raise ValueError('Weather units do not match the trained feature contract')
    solar = _forecast_table(snapshot['solar'], 'period_start', ['pv_estimate'])
    weather = _forecast_table(snapshot['weather'], 'datetime', ['temperature', 'wind_speed'])
    if solar.pv_estimate.lt(0).any() or weather.wind_speed.lt(0).any():
        raise ValueError('Negative solar or wind forecast')
    model_dir = Path(model_dir)
    model = xgb.XGBRegressor()
    model.load_model(model_dir / 'energy_model.json')
    features = json.loads((model_dir / 'model_features.json').read_text())
    metadata = json.loads((model_dir / 'model_training_metadata.json').read_text())
    if not metadata.get('last_eligible') or _timestamp(metadata['last_eligible']).timestamp() >= history.cutoff:
        raise ValueError('Model training must precede the forecast origin')
    booster = model.get_booster()
    validate_baseload_model(features, metadata, booster.attr('baseload_feature_contract'), booster.feature_names)
    lag1, lag24, leaf1, leaf_energy = archive_inference_anchors(history)
    current = {'temp_val': _finite(snapshot['current_temperature']['state']),
               'wind_val': _finite(attrs['wind_speed']), 'lag1h_val': lag1,
               'lag24h_val': lag24, 'leaf_lag_val': leaf1, 'leaf_energy_val': leaf_energy}
    # process_frame normalizes the training index to UTC before extracting
    # calendar features. Do not reproduce the live predictor's local-clock
    # mismatch in this candidate experiment; live behavior is not changed here.
    start = origin.ceil('15min')
    end = min(solar.index.max(), weather.index.max(), origin + pd.Timedelta(hours=48)).tz_convert(start.tzinfo)
    if start > end:
        raise ValueError('No future forecast horizon')
    targets = pd.date_range(start, end, freq='15min')
    nearest = weather.index.get_indexer(targets, method='nearest')
    if any(abs(t - weather.index[int(i)]) >= pd.Timedelta(hours=2) for t, i in zip(targets, nearest)):
        raise ValueError('Weather forecast has an unsupported horizon gap')
    solar_nearest = solar.index.get_indexer(targets, method='nearest')
    if any(abs(t - solar.index[int(i)]) >= pd.Timedelta(hours=1) for t, i in zip(targets, solar_nearest)):
        raise ValueError('Solar forecast has an unsupported horizon gap')
    data, timestamps = generate_inference_data(start.to_pydatetime(), end.to_pydatetime(), 15,
        solar, weather, current, {'now': origin.to_pydatetime()})
    matrix = pd.DataFrame(data)[features]
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError('Nonfinite shadow inference features')
    predictions = model.predict(matrix)
    if len(predictions) != len(timestamps) or not np.isfinite(predictions).all():
        raise ValueError('Nonfinite or missing shadow model outputs')
    # The one-hour persistence baseline is held fixed at forecast origin, not
    # refreshed with future actuals for each target as in hindsight comparisons.
    floor = pd.Timestamp(history.cutoff, unit='s', tz='UTC').floor('15min')
    raw = history.raw((floor-pd.Timedelta(days=1)).timestamp(), floor.timestamp(),
                      measurements=list(METERS), interval_seconds=900)
    prior = apply_training_policy(raw, pd.DataFrame(index=raw.index), history.legacy_policy)
    prior_values = prior.baseload_power.where(prior.baseload_training_eligible)
    references = {_timestamp(row['timestamp']): _finite(row['predicted_baseload'])
                  for row in reference or [] if 'predicted_baseload' in row}
    generated = _timestamp(now if now is not None else datetime.now(timezone.utc))
    rows = []
    for timestamp, prediction, inputs in zip(timestamps, predictions, data):
        target = _timestamp(timestamp)
        if target <= generated:  # Never backdate an output into an elapsed slot.
            continue
        previous = prior_values.get(target-pd.Timedelta(days=1), np.nan)
        rows.append({'timestamp': target.isoformat(), 'candidate_kw': float(prediction),
                     'one_hour_origin_baseline_kw': lag1,
                     'previous_day_profile_kw': float(previous) if pd.notna(previous) else None,
                     'reference_baseload_kw': references.get(target),
                     'solar_forecast_kw': _finite(inputs['solar_forecast']),
                     'outside_temp': _finite(inputs['outside_temp']), 'wind_speed': _finite(inputs['wind_speed'])})
    if not rows:
        raise ValueError('No strictly future targets remain')
    return {'format_version': 1, 'shadow_only': True, 'generated_at': generated.isoformat(),
            'calendar_feature_timezone': 'UTC',
            'history_cutoff': history.cutoff, 'capture_completed_at': snapshot['completed_at'],
            'interval_seconds': 900, 'provenance': provenance,
            'model_version': metadata.get('model_version'),
            'model_sha256': hashlib.sha256((model_dir / 'energy_model.json').read_bytes()).hexdigest(),
            'input_snapshot_sha256': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
            'forecasts': rows,
            'notes': ['Never feed shadow outputs to control or publish them.',
                      'UTC calendar features match archived training; the unchanged operational predictor uses local time.',
                      'Historical anchors are fixed at history_cutoff; input capture and generation times are retained.',
                      'Previous-day profile is unavailable beyond observed history; no future actual baseline fill.']}


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    if not actual:
        return {'rows': 0, 'mae_kw': None, 'rmse_kw': None, 'bias_kw': None}
    error = np.asarray(predicted)-np.asarray(actual)
    return {'rows': len(actual), 'mae_kw': float(np.abs(error).mean()),
            'rmse_kw': float(np.sqrt((error**2).mean())), 'bias_kw': float(error.mean())}


def score_shadow(history: ArchiveHistory, forecast: dict[str, Any]) -> dict[str, Any]:
    if (forecast.get('format_version') != 1 or forecast.get('shadow_only') is not True
            or forecast.get('calendar_feature_timezone') != 'UTC'
            or forecast.get('interval_seconds') != 900 or _provenance(history) != forecast.get('provenance')):
        raise ValueError('Shadow provenance mismatch')
    generated = _timestamp(forecast['generated_at'])
    rows = forecast['forecasts']
    stamps = [_timestamp(row['timestamp']) for row in rows]
    if len(set(stamps)) != len(stamps) or any(t <= generated or t.timestamp() % 900 for t in stamps):
        raise ValueError('Shadow targets must be unique, aligned and strictly future at issue time')
    settled = [(t, row) for t, row in zip(stamps, rows) if t.timestamp()+900 <= history.cutoff]
    matched = []
    if settled:
        raw = history.raw(min(t.timestamp() for t, _ in settled), max(t.timestamp()+900 for t, _ in settled),
                          measurements=list(METERS), interval_seconds=900)
        targets = apply_training_policy(raw, pd.DataFrame(index=raw.index), history.legacy_policy)
        for t, row in settled:
            if bool(targets.loc[t, 'baseload_training_eligible']):
                matched.append((t, _finite(targets.loc[t, 'baseload_power']), row))
    actuals = [actual for _, actual, _ in matched]
    predictions = [_finite(row['candidate_kw']) for _, _, row in matched]
    comparisons = {}
    for name in ('one_hour_origin_baseline_kw', 'previous_day_profile_kw', 'reference_baseload_kw'):
        pairs = [(actual, row) for _, actual, row in matched if row.get(name) is not None]
        comparisons[name] = {
            'candidate': _metrics([a for a, _ in pairs], [_finite(row['candidate_kw']) for _, row in pairs]),
            'baseline': _metrics([a for a, _ in pairs], [_finite(row[name]) for _, row in pairs])}
    leads = {}
    for label, lower, upper in [('0-1h', 0, 1), ('1-6h', 1, 6), ('6-24h', 6, 24), ('24h+', 24, math.inf)]:
        group = [(actual, row) for t, actual, row in matched if lower <= (t-generated).total_seconds()/3600 < upper]
        leads[label] = _metrics([a for a, _ in group], [_finite(row['candidate_kw']) for _, row in group])
    return {'shadow_only': True, 'generated_at': forecast['generated_at'], 'actuals_cutoff': history.cutoff,
            'calendar_feature_timezone': forecast['calendar_feature_timezone'],
            'matched_actual_quarters': len(matched), 'not_yet_settled_quarters': len(rows)-len(settled),
            'invalid_actual_quarters': len(settled)-len(matched), 'candidate': _metrics(actuals, predictions),
            'comparisons': comparisons, 'candidate_by_lead': leads,
            'note': 'Scores retain one issued vintage and matched timestamps; missing actuals never become zero.'}


def write_new(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    if path.name in {'hepo.db', 'future_predictions.json', 'future_predictions_metadata.json',
                     'optimization_plan.json', 'energy_model.json', 'model_features.json',
                     'model_training_metadata.json', 'net_metering_state.json'}:
        raise ValueError('Shadow output must not use an active artifact filename')
    with path.open('x') as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write('\n')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture', help='Host-only HA reads; never a control call')
    capture.add_argument('--env-file', type=Path, required=True)
    capture.add_argument('--output', type=Path, required=True)
    capture.add_argument('--csv', type=Path, help='Optional recent CSV to freeze before HA capture')
    capture.add_argument('--csv-output', type=Path, help='New explicit frozen CSV sidecar path; requires --csv')
    capture.add_argument('--wait-for-csv-seconds', type=float, default=0,
                         help='Wait up to this many seconds for normal CSV refresh before HA reads (0..1800)')
    for command in ('generate', 'score'):
        sub = commands.add_parser(command, help='Offline; no HA/PostgreSQL/feedback DB calls')
        add_history_arguments(sub)
        sub.add_argument('--output', type=Path, required=True)
        if command == 'generate':
            sub.add_argument('--model-dir', type=Path, required=True)
            sub.add_argument('--snapshot', type=Path, required=True)
            sub.add_argument('--reference-forecast', type=Path)
        else:
            sub.add_argument('--forecast', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must be a new explicit path')
    if args.command == 'capture':
        if bool(args.csv) != bool(args.csv_output):
            parser.error('--csv and --csv-output must be supplied together')
        if args.wait_for_csv_seconds != 0 and args.csv is None:
            parser.error('--wait-for-csv-seconds requires --csv and --csv-output')
        if args.csv_output is not None and args.csv_output.resolve() == args.output.resolve():
            parser.error('CSV and snapshot output paths must be distinct')
        import requests
        from dotenv import dotenv_values
        config = {**dotenv_values(args.env_file), **os.environ}
        with requests.Session() as session:
            if args.csv is not None and args.csv_output is not None:
                result = capture_with_csv(session, config.get('HA_HOST') or '', config.get('HA_TOKEN') or '',
                                          args.csv, args.csv_output, wait_seconds=args.wait_for_csv_seconds)
            else:
                result = capture_inputs(session, config.get('HA_HOST') or '', config.get('HA_TOKEN') or '')
    else:
        history = history_from_arguments(args)
        if history is None:
            parser.error('Explicit --archive and --cutoff required')
        if args.command == 'generate':
            reference = json.loads(args.reference_forecast.read_text()) if args.reference_forecast else None
            result = generate_shadow(history, args.model_dir, json.loads(args.snapshot.read_text()), reference=reference)
        else:
            result = score_shadow(history, json.loads(args.forecast.read_text()))
    write_new(args.output, result)
    print(f'Shadow-only {args.command} saved to {args.output}')


if __name__ == '__main__':
    main()
