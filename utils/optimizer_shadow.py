"""Compare two candidate plans from a single explicit frozen input bundle.

The bundle is supplied by the caller; this module does not capture live state
or prove when an input became available. Plans are candidates before the
operational current-slot latch, not published control plans.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite number')
    return float(value)


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError('Expected an explicit boolean')
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError('Expected an ISO timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return parsed.astimezone(timezone.utc)


def compare_frozen_plans(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute isolated candidate plans; no capture, publish, or feedback writes."""
    # optimize_plan imports dotenv at module load. Refuse to import it unless
    # dotenv is explicitly disabled; never load host credentials for a replay.
    if os.getenv('PYTHON_DOTENV_DISABLED', '').lower() not in {'1', 'true', 'yes', 'on'}:
        raise ValueError('Set PYTHON_DOTENV_DISABLED=1 for offline comparison')
    from optimize_plan import FrozenOptimizationRuntime, calculate_optimization

    # Copy before validation so route computations share the same frozen bundle.
    try:
        encoded = json.dumps(bundle, sort_keys=True, allow_nan=False)
        data = json.loads(encoded)
        if data['format_version'] != 1:
            raise ValueError('Unsupported bundle format')
        model_hash = data['model_sha256']
        if not isinstance(model_hash, str) or len(model_hash) != 64 or any(c not in '0123456789abcdef' for c in model_hash):
            raise ValueError('Expected lowercase SHA256 model identity')
        origin = _timestamp(data['origin'])
        interval = data['interval_minutes']
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError('Invalid interval')
        config = data['config']
        if not isinstance(config, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in config.items()):
            raise ValueError('Configuration must contain strings only')
        if int(config.get('PLAN_INTERVAL_MINUTES', str(interval))) != interval:
            raise ValueError('Planning interval disagrees with configuration')
        config['PLAN_INTERVAL_MINUTES'] = str(interval)
        output_timezone = data['output_timezone']
        ZoneInfo(output_timezone)
        tomorrow_valid = _boolean(data['tomorrow_valid'])
        rows = data['horizon']
        if not isinstance(rows, list) or not rows:
            raise ValueError('Empty horizon')
        timestamps = [_timestamp(row['timestamp']) for row in rows]
        if any(t <= origin or t.timestamp() % (interval * 60) for t in timestamps):
            raise ValueError('Targets must be aligned and after origin')
        if any((b - a).total_seconds() != interval * 60 for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError('Horizon must be consecutive')
        runtime = data['runtime']
        battery_enabled = _boolean(runtime['battery_enabled'])
        allow_export = _boolean(runtime['allow_export'])
        gshp_failed = _boolean(runtime['gshp_failed'])
        compressor_running = _boolean(runtime['compressor_running'])
        resistive_running = _boolean(runtime['resistive_running'])
        fireplace_on = _boolean(runtime['fireplace_on'])
        current_temp = _number(runtime['current_acc_temp'])
        soc = runtime['battery_soc_pct']
        soc = None if soc is None else _number(soc)
        if soc is not None and not 0 <= soc <= 100:
            raise ValueError('Invalid battery SOC')
        model_version = runtime['model_version']
        if not isinstance(model_version, str) or not model_version:
            raise ValueError('Missing model version')
        ev_state = runtime['ev_soc_state']
        if ev_state is not None and not isinstance(ev_state, dict):
            raise ValueError('Invalid EV state')
        if ev_state is not None:
            ev_value = ev_state.get('state')
            if ev_value not in ('unknown', 'unavailable'):
                if isinstance(ev_value, bool) or not isinstance(ev_value, (int, float, str)):
                    raise ValueError('Invalid EV SOC')
                try:
                    ev_soc = float(ev_value)
                except ValueError as error:
                    raise ValueError('Invalid EV SOC') from error
                if not math.isfinite(ev_soc) or not 0 <= ev_soc <= 100:
                    raise ValueError('Invalid EV SOC')
        numeric = ('outside_temp', 'solar_kw', 'solar_p10_kw', 'solar_p90_kw',
                   'market_price', 'import_price', 'export_price')
        for row in rows:
            for field in numeric:
                _number(row[field])
            for field in ('is_sauna_active', 'ev_position', 'fallback_price'):
                if type(row[field]) is not int or row[field] not in (0, 1):
                    raise ValueError(f'Invalid {field}')
            for field in ('sarima_lower', 'sarima_upper'):
                if row[field] is not None:
                    _number(row[field])
            if any(row[field] < 0 for field in ('solar_kw', 'solar_p10_kw', 'solar_p90_kw')):
                raise ValueError('Negative solar forecast')
        routes = data['routes']
        if not isinstance(routes, dict) or set(routes) != {'archive', 'compatibility'}:
            raise ValueError('Both explicitly named routes are required')
        for route in routes.values():
            for key in ('baseload_kw', 'uncontrolled_heating_kw'):
                values = route[key]
                if not isinstance(values, list) or len(values) != len(rows):
                    raise ValueError('Route values must match the horizon')
                for value in values:
                    if _number(value) < 0:
                        raise ValueError(f'{key} cannot be negative')
        if battery_enabled and soc is None:
            raise ValueError('Enabled battery requires explicit SOC')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Malformed frozen optimizer bundle') from error

    plans: dict[str, list[dict[str, Any]]] = {}
    for name in ('compatibility', 'archive'):
        route = routes[name]
        frozen = FrozenOptimizationRuntime(
            ev_soc_state=ev_state, allow_export=allow_export, battery_soc_pct=soc,
            battery_enabled=battery_enabled,
            uncontrolled_heating_kw=np.array(route['uncontrolled_heating_kw'], dtype=float),
            model_version=model_version, gshp_failed=gshp_failed,
            output_timezone=output_timezone, resistive_running=resistive_running, config=config,
        )
        plans[name] = calculate_optimization(
            rows, np.array(route['baseload_kw'], dtype=float), timestamps,
            pd.Series([float('nan') if row['sarima_lower'] is None else row['sarima_lower'] for row in rows]),
            pd.Series([float('nan') if row['sarima_upper'] is None else row['sarima_upper'] for row in rows]),
            [row['market_price'] for row in rows], np.array([row['import_price'] for row in rows]),
            np.array([row['export_price'] for row in rows]), [row['fallback_price'] for row in rows],
            tomorrow_valid, np.array([row['solar_kw'] for row in rows]),
            np.array([row['solar_p10_kw'] for row in rows]), np.array([row['solar_p90_kw'] for row in rows]),
            current_temp, compressor_running, fireplace_on, history=None, frozen_runtime=frozen,
        )
    fields = ('battery_action', 'gshp_intent', 'resistive_heater_intent', 'bulk_heater_intent', 'leaf_intent')
    differences = {field: sum(a.get(field) != b.get(field) for a, b in zip(plans['archive'], plans['compatibility']))
                   for field in fields}
    numeric_fields = ('predicted_baseload_kw', 'uncontrolled_heating_kw', 'predicted_usage_kw',
                      'battery_power_kw', 'soc_pct', 'grid_import_kwh', 'grid_export_kwh',
                      'planned_gshp_kw', 'planned_resistive_kw', 'planned_bulk_heater_kw',
                      'planned_leaf_kw', 'gshp_temp_simulated', 'charge_from_solar_kwh',
                      'charge_from_grid_kwh', 'discharge_to_load_kwh', 'discharge_to_export_kwh',
                      'discharge_budget_kwh', 'effective_cost', 'estimated_hour_cost')
    numeric_deltas = {}
    for field in numeric_fields:
        deltas = [_number(a[field]) - _number(b[field])
                  for a, b in zip(plans['archive'], plans['compatibility'])
                  if a.get(field) is not None and b.get(field) is not None]
        numeric_deltas[field] = {
            'matched_intervals': len(deltas),
            'mean_signed': sum(deltas) / len(deltas) if deltas else None,
            'mean_abs': sum(abs(value) for value in deltas) / len(deltas) if deltas else None,
            'max_abs': max(abs(value) for value in deltas) if deltas else None,
        }
    return {
        'shadow_only': True, 'format_version': 1,
        'bundle_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
        'model_sha256': model_hash, 'origin': data['origin'],
        'plans': plans,
        'comparison': {'matched_intervals': len(rows), 'different_intents': differences,
                       'numeric_deltas': numeric_deltas},
        'note': 'Candidate plans before operational latching; caller-supplied inputs and provenance are not independently verified.',
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Explicit frozen input bundle')
    parser.add_argument('--output-dir', type=Path, required=True, help='New isolated output directory')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {args.output_dir}')
    result = compare_frozen_plans(json.loads(args.input.read_text()))
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    args.output_dir.mkdir()
    output = args.output_dir / 'optimizer-shadow-comparison.json'
    with output.open('x') as stream:
        stream.write(encoded)
    print(f'Shadow-only comparison saved to {output}')


if __name__ == '__main__':
    main()
