"""Read measurement windows and generate existing features without CSV files.

Archive mode is explicit. This module never ingests, contacts HA/SQL, or falls
back to operational CSVs. Legacy eligibility never implies source verification.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd

from process_data import process_frame
from utils.archive_quality import LEGACY_POLICIES, apply_training_policy
from utils.measurement_reader import MeasurementReader, interval_index
from utils.measurement_resampling import resample_numeric
from utils.measurement_sources import MeasurementSource, load_sources, parse_sources
from utils.measurement_policy import load_policy_registry, parse_policies, require_policy_entries


BASELOAD_INPUTS = ('outside_temp', 'wind_speed', 'solar_forecast', 'total_power',
                   'solar_actual', 'battery_power', 'gshp_power', 'leaf_power')
_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'measurement_sources.yaml'
_QUALITY_SUFFIXES = ('_source_valid', '_coverage_fraction', '_provenance', '_conflict', '_import_ids', '_entities', '_unit_verified')
_CANONICAL_UNITS = {
    **dict.fromkeys(('total_power', 'solar_actual', 'solar_forecast', 'mummun_power'), 'kW'),
    **dict.fromkeys(('battery_power', 'gshp_power', 'leaf_power', 'aahp_living_power',
                     'aahp_cabin_power', 'resistive_heater_power', 'bulk_heater_power'), 'W'),
    **dict.fromkeys(('outside_temp', 'gshp_supply_temp', 'gshp_return_temp', 'sauna_temp',
                     'gshp_pump_temp', 'accumulator_temp'), 'C'),
    'ev_soc': '%', 'ev_position': '1',
}


def default_sources() -> dict[str, list[MeasurementSource]]:
    """Existing extraction defaults, evaluated at call time, not import time.

    Dated YAML overrides are applied separately by the loader. Optional disabled
    accumulator instrumentation is omitted rather than inventing an entity.
    """
    definitions = {
        'outside_temp': os.getenv('OUTSIDE_TEMP_ENTITY', 'sensor.ulkona_temperature_2'),
        'gshp_power': os.getenv('GSHP_POWER_ENTITY', 'sensor.mlp_teho'),
        'resistive_heater_power': os.getenv('RESISTIVE_HEATER_ENTITY', 'switch.mlp_vastus_output_0'),
        'bulk_heater_power': os.getenv('BULK_HEATER_ENTITY', 'switch.mlp_vastus_output_1'),
        'aahp_living_power': 'sensor.saikaan_olohuone_current_power',
        'aahp_cabin_power': 'sensor.mokkimokin_ilp_power',
        'mummun_energy': 'sensor.mummun_energy',
        'gshp_supply_temp': os.getenv('GSHP_SUPPLY_TEMP_ENTITY', 'sensor.gshp_supply_temp'),
        'gshp_return_temp': os.getenv('GSHP_RETURN_TEMP_ENTITY', 'sensor.gshp_return_temp'),
        'ev_soc': 'sensor.xpz_491_battery_level',
        'ev_position': 'device_tracker.xpz_491_position',
        'leaf_power': 'sensor.tasmota_energy_power_3',
        'total_power': 'sensor.sahkokauppa_nyt',
        'solar_forecast': 'sensor.solcast_pv_forecast_forecast_tomorrow',
        'solar_actual': os.getenv('SOLAR_PRODUCTION_ENTITY', 'sensor.solarh_63038_real_power_kw'),
        'sauna_temp': 'sensor.sauna_temperature_2',
        'gshp_pump_temp': 'sensor.mlp_pumpun_lampotla',
        'battery_power': 'sensor.be_stat_batt_power',
    }
    accumulator = os.getenv('GSHP_ACCUMULATOR_TEMP_ENTITY', 'sensor.mlp_varaajan_lampotila').strip()
    if accumulator:
        definitions['accumulator_temp'] = accumulator
    config = {name: [{'entity': entity}] for name, entity in definitions.items()}
    config['wind_speed'] = [{'entity': 'weather.home', 'attribute': 'wind_speed'}]
    return parse_sources(config)


def load_archive_raw(
    path: str | Path, start: float, end: float,
    *, sources: dict[str, list[MeasurementSource]] | None = None,
    sources_path: str | Path = _CONFIG, measurements: list[str] | None = None,
    interval_seconds: int = 60, chunk_seconds: int = 86400,
    allow_legacy: bool = False, preferred_imports: list[int] | None = None,
    policy_path: str | Path | None = None, policies: dict | None = None,
    require_policies: bool = False,
) -> pd.DataFrame:
    """Resample bounded chunks under a single read-only archive snapshot.

    Units match the existing processor: grid/solar are kW, shared heating,
    Leaf, battery and relay estimates are W. Source units must be audited before
    cutover. Energy-counter and category policies require an explicit registry.
    require_policies enforces registry completeness; it does not certify units.
    """
    interval_index(start, end, interval_seconds)
    if not isinstance(chunk_seconds, int) or chunk_seconds <= 0 or chunk_seconds % interval_seconds:
        raise ValueError('Chunk size must be positive and divisible by the interval')
    if sources is None:
        sources = default_sources()
        sources.update(load_sources(sources_path))
    if policy_path is not None and policies is not None:
        raise ValueError('Use either policy_path or policies, not both')
    registry, policy_sha256 = (load_policy_registry(policy_path) if policy_path is not None
                               else (parse_policies(policies if policies is not None else {}), None))
    for name, policy in registry.items():
        if name in _CANONICAL_UNITS and policy.canonical_unit != _CANONICAL_UNITS[name]:
            raise ValueError(f'Invalid canonical unit for {name}: expected {_CANONICAL_UNITS[name]}')
        if ((name == 'ev_position' and policy.kind != 'presence')
                or (name in {'mummun_energy', 'mummun_power'} and policy.kind != 'counter')):
            raise ValueError(f'Invalid measurement kind for {name}')
    if 'mummun_power' in registry and 'mummun_power' not in sources and 'mummun_energy' in sources:
        sources = {**sources, 'mummun_power': sources['mummun_energy']}
    names = list(dict.fromkeys(measurements if measurements is not None else BASELOAD_INPUTS))
    if require_policies:
        require_policy_entries(registry, names)
    if not names or any(name not in sources for name in names):
        raise ValueError('Unknown or empty measurement selection')
    if any(name in {'ev_position', 'mummun_energy', 'mummun_power'} and name not in registry for name in names):
        raise ValueError('Category/energy-counter measurements require an explicit policy')
    frames = []
    with MeasurementReader(path) as reader:
        imports = [dict(row) for row in reader.db.execute(
            'SELECT id, sha256, source_path, interval_seconds FROM legacy_imports ORDER BY id')]
        left = start
        while left < end:
            right = min(end, left + chunk_seconds)
            columns = {}
            for name in names:
                state_values = None
                if name not in registry and name in {'resistive_heater_power', 'bulk_heater_power'}:
                    var = 'RESISTIVE_HEATER_POWER_KW' if name == 'resistive_heater_power' else 'BULK_HEATER_POWER_KW'
                    watts = float(os.getenv(var, '6.0')) * 1000
                    if not math.isfinite(watts) or watts <= 0:
                        raise ValueError(f'Invalid heater rating: {var}')
                    state_values = {'on': watts, 'off': 0.0}
                result = resample_numeric(reader, sources, name, left, right,
                    interval_seconds=interval_seconds, allow_legacy=allow_legacy,
                    preferred_imports=preferred_imports, state_values=state_values,
                    policy=registry.get(name),
                    nonnegative=name in {'solar_actual', 'gshp_power', 'leaf_power', 'wind_speed',
                                         'resistive_heater_power', 'bulk_heater_power'})
                columns[name] = result['value']
                for suffix in _QUALITY_SUFFIXES:
                    columns[name + suffix] = result[suffix[1:]]
            frames.append(pd.DataFrame(columns))
            left = right
    out = pd.concat(frames)
    out.attrs['archive'] = {'archive_path': str(Path(path).resolve()),
                             'start': start, 'end': end, 'interval_seconds': interval_seconds,
                             'allow_legacy': allow_legacy, 'preferred_imports': preferred_imports or [],
                             'policy_path': str(policy_path) if policy_path is not None else None,
                             'policies': {name: asdict(policy) for name, policy in registry.items()},
                             'implicit_policy_measurements': sorted(set(names) - registry.keys()),
                             'policy_sha256': policy_sha256,
                             'policy_semantic_sha256': hashlib.sha256(json.dumps(
                                 {name: asdict(policy) for name, policy in registry.items()},
                                 sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                            'imports': imports}
    return out


def load_archive_features(path: str | Path, start: float, end: float, *,
                          interval_seconds: int = 60, legacy_policy: str = 'verified-only',
                          **options) -> pd.DataFrame:
    """Include a full preceding day for exact lags, then trim the output window."""
    interval_index(start, end, interval_seconds)
    if legacy_policy not in LEGACY_POLICIES:
        raise ValueError('Unknown legacy training policy')
    if legacy_policy != 'verified-only' and not options.get('allow_legacy', False):
        raise ValueError('Legacy training policy requires allow_legacy=True')
    if interval_seconds < 60 or 86400 % interval_seconds:
        raise ValueError('Feature resolution must be whole minutes dividing a day')
    if interval_seconds % 60:
        raise ValueError('Feature resolution must be whole minutes')
    raw = load_archive_raw(path, start - 86400, end, interval_seconds=interval_seconds, **options)
    metadata = [name for name in raw if name.endswith(_QUALITY_SUFFIXES) and not name.endswith('_source_valid')]
    features = process_frame(raw.drop(columns=metadata))
    # The compatibility processor treats all float positions as strings and
    # smooths/fills optional states and counters. These explicit policy outputs
    # are already interpreted; do not turn missing history into away/zero.
    for name, policy in raw.attrs['archive']['policies'].items():
        if name in raw and policy['kind'] in {'state', 'presence', 'counter'}:
            features[name] = raw[name]
    # Generic numeric processing must not smooth or fabricate archive metadata.
    for name in raw:
        if name.endswith(_QUALITY_SUFFIXES):
            features[name] = raw[name]
        else:
            # Numeric compatibility denoising is retained only where input is
            # actually present; its zero/forward fills cannot establish history.
            features[name] = features[name].where(raw[name].notna())
    features = apply_training_policy(raw, features, legacy_policy)
    features.attrs['archive_training'].update({'start': start, 'end': end, 'context_start': start - 86400})
    lower, upper = pd.to_datetime(start, unit='s', utc=True), pd.to_datetime(end, unit='s', utc=True)
    return features.loc[(features.index >= lower) & (features.index < upper)].copy()
