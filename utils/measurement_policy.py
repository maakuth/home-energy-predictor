"""Explicit measurement interpretation, independent of historical source YAML.

No operational unit or freshness evidence is inferred by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from utils.measurement_reader import numeric
from utils.measurement_sources import _UniqueSafeLoader


# dimension, multiplier, offset into the dimension's reference unit
_UNITS = {'W': ('power', 1., 0.), 'kW': ('power', 1000., 0.),
          'Wh': ('energy', 1., 0.), 'kWh': ('energy', 1000., 0.),
          'C': ('temperature', 1., 0.), '°C': ('temperature', 1., 0.),
          '°F': ('temperature', 5 / 9, -32 * 5 / 9),
          'm/s': ('speed', 1., 0.), 'km/h': ('speed', 1 / 3.6, 0.),
          '%': ('percent', 1., 0.), '1': ('ratio', 1., 0.),
          'state': ('state', 1., 0.)}


def convert_unit(value: float, input_unit: str, canonical_unit: str) -> float:
    """Convert compatible units; never silently convert energy to power."""
    first, second = _UNITS[input_unit], _UNITS[canonical_unit]
    if first[0] != second[0]:
        raise ValueError('Incompatible measurement units')
    return (value * first[1] + first[2] - second[2]) / second[1]


@dataclass(frozen=True)
class MeasurementPolicy:
    kind: str
    input_unit: str
    canonical_unit: str
    sign: int
    require_unit: bool = False
    attribute: str | None = None
    unit_attribute: str = 'unit_of_measurement'
    max_age_seconds: float | None = None
    nonnegative: bool = False
    state_values: dict[str, float] | None = None
    aggregation: str = 'duration'
    min_value: float | None = None
    max_value: float | None = None

    def observation(self, state: object, attributes: object) -> tuple[float, bool]:
        """Invalid states/attributes/units are missing, not zero or verified."""
        if str(state).lower() in {'unknown', 'unavailable', 'none'}:
            return math.nan, False
        try:
            attrs = json.loads(str(attributes)) if attributes is not None else {}
        except (ValueError, TypeError):
            return math.nan, False
        if not isinstance(attrs, dict):
            return math.nan, False
        unit = attrs.get(self.unit_attribute)
        verified = isinstance(unit, str) and unit in _UNITS and (
            _UNITS[unit] == _UNITS[self.input_unit])
        if (unit is not None and not verified) or (self.require_unit and not verified):
            return math.nan, False
        raw = attrs.get(self.attribute) if self.attribute else state
        if self.state_values is not None:
            value = self.state_values.get(str(raw).lower(), math.nan)
        else:
            value = numeric(raw)
        if self.kind != 'counter':
            # State mappings already express canonical output values.
            value = value if self.state_values is not None else convert_unit(
                value, self.input_unit, self.canonical_unit)
            value *= self.sign
        if self.nonnegative and value < 0:
            value = math.nan
        # Inclusive physical bounds use canonical units after sign conversion.
        # Preserve raw archive events; a correction is not verified meter truth.
        if ((self.min_value is not None and value < self.min_value)
                or (self.max_value is not None and value > self.max_value)):
            value = math.nan
        return value, bool(verified and math.isfinite(value))


def parse_policies(config: dict[str, Any]) -> dict[str, MeasurementPolicy]:
    if not isinstance(config, dict):
        raise ValueError('Expected a measurement policy mapping')
    result = {}
    fields = set(MeasurementPolicy.__dataclass_fields__)
    required = {'kind', 'input_unit', 'canonical_unit', 'sign'}
    for name, entry in config.items():
        if isinstance(entry, MeasurementPolicy):
            entry = asdict(entry)
        if (not isinstance(name, str) or not name or not isinstance(entry, dict)
                or set(entry) - fields or not required <= set(entry)):
            raise ValueError('Invalid measurement policy fields')
        if not isinstance(entry['kind'], str) or entry['kind'] not in {'numeric', 'state', 'presence', 'counter'}:
            raise ValueError('Unknown measurement kind')
        if any(not isinstance(entry[key], str) or entry[key] not in _UNITS
               for key in ('input_unit', 'canonical_unit')):
            raise ValueError('Unknown measurement unit')
        if type(entry['sign']) is not int or entry['sign'] not in {-1, 1}:
            raise ValueError('Sign must be +1 or -1')
        for key in ('require_unit', 'nonnegative'):
            if key in entry and type(entry[key]) is not bool:
                raise ValueError(f'{key} must be boolean')
        for key in ('attribute', 'unit_attribute'):
            value = entry.get(key)
            if (key in entry and not (key == 'attribute' and value is None)
                    and (not isinstance(value, str) or not value.strip())):
                raise ValueError(f'Invalid {key}')
        age = entry.get('max_age_seconds')
        if age is not None and (type(age) not in (int, float) or not math.isfinite(age) or age <= 0):
            raise ValueError('State expiry must be positive and finite')
        minimum, maximum = entry.get('min_value'), entry.get('max_value')
        for bound in (minimum, maximum):
            if bound is not None and (type(bound) not in (int, float) or not math.isfinite(bound)):
                raise ValueError('Physical bounds must be finite numbers')
        if minimum is not None or maximum is not None:
            if entry['kind'] != 'numeric' or (minimum is not None and maximum is not None and minimum > maximum):
                raise ValueError('Physical bounds require numeric kind and min_value <= max_value')
        mapping = entry.get('state_values')
        categorical = entry['kind'] in {'state', 'presence'}
        if categorical:
            if (entry['input_unit'] != 'state' or not isinstance(mapping, dict) or not mapping
                    or any(not isinstance(k, str) or not k or k != k.lower()
                           or k in {'unknown', 'unavailable', 'none'}
                           or type(v) not in (int, float) or not math.isfinite(v)
                           for k, v in mapping.items())):
                raise ValueError('States require explicit finite lower-case mappings')
        elif mapping is not None:
            raise ValueError('Only categorical policies accept state mappings')
        aggregation = entry.get('aggregation', 'duration')
        if not isinstance(aggregation, str) or aggregation not in {'duration', 'any_home'} or (aggregation == 'any_home' and (
                entry['kind'] != 'presence' or entry['canonical_unit'] != '1'
                or entry['sign'] != 1 or not isinstance(mapping, dict) or set(mapping.values()) != {0, 1}
                or mapping.get('home') != 1)):
            raise ValueError('any_home requires explicit home=1 and away-state=0 mappings')
        if entry['kind'] == 'presence' and (entry['canonical_unit'] != '1' or entry['sign'] != 1
                or not isinstance(mapping, dict) or not set(mapping.values()) <= {0, 1}):
            raise ValueError('Presence output must map to dimensionless 0/1 with positive sign')
        if categorical and entry['canonical_unit'] == 'state':
            raise ValueError('State mappings require numeric canonical output units')
        if entry['kind'] == 'counter':
            if (_UNITS[entry['input_unit']][0] != 'energy'
                    or _UNITS[entry['canonical_unit']][0] != 'power' or entry['sign'] != 1):
                raise ValueError('Counters require energy input and positive power output')
        elif not categorical:
            if entry['input_unit'] == 'state' or entry['canonical_unit'] == 'state':
                raise ValueError('Numeric policies cannot interpret category units')
            convert_unit(0, entry['input_unit'], entry['canonical_unit'])
        result[name] = MeasurementPolicy(**entry)
    return result


def load_policies(path: str | Path) -> dict[str, MeasurementPolicy]:
    return load_policy_registry(path)[0]


def load_policy_registry(path: str | Path) -> tuple[dict[str, MeasurementPolicy], str]:
    """Parse and hash exactly the same bytes, not two racy reads."""
    contents = Path(path).read_bytes()
    try:
        config = yaml.load(contents.decode('utf-8'), Loader=_UniqueSafeLoader)
    except (yaml.YAMLError, UnicodeError) as error:
        raise ValueError(f'Invalid measurement policy YAML: {error}') from error
    return parse_policies(config), hashlib.sha256(contents).hexdigest()


def require_policy_entries(policies: dict[str, MeasurementPolicy], measurements: list[str]) -> None:
    """Pipeline preflight: explicit coverage, not proof of operational evidence."""
    missing = sorted(set(measurements) - policies.keys())
    if missing:
        raise ValueError(f'Measurement policies missing entries: {", ".join(missing)}')


def counter_power(first: float, last: float, seconds: float,
                  policy: MeasurementPolicy) -> float:
    """Boundary delta only; reset/invalid/gap handling belongs to the caller."""
    if not all(math.isfinite(v) for v in (first, last, seconds)) or seconds <= 0 or last < first:
        return math.nan
    return (last - first) * _UNITS[policy.input_unit][1] * 3600 / seconds / _UNITS[policy.canonical_unit][1]
