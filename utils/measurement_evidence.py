"""Operator-run, read-only recorder/unit evidence. No archive or control writes."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import psycopg2

from dotenv import load_dotenv

_METADATA = ('unit_of_measurement', 'state_class', 'device_class', 'wind_speed_unit',
             'temperature_unit')


def aware_epoch(value: str) -> float:
    try:
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError('Timezone required')
        return timestamp.timestamp()
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected ISO timestamp with explicit timezone') from error


@dataclass(frozen=True)
class _Source:
    entity: str
    attribute: str | None = None
    start: float | None = None
    end: float | None = None


def _parse_sources(config: Any) -> dict[str, list[_Source]]:
    if not isinstance(config, dict) or not config:
        raise ValueError('Expected a nonempty source mapping')
    result = {}
    for name, entries in config.items():
        if not isinstance(name, str) or not isinstance(entries, list) or not entries:
            raise ValueError('Invalid measurement/source list')
        parsed = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {'entity', 'attribute', 'from', 'to'}:
                raise ValueError('Invalid source fields')
            entity, attribute = entry.get('entity'), entry.get('attribute')
            if not isinstance(entity, str) or not entity.strip() or (attribute is not None and not isinstance(attribute, str)):
                raise ValueError('Invalid entity/attribute')
            try:
                start = aware_epoch(str(entry['from'])) if entry.get('from') is not None else None
                end = aware_epoch(str(entry['to'])) if entry.get('to') is not None else None
            except argparse.ArgumentTypeError as error:
                raise ValueError(str(error)) from error
            if start is not None and end is not None and start >= end:
                raise ValueError('Source bounds must increase')
            parsed.append(_Source(entity, attribute, start, end))
        parsed.sort(key=lambda source: source.start if source.start is not None else -math.inf)
        for previous, current in zip(parsed, parsed[1:]):
            if previous.end is None or current.start is None or previous.end > current.start:
                raise ValueError(f'Overlapping source periods: {name}')
        result[name] = parsed
    return result


def load_sources(path: Path) -> dict[str, list[_Source]]:
    try:
        import yaml
    except ImportError as error:
        raise ValueError('Custom --sources requires PyYAML; omit it to use the embedded inventory') from error
    class UniqueLoader(yaml.SafeLoader):
        def construct_mapping(self, node: Any, deep: bool = False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in result:
                    raise ValueError('Invalid/duplicate source YAML key')
                result[key] = self.construct_object(value_node, deep=deep)
            return result
    try:
        return _parse_sources(yaml.load(path.read_text(), Loader=UniqueLoader))
    except yaml.YAMLError as error:
        raise ValueError('Invalid source YAML') from error


def default_sources() -> dict[str, list[_Source]]:
    """Portable audit inventory; does not import any application/control code."""
    definitions = {
        'outside_temp': os.getenv('OUTSIDE_TEMP_ENTITY', 'sensor.ulkona_temperature_2'),
        'gshp_power': os.getenv('GSHP_POWER_ENTITY', 'sensor.mlp_teho'),
        'total_power': 'sensor.sahkokauppa_nyt', 'solar_actual': os.getenv('SOLAR_PRODUCTION_ENTITY', 'sensor.solarh_63038_real_power_kw'),
        'battery_power': 'sensor.be_stat_batt_power', 'leaf_power': 'sensor.tasmota_energy_power_3',
        'solar_forecast': 'sensor.solcast_pv_forecast_forecast_tomorrow',
        'sauna_temp': 'sensor.sauna_temperature_2', 'ev_soc': 'sensor.xpz_491_battery_level',
        'ev_position': 'device_tracker.xpz_491_position', 'mummun_energy': 'sensor.mummun_energy',
        'gshp_supply_temp': os.getenv('GSHP_SUPPLY_TEMP_ENTITY', 'sensor.gshp_supply_temp'),
        'gshp_return_temp': os.getenv('GSHP_RETURN_TEMP_ENTITY', 'sensor.gshp_return_temp'),
        'gshp_pump_temp': 'sensor.mlp_pumpun_lampotla',
        'accumulator_temp': os.getenv('GSHP_ACCUMULATOR_TEMP_ENTITY', 'sensor.mlp_varaajan_lampotila'),
        'resistive_heater_power': os.getenv('RESISTIVE_HEATER_ENTITY', 'switch.mlp_vastus_output_0'),
        'bulk_heater_power': os.getenv('BULK_HEATER_ENTITY', 'switch.mlp_vastus_output_1'),
        'aahp_living_power': 'sensor.saikaan_olohuone_current_power', 'aahp_cabin_power': 'sensor.mokkimokin_ilp_power',
    }
    result = {name: [_Source(entity)] for name, entity in definitions.items() if entity.strip()}
    result['wind_speed'] = [_Source('weather.home', 'wind_speed')]
    return result


def _confirmed_sources() -> dict[str, list[_Source]]:
    return _parse_sources({
        'outside_temp': [dict(entity='sensor.ulkona_temperature_2', to='2026-10-19T00:00:00+03:00'),
                         {'entity': 'sensor.ulkolampotila', 'from': '2026-10-19T00:00:00+03:00'}],
        'gshp_power': [dict(entity='sensor.mlp_teho', to='2026-10-20T08:30:00+03:00'),
                       {'entity': 'sensor.instantaneous_used_power_32167', 'from': '2026-10-20T08:30:00+03:00'}],
    })


def mapped_ranges(sources: dict[str, list[_Source]], start: float, end: float,
                   measurements: list[str] | None):
    selected = list(sources) if measurements is None else list(dict.fromkeys(measurements))
    if not selected or any(name not in sources for name in selected):
        raise ValueError('Unknown or empty measurement selection')
    return [(name, source, max(start, source.start if source.start is not None else start),
             min(end, source.end if source.end is not None else end))
            for name in selected for source in sources[name]
            if max(start, source.start if source.start is not None else start)
            < min(end, source.end if source.end is not None else end)]


def get_db_connection():
    return psycopg2.connect(host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'), connect_timeout=10)


def summarize_observations(rows: list[tuple[Any, ...]], *, attribute: str | None) -> dict[str, Any]:
    variants: Counter[str] = Counter()
    values: list[float] = []
    timestamps: list[float] = []
    malformed = invalid = 0
    forecast_keys: set[str] = set()
    for timestamp, state, attributes in rows:
        timestamps.append(float(timestamp))
        try:
            attrs = json.loads(attributes) if isinstance(attributes, str) else (attributes or {})
            if not isinstance(attrs, dict):
                raise ValueError('Attributes must be an object')
        except (ValueError, TypeError):
            malformed += 1
            attrs = {}
        metadata = {key: value[:128] if isinstance(value, str) else value for key, value in attrs.items()
                    if key in _METADATA and (value is None or isinstance(value, (str, bool, int)))}
        variants[json.dumps(metadata, sort_keys=True)] += 1
        for key in ('detailedHourly', 'detailedForecast'):
            entries = attrs.get(key)
            if isinstance(entries, list):
                for entry in entries[:2]:
                    if isinstance(entry, dict):
                        forecast_keys.update(str(name) for name in entry)
        raw = attrs.get(attribute) if attribute is not None else state
        try:
            value = float(raw) if raw is not None and str(state).lower() not in {'unknown', 'unavailable', 'none'} else math.nan
        except (ValueError, TypeError):
            value = math.nan
        if math.isfinite(value):
            values.append(value)
        else:
            invalid += 1
    ordered = sorted(set(timestamps))
    return {
        'sampled_events': len(rows), 'attribute': attribute,
        'sample_first_epoch': min(timestamps) if timestamps else None,
        'sample_last_epoch': max(timestamps) if timestamps else None,
        'sample_max_update_gap_seconds': max((b - a for a, b in zip(ordered, ordered[1:])), default=None),
        'metadata_variants': [{'metadata': json.loads(key), 'events': count} for key, count in variants.items()],
        'malformed_attributes': malformed, 'nonfinite_or_nonnumeric_events': invalid,
        'numeric_values': {'count': len(values), 'minimum': min(values) if values else None,
                           'maximum': max(values) if values else None,
                           'negative': sum(value < 0 for value in values),
                           'zero': sum(value == 0 for value in values),
                           'positive': sum(value > 0 for value in values)},
        'forecast_entry_keys': sorted(forecast_keys),
        'source_verification': 'not_established',
        'note': 'Bounded sample, not complete coverage. Update gaps do not establish a freshness limit; '
                'numeric sign alone does not establish charge/discharge semantics. Units are not converted.',
    }


def collect_postgres(connection: Any, requests: list[tuple[str, str, str | None]],
                     start: float, end: float, *, sample_limit: int = 200,
                     source_ranges: dict[tuple[str, str, str | None], list[tuple[float, float]]] | None = None) -> dict[str, Any]:
    if not 1 <= sample_limit <= 10000 or not all(math.isfinite(v) for v in (start, end)) or start >= end:
        raise ValueError('Require increasing finite bounds and sample limit 1..10000')
    connection.set_session(readonly=True, autocommit=False)
    cursor = connection.cursor()
    errors: list[dict[str, str]] = []

    def query(label: str, statement: str, parameters=None, *, many: bool = False):
        try:
            cursor.execute("SET LOCAL statement_timeout = '15s'")
            cursor.execute("SET LOCAL lock_timeout = '3s'")
            cursor.execute(statement, parameters)
            return cursor.fetchall() if many else cursor.fetchone()
        except Exception as error:
            connection.rollback()
            errors.append({'operation': label, 'error_type': type(error).__name__, 'error': str(error)})
            return None

    try:
        cursor.execute('SHOW transaction_read_only')
        row = cursor.fetchone()
        if row is None or row[0] != 'on':
            raise ValueError('Server did not confirm a read-only transaction; refusing inspection')
        row = query('database_identity', '''SELECT current_database(), current_setting('server_version'),
            (SELECT oid FROM pg_database WHERE datname=current_database()), 'states'::regclass::oid''')
        identity: dict[str, Any] = dict(zip(('database', 'server_version', 'database_oid', 'states_table_oid'), row)) if row else {}
        system = query('cluster_identity', 'SELECT system_identifier::text FROM pg_control_system()')
        identity['cluster_system_identifier'] = system[0] if system else None
        identity['namespace_confirmation'] = 'operator_required; restore/truncate/rebuild may reuse identity evidence'
        result = []
        for name, entity, attribute in dict.fromkeys(requests):
            metadata = query(f'{entity}:metadata', 'SELECT metadata_id FROM states_meta WHERE entity_id=%s', (entity,))
            item: dict[str, Any] = {'measurement': name, 'entity': entity, 'attribute': attribute,
                                    'recorder_entity_found': metadata is not None}
            if metadata is not None:
                metadata_id = metadata[0]
                for key, direction in (('retained_first_epoch', 'ASC'), ('retained_last_epoch', 'DESC')):
                    endpoint = query(f'{entity}:{key}', f'''SELECT last_updated_ts FROM states
                        WHERE metadata_id=%s AND last_updated_ts IS NOT NULL
                        ORDER BY last_updated_ts {direction}, state_id {direction} LIMIT 1''', (metadata_id,))
                    item[key] = float(endpoint[0]) if endpoint else None
                windows = (source_ranges or {}).get((name, entity, attribute), [(start, end)])
                item['sample_windows'] = windows
                sampled = []
                for left, right in windows:
                    rows = query(f'{entity}:sample', '''SELECT s.last_updated_ts, s.state, LEFT(a.shared_attrs::text, 65536) AS shared_attrs
                        FROM states s LEFT JOIN state_attributes a ON a.attributes_id=s.attributes_id
                        WHERE s.metadata_id=%s AND s.last_updated_ts>=%s AND s.last_updated_ts<%s
                        ORDER BY s.last_updated_ts DESC, s.state_id DESC LIMIT %s''',
                        (metadata_id, left, right, sample_limit), many=True)
                    sampled.extend(rows or [])
                sampled.sort(key=lambda row: row[0], reverse=True)
                item['sample'] = summarize_observations(sampled[:sample_limit], attribute=attribute)
                item['sample']['attribute_characters_cap'] = 65536
            result.append(item)
        return {'read_only': True, 'identity_evidence': identity, 'sources': result, 'errors': errors,
                'sample_limit_per_source': sample_limit,
                'sampling': 'Latest bounded events within mapped source windows; attributes capped at 65536 characters '
                            '(truncated JSON is reported malformed, never trusted). Endpoint dates use LIMIT 1 queries.'}
    finally:
        connection.rollback()
        cursor.close()


def collect_home_assistant(entities: list[str]) -> dict[str, Any]:
    host = os.getenv('HA_HOST', '').strip().rstrip('/')
    if host and '://' not in host:
        host = 'http://' + host
    token = os.getenv('HA_TOKEN', '').strip()
    parts = urlsplit(host)
    if (not token or parts.scheme not in {'http', 'https'} or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError('--ha requires configured HA_HOST and HA_TOKEN; credentials are never written to the report')
    result = {}
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = build_opener(NoRedirect())
    for entity in dict.fromkeys(entities):
        request = Request(host + '/api/states/' + quote(entity, safe=''),
                          headers={'Authorization': 'Bearer ' + token}, method='GET')
        try:
            with opener.open(request, timeout=10) as response:
                payload = response.read(2 * 1024 * 1024 + 1)
                if len(payload) > 2 * 1024 * 1024:
                    raise ValueError('State response exceeded inspection bound')
                state = json.loads(payload)
                if not isinstance(state, dict):
                    raise ValueError('Expected a state object')
            result[entity] = summarize_observations(
                [(time.time(), state.get('state'), state.get('attributes'))], attribute=None)
        except Exception as error:
            # Do not echo request headers, response bodies or tokens on failure.
            result[entity] = {'error_type': type(error).__name__}
    return {'method': 'GET /api/states/<entity>', 'current_metadata_only': True, 'sources': result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, help='Optional dated YAML; default is embedded confirmed mappings plus runtime inventory')
    parser.add_argument('--include-default-sources', action='store_true')
    parser.add_argument('--measurement', action='append')
    parser.add_argument('--from', dest='start', type=aware_epoch)
    parser.add_argument('--to', dest='end', type=aware_epoch)
    parser.add_argument('--days', type=int, default=7, help='Default bounded sample window (1..31 days)')
    parser.add_argument('--sample-limit', type=int, default=200, help='Latest events per source, 1..10000')
    parser.add_argument('--ha', action='store_true', help='Also GET current entity metadata; never call services')
    parser.add_argument('--output', type=Path, help='NEW JSON file, mode 0600; otherwise stdout')
    args = parser.parse_args(argv)
    if args.output and (args.output.exists() or args.output.suffix.lower() != '.json'
                        or args.output.name in {'optimization_plan.json', 'future_predictions.json'}):
        parser.error('--output must be a NEW non-operational JSON path')
    if not 1 <= args.days <= 31 or not 1 <= args.sample_limit <= 10000:
        parser.error('days must be 1..31 and sample-limit 1..10000')
    if (args.start is None) != (args.end is None):
        parser.error('Supply both --from and --to, or neither')
    end = args.end if args.end is not None else math.floor(time.time() / 60) * 60
    start = args.start if args.start is not None else end - args.days * 86400
    if not all(math.isfinite(value) for value in (start, end)) or start >= end or end - start > 31 * 86400:
        parser.error('Inspection window must be finite, increasing, and at most 31 days')
    try:
        # A standalone copy may live in /tmp; configuration belongs to the
        # operator's project working directory, never the script's parent.
        load_dotenv(dotenv_path=Path.cwd() / '.env', override=False)
        sources = load_sources(args.sources) if args.sources else {**default_sources(), **_confirmed_sources()}
        if args.sources and args.include_default_sources:
            sources = {**default_sources(), **sources}
        ranges = mapped_ranges(sources, start, end, args.measurement)
        requests = [(name, source.entity, source.attribute) for name, source, _, _ in ranges]
        if not requests:
            raise ValueError('No selected source periods intersect the inspection window')
        source_ranges: dict[tuple[str, str, str | None], list[tuple[float, float]]] = {}
        for name, source, left, right in ranges:
            source_ranges.setdefault((name, source.entity, source.attribute), []).append((left, right))
        connection = get_db_connection()
        try:
            postgres = collect_postgres(connection, requests, start, end, sample_limit=args.sample_limit,
                                        source_ranges=source_ranges)
        finally:
            connection.close()
        report = {'format_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
                  'window': {'start': start, 'end': end}, 'postgres': postgres,
                  'safety': 'Read-only PostgreSQL and optional HA GETs only; no archive/model/control/service writes.',
                  'remaining_evidence': ['Confirm recorder namespace continuity; identity fields are evidence, not an automatic binding.',
                      'Confirm battery charging sign using independent operational evidence.',
                      'Confirm historical forecast feature matches inference hourly pv_estimate dimensions.',
                      'Choose freshness limits from source behavior/equipment evidence, not this bounded sample alone.']}
        if args.ha:
            report['home_assistant'] = collect_home_assistant([entity for _, entity, _ in requests])
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, 'w') as output:
                output.write(text)
            print(f'Evidence saved to {args.output}; review before sharing (contains entity names and bounded numeric summaries).')
        else:
            print(text, end='')
    except (OSError, ValueError, psycopg2.Error) as error:
        parser.exit(1, f'Evidence inspection stopped: {error}\n')


if __name__ == '__main__':
    main()
