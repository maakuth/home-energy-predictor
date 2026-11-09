"""Validate half-open, timezone-aware historical source mappings."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MeasurementSource:
    entity: str
    start: datetime | None
    end: datetime | None
    attribute: str | None = None


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('Source boundary must be an ISO timestamp')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError(f'Invalid source boundary: {value}') from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Source boundaries must have a timezone')
    return result.astimezone(timezone.utc)


def parse_sources(config: dict[str, Any]) -> dict[str, list[MeasurementSource]]:
    if not isinstance(config, dict) or not config:
        raise ValueError('Expected a nonempty measurement mapping')
    result: dict[str, list[MeasurementSource]] = {}
    for measurement, entries in config.items():
        if not isinstance(measurement, str) or not measurement or not isinstance(entries, list) or not entries:
            raise ValueError('Measurements require a nonempty source list')
        parsed: list[MeasurementSource] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {'entity', 'attribute', 'from', 'to'}:
                raise ValueError('Invalid source fields')
            entity = entry.get('entity')
            attr = entry.get('attribute')
            if not isinstance(entity, str) or not entity.strip() or (attr is not None and (not isinstance(attr, str) or not attr)):
                raise ValueError('Invalid entity or attribute')
            start, end = _timestamp(entry.get('from')), _timestamp(entry.get('to'))
            if start is not None and end is not None and start >= end:
                raise ValueError('Source start must precede end')
            parsed.append(MeasurementSource(entity, start, end, attr))
        parsed.sort(key=lambda s: s.start or datetime.min.replace(tzinfo=timezone.utc))
        for previous, current in zip(parsed, parsed[1:]):
            if previous.end is None or current.start is None or previous.end > current.start:
                raise ValueError(f'Overlapping sources for {measurement}')
        result[measurement] = parsed
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    """Reject silent YAML key overwrites as well as unsafe YAML types."""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError('Expected a YAML mapping')
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError(f'Invalid or duplicate YAML key: {key}')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_sources(path: str | Path) -> dict[str, list[MeasurementSource]]:
    try:
        with Path(path).open(encoding='utf-8') as stream:
            config = yaml.load(stream, Loader=_UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid source YAML: {error}') from error
    return parse_sources(config)


def mapped_ranges(config: dict[str, list[MeasurementSource]], start: float, end: float,
                  measurements: list[str] | None = None) -> list[tuple[str, MeasurementSource, float, float]]:
    """Intersect requested bounds with source periods; never fill mapping gaps."""
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError('Invalid ingestion interval')
    selected = list(config) if measurements is None else list(dict.fromkeys(measurements))
    if not selected or any(name not in config for name in selected):
        raise ValueError('Unknown or empty measurement selection')
    result = []
    for name in selected:
        for source in config[name]:
            left = max(start, source.start.timestamp()) if source.start is not None else start
            right = min(end, source.end.timestamp()) if source.end is not None else end
            if left < right:
                result.append((name, source, left, right))
    return result


def source_at(sources: list[MeasurementSource], timestamp: datetime) -> MeasurementSource | None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError('Timestamp must have a timezone')
    return next((source for source in sources
                 if (source.start is None or source.start <= timestamp)
                 and (source.end is None or timestamp < source.end)), None)
