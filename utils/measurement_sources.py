"""Validate half-open, timezone-aware historical source mappings."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


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


def source_at(sources: list[MeasurementSource], timestamp: datetime) -> MeasurementSource | None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError('Timestamp must have a timezone')
    return next((source for source in sources
                 if (source.start is None or source.start <= timestamp)
                 and (source.end is None or timestamp < source.end)), None)
