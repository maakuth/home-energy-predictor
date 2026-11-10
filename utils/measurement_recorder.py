"""Read bounded HA recorder events; caller supplies the database connection."""
from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Any

from utils.measurement_archive import Event


def recorder_predecessor(connection: Any, entity: str, start: float) -> Event | None:
    """Latest actual state before a successful range's left boundary."""
    if not entity or not math.isfinite(start):
        raise ValueError('Invalid predecessor query')
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT metadata_id FROM states_meta WHERE entity_id = %s', (entity,))
        metadata = cursor.fetchone()
        if metadata is None:
            raise ValueError(f'Recorder entity not found: {entity}')
        cursor.execute('''SELECT s.state_id, s.last_updated_ts, s.state, a.shared_attrs
            FROM states s LEFT JOIN state_attributes a ON s.attributes_id = a.attributes_id
            WHERE s.metadata_id = %s AND s.last_updated_ts < %s
            ORDER BY s.last_updated_ts DESC, s.state_id DESC LIMIT 1''', (metadata[0], start))
        row = cursor.fetchone()
        return None if row is None else (int(row[0]), float(row[1]), row[2], row[3])
    finally:
        cursor.close()


def recorder_events(connection: Any, entity: str, start: float, end: float,
                    batch_size: int = 2000) -> Iterator[Event]:
    if (not entity or not math.isfinite(start) or not math.isfinite(end)
            or start >= end or not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError('Invalid recorder query')
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT metadata_id FROM states_meta WHERE entity_id = %s', (entity,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f'Recorder entity not found: {entity}')
        # A normal psycopg cursor can buffer the whole result even when callers
        # use fetchmany. Bound each SQL result and advance by the stable tuple,
        # including recorder ID to preserve events sharing a timestamp.
        after: tuple[float, int] | None = None
        while True:
            predicate = '' if after is None else ' AND (s.last_updated_ts, s.state_id) > (%s, %s)'
            parameters = (row[0], start, end, *(after or ()), batch_size)
            cursor.execute('''SELECT s.state_id, s.last_updated_ts, s.state, a.shared_attrs
                FROM states s LEFT JOIN state_attributes a ON s.attributes_id = a.attributes_id
                WHERE s.metadata_id = %s AND s.last_updated_ts >= %s AND s.last_updated_ts < %s'''
                + predicate + ' ORDER BY s.last_updated_ts, s.state_id LIMIT %s', parameters)
            batch = cursor.fetchall()
            if not batch:
                break
            for recorder_id, timestamp, state, attrs in batch:
                yield (int(recorder_id), float(timestamp), state, attrs)
            after = (float(batch[-1][1]), int(batch[-1][0]))
    finally:
        cursor.close()
