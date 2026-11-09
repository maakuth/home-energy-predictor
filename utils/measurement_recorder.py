"""Read bounded HA recorder events; caller supplies the database connection."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from utils.measurement_archive import Event


def recorder_events(connection: Any, entity: str, start: float, end: float,
                    batch_size: int = 2000) -> Iterator[Event]:
    if not entity or start >= end or batch_size <= 0:
        raise ValueError('Invalid recorder query')
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT metadata_id FROM states_meta WHERE entity_id = %s', (entity,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f'Recorder entity not found: {entity}')
        cursor.execute('''SELECT s.state_id, s.last_updated_ts, s.state, a.shared_attrs
            FROM states s LEFT JOIN state_attributes a ON s.attributes_id = a.attributes_id
            WHERE s.metadata_id = %s AND s.last_updated_ts >= %s AND s.last_updated_ts < %s
            ORDER BY s.last_updated_ts, s.state_id''', (row[0], start, end))
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            for recorder_id, timestamp, state, attrs in batch:
                yield (int(recorder_id), float(timestamp), state, attrs)
    finally:
        cursor.close()
