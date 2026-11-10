"""Compose archive updates with an explicitly supplied recorder connection."""
from __future__ import annotations

from typing import Any

from utils.measurement_archive import MeasurementArchive
from utils.measurement_ingestion import update_source
from utils.measurement_recorder import recorder_events, recorder_predecessor


def update_recorder_sources(archive: MeasurementArchive, connection: Any,
                            sources: list[tuple[str, str | None]],
                            start: float, end: float, overlap: float, *,
                            chunk_seconds: float = 86400, retries: int = 0,
                            fetch_predecessor: bool = False) -> dict[str, str]:
    """Update each source independently; report failures rather than masking them."""
    failures: dict[str, str] = {}
    for entity, attribute in sources:
        key = entity if attribute is None else f'{entity}#{attribute}'
        try:
            update_source(archive, entity, attribute, start, end, overlap,
                          lambda left, right: recorder_events(connection, entity, left, right),
                          predecessor=(lambda left: recorder_predecessor(connection, entity, left))
                          if fetch_predecessor else None,
                          chunk_seconds=chunk_seconds, retries=retries, rollback=connection.rollback)
        except Exception as error:
            # PostgreSQL marks failed transactions aborted until rollback.
            connection.rollback()
            failures[key] = str(error)
    return failures
