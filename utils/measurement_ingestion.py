"""Bounded incremental source updates; no recorder connections at import time."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from utils.measurement_archive import Event, MeasurementArchive


def update_source(archive: MeasurementArchive, entity: str, attribute: str | None,
                  start: float, end: float, overlap: float,
                  fetch: Callable[[float, float], Iterable[Event]], *,
                  predecessor: Callable[[float], Event | None] | None = None,
                  chunk_seconds: float = 86400, retries: int = 0,
                  rollback: Callable[[], object] | None = None) -> None:
    """Fill uncovered intervals and refresh the latest contiguous overlap.

    Every fetch range commits separately. Failures roll back their observations
    and coverage; completed ranges remain available for retry.
    """
    if not all(math.isfinite(x) for x in (start, end, overlap)) or start >= end or overlap < 0:
        raise ValueError('Invalid update interval or overlap')
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0 or not isinstance(retries, int) or retries < 0:
        raise ValueError('Invalid chunk size or retries')
    coverage = archive.coverage(entity, attribute)
    gaps: list[tuple[float, float]] = []
    cursor = start
    for left, right in coverage:
        if right <= start or left >= end:
            continue
        if left > cursor:
            gaps.append((cursor, min(left, end)))
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append((cursor, end))
    # Refresh only the trailing covered segment; do not bridge unqueried gaps.
    for left, right in reversed(coverage):
        if left < end and right >= end:
            refresh = max(start, left, end - overlap)
            if refresh < end:
                gaps.append((refresh, end))
            break
        if start < right < end:
            refresh = max(start, left, right - overlap)
            if gaps and gaps[-1][0] == right:
                gaps[-1] = (refresh, gaps[-1][1])
            break
    for left, right in gaps:
        while left < right:
            boundary = min(right, left + chunk_seconds)
            for attempt in range(retries + 1):
                count = 0
                def events() -> Iterable[Event]:
                    nonlocal count
                    for event in fetch(left, boundary):
                        count += 1
                        yield event
                try:
                    previous = predecessor(left) if predecessor else None
                    archive.ingest(entity, attribute, left, boundary, events(), predecessor=previous)
                except Exception as error:
                    if rollback is not None:
                        rollback()
                    with archive.db:
                        archive.db.execute('''INSERT INTO ingestion_runs
                            (entity, attribute, start, end, attempt, status, error, event_count)
                            VALUES (?, ?, ?, ?, ?, 'failed', ?, ?)''',
                            (entity, attribute or '', left, boundary, attempt + 1, str(error), count))
                    if attempt == retries:
                        raise
                else:
                    with archive.db:
                        archive.db.execute('''INSERT INTO ingestion_runs
                            (entity, attribute, start, end, attempt, status, event_count)
                            VALUES (?, ?, ?, ?, ?, 'success', ?)''',
                            (entity, attribute or '', left, boundary, attempt + 1, count))
                    break
            left = boundary
