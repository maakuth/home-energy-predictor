"""Bounded incremental source updates; no recorder connections at import time."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from utils.measurement_archive import Event, MeasurementArchive


def update_source(archive: MeasurementArchive, entity: str, attribute: str | None,
                  start: float, end: float, overlap: float,
                  fetch: Callable[[float, float], Iterable[Event]]) -> None:
    """Fill uncovered intervals and refresh the latest contiguous overlap.

    Every fetch range commits separately. Failures roll back their observations
    and coverage; completed ranges remain available for retry.
    """
    if not all(math.isfinite(x) for x in (start, end, overlap)) or start >= end or overlap < 0:
        raise ValueError('Invalid update interval or overlap')
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
        archive.ingest(entity, attribute, left, right, fetch(left, right))
