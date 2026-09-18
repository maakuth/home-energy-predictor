from __future__ import annotations

from datetime import datetime
from typing import Any


def slot_start(now: datetime, interval_minutes: int) -> datetime:
    """Return the epoch-aligned start of the interval containing ``now``."""
    if now.tzinfo is None:
        raise ValueError('now must be timezone-aware')
    if interval_minutes <= 0:
        raise ValueError('interval_minutes must be positive')
    interval_seconds = interval_minutes * 60
    start_epoch = int(now.timestamp()) // interval_seconds * interval_seconds
    return datetime.fromtimestamp(start_epoch, tz=now.tzinfo)


def current_plan_entry(
    plan: list[dict[str, Any]],
    now: datetime,
    interval_minutes: int,
) -> dict[str, Any] | None:
    """Return the exact current entry; never substitute another interval."""
    current_start = slot_start(now, interval_minutes)
    for entry in plan:
        try:
            timestamp = datetime.fromisoformat(str(entry['timestamp']))
            if timestamp.tzinfo is None:
                continue
            if timestamp.astimezone(current_start.tzinfo) == current_start:
                return entry
        except (KeyError, TypeError, ValueError):
            continue
    return None
