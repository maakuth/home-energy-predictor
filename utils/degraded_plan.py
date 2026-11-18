"""Conservative local plan when archive prediction or optimization cannot run.

No telemetry, forecast or price is inferred here. The normal HA publisher is not
compatible with unknown values and must not publish this artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from utils.archive_history import ArchiveHistory


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=f'.{path.name}.',
                                         delete=False) as stream:
            name = stream.name
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def write_degraded_plan(history: ArchiveHistory, failed_step: str, reason: str) -> None:
    interval = int(os.getenv('PLAN_INTERVAL_MINUTES', '15'))
    if interval <= 0 or 1440 % interval:
        raise ValueError('Invalid PLAN_INTERVAL_MINUTES for degraded plan')
    slot = int(history.cutoff // (interval * 60)) * interval * 60
    timestamp = datetime.fromtimestamp(slot, timezone.utc).isoformat()
    entry = {'timestamp': timestamp, 'plan_quality': 'degraded',
             'degraded_reason': reason, 'battery_action': 'idle', 'battery_power_kw': 0.0,
             # Null avoids SG Ready interpreting an absent cost as a cheap zero.
             'effective_cost': None, 'gshp_intent': 'STOP',
             'resistive_heater_intent': 'OFF', 'bulk_heater_intent': 'OFF',
             'leaf_intent': 'OFF'}
    metadata = {'plan_quality': 'degraded', 'failed_step': failed_step, 'reason': reason,
                'history_cutoff': history.cutoff, 'archive_path': str(Path(history.path).resolve()),
                'legacy_policy': history.legacy_policy, 'price_source': None,
                 'published_to_ha': False,
                 'ha_sensor_status': 'stale_until_invalidation_confirmed',
                 'ha_invalidation': 'attempted_by_pipeline_publisher_not_verified_by_sidecar'}
    path = Path(os.getenv('TEST_PLAN_FILE', 'state/optimization_plan.json'))
    # Metadata first: the plan itself carries its label, so unlocked readers
    # never need to join two independently replaced files to recognize it.
    _atomic_json(path.with_name('optimization_plan_metadata.json'), metadata)
    _atomic_json(path, [entry])
