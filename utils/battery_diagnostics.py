"""One grep-friendly JSON diagnostic per battery-control cycle."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from utils.git_utils import get_model_version


BATTERY_DIAGNOSTIC_PREFIX = 'HEPO_BATTERY_CONTROL '


def new_battery_diagnostic(**fields: Any) -> dict[str, Any]:
    """Stable schema; unavailable measurements and unattempted commands stay null."""
    record: dict[str, Any] = dict.fromkeys((
        'plan_slot_id', 'plan_mtime', 'planned_action', 'effective_action', 'control_action',
        'decision_detail', 'deadband_w', 'was_native_for_hysteresis',
        'mode_requested', 'previous_requested_mode', 'previous_request_slot_id',
        'previous_request_action', 'previous_request_at', 'previous_command_status',
        'command_error', 'command_completed_at',
        'battery_power_w', 'grid_power_w', 'solar_power_w', 'solar_used_w',
        'battery_last_updated', 'grid_last_updated', 'soc_pct', 'phase_currents_a',
        'import_meter_kwh', 'export_meter_kwh', 'planned_power_w',
        'directional_planned_power_w', 'smoothed_planned_power_w',
        'native_estimate_w', 'native_clamped_estimate_w', 'guarded_target_w',
        'fixed_setpoint_w', 'bridge_set_power_w', 'discharge_cap_w',
        'discharge_reentry_cap_w', 'max_charge_w', 'max_discharge_w',
        'soc_floor_pct', 'main_fuse_a', 'discharge_budget_kwh', 'discharge_used_kwh',
        'discharge_remaining_kwh', 'budget_before_w', 'budget_after_w',
        'fuse_before_w', 'fuse_after_w', 'ramp_before_w', 'ramp_after_w',
    ))
    record.update(
        model_version=get_model_version(), override_mode='auto',
        command_status='not_attempted', native_selected_before_guards=False,
        discharge_budget_active=False, decision_reasons=[], inverter_mode_verified=False,
    )
    record.update(fields)
    return record


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def emit_battery_diagnostic(record: dict[str, Any]) -> None:
    payload = {'schema_version': 1, 'logged_at': datetime.now().astimezone().isoformat(), **record}
    print(BATTERY_DIAGNOSTIC_PREFIX + json.dumps(
        _json_safe(payload), sort_keys=True, separators=(',', ':'), allow_nan=False,
    ))


def parse_battery_diagnostic(line: str) -> dict[str, Any] | None:
    """Accept plain stdout and journal/syslog-prefixed diagnostic lines."""
    _, marker, payload = line.partition(BATTERY_DIAGNOSTIC_PREFIX)
    if not marker:
        return None
    try:
        record = json.loads(payload)
        return record if isinstance(record, dict) else None
    except ValueError:
        return None
