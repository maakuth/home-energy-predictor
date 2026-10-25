"""Best-effort direction guards for inverter-native self consumption."""

from __future__ import annotations

import json
import math
import os
from typing import Any


NATIVE_FOLLOW_ACTIONS = {'follow', 'charge_solar', 'discharge_load'}
# Match the existing software follow controller's 200 W noise deadband.
DIRECTION_DEADBAND_KW = 0.2


def choose_native_following(
    action: str,
    planned_kw: float,
    battery_w: float | None,
    grid_w: float | None,
    *,
    was_native: bool = False,
    max_charge_kw: float = 10.0,
    max_discharge_kw: float = 10.0,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[bool, float, str]:
    """Return mode eligibility and a fixed fallback (positive kW = charging).

    Native power is estimated as battery minus grid. For restricted actions,
    enter only on a clear allowed-direction signal, but remain native through
    small zero crossings. Measured wrong-direction power also forces fixed
    control, even if the estimated next response has the correct direction.
    Power caps and the external SoC/budget/fuse guards override this deadband.
    """
    reasons: list[str] = []
    if diagnostics is not None:
        diagnostics.update(decision_reasons=reasons, deadband_w=DIRECTION_DEADBAND_KW * 1000.0)

    def result(native: bool, target: float, message: str) -> tuple[bool, float, str]:
        if diagnostics is not None:
            diagnostics['native_selected_before_guards'] = native
        return native, target, message

    if action not in NATIVE_FOLLOW_ACTIONS:
        reasons.append('action_not_native')
        return result(False, planned_kw, '')
    if battery_w is None or grid_w is None or not all(map(math.isfinite, (battery_w, grid_w))):
        if battery_w is None or not math.isfinite(battery_w):
            reasons.append('battery_telemetry_missing')
        if grid_w is None or not math.isfinite(grid_w):
            reasons.append('grid_telemetry_missing')
        return result(False, 0.0, 'native blocked: battery/grid telemetry unavailable')

    battery_kw = battery_w / 1000.0
    target_kw = max(-max_discharge_kw, min(max_charge_kw, (battery_w - grid_w) / 1000.0))
    if diagnostics is not None:
        diagnostics.update(native_estimate_w=battery_w - grid_w, native_clamped_estimate_w=target_kw * 1000.0)
    if action == 'follow':
        return result(True, target_kw, 'native self_consumption')

    deadband = DIRECTION_DEADBAND_KW
    if action == 'charge_solar':
        wrong_direction = battery_kw < -deadband or target_kw < -deadband
        allowed = target_kw >= -deadband if was_native else target_kw > deadband
        fixed_kw = max(0.0, target_kw)
        if battery_kw < -deadband:
            reasons.append('observed_wrong_direction')
        if target_kw < -deadband:
            reasons.append('expected_wrong_direction')
        elif not allowed:
            reasons.append('direction_hysteresis')
    else:
        cap_kw = min(max_discharge_kw, max(0.0, -planned_kw))
        wrong_direction = battery_kw > deadband or target_kw > deadband
        # A lower re-entry threshold keeps measurements near the power cap from
        # repeatedly switching between fixed and native mode.
        entry_cap_kw = cap_kw if was_native else max(0.0, cap_kw - deadband)
        if diagnostics is not None:
            diagnostics.update(discharge_cap_w=cap_kw * 1000.0, discharge_reentry_cap_w=entry_cap_kw * 1000.0)
        allowed = (
            cap_kw > 0
            and (target_kw <= deadband if was_native else target_kw < -deadband)
            and target_kw >= -entry_cap_kw
            and battery_kw >= -cap_kw
        )
        fixed_kw = max(-cap_kw, min(0.0, target_kw))
        if battery_kw > deadband:
            reasons.append('observed_wrong_direction')
        if target_kw > deadband:
            reasons.append('expected_wrong_direction')
        elif not was_native and target_kw >= -deadband:
            reasons.append('direction_hysteresis')
        if cap_kw <= 0:
            reasons.append('zero_discharge_cap')
        if target_kw < -cap_kw:
            reasons.append('expected_discharge_cap')
        elif target_kw < -entry_cap_kw:
            reasons.append('power_cap_hysteresis')
        if battery_kw < -cap_kw:
            reasons.append('observed_discharge_cap')

    if wrong_direction or not allowed:
        return result(False, fixed_kw, 'native blocked: direction/power guard')
    return result(True, target_kw, 'native self_consumption')


def _state_path() -> str:
    return os.getenv('HEPO_NATIVE_FOLLOW_STATE_FILE', 'state/native_follow_control.json')


def read_native_follow_state() -> dict[str, Any]:
    """Read the controller cache, without claiming actual inverter state."""
    try:
        with open(_state_path()) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def load_native_follow_state(slot_id: str, action: str, override_mode: str) -> bool:
    """Load the last successful request, not an assertion of inverter state."""
    state = read_native_follow_state()
    return (
        state.get('slot_id') == slot_id
        and state.get('action') == action
        and state.get('override_mode') == override_mode
        and state.get('native') is True
    )


def save_native_follow_state(
    slot_id: str, action: str, override_mode: str, native: bool,
    *, command_status: str | None = None, requested_at: str | None = None,
) -> None:
    """Persist hysteresis only after the caller successfully requests a mode."""
    path = _state_path()
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(f'{path}.tmp', 'w') as f:
            json.dump({'slot_id': slot_id, 'action': action, 'override_mode': override_mode,
                       'native': native, 'command_status': command_status,
                       'requested_at': requested_at}, f)
        os.replace(f'{path}.tmp', path)
    except OSError as exc:
        print(f'Could not persist native following state: {exc}')
