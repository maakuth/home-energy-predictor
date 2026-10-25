"""Best-effort direction guards for inverter-native self consumption."""

from __future__ import annotations

import json
import math
import os


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
) -> tuple[bool, float, str]:
    """Return mode eligibility and a fixed fallback (positive kW = charging).

    Native power is estimated as battery minus grid. For restricted actions,
    enter only on a clear allowed-direction signal, but remain native through
    small zero crossings. Measured wrong-direction power also forces fixed
    control, even if the estimated next response has the correct direction.
    Power caps and the external SoC/budget/fuse guards override this deadband.
    """
    if action not in NATIVE_FOLLOW_ACTIONS:
        return False, planned_kw, ''
    if battery_w is None or grid_w is None or not all(map(math.isfinite, (battery_w, grid_w))):
        return False, 0.0, 'native blocked: battery/grid telemetry unavailable'

    battery_kw = battery_w / 1000.0
    target_kw = max(-max_discharge_kw, min(max_charge_kw, (battery_w - grid_w) / 1000.0))
    if action == 'follow':
        return True, target_kw, 'native self_consumption'

    deadband = DIRECTION_DEADBAND_KW
    if action == 'charge_solar':
        wrong_direction = battery_kw < -deadband or target_kw < -deadband
        allowed = target_kw >= -deadband if was_native else target_kw > deadband
        fixed_kw = max(0.0, target_kw)
    else:
        cap_kw = min(max_discharge_kw, max(0.0, -planned_kw))
        wrong_direction = battery_kw > deadband or target_kw > deadband
        # A lower re-entry threshold keeps measurements near the power cap from
        # repeatedly switching between fixed and native mode.
        entry_cap_kw = cap_kw if was_native else max(0.0, cap_kw - deadband)
        allowed = (
            cap_kw > 0
            and (target_kw <= deadband if was_native else target_kw < -deadband)
            and target_kw >= -entry_cap_kw
            and battery_kw >= -cap_kw
        )
        fixed_kw = max(-cap_kw, min(0.0, target_kw))

    if wrong_direction or not allowed:
        return False, fixed_kw, 'native blocked: direction/power guard'
    return True, target_kw, 'native self_consumption'


def _state_path() -> str:
    return os.getenv('HEPO_NATIVE_FOLLOW_STATE_FILE', 'state/native_follow_control.json')


def load_native_follow_state(slot_id: str, action: str, override_mode: str) -> bool:
    """Load the last successful request, not an assertion of inverter state."""
    try:
        with open(_state_path()) as f:
            state = json.load(f)
        return (
            isinstance(state, dict)
            and state.get('slot_id') == slot_id
            and state.get('action') == action
            and state.get('override_mode') == override_mode
            and state.get('native') is True
        )
    except (OSError, ValueError):
        return False


def save_native_follow_state(slot_id: str, action: str, override_mode: str, native: bool) -> None:
    """Persist hysteresis only after the caller successfully requests a mode."""
    path = _state_path()
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(f'{path}.tmp', 'w') as f:
            json.dump({'slot_id': slot_id, 'action': action, 'override_mode': override_mode,
                       'native': native}, f)
        os.replace(f'{path}.tmp', path)
    except OSError as exc:
        print(f'Could not persist native following state: {exc}')
