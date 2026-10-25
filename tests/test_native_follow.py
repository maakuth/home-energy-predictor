import pytest

from utils.native_follow import choose_native_following, load_native_follow_state, save_native_follow_state


@pytest.mark.parametrize('action,planned,battery,grid,native,target', [
    ('charge_solar', 3, 1000, -1000, True, 2),
    ('charge_solar', 3, 0, 1000, False, 0),
    ('charge_solar', 3, -1000, -3000, False, 2),
    ('discharge_load', -3, -1000, 1000, True, -2),
    ('discharge_load', -3, 0, -1000, False, 0),
    ('discharge_load', -3, 1000, 3000, False, -2),
    ('discharge_load', -1, 0, 2000, False, -1),
    ('discharge_load', -1, -2000, -1500, False, -.5),
    ('discharge_load', 0, 0, 2000, False, 0),
    ('charge_grid', 3, 0, -1000, False, 3),
])
def test_direction_and_power_checks(action, planned, battery, grid, native, target):
    enabled, kw, _ = choose_native_following(action, planned, battery, grid)
    assert enabled is native
    assert kw == pytest.approx(target)


@pytest.mark.parametrize('action,planned,sign', [('charge_solar', 3, 1), ('discharge_load', -3, -1)])
def test_direction_hysteresis(action, planned, sign):
    # Enter on a clear allowed-direction signal, stay active around zero,
    # exit on clear reversal, and do not re-enter on small noisy readings.
    active = False
    decisions = []
    for target in (1, .1, -.1, -.3, -.1, .1, 1):
        active, _, _ = choose_native_following(
            action, planned, 0, -target * sign * 1000, was_native=active,
        )
        decisions.append(active)
    assert decisions == [True, True, True, False, False, False, True]


def test_discharge_cap_hysteresis():
    active = False
    decisions = []
    for target in (-1, -2.1, -1.9, -1.7):
        active, kw, _ = choose_native_following(
            'discharge_load', -2, 0, -target * 1000, was_native=active,
        )
        assert kw >= -2
        decisions.append(active)
    assert decisions == [True, False, False, True]


def test_native_estimate_respects_physical_limits():
    assert choose_native_following('follow', 0, 0, 20000, max_discharge_kw=4)[1] == -4
    assert choose_native_following('follow', 0, 0, -20000, max_charge_kw=5)[1] == 5


@pytest.mark.parametrize('battery,grid', [(None, 1000), (0, None), (float('nan'), 1000), (0, float('inf'))])
def test_missing_or_nonfinite_telemetry_fails_closed(battery, grid):
    active, target, _ = choose_native_following('charge_solar', 3, battery, grid, was_native=True)
    assert not active
    assert target == 0


def test_native_follow_state_is_scoped_and_handles_bad_files(tmp_path, monkeypatch):
    path = tmp_path / 'state' / 'native.json'
    monkeypatch.setenv('HEPO_NATIVE_FOLLOW_STATE_FILE', str(path))
    assert not load_native_follow_state('slot', 'charge_solar', 'auto')
    save_native_follow_state('slot', 'charge_solar', 'auto', True)
    assert load_native_follow_state('slot', 'charge_solar', 'auto')
    assert not load_native_follow_state('next_slot', 'charge_solar', 'auto')
    assert not load_native_follow_state('slot', 'discharge_load', 'auto')
    assert not load_native_follow_state('slot', 'charge_solar', 'charge_solar')
    save_native_follow_state('slot', 'charge_solar', 'auto', False)
    assert not load_native_follow_state('slot', 'charge_solar', 'auto')
    for text in ('{broken', 'null', '[]'):
        path.write_text(text)
        assert not load_native_follow_state('slot', 'charge_solar', 'auto')
