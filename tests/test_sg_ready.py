from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from utils.sg_ready import (
    derive_sg_ready_bits,
    derive_sg_ready_mode,
    update_virtual_accumulator_temp,
    get_virtual_accumulator_temp,
)


def test_start_cheap_low_price():
    current = {"gshp_intent": "START", "effective_cost": 0.05, "planned_gshp_kw": 3.4}
    plan = [current] + [{"effective_cost": 0.20} for _ in range(5)]
    assert derive_sg_ready_bits(current=current, plan=plan) == (0, 1)
    assert derive_sg_ready_mode(current=current, plan=plan) == "low_price"


def test_start_expensive_overcapacity():
    current = {"gshp_intent": "START", "effective_cost": 0.20, "planned_gshp_kw": 3.4}
    plan = [current] + [{"effective_cost": 0.05} for _ in range(5)]
    assert derive_sg_ready_bits(current=current, plan=plan) == (1, 1)


def test_stop_expensive_blocking():
    current = {"gshp_intent": "STOP", "effective_cost": 0.20}
    plan = [current] + [{"effective_cost": 0.05} for _ in range(5)]
    assert derive_sg_ready_bits(current=current, plan=plan) == (1, 0)


def test_stop_cheap_normal():
    current = {"gshp_intent": "STOP", "effective_cost": 0.05}
    plan = [current] + [{"effective_cost": 0.20} for _ in range(5)]
    assert derive_sg_ready_bits(current=current, plan=plan) == (0, 0)


def test_boost_bulk_overcapacity():
    current = {"gshp_intent": "START", "effective_cost": 0.05, "bulk_heater_intent": "ON", "planned_gshp_kw": 3.4}
    plan = [current] + [{"effective_cost": 0.20} for _ in range(5)]
    assert derive_sg_ready_bits(current=current, plan=plan) == (1, 1)


def test_virtual_accumulator_advances():
    state_file = os.path.join(os.environ["TEST_TMP_DIR"], "va.json")
    os.environ["GSHP_VIRTUAL_ACCUMULATOR_STATE_FILE"] = state_file
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    temp0 = update_virtual_accumulator_temp(4.0, 0.0, now=t0, path=state_file, cop=3.5, reservoir_liters=500.0)
    temp1 = update_virtual_accumulator_temp(0.0, 0.0, now=t1, path=state_file, cop=3.5, reservoir_liters=500.0)
    assert temp1 != temp0
    assert get_virtual_accumulator_temp(path=state_file) == temp1