from __future__ import annotations
"""Tests for the per-interval discharge budget.

The budget caps how much energy the battery may spend load-following during a
planning interval, scaled by price attractiveness so cheap intervals conserve
energy for higher-profit periods.
"""

import os
import tempfile
import shutil
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

import numpy as np

from optimize_plan import plan_battery_dispatch
from battery_planners.heuristic import _compute_discharge_budget
from utils.battery_utils import (
    apply_discharge_budget,
    accumulate_interval_discharge,
    _discharge_budget_state_path,
)


@contextmanager
def patched_env(overrides):
    original = {}
    missing = object()
    for key, value in overrides.items():
        original[key] = os.environ.get(key, missing)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key, old in original.items():
            if old is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


class DischargeBudgetFormulaTests(unittest.TestCase):
    """Unit tests for the planner-side budget formula."""

    def test_expensive_interval_gets_larger_budget(self):
        prices = np.array([0.03, 0.03, 0.08, 0.15, 0.20])
        cheap = _compute_discharge_budget(
            30.0, 5.0, 0.95, 10.0, 0.25, prices[0], prices)
        expensive = _compute_discharge_budget(
            30.0, 5.0, 0.95, 10.0, 0.25, prices[-1], prices)
        self.assertGreater(expensive, cheap)

    def test_budget_floor_keeps_some_discharge_in_cheapest_interval(self):
        prices = np.array([0.03, 0.10, 0.15, 0.20])
        budget = _compute_discharge_budget(
            30.0, 5.0, 0.95, 10.0, 0.25, prices[0], prices, min_factor=0.10)
        floor = 0.10 * 10.0 * 0.25
        # Even the cheapest interval keeps at least the floor budget, so the
        # battery still contributes some load-following (vs the old idle logic).
        self.assertGreaterEqual(budget, floor)
        # And it stays well below the full max interval discharge.
        self.assertLessEqual(budget, 10.0 * 0.25)

    def test_equal_prices_allow_full_budget(self):
        # When current price ties all future prices, discharging now is as good
        # as later, so the budget should be the full max interval discharge.
        prices = np.array([0.03, 0.03, 0.03])
        budget = _compute_discharge_budget(
            30.0, 5.0, 0.95, 10.0, 0.25, prices[0], prices, min_factor=0.10)
        self.assertAlmostEqual(budget, 10.0 * 0.25, places=3)

    def test_zero_when_battery_empty(self):
        prices = np.array([0.03, 0.15, 0.20])
        budget = _compute_discharge_budget(
            5.0, 5.0, 0.95, 10.0, 0.25, prices[0], prices)
        self.assertEqual(budget, 0.0)

    def test_budget_capped_by_max_discharge_power(self):
        prices = np.array([0.03, 0.15, 0.20])
        budget = _compute_discharge_budget(
            30.0, 5.0, 0.95, 10.0, 0.25, prices[-1], prices, min_factor=0.10)
        self.assertLessEqual(budget, 10.0 * 0.25)


class ApplyDischargeBudgetTests(unittest.TestCase):
    """Unit tests for the execution-side budget enforcement."""

    def test_passthrough_when_no_budget(self):
        self.assertEqual(apply_discharge_budget(-2.0, None, 0.0), (-2.0, ""))

    def test_passthrough_when_charging(self):
        self.assertEqual(apply_discharge_budget(3.0, 1.0, 0.0), (3.0, ""))

    def test_forces_zero_when_budget_exhausted(self):
        kw, msg = apply_discharge_budget(-2.0, 1.0, 1.2)
        self.assertEqual(kw, 0.0)
        self.assertIn('budget exhausted', msg)

    def test_caps_large_discharge_to_remaining_budget(self):
        # budget 1 kWh, none used, interval just started -> allowed avg 4 kW
        kw, msg = apply_discharge_budget(-10.0, 1.0, 0.0, elapsed_minutes=0)
        self.assertAlmostEqual(kw, -4.0, places=3)
        self.assertIn('budget cap', msg)

    def test_does_not_cap_small_discharge(self):
        # budget 1 kWh, none used, 10 min elapsed -> allowed avg 12 kW
        kw, msg = apply_discharge_budget(-1.0, 1.0, 0.0, elapsed_minutes=10)
        self.assertEqual(kw, -1.0)
        self.assertEqual(msg, "")

    def test_reduces_allowed_as_interval_progresses(self):
        early = apply_discharge_budget(-10.0, 1.0, 0.0, elapsed_minutes=0)[0]
        late = apply_discharge_budget(-10.0, 1.0, 0.0, elapsed_minutes=13)[0]
        self.assertGreater(early, late)  # early allows more discharge
        self.assertAlmostEqual(early, -4.0, places=3)
        self.assertAlmostEqual(late, -10.0, places=3)  # 1 kWh / (2 min / 60)

    def test_zero_budget_blocks_discharge(self):
        kw, _ = apply_discharge_budget(-5.0, 0.0, 0.0)
        self.assertEqual(kw, 0.0)


class AccumulateIntervalDischargeTests(unittest.TestCase):
    """Tests for the cumulative discharge tracker used across 20s ticks."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.state_file = os.path.join(self.tmp, 'state', 'discharge_budget_state.json')
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_accumulates_discharge_across_calls(self):
        base = datetime.now(timezone.utc)
        d1 = base
        d2 = base + timedelta(seconds=20)
        d3 = base + timedelta(seconds=40)
        # discharging at 2 kW for 40s => 2 * 40/3600 = 0.0222 kWh
        used1 = accumulate_interval_discharge(-2000.0, now=d1, state_file=self.state_file)
        used2 = accumulate_interval_discharge(-2000.0, now=d2, state_file=self.state_file)
        used3 = accumulate_interval_discharge(-2000.0, now=d3, state_file=self.state_file)
        self.assertAlmostEqual(used3, 2.0 * 40.0 / 3600.0, places=5)

    def test_ignores_charging(self):
        base = datetime.now(timezone.utc)
        used = accumulate_interval_discharge(3000.0, now=base, state_file=self.state_file)
        used = accumulate_interval_discharge(3000.0, now=base + timedelta(seconds=20),
                                             state_file=self.state_file)
        self.assertEqual(used, 0.0)

    def test_resets_at_interval_boundary(self):
        base = datetime.now(timezone.utc)
        # First interval: accumulate some discharge
        accumulate_interval_discharge(-2000.0, now=base, state_file=self.state_file)
        accumulate_interval_discharge(-2000.0, now=base + timedelta(seconds=20),
                                      state_file=self.state_file)
        used = accumulate_interval_discharge(-2000.0, now=base + timedelta(seconds=40),
                                             state_file=self.state_file)
        self.assertAlmostEqual(used, 2.0 * 40.0 / 3600.0, places=5)

        # Jump to a future interval (e.g. 16 minutes later) -> reset to zero
        later = base + timedelta(minutes=16)
        used_after_reset = accumulate_interval_discharge(-2000.0, now=later,
                                                         state_file=self.state_file)
        self.assertAlmostEqual(used_after_reset, 0.0, places=5)


class PlannerBudgetIntegrationTests(unittest.TestCase):
    """The planner emits a discharge_budget_kwh per interval and scales it by price."""

    def _plan(self, prices, env=None, predictions=None, solar=None):
        overrides = {
            "BATTERY_CAPACITY_KWH": "40",
            "BATTERY_MIN_SOC_PCT": "10",
            "BATTERY_MAX_SOC_PCT": "90",
            "BATTERY_INITIAL_SOC_PCT": "80",
            "BATTERY_MAX_CHARGE_KW": "10",
            "BATTERY_MAX_DISCHARGE_KW": "10",
            "BATTERY_CHARGE_EFFICIENCY": "1.0",
            "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
            "BATTERY_ALLOW_EXPORT": "false",
            "PLAN_INTERVAL_MINUTES": "60",
        }
        if env:
            overrides.update(env)
        if predictions is None:
            predictions = np.full(len(prices), 2.0)
        if solar is None:
            solar = np.zeros(len(prices))
        with patched_env(overrides):
            return plan_battery_dispatch(predictions, solar, prices, prices)

    def test_every_entry_has_budget(self):
        plan = self._plan([0.05, 0.10, 0.15, 0.20])
        self.assertEqual(len(plan), 4)
        for entry in plan:
            self.assertIn('discharge_budget_kwh', entry)

    def test_expensive_interval_budget_exceeds_cheap(self):
        prices = [0.05, 0.10, 0.15, 0.20]
        plan = self._plan(prices)
        self.assertGreater(
            plan[3]['discharge_budget_kwh'], plan[0]['discharge_budget_kwh'])

    def test_budget_never_undercuts_planned_discharge(self):
        # All intervals same high price -> full discharge_to_load planned.
        prices = [0.20, 0.20]
        plan = self._plan(prices)
        for entry in plan:
            self.assertGreaterEqual(
                entry['discharge_budget_kwh'],
                entry['discharge_to_load_kwh'] - 1e-9,
            )

    def test_min_factor_tunable_raises_budget(self):
        prices = [0.05, 0.10, 0.15, 0.20]
        base = self._plan(prices)
        high_floor = self._plan(prices, env={"BATTERY_FOLLOW_BUDGET_MIN_FACTOR": "0.9"})
        # The cheap first interval should get a larger budget with a high floor.
        self.assertGreater(
            high_floor[0]['discharge_budget_kwh'], base[0]['discharge_budget_kwh'])


if __name__ == '__main__':
    unittest.main()
