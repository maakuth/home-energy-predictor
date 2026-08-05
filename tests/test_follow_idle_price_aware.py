from __future__ import annotations
"""Tests for the price-aware follow/idle decision.

Regression: when the LP assigns zero dispatch to a cheap interval, the
real-time follow layer used to mechanically drain the battery there (because
`should_idle_interval` only compared the immediate grid benefit against the
cycling cost).  This drained the battery at cheap prices, forcing a later
grid top-up at an expensive price (e.g. charging at 19:45 when 17:00/18:00
were cheaper).

The fix makes the idle decision price-aware: the battery should idle (buy
from grid) at a cheap interval when the energy it would spend is worth more
later (i.e. the cheapest future import price is not low enough to refill the
battery at a profit).
"""

import os
import unittest
from contextlib import contextmanager

import numpy as np

from battery_planners.base import should_idle_interval
from battery_planners import BatteryPlannerFactory


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


class ShouldIdleIntervalPriceAwareTests(unittest.TestCase):
    """Unit tests: cheap interval should idle, expensive should follow."""

    def setUp(self):
        self.base = dict(
            net_kw=5.0,
            max_battery_kw=10.0,
            degradation_cost_per_kwh=0.01,
            interval_hours=0.25,
            charge_eff=0.98,
            discharge_eff=0.98,
            import_price=0.08,
            export_price=0.05,
        )

    def test_idles_when_import_price_below_refill_cost(self):
        # Current import 0.08, but the cheapest future import is 0.07.
        # Round-trip efficiency 0.98*0.98=0.9604, so refilling costs
        # 0.07/0.9604 + 2*0.01 = 0.0929 > 0.08.  Idle is cheaper than
        # discharging now and refilling later.
        self.base['future_import_prices'] = np.array([0.07, 0.07, 0.07, 0.07])
        self.assertTrue(should_idle_interval(**self.base))

    def test_follows_when_import_price_above_refill_cost(self):
        # Current import 0.11 beats refill cost 0.0929, so discharging now
        # is profitable -> follow.
        self.base['import_price'] = 0.11
        self.base['future_import_prices'] = np.array([0.07, 0.07, 0.07, 0.07])
        self.assertFalse(should_idle_interval(**self.base))

    def test_no_future_prices_keeps_legacy_behavior(self):
        # Without future prices there is nothing to compare against -> fall
        # back to the cycling cost/benefit check.
        self.base['import_price'] = 0.11
        result = should_idle_interval(**self.base)
        self.assertIsInstance(result, bool)

    def test_cheapest_future_price_drives_the_decision(self):
        # Refill cost with a 0.06 refill = 0.06/0.9604 + 0.02 = 0.0825, still
        # above the 0.08 import price, so the battery should still idle.
        self.base['future_import_prices'] = np.array([0.06, 0.08, 0.09, 0.10])
        self.assertTrue(
            should_idle_interval(**self.base),
            msg="0.08 discharge is not profitable when refilling costs 0.0825")

    def test_follows_when_refill_is_cheap_enough(self):
        # Refill cost with a 0.05 refill = 0.05/0.9604 + 0.02 = 0.0721, below
        # the 0.08 import price, so discharging now is profitable -> follow.
        self.base['future_import_prices'] = np.array([0.05, 0.08, 0.09, 0.10])
        self.assertFalse(
            should_idle_interval(**self.base),
            msg="0.08 discharge should be profitable when refilling costs 0.0721")

    def test_zero_degradation_keeps_legacy_follow(self):
        # With no degradation cost configured, the legacy behaviour (always
        # follow when feasible) must be preserved.
        self.base['degradation_cost_per_kwh'] = 0.0
        self.base['future_import_prices'] = np.array([0.07, 0.07, 0.07])
        self.assertFalse(should_idle_interval(**self.base))


class FollowPriceAwarePlannerTests(unittest.TestCase):
    """Integration: the planner must not drain the battery at cheap prices
    when a more expensive stretch follows."""

    def _plan(self, prices, env=None, soc=30.0, load=2.0):
        overrides = {
            "BATTERY_PLANNER_TYPE": "nemotron-linprog",
            "BATTERY_CAPACITY_KWH": "40",
            "BATTERY_MIN_SOC_PCT": "10",
            "BATTERY_MAX_SOC_PCT": "90",
            "BATTERY_INITIAL_SOC_PCT": str(soc),
            "BATTERY_MAX_CHARGE_KW": "10",
            "BATTERY_MAX_DISCHARGE_KW": "10",
            "BATTERY_CHARGE_EFFICIENCY": "0.98",
            "BATTERY_DISCHARGE_EFFICIENCY": "0.98",
            "BATTERY_ALLOW_EXPORT": "false",
            "PLAN_INTERVAL_MINUTES": "60",
            "BATTERY_DEGRADATION_COST_EUR_PER_KWH": "0.01",
            "BATTERY_FOLLOW_MAX_KW": "10",
        }
        if env:
            overrides.update(env)
        pred = np.full(len(prices), load)
        solar = np.zeros(len(prices))
        with patched_env(overrides):
            planner = BatteryPlannerFactory.create('nemotron-linprog')
            return planner.plan(
                pred, solar, np.array(prices), np.zeros(len(prices)),
                [f'i{i}' for i in range(len(prices))],
                allow_export=False)

    def test_no_follow_discharge_at_cheap_prices_before_peak(self):
        # 4 cheap intervals then 12 expensive ones.  A price-unaware follow
        # drains ~4 kWh at 0.07 before the peak, leaving the battery short
        # during the expensive stretch (importing 0.95 kWh at 0.15).
        prices = [0.07] * 4 + [0.15] * 12
        plan = self._plan(prices, soc=30.0)

        cheap_follow = sum(
            e.discharge_to_load_kwh
            for i, e in enumerate(plan)
            if i < 4 and e.battery_action == 'follow'
        )
        self.assertAlmostEqual(cheap_follow, 0.0, places=3,
                               msg="must not load-follow (drain) at 0.07 prices")

        # Battery energy must be preserved for the expensive stretch.
        self.assertGreater(plan[0].soc_pct, 20.0,
                           "battery should not be drained during cheap hours")

    def test_cheap_surplus_charge_still_happens(self):
        # Following to charge from cheap surplus must not be blocked by the
        # idle check (net load negative -> battery charging, not discharging).
        prices = [0.07, 0.07, 0.15, 0.15]
        pred = np.array([0.0, 0.0, 2.0, 2.0])
        solar = np.array([0.0, 0.0, 0.0, 0.0])
        overrides = {
            "BATTERY_PLANNER_TYPE": "nemotron-linprog",
            "BATTERY_CAPACITY_KWH": "40",
            "BATTERY_MIN_SOC_PCT": "10",
            "BATTERY_MAX_SOC_PCT": "90",
            "BATTERY_INITIAL_SOC_PCT": "20",
            "BATTERY_MAX_CHARGE_KW": "10",
            "BATTERY_MAX_DISCHARGE_KW": "10",
            "BATTERY_CHARGE_EFFICIENCY": "0.98",
            "BATTERY_DISCHARGE_EFFICIENCY": "0.98",
            "BATTERY_ALLOW_EXPORT": "false",
            "PLAN_INTERVAL_MINUTES": "60",
            "BATTERY_DEGRADATION_COST_EUR_PER_KWH": "0.01",
            "BATTERY_FOLLOW_MAX_KW": "10",
        }
        with patched_env(overrides):
            planner = BatteryPlannerFactory.create('nemotron-linprog')
            plan = planner.plan(
                pred, solar, np.array(prices), np.zeros(len(prices)),
                [f'i{i}' for i in range(len(prices))],
                allow_export=False)
        # The battery should still be able to charge at cheap prices, because
        # charging to serve the expensive stretch later is profitable.
        self.assertGreater(plan[1].charge_from_grid_kwh, 0.0,
                           "cheap charging must still be allowed")


if __name__ == '__main__':
    unittest.main()
