from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from battery_planners.nemotron_linprog import NemotronLinprogPlanner


class NemotronLinprogFlowTests(unittest.TestCase):
    """Physical invariants for the LP-native battery flow model."""

    def _plan(
        self,
        predictions: list[float],
        solar: list[float],
        prices: list[float],
        *,
        committed: list[float] | None = None,
        allow_export: bool = True,
        **overrides: str,
    ):
        env = {
            'BATTERY_CAPACITY_KWH': '40',
            'BATTERY_INITIAL_SOC_PCT': '80',
            'BATTERY_MIN_SOC_PCT': '10',
            'BATTERY_RESERVE_SOC_PCT': '10',
            'BATTERY_MAX_SOC_PCT': '90',
            'BATTERY_MAX_CHARGE_KW': '10',
            'BATTERY_MAX_DISCHARGE_KW': '10',
            'BATTERY_CHARGE_EFFICIENCY': '1.0',
            'BATTERY_DISCHARGE_EFFICIENCY': '1.0',
            'BATTERY_LP_HORIZON': '96',
            'BATTERY_LP_HORIZON_FALLBACK': '96',
            'BATTERY_LP_DISCOUNT': '1.0',
            'BATTERY_TERMINAL_VALUE_PERCENTILE': '0',
            'BATTERY_DEGRADATION_COST_EUR_PER_KWH': '0',
            'PLAN_INTERVAL_MINUTES': '60',
        }
        env.update(overrides)
        with patch.dict(os.environ, env, clear=False):
            return NemotronLinprogPlanner().plan(
                np.asarray(predictions, dtype=float),
                np.asarray(solar, dtype=float),
                np.asarray(prices, dtype=float),
                np.asarray(prices, dtype=float),
                [f'i{i}' for i in range(len(predictions))],
                committed_load_kwh=(
                    np.asarray(committed, dtype=float) if committed is not None else None
                ),
                allow_export=allow_export,
                context={'tomorrow_valid': True},
            )

    def test_committed_load_is_not_powered_by_the_house_battery(self):
        plan = self._plan(
            [0.0], [0.0], [1.0], committed=[2.0], allow_export=False,
        )

        self.assertAlmostEqual(plan[0].discharge_to_load_kwh, 0.0)
        self.assertAlmostEqual(plan[0].grid_import_kwh, 2.0)

    def test_reserve_soc_is_a_planning_floor(self):
        plan = self._plan(
            [2.0, 2.0], [0.0, 0.0], [1.0, 1.0],
            BATTERY_INITIAL_SOC_PCT='25',
            BATTERY_RESERVE_SOC_PCT='20',
        )

        self.assertTrue(all(entry.soc_pct >= 20.0 - 1e-8 for entry in plan))

    def test_disabled_export_uses_curtailment_not_grid_export(self):
        plan = self._plan(
            [0.0], [10.0], [0.10], allow_export=False,
            BATTERY_INITIAL_SOC_PCT='90',
            BATTERY_MAX_SOC_PCT='90',
        )

        self.assertAlmostEqual(plan[0].grid_export_kwh, 0.0)
        self.assertAlmostEqual(plan[0].discharge_to_export_kwh, 0.0)

    def test_reported_flows_obey_the_interval_energy_balance(self):
        predictions = [1.0, 0.0, 2.0]
        solar = [0.0, 3.0, 0.0]
        committed = [0.5, 0.0, 1.0]
        plan = self._plan(predictions, solar, [0.10, 0.05, 0.30], committed=committed)

        for i, entry in enumerate(plan):
            expected_net = (
                predictions[i] - solar[i] + committed[i]
                + entry.charge_from_solar_kwh + entry.charge_from_grid_kwh
                - entry.discharge_to_load_kwh - entry.discharge_to_export_kwh
            )
            self.assertAlmostEqual(
                entry.grid_import_kwh - entry.grid_export_kwh,
                expected_net,
                places=7,
            )

    def test_lp_budget_preserves_energy_reserved_for_a_future_peak(self):
        plan = self._plan(
            [3.0, 3.0, 3.0], [0.0, 0.0, 0.0], [0.03, 0.30, 0.30],
            BATTERY_INITIAL_SOC_PCT='30',
            BATTERY_LP_HEADROOM_COST_TOLERANCE_EUR='0',
        )

        self.assertAlmostEqual(plan[0].discharge_to_load_kwh, 0.0)
        budget = plan[0].discharge_budget_kwh
        if budget is None:
            self.fail('current interval must have an LP-derived discharge budget')
        self.assertAlmostEqual(budget, 0.0)


class NemotronLinprogSolarHedgeTests(unittest.TestCase):
    """BATTERY_SOLAR_HEDGE_ALPHA blends the LP's solar input toward p10.

    The blend affects only this planner's solar series; callers keep p50 for
    everything else (XGBoost features, GSHP planning). Default alpha=1.0 must
    reproduce legacy behaviour exactly.
    """

    LOAD = [0.5] * 12
    SOLAR = [0.0] * 4 + [4.0] * 8       # free midday solar fills the battery
    PRICES = [0.05] * 4 + [0.15] * 8    # cheap morning window, expensive rest
    EXPORT = [0.02] * 12

    def _plan(self, *, alpha: str | None, p10: list[float] | None = None,
              solar: list[float] | None = None):
        solar = list(self.SOLAR if solar is None else solar)
        env = {
            'BATTERY_CAPACITY_KWH': '40',
            'BATTERY_INITIAL_SOC_PCT': '10',
            'BATTERY_MIN_SOC_PCT': '10',
            'BATTERY_RESERVE_SOC_PCT': '10',
            'BATTERY_MAX_SOC_PCT': '90',
            'BATTERY_MAX_CHARGE_KW': '10',
            'BATTERY_MAX_DISCHARGE_KW': '10',
            'BATTERY_CHARGE_EFFICIENCY': '1.0',
            'BATTERY_DISCHARGE_EFFICIENCY': '1.0',
            'BATTERY_LP_HORIZON': '96',
            'BATTERY_LP_HORIZON_FALLBACK': '96',
            'BATTERY_LP_DISCOUNT': '1.0',
            'BATTERY_TERMINAL_VALUE_PERCENTILE': '0',
            'BATTERY_DEGRADATION_COST_EUR_PER_KWH': '0',
            'PLAN_INTERVAL_MINUTES': '60',
        }
        if alpha is not None:
            env['BATTERY_SOLAR_HEDGE_ALPHA'] = alpha
        context = {'tomorrow_valid': True}
        if p10 is not None:
            context['solar_p10_kwh'] = np.asarray(p10, dtype=float)
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop('BATTERY_SOLAR_HEDGE_ALPHA', None) if alpha is None else None
            return NemotronLinprogPlanner().plan(
                np.asarray(self.LOAD, dtype=float),
                np.asarray(solar, dtype=float),
                np.asarray(self.PRICES, dtype=float),
                np.asarray(self.EXPORT, dtype=float),
                [f'i{i}' for i in range(len(solar))],
                allow_export=True,
                context=context,
            )

    def _flows(self, plan):
        return [
            (e.charge_from_grid_kwh, e.charge_from_solar_kwh,
             e.discharge_to_load_kwh, e.discharge_to_export_kwh, e.soc_kwh)
            for e in plan
        ]

    def _assert_same_plan(self, a, b):
        for fa, fb in zip(self._flows(a), self._flows(b)):
            for va, vb in zip(fa, fb):
                self.assertAlmostEqual(va, vb, places=7)

    def test_default_alpha_ignores_p10(self):
        baseline = self._plan(alpha=None)
        with_p10 = self._plan(alpha=None, p10=[0.0] * 12)
        self._assert_same_plan(baseline, with_p10)

    def test_explicit_alpha_one_ignores_p10(self):
        baseline = self._plan(alpha='1.0')
        with_p10 = self._plan(alpha='1.0', p10=[0.0] * 12)
        self._assert_same_plan(baseline, with_p10)

    def test_full_hedge_equals_planning_on_p10_directly(self):
        p10 = [s * 0.5 for s in self.SOLAR]
        hedged = self._plan(alpha='0.0', p10=p10)
        reference = self._plan(alpha=None, solar=p10)
        self._assert_same_plan(hedged, reference)

    def test_partial_hedge_equals_equivalent_solar_series(self):
        p10 = [s * 0.5 for s in self.SOLAR]
        hedged = self._plan(alpha='0.5', p10=p10)
        blended = [0.5 * s + 0.5 * p for s, p in zip(self.SOLAR, p10)]
        reference = self._plan(alpha=None, solar=blended)
        self._assert_same_plan(hedged, reference)

    def test_missing_p10_falls_back_to_p50(self):
        baseline = self._plan(alpha=None)
        hedged = self._plan(alpha='0.5')  # no p10 in context
        self._assert_same_plan(baseline, hedged)

    def test_nan_p10_falls_back_to_p50(self):
        baseline = self._plan(alpha=None)
        p10 = [float('nan')] * 12
        hedged = self._plan(alpha='0.5', p10=p10)
        self._assert_same_plan(baseline, hedged)

    def test_hedge_increases_cheap_window_grid_charging(self):
        """With solar untrusted (p10=0), the LP must buy in the cheap window."""
        baseline = self._plan(alpha=None)
        hedged = self._plan(alpha='0.0', p10=[0.0] * 12)
        cheap = slice(0, 4)
        base_charge = sum(e.charge_from_grid_kwh for e in baseline[cheap])
        hedged_charge = sum(e.charge_from_grid_kwh for e in hedged[cheap])
        self.assertGreater(hedged_charge, base_charge + 1e-6)


if __name__ == '__main__':
    unittest.main()
