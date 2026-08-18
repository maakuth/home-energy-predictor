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


if __name__ == '__main__':
    unittest.main()
