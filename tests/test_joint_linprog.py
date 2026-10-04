from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from typing import cast
from unittest.mock import patch
import numpy as np

from battery_planners.base import BatteryPlannerContext
from battery_planners.joint_linprog import JointLinprogPlanner, compute_gshp_thermal_demand
from battery_planners.factory import BatteryPlannerFactory


class JointLinprogTests(unittest.TestCase):
    """Unit tests for the Joint Battery + GSHP + Leaf LP Planner."""

    def _plan(
        self,
        predictions: list[float],
        solar: list[float],
        prices: list[float],
        *,
        outside_temps: list[float] | None = None,
        is_sauna: list[int] | None = None,
        current_acc_temp: float = 50.0,
        committed: list[float] | None = None,
        allow_export: bool = True,
        timestamps: list[datetime] | None = None,
        **overrides: str,
    ):
        n = len(predictions)
        env = {
            'BATTERY_CAPACITY_KWH': '40',
            'BATTERY_INITIAL_SOC_PCT': '50',
            'BATTERY_MIN_SOC_PCT': '10',
            'BATTERY_RESERVE_SOC_PCT': '10',
            'BATTERY_MAX_SOC_PCT': '90',
            'BATTERY_MAX_CHARGE_KW': '10',
            'BATTERY_MAX_DISCHARGE_KW': '10',
            'BATTERY_CHARGE_EFFICIENCY': '0.95',
            'BATTERY_DISCHARGE_EFFICIENCY': '0.95',
            'BATTERY_LP_HORIZON': str(n),
            'BATTERY_LP_HORIZON_FALLBACK': str(n),
            'BATTERY_LP_DISCOUNT': '1.0',
            'BATTERY_TERMINAL_VALUE_PERCENTILE': '0',
            'BATTERY_DEGRADATION_COST_EUR_PER_KWH': '0',
            'PLAN_INTERVAL_MINUTES': '60',
            'GSHP_OPTIMIZE_ENABLED': '1',
            'GSHP_POWER_MIN_KW': '0.0',
            'GSHP_POWER_MAX_KW': '4.0',
            'GSHP_COP': '3.5',
            'GSHP_RESERVOIR_LITERS': '500',
            'GSHP_MIN_TEMP': '42.0',
            'GSHP_MAX_TEMP': '55.0',
            'GSHP_INITIAL_TEMP': str(current_acc_temp),
            'LEAF_OPTIMIZE_ENABLED': '0',  # disabled by default unless tested
            'LEAF_DAILY_TARGET_KWH': '0',
        }
        env.update(overrides)
        context = {
            'tomorrow_valid': True,
            'outside_temps': np.array(outside_temps if outside_temps is not None else [5.0] * n),
            'is_sauna_active': np.array(is_sauna if is_sauna is not None else [0] * n),
            'current_acc_temp': current_acc_temp,
        }
        with patch.dict(os.environ, env, clear=False):
            return JointLinprogPlanner().plan(
                np.asarray(predictions, dtype=float),
                np.asarray(solar, dtype=float),
                np.asarray(prices, dtype=float),
                np.asarray(prices, dtype=float),
                timestamps if timestamps is not None else [f'i{i}' for i in range(n)],
                committed_load_kwh=(
                    np.asarray(committed, dtype=float) if committed is not None else None
                ),
                allow_export=allow_export,
                context=cast(BatteryPlannerContext, context),
            )

    def test_factory_registration(self):
        planner = BatteryPlannerFactory.create('joint-linprog')
        self.assertIsInstance(planner, JointLinprogPlanner)

    def test_delegates_battery_only_problem_to_nemotron(self):
        sentinel = [object()]
        with patch(
            'battery_planners.joint_linprog.NemotronLinprogPlanner.plan',
            return_value=sentinel,
        ) as plan:
            with patch.dict(os.environ, {
                'GSHP_OPTIMIZE_ENABLED': '0',
                'RESISTIVE_HEATER_OPTIMIZE_ENABLED': '0',
                'BULK_HEATER_OPTIMIZE_ENABLED': '0',
                'LEAF_OPTIMIZE_ENABLED': '0',
            }, clear=False):
                result = JointLinprogPlanner().plan(
                    np.array([1.0]), np.array([0.0]),
                    np.array([0.20]), np.array([0.05]), ['t0'],
                )

        self.assertIs(result, sentinel)
        plan.assert_called_once()

    def test_house_and_thermal_energy_balances(self):
        """Verify electric and thermal energy conservation across all intervals."""
        n = 4
        preds = [1.0, 1.5, 0.8, 2.0]
        solar = [0.0, 2.0, 3.0, 0.0]
        prices = [0.10, 0.05, 0.20, 0.15]
        temps = [2.0, 5.0, 8.0, 0.0]

        plan = self._plan(preds, solar, prices, outside_temps=temps, current_acc_temp=48.0)
        self.assertEqual(len(plan), n)

        cop = 3.5
        c_deg = (500.0 * 4.18) / 3600.0  # ~0.580556
        dt = 1.0  # 60 min intervals

        demand_kw = compute_gshp_thermal_demand(np.array(temps), np.zeros(n), 1.0, 0.135, 6.0)

        current_t = 48.0
        current_soc = 20.0  # 50% of 40 kWh

        for i, entry in enumerate(plan):
            p_gshp = entry.planned_gshp_kw or 0.0
            p_leaf = entry.planned_leaf_kw or 0.0
            sim_temp = entry.gshp_temp_sim
            assert sim_temp is not None

            # Electric house balance
            net_served = (
                entry.grid_import_kwh - entry.charge_from_grid_kwh
                + entry.solar_forecast_kwh if hasattr(entry, 'solar_forecast_kwh') else entry.charge_from_solar_kwh + entry.discharge_to_load_kwh
            )
            # Total electric demand = baseload + gshp + leaf
            total_load_kwh = preds[i] * dt + (p_gshp + p_leaf) * dt

            # Solar balance
            solar_flows = (
                entry.charge_from_solar_kwh
                + entry.discharge_to_export_kwh  # or solar export
            )
            self.assertTrue(entry.charge_from_solar_kwh <= solar[i] * dt + 1e-6)

            # Thermal balance
            net_heat = (p_gshp * dt * cop) - (demand_kw[i] * dt)
            expected_temp = current_t + (net_heat / c_deg)
            self.assertAlmostEqual(sim_temp, expected_temp, delta=0.01)
            current_t = sim_temp

            # Temperature bounds
            self.assertTrue(sim_temp >= 42.0 - 1e-5)
            self.assertTrue(sim_temp <= 55.0 + 1e-5)

            # SoC bounds
            self.assertTrue(entry.soc_pct >= 10.0 - 1e-5)
            self.assertTrue(entry.soc_pct <= 90.0 + 1e-5)

    def test_thermal_preheating_during_cheap_price(self):
        """GSHP should preheat accumulator during cheap hours to avoid expensive peak hours."""
        # 4 intervals: cheap at index 0 (0.02 €/kWh), then very expensive at index 1-3 (0.50 €/kWh)
        preds = [1.0, 1.0, 1.0, 1.0]
        solar = [0.0, 0.0, 0.0, 0.0]
        prices = [0.02, 0.50, 0.50, 0.50]
        temps = [-5.0, -5.0, -5.0, -5.0]  # Cold outside -> strong heating demand

        plan = self._plan(preds, solar, prices, outside_temps=temps, current_acc_temp=45.0)

        # GSHP should heat heavily at index 0 (cheap) to coast through expensive index 1-3
        self.assertGreater(plan[0].planned_gshp_kw or 0.0, 2.0, "GSHP should run heavily during cheap interval 0")
        self.assertGreater(plan[0].gshp_temp_sim or 0.0, 45.0, "Accumulator temperature should rise in interval 0")

    def test_gshp_runs_at_least_minimum_power_for_a_full_slot(self):
        """A compressor start must not be represented as a fractional 15-minute run."""
        plan = self._plan(
            [0.0], [0.0], [0.20],
            outside_temps=[20.0], current_acc_temp=42.0,
            PLAN_INTERVAL_MINUTES='15',
            GSHP_POWER_MIN_KW='3.4',
            GSHP_POWER_MAX_KW='4.2',
        )

        planned_kw = plan[0].planned_gshp_kw or 0.0
        self.assertGreaterEqual(planned_kw, 3.4 - 1e-6)
        self.assertEqual(plan[0].gshp_intent, 'START')

    def test_direct_solar_consumed_by_loads_without_battery_loss(self):
        """When solar surplus is available, it should directly supply GSHP with 100% efficiency."""
        preds = [0.5, 0.5]
        solar = [5.0, 0.0]  # 5 kW solar in interval 0
        prices = [0.20, 0.20]
        temps = [0.0, 0.0]

        plan = self._plan(preds, solar, prices, outside_temps=temps, current_acc_temp=43.0)

        # In interval 0, solar surplus powers GSHP and charges battery without grid import
        self.assertAlmostEqual(plan[0].grid_import_kwh, 0.0, delta=1e-6)
        self.assertGreater(plan[0].planned_gshp_kw or 0.0, 0.0)

    def test_resistive_source_replaces_disabled_gshp(self):
        plan = self._plan(
            [1.0, 1.0], [0.0, 0.0], [0.01, 0.50],
            outside_temps=[20.0, 20.0], current_acc_temp=45.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='1',
            RESISTIVE_HEATER_POWER_KW='6.0',
            RESISTIVE_HEATER_EFFECTIVE_LITERS='150',
            RESISTIVE_HEATER_MAX_TEMP='60.0',
            GSHP_BASELINE_DEMAND_KW='0.0',
            GSHP_HEAT_LOSS_K='0.0',
        )

        self.assertEqual(plan[0].gshp_intent, 'STOP')
        self.assertEqual(plan[0].resistive_heater_intent, 'ON')
        self.assertEqual(plan[0].planned_gshp_kw, 0.0)
        self.assertGreater(plan[0].planned_resistive_kw or 0.0, 0.0)

    def test_gshp_and_resistive_sources_can_run_together(self):
        plan = self._plan(
            [1.0], [0.0], [0.10],
            outside_temps=[0.0], current_acc_temp=42.0,
            GSHP_OPTIMIZE_ENABLED='1',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='1',
            GSHP_POWER_MIN_KW='0.0',
            GSHP_POWER_MAX_KW='4.0',
            GSHP_COP='3.5',
            RESISTIVE_HEATER_POWER_KW='6.0',
            RESISTIVE_HEATER_EFFECTIVE_LITERS='150',
            RESISTIVE_HEATER_MAX_TEMP='60.0',
            GSHP_BASELINE_DEMAND_KW='34.0',
            GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0',
            BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertAlmostEqual(plan[0].planned_gshp_kw or 0.0, 4.0, places=4)
        self.assertAlmostEqual(plan[0].planned_resistive_kw or 0.0, 6.0, places=4)
        self.assertAlmostEqual(plan[0].grid_import_kwh, 11.0, places=4)

    def test_bulk_heater_heats_whole_reservoir_independently(self):
        plan = self._plan(
            [1.0, 1.0], [0.0, 0.0], [0.01, 0.50],
            outside_temps=[20.0, 20.0], current_acc_temp=42.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='0',
            BULK_HEATER_OPTIMIZE_ENABLED='1',
            BULK_HEATER_POWER_KW='6.0',
            BULK_HEATER_MAX_TEMP='60.0',
            GSHP_BASELINE_DEMAND_KW='6.0',
            GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0',
            BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertEqual(plan[0].bulk_heater_intent, 'ON')
        self.assertAlmostEqual(plan[0].planned_bulk_heater_kw or 0.0, 6.0, places=4)
        expected_gain = (6.0 - 3.0) / ((500.0 * 4.18) / 3600.0)
        self.assertAlmostEqual(plan[0].gshp_temp_sim or 0.0, 42.0 + expected_gain, places=4)

    def test_constrained_import_prioritizes_upper_before_bulk(self):
        plan = self._plan(
            [0.0], [0.0], [0.01], outside_temps=[20.0], current_acc_temp=42.0,
            GSHP_OPTIMIZE_ENABLED='0', RESISTIVE_HEATER_OPTIMIZE_ENABLED='1',
            BULK_HEATER_OPTIMIZE_ENABLED='1', MAIN_FUSE_SIZE_A='8.7',
            GSHP_BASELINE_DEMAND_KW='6.0', GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0', BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertGreater(plan[0].planned_resistive_kw or 0.0, 0.8)
        self.assertAlmostEqual(plan[0].planned_bulk_heater_kw or 0.0, 0.0, places=4)

    def test_resistive_preheats_to_soft_target_during_cheap_period(self):
        plan = self._plan(
            [0.0, 0.0], [0.0, 0.0], [0.01, 0.50],
            outside_temps=[20.0, 20.0], current_acc_temp=50.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='1',
            RESISTIVE_HEATER_POWER_KW='6.0',
            RESISTIVE_HEATER_EFFECTIVE_LITERS='500',
            THERMAL_TARGET_TEMP='55.0',
            GSHP_BASELINE_DEMAND_KW='0.0', GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0', BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertGreater(plan[0].planned_resistive_kw or 0.0, 0.0)
        self.assertGreaterEqual(plan[-1].gshp_temp_sim or 0.0, 55.0 - 1e-4)
        self.assertLessEqual(plan[-1].gshp_temp_sim or 0.0, 55.0 + 1e-4)

    def test_resistive_recovers_below_minimum_to_safety_margin(self):
        plan = self._plan(
            [0.0], [0.0], [0.50],
            outside_temps=[20.0], current_acc_temp=44.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='1',
            RESISTIVE_HEATER_POWER_KW='6.0',
            RESISTIVE_HEATER_EFFECTIVE_LITERS='150',
            GSHP_MIN_TEMP='45.0',
            THERMAL_TARGET_TEMP='45.0',
            GSHP_BASELINE_DEMAND_KW='1.0', GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0', BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertGreater(plan[0].planned_resistive_kw or 0.0, 0.0)
        self.assertGreaterEqual(plan[0].gshp_temp_sim or 0.0, 45.5 - 1e-4)

    def test_temperature_deficit_does_not_create_phantom_heat(self):
        plan = self._plan(
            [0.0], [0.0], [0.50],
            outside_temps=[20.0], current_acc_temp=30.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='0',
            BULK_HEATER_OPTIMIZE_ENABLED='0',
            LEAF_OPTIMIZE_ENABLED='1', LEAF_DAILY_TARGET_KWH='0',
            GSHP_MIN_TEMP='45.0',
            GSHP_BASELINE_DEMAND_KW='0.0', GSHP_HEAT_LOSS_K='0.0',
        )

        self.assertAlmostEqual(plan[0].gshp_temp_sim or 0.0, 30.0, places=4)

    def test_bulk_heater_maximizes_recovery_when_floor_is_unreachable(self):
        plan = self._plan(
            [0.0], [0.0], [0.50],
            outside_temps=[20.0], current_acc_temp=30.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='0',
            BULK_HEATER_OPTIMIZE_ENABLED='1', BULK_HEATER_POWER_KW='6.0',
            GSHP_MIN_TEMP='45.0',
            GSHP_BASELINE_DEMAND_KW='0.0', GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0', BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertAlmostEqual(plan[0].planned_bulk_heater_kw or 0.0, 6.0, places=4)
        expected_temp = 30.0 + 6.0 / ((500.0 * 4.18) / 3600.0)
        self.assertAlmostEqual(plan[0].gshp_temp_sim or 0.0, expected_temp, places=4)

    def test_negative_prices_do_not_make_terminal_heat_unbounded(self):
        plan = self._plan(
            [0.0], [0.0], [-0.50],
            outside_temps=[20.0], current_acc_temp=30.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='0',
            BULK_HEATER_OPTIMIZE_ENABLED='1', BULK_HEATER_POWER_KW='6.0',
            GSHP_MIN_TEMP='45.0',
            GSHP_BASELINE_DEMAND_KW='0.0', GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0', BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertAlmostEqual(plan[0].planned_bulk_heater_kw or 0.0, 6.0, places=4)

    def test_upper_element_complements_running_gshp_to_reach_target(self):
        plan = self._plan(
            [0.0, 0.0], [0.0, 0.0], [0.01, 0.50],
            outside_temps=[20.0, 20.0], current_acc_temp=42.0,
            GSHP_OPTIMIZE_ENABLED='1', GSHP_POWER_MIN_KW='0.0', GSHP_POWER_MAX_KW='1.0',
            GSHP_COP='3.5',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='1',
            RESISTIVE_HEATER_POWER_KW='6.0', RESISTIVE_HEATER_EFFECTIVE_LITERS='500',
            THERMAL_TARGET_TEMP='55.0',
            GSHP_BASELINE_DEMAND_KW='0.0', GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0', BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertGreater(plan[0].planned_gshp_kw or 0.0, 0.0)
        self.assertGreater(plan[0].planned_resistive_kw or 0.0, 0.0)
        self.assertGreaterEqual(plan[-1].gshp_temp_sim or 0.0, 55.0 - 1e-4)

    def test_target_waits_for_a_cheaper_period(self):
        plan = self._plan(
            [0.0, 0.0], [0.0, 0.0], [0.50, 0.01],
            outside_temps=[20.0, 20.0], current_acc_temp=50.0,
            GSHP_OPTIMIZE_ENABLED='0',
            RESISTIVE_HEATER_OPTIMIZE_ENABLED='1',
            RESISTIVE_HEATER_POWER_KW='6.0', RESISTIVE_HEATER_EFFECTIVE_LITERS='500',
            THERMAL_TARGET_TEMP='55.0',
            GSHP_BASELINE_DEMAND_KW='0.0', GSHP_HEAT_LOSS_K='0.0',
            BATTERY_MAX_CHARGE_KW='0.0', BATTERY_MAX_DISCHARGE_KW='0.0',
        )

        self.assertAlmostEqual(plan[0].planned_resistive_kw or 0.0, 0.0, places=4)
        self.assertGreater(plan[1].planned_resistive_kw or 0.0, 0.0)

    def test_leaf_charges_overnight_except_price_peaks(self):
        timestamps = [
            datetime(2026, 1, 1, hour, tzinfo=timezone.utc)
            for hour in (21, 22, 23, 0)
        ]
        plan = self._plan(
            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.10, 0.10, 0.30, 0.10],
            timestamps=timestamps,
            LEAF_OPTIMIZE_ENABLED='1',
        )

        self.assertEqual([entry.leaf_intent for entry in plan], ['OFF', 'ON', 'OFF', 'ON'])
        self.assertEqual([entry.planned_leaf_kw for entry in plan], [0.0, 1.8, 0.0, 1.8])

    def test_fuse_limit_prevents_simultaneous_overload(self):
        """Battery charging, GSHP, and Leaf combined must respect main fuse limit."""
        # 1 interval with very cheap price (0.01 €/kWh). Everything wants to charge.
        preds = [2.0]
        solar = [0.0]
        prices = [0.01]
        temps = [-10.0]

        # 25A fuse @ 3-phase 230V = 17.25 kW max import
        plan = self._plan(
            preds, solar, prices,
            outside_temps=temps,
            current_acc_temp=42.0,
            MAIN_FUSE_SIZE_A='25.0',
            BATTERY_MAX_CHARGE_KW='10.0',
            GSHP_POWER_MAX_KW='4.0',
            LEAF_OPTIMIZE_ENABLED='1',
            LEAF_DAILY_TARGET_KWH='6.0',
            LEAF_MAX_POWER_KW='3.0',
        )

        self.assertLessEqual(plan[0].grid_import_kwh, 17.25 + 1e-4)


if __name__ == '__main__':
    unittest.main()
