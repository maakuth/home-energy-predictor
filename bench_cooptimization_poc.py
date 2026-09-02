from __future__ import annotations
"""
Proof of Concept (PoC) Benchmark: Battery-Aware Load Co-Optimization.

Compares:
1. Baseline: Decoupled Load Governance (GSHP & Leaf planned independently based
   on spot prices + solar, then passed as fixed load to NemotronLinprog).
2. Co-Optimized: Joint LP Planner (GSHP thermal dynamics, Leaf charging, and
   Battery flows co-optimized in a single unified LP).

Evaluates financial differences (€ and %), round-trip battery losses avoided,
direct solar utilization, fuse limit adherence, and thermal storage efficiency.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.battery_test_data import BatteryTestData
from battery_planners.nemotron_linprog import NemotronLinprogPlanner
from battery_planners.joint_linprog import JointLinprogPlanner
from battery_planners.base import BatteryPlannerContext
from optimize_plan import plan_gshp_dispatch


def run_decoupled_baseline(
    predictions_kwh: np.ndarray,
    solar_kwh: np.ndarray,
    import_prices: np.ndarray,
    export_prices: np.ndarray,
    timestamps: list[datetime],
    outside_temps: np.ndarray,
    is_sauna_active: np.ndarray,
    initial_soc_pct: float,
    current_acc_temp: float,
    leaf_target_kwh: float = 10.0,
    interval_hours: float = 0.25,
) -> tuple[float, list[dict], dict]:
    """Run decoupled baseline: GSHP -> Leaf -> Nemotron Battery LP."""
    horizon = len(timestamps)
    solar_kw = solar_kwh / interval_hours

    # 1. Plan GSHP in isolation (battery-agnostic)
    os.environ['GSHP_INITIAL_TEMP'] = str(current_acc_temp)
    gshp_plan = plan_gshp_dispatch(
        timestamps,
        is_sauna_active.tolist(),
        list(outside_temps),
        import_prices,
        export_prices,
        solar_kw,
    )
    planned_gshp_kw = np.array([g['gshp_electric_kw'] for g in gshp_plan])
    planned_gshp_kwh = planned_gshp_kw * interval_hours

    # 2. Plan Leaf in isolation (battery-agnostic)
    night_window_indices = [
        i for i, ts in enumerate(timestamps)
        if ts.hour >= 22 or ts.hour < 7
    ]
    night_prices = [(import_prices[i], i) for i in night_window_indices]
    night_prices.sort()
    leaf_backup_hours = 4.0
    leaf_intervals_backup = max(1, int(round(leaf_backup_hours / interval_hours)))
    leaf_backup_indices = [idx for price, idx in night_prices[:leaf_intervals_backup]]
    leaf_price_threshold_day = np.percentile(import_prices, 35)

    leaf_intents = []
    for i, ts in enumerate(timestamps):
        price = import_prices[i]
        solar = solar_kw[i]
        is_day = 7 <= ts.hour < 22
        intent = 'OFF'
        if i in leaf_backup_indices:
            intent = 'ON'
        elif is_day and (price <= leaf_price_threshold_day or solar >= 2.0):
            intent = 'ON'
        leaf_intents.append(intent)

    num_on = sum(1 for x in leaf_intents if x == 'ON')
    scaled_target_kwh = leaf_target_kwh * (horizon * interval_hours / 24.0)
    leaf_avg_power = (scaled_target_kwh / (num_on * interval_hours)) if num_on > 0 else 0.0
    leaf_avg_power = min(leaf_avg_power, 3.0)
    planned_leaf_kw = np.array([leaf_avg_power if intent == 'ON' else 0.0 for intent in leaf_intents])
    planned_leaf_kwh = planned_leaf_kw * interval_hours

    # 3. Combine loads into fixed total load
    fixed_load_kwh = predictions_kwh + planned_gshp_kwh + planned_leaf_kwh

    # 4. Run Battery LP
    nemotron = NemotronLinprogPlanner()
    context: BatteryPlannerContext = {
        'tomorrow_valid': True,
        'outside_temps': outside_temps,
        'is_sauna_active': is_sauna_active,
        'current_acc_temp': current_acc_temp,
        'planned_gshp_kw': planned_gshp_kw,
    }
    battery_plan = nemotron.plan(
        predictions_kwh=fixed_load_kwh,
        solar_kwh=solar_kwh,
        import_prices=import_prices,
        export_prices=export_prices,
        prediction_timestamps=timestamps,
        allow_export=True,
        initial_soc_pct=initial_soc_pct,
        context=context,
    )

    total_cost = 0.0
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_battery_discharge_kwh = 0.0
    total_battery_charge_kwh = 0.0
    total_direct_solar_kwh = 0.0
    records = []

    for i, entry in enumerate(battery_plan):
        cost = entry.grid_import_kwh * import_prices[i] - entry.grid_export_kwh * export_prices[i]
        total_cost += cost
        total_import_kwh += entry.grid_import_kwh
        total_export_kwh += entry.grid_export_kwh
        total_battery_discharge_kwh += (entry.discharge_to_load_kwh + entry.discharge_to_export_kwh)
        total_battery_charge_kwh += (entry.charge_from_solar_kwh + entry.charge_from_grid_kwh)
        
        # Direct solar used = solar generated - solar charged to batt - solar exported
        direct_solar = max(0.0, solar_kwh[i] - entry.charge_from_solar_kwh - entry.discharge_to_export_kwh)
        total_direct_solar_kwh += direct_solar

        records.append({
            'timestamp': timestamps[i],
            'import_price': import_prices[i],
            'export_price': export_prices[i],
            'baseload_kw': predictions_kwh[i] / interval_hours,
            'solar_kw': solar_kw[i],
            'gshp_kw': planned_gshp_kw[i],
            'gshp_temp': gshp_plan[i]['gshp_temp_sim'],
            'leaf_kw': planned_leaf_kw[i],
            'grid_import_kwh': entry.grid_import_kwh,
            'grid_export_kwh': entry.grid_export_kwh,
            'charge_solar_kwh': entry.charge_from_solar_kwh,
            'charge_grid_kwh': entry.charge_from_grid_kwh,
            'discharge_load_kwh': entry.discharge_to_load_kwh,
            'discharge_export_kwh': entry.discharge_to_export_kwh,
            'soc_pct': entry.soc_pct,
            'battery_action': entry.battery_action,
            'cost_eur': cost,
        })

    summary = {
        'total_cost_eur': total_cost,
        'grid_import_kwh': total_import_kwh,
        'grid_export_kwh': total_export_kwh,
        'battery_throughput_kwh': total_battery_charge_kwh + total_battery_discharge_kwh,
        'battery_roundtrip_losses_kwh': total_battery_charge_kwh * 0.05 + total_battery_discharge_kwh * (1.0/0.95 - 1.0),
        'direct_solar_kwh': total_direct_solar_kwh,
        'gshp_total_electric_kwh': float(np.sum(planned_gshp_kwh)),
        'leaf_total_electric_kwh': float(np.sum(planned_leaf_kwh)),
        'final_soc_pct': battery_plan[-1].soc_pct if battery_plan else initial_soc_pct,
        'final_acc_temp': gshp_plan[-1]['gshp_temp_sim'] if gshp_plan else current_acc_temp,
    }
    return total_cost, records, summary


def run_joint_cooptimization(
    predictions_kwh: np.ndarray,
    solar_kwh: np.ndarray,
    import_prices: np.ndarray,
    export_prices: np.ndarray,
    timestamps: list[datetime],
    outside_temps: np.ndarray,
    is_sauna_active: np.ndarray,
    initial_soc_pct: float,
    current_acc_temp: float,
    leaf_target_kwh: float = 10.0,
    interval_hours: float = 0.25,
) -> tuple[float, list[dict], dict]:
    """Run Joint Co-Optimization LP (Battery + GSHP + Leaf simultaneous)."""
    horizon = len(timestamps)
    solar_kw = solar_kwh / interval_hours

    planner = JointLinprogPlanner()
    context: BatteryPlannerContext = {
        'tomorrow_valid': True,
        'outside_temps': outside_temps,
        'is_sauna_active': is_sauna_active,
        'current_acc_temp': current_acc_temp,
    }

    os.environ['GSHP_OPTIMIZE_ENABLED'] = '1'
    os.environ['LEAF_OPTIMIZE_ENABLED'] = '1'
    os.environ['LEAF_DAILY_TARGET_KWH'] = str(leaf_target_kwh)

    plan = planner.plan(
        predictions_kwh=predictions_kwh,
        solar_kwh=solar_kwh,
        import_prices=import_prices,
        export_prices=export_prices,
        prediction_timestamps=timestamps,
        allow_export=True,
        initial_soc_pct=initial_soc_pct,
        context=context,
    )

    total_cost = 0.0
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_battery_discharge_kwh = 0.0
    total_battery_charge_kwh = 0.0
    total_direct_solar_kwh = 0.0
    total_gshp_kwh = 0.0
    total_leaf_kwh = 0.0
    records = []

    for i, entry in enumerate(plan):
        cost = entry.grid_import_kwh * import_prices[i] - entry.grid_export_kwh * export_prices[i]
        total_cost += cost
        total_import_kwh += entry.grid_import_kwh
        total_export_kwh += entry.grid_export_kwh
        total_battery_discharge_kwh += (entry.discharge_to_load_kwh + entry.discharge_to_export_kwh)
        total_battery_charge_kwh += (entry.charge_from_solar_kwh + entry.charge_from_grid_kwh)
        p_gshp = entry.planned_gshp_kw or 0.0
        p_leaf = entry.planned_leaf_kw or 0.0
        total_gshp_kwh += p_gshp * interval_hours
        total_leaf_kwh += p_leaf * interval_hours

        direct_solar = max(0.0, solar_kwh[i] - entry.charge_from_solar_kwh - entry.discharge_to_export_kwh)
        total_direct_solar_kwh += direct_solar

        records.append({
            'timestamp': timestamps[i],
            'import_price': import_prices[i],
            'export_price': export_prices[i],
            'baseload_kw': predictions_kwh[i] / interval_hours,
            'solar_kw': solar_kw[i],
            'gshp_kw': p_gshp,
            'gshp_temp': entry.gshp_temp_sim,
            'leaf_kw': p_leaf,
            'grid_import_kwh': entry.grid_import_kwh,
            'grid_export_kwh': entry.grid_export_kwh,
            'charge_solar_kwh': entry.charge_from_solar_kwh,
            'charge_grid_kwh': entry.charge_from_grid_kwh,
            'discharge_load_kwh': entry.discharge_to_load_kwh,
            'discharge_export_kwh': entry.discharge_to_export_kwh,
            'soc_pct': entry.soc_pct,
            'battery_action': entry.battery_action,
            'cost_eur': cost,
        })

    summary = {
        'total_cost_eur': total_cost,
        'grid_import_kwh': total_import_kwh,
        'grid_export_kwh': total_export_kwh,
        'battery_throughput_kwh': total_battery_charge_kwh + total_battery_discharge_kwh,
        'battery_roundtrip_losses_kwh': total_battery_charge_kwh * 0.05 + total_battery_discharge_kwh * (1.0/0.95 - 1.0),
        'direct_solar_kwh': total_direct_solar_kwh,
        'gshp_total_electric_kwh': total_gshp_kwh,
        'leaf_total_electric_kwh': total_leaf_kwh,
        'final_soc_pct': plan[-1].soc_pct if plan else initial_soc_pct,
        'final_acc_temp': plan[-1].gshp_temp_sim if plan else current_acc_temp,
    }
    return total_cost, records, summary


def evaluate_fixture(
    fixture_path: str,
    window_hours: int = 48,
    leaf_target_kwh: float = 10.0,
) -> dict:
    """Evaluate a fixture dataset with rolling window comparison."""
    data = BatteryTestData.load(fixture_path)
    preds_archive = data._data.get('history', {}).get('predictions_archive', [])

    if not preds_archive:
        return {'fixture': Path(fixture_path).name, 'error': 'No data'}

    df_preds = pd.DataFrame(preds_archive)
    if 'target_timestamp' in df_preds.columns:
        df_preds['target_timestamp'] = pd.to_datetime(df_preds['target_timestamp'], utc=True)
        if 'generated_at' in df_preds.columns:
            df_preds['generated_at'] = pd.to_datetime(df_preds['generated_at'], utc=True)
            df_preds = df_preds.sort_values(['target_timestamp', 'generated_at']).groupby('target_timestamp').last().reset_index()
        df_preds = df_preds.set_index('target_timestamp').sort_index()

    if df_preds.empty:
        return {'fixture': Path(fixture_path).name, 'error': 'Empty predictions'}

    intervals_in_window = int(window_hours * 4)
    if len(df_preds) < intervals_in_window:
        intervals_in_window = len(df_preds)

    df_window = df_preds.iloc[:intervals_in_window]
    timestamps = [ts.to_pydatetime() for ts in df_window.index]
    horizon = len(timestamps)

    interval_hours = 0.25
    baseload_kw = df_window.get('predicted_usage_kw', pd.Series(1.0, index=df_window.index)).fillna(1.0).values
    predictions_kwh = np.asarray(baseload_kw, dtype=float) * interval_hours

    solar_kw = df_window.get('solar_forecast_kw', pd.Series(0.0, index=df_window.index)).fillna(0.0).values
    solar_kwh = np.asarray(solar_kw, dtype=float) * interval_hours

    import_prices = df_window.get('import_price', pd.Series(0.15, index=df_window.index)).fillna(0.15).to_numpy(dtype=float)
    export_prices = df_window.get('export_price', pd.Series(0.05, index=df_window.index)).fillna(0.05).to_numpy(dtype=float)

    month = timestamps[0].month
    base_temp = -5.0 if month in [12, 1, 2] else (5.0 if month in [3, 4, 10, 11] else 18.0)
    outside_temps = np.full(horizon, base_temp)
    is_sauna_active = np.zeros(horizon, dtype=int)

    initial_soc_pct = 50.0
    initial_acc_temp = 48.0

    cost_base, recs_base, sum_base = run_decoupled_baseline(
        predictions_kwh, solar_kwh, import_prices, export_prices,
        timestamps, outside_temps, is_sauna_active,
        initial_soc_pct, initial_acc_temp, leaf_target_kwh, interval_hours,
    )

    cost_joint, recs_joint, sum_joint = run_joint_cooptimization(
        predictions_kwh, solar_kwh, import_prices, export_prices,
        timestamps, outside_temps, is_sauna_active,
        initial_soc_pct, initial_acc_temp, leaf_target_kwh, interval_hours,
    )

    savings_eur = cost_base - cost_joint
    # Baseline vs joint relative difference:
    abs_base = abs(cost_base) if abs(cost_base) > 0.01 else 1.0
    savings_pct = (savings_eur / abs_base * 100.0)

    return {
        'fixture': Path(fixture_path).name,
        'horizon_hours': horizon * interval_hours,
        'intervals': horizon,
        'cost_base_eur': cost_base,
        'cost_joint_eur': cost_joint,
        'savings_eur': savings_eur,
        'savings_pct': savings_pct,
        'summary_base': sum_base,
        'summary_joint': sum_joint,
        'records_base': recs_base,
        'records_joint': recs_joint,
    }


def print_detailed_case_study(records_base: list[dict], records_joint: list[dict], n_intervals: int = 16):
    """Print an interval-by-interval diff illustrating where co-optimization acts differently."""
    print("\n" + "=" * 115)
    print("  CASE STUDY: INTERVAL-BY-INTERVAL COMPARISON (Decoupled Baseline vs. Joint Co-Optimization)")
    print("=" * 115)
    print(f"{'Time':<16} | {'Price':<7} | {'Solar':<5} | {'Base GSHP':<9} | {'Joint GSHP':<10} | {'Base Leaf':<9} | {'Joint Leaf':<10} | {'Base SoC':<8} | {'Joint SoC':<9} | {'Cost Diff':<9}")
    print("-" * 115)

    for i in range(min(n_intervals, len(records_base))):
        rb = records_base[i]
        rj = records_joint[i]
        ts_str = rb['timestamp'].strftime('%m-%d %H:%M')
        price_str = f"{rb['import_price']:.3f}"
        solar_str = f"{rb['solar_kw']:.1f}"
        bg_str = f"{rb['gshp_kw']:.1f}kW"
        jg_str = f"{rj['gshp_kw']:.1f}kW"
        bl_str = f"{rb['leaf_kw']:.1f}kW"
        jl_str = f"{rj['leaf_kw']:.1f}kW"
        bsoc_str = f"{rb['soc_pct']:.1f}%"
        jsoc_str = f"{rj['soc_pct']:.1f}%"
        cdiff = rb['cost_eur'] - rj['cost_eur']
        cdiff_str = f"{cdiff:+.3f}€"

        print(f"{ts_str:<16} | {price_str:<7} | {solar_str:<5} | {bg_str:<9} | {jg_str:<10} | {bl_str:<9} | {jl_str:<10} | {bsoc_str:<8} | {jsoc_str:<9} | {cdiff_str:<9}")

    print("=" * 115)


def main():
    print("=" * 95)
    print("   HEPO PROOF OF CONCEPT: BATTERY-AWARE LOAD CO-OPTIMIZATION BENCHMARK REPORT")
    print("=" * 95)

    fixture_files = sorted(list(Path('tests/fixtures').glob('*.pkl')))
    if not fixture_files:
        print("No fixtures found in tests/fixtures")
        return

    results = []
    for f in fixture_files:
        res = evaluate_fixture(str(f), window_hours=48, leaf_target_kwh=10.0)
        if 'error' in res:
            continue
        results.append(res)

    print(f"{'Season / Fixture':<18} | {'Horizon':<7} | {'Base Cost':<11} | {'Joint Cost':<11} | {'Net Savings':<12} | {'Avoided Loss':<12} | {'Direct Solar'}")
    print("-" * 95)

    total_base = 0.0
    total_joint = 0.0
    total_loss_base = 0.0
    total_loss_joint = 0.0
    total_solar_base = 0.0
    total_solar_joint = 0.0

    for r in results:
        sb = r['summary_base']
        sj = r['summary_joint']
        total_base += r['cost_base_eur']
        total_joint += r['cost_joint_eur']
        total_loss_base += sb['battery_roundtrip_losses_kwh']
        total_loss_joint += sj['battery_roundtrip_losses_kwh']
        total_solar_base += sb['direct_solar_kwh']
        total_solar_joint += sj['direct_solar_kwh']

        avoided_loss = sb['battery_roundtrip_losses_kwh'] - sj['battery_roundtrip_losses_kwh']
        name = r['fixture'].replace('.pkl', '')
        desc = {
            'jan': 'Winter (Jan)',
            'may': 'Spring (May)',
            'jul': 'Summer (Jul)',
            'oct': 'Autumn (Oct)',
        }.get(name, name)

        print(
            f"{desc:<18} | {r['horizon_hours']:4.0f}h   | {r['cost_base_eur']:8.3f} €  | {r['cost_joint_eur']:8.3f} €  | "
            f"{r['savings_eur']:8.3f} €    | {avoided_loss:+6.2f} kWh   | Base {sb['direct_solar_kwh']:4.1f} vs Joint {sj['direct_solar_kwh']:4.1f} kWh"
        )

    tot_savings = total_base - total_joint
    tot_loss_diff = total_loss_base - total_loss_joint
    print("-" * 95)
    print(
        f"{'TOTAL / 4 SEASONS':<18} | {'192h':<7} | {total_base:8.3f} €  | {total_joint:8.3f} €  | "
        f"{tot_savings:8.3f} €    | {tot_loss_diff:+6.2f} kWh   | Base {total_solar_base:4.1f} vs Joint {total_solar_joint:4.1f} kWh"
    )
    print("=" * 95)

    # Print interval comparison case study from the Winter / Jan fixture
    jan_res = next((r for r in results if 'jan' in r['fixture']), results[0])
    print_detailed_case_study(jan_res['records_base'], jan_res['records_joint'], n_intervals=16)


if __name__ == '__main__':
    main()
