#!/usr/bin/env python3
from __future__ import annotations
"""Solar hedge (BATTERY_SOLAR_HEDGE_ALPHA) fixture benchmark.

Evaluates blending the LP's solar input toward the worst-case p10 bound:
    solar_hedged = alpha * p50 + (1 - alpha) * p10

The four replay fixtures (jan/jul/may/oct) predate p10 archiving, so this
bench injects a SYNTHETIC p10 series into each fixture's predictions archive:
p10 = ratio * p50, with ratio sampled i.i.d. per interval from the empirical
*day-ahead* (20-28 h lag) p10/p50 distribution measured from the production
archive — matching the fixtures' fixed 24 h forecast lag. The same seeded
synthetic p10 is used for every alpha (paired comparison).

Caveats (treat results as indicative, not definitive):
  - ratios are sampled independently per interval; real forecast-error
    clustering at day level is not reproduced,
  - the ratio distribution is estimated from Aug 2026 (summer) data only,
  - fixtures' actuals/loads/prices stay fully real; only p10 is synthetic.

Usage:
    venv/bin/python3 bench_solar_hedge.py [--alphas 1.0,0.85,0.7] [--seed 42]
    venv/bin/python3 bench_solar_hedge.py --rebuild-ratio-model [--db PATH]

Full-length runs only (see AGENTS.md: short windows mislead). Takes ~30 min.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RATIO_MODEL_PATH = Path(__file__).parent / 'tests' / 'fixtures' / 'solar_p10_ratio_model.json'
QUANTILE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]


def build_ratio_model(db_path: str) -> dict:
    """Extract the empirical day-ahead p10/p50 ratio distribution from the DB."""
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    df = pd.read_sql_query(
        """SELECT target_timestamp, generated_at,
                  solar_forecast_kw AS p50, solar_forecast_p10_kw AS p10
           FROM predictions
           WHERE solar_forecast_p10_kw IS NOT NULL AND solar_forecast_kw > 0.05""",
        conn,
    )
    conn.close()
    df['target_timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
    df['generated_at'] = pd.to_datetime(df['generated_at'], utc=True)
    lag_h = (df['target_timestamp'] - df['generated_at']).dt.total_seconds() / 3600.0
    da = df[(lag_h >= 20) & (lag_h <= 28)].copy()
    da['ratio'] = (da['p10'] / da['p50']).clip(0.0, 1.0)

    bins = [0.05, 0.3, 1.0, 2.0, 3.0, 4.0, float('inf')]
    per_bin = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = da[(da['p50'] >= lo) & (da['p50'] < hi)]
        if len(b) >= 30:
            per_bin[f'{lo},{hi}'] = np.percentile(b['ratio'], np.array(QUANTILE_LEVELS) * 100).tolist()
    return {
        'description': 'Empirical day-ahead (20-28h lag) Solcast p10/p50 ratio quantiles',
        'source_db': db_path,
        'quantile_levels': QUANTILE_LEVELS,
        'global': np.percentile(da['ratio'], np.array(QUANTILE_LEVELS) * 100).tolist(),
        'per_p50_bin': per_bin,
        'n_pairs_day_ahead': int(len(da)),
    }


def sample_p10_ratios(p50_kw: np.ndarray, model: dict, rng: np.random.Generator) -> np.ndarray:
    """Sample per-interval p10/p50 ratios conditioned on p50 magnitude."""
    levels = np.asarray(model['quantile_levels'])
    ratios = np.empty_like(p50_kw, dtype=float)
    bins = sorted(
        (tuple(float(x) for x in k.split(',')), v) for k, v in model['per_p50_bin'].items()
    )
    global_q = np.asarray(model['global'])
    u = rng.random(len(p50_kw))
    for i, p50 in enumerate(p50_kw):
        q = global_q
        for (lo, hi), values in bins:
            if lo <= p50 < hi:
                q = np.asarray(values)
                break
        ratios[i] = np.interp(u[i], levels, q)
    return np.clip(ratios, 0.0, 1.0)


def inject_synthetic_p10(fixture, model: dict, seed: int, rho: float = 0.0) -> None:
    """Add solar_forecast_p10_kw to a loaded fixture's predictions archive.

    ``rho`` controls the rank correlation between the sampled ratio and the
    interval's *realized* forecast shortfall (actual below p50), using the
    fixture's measured solar:
      rho=0.0 — p10 is uninformative noise with the right marginal
                distribution (lower bound for the hedge's value),
      rho=1.0 — low p10 perfectly flags the intervals that actually
                underdeliver (upper bound).
    The real August fixture (real Solcast p10) will land somewhere between.
    """
    archive = fixture._data.get('history', {}).get('predictions_archive', [])
    if not archive:
        raise ValueError('fixture has no predictions archive')
    p50 = np.asarray([float(r.get('solar_forecast_kw', 0.0) or 0.0) for r in archive])
    rng = np.random.default_rng(seed)
    ratios = sample_p10_ratios(p50, model, rng)

    if rho > 0.0:
        # Realized shortfall per archive row (aligned on target_timestamp).
        measurements = fixture._data.get('history', {}).get('measurements', [])
        actual_by_ts = {
            pd.to_datetime(m['timestamp'], utc=True): float(m.get('solar_actual_kw', 0.0) or 0.0)
            for m in measurements
        }
        shortfall = np.asarray([
            max(0.0, p50[i] - actual_by_ts.get(pd.to_datetime(r['target_timestamp'], utc=True), 0.0))
            for i, r in enumerate(archive)
        ])
        # Gaussian-copula-style rank induction: score = rho*z(shortfall rank)
        # + sqrt(1-rho^2)*z(noise); lowest ratios go to highest scores.
        from scipy.stats import norm
        n = len(archive)
        z_signal = norm.ppf((shortfall.argsort().argsort() + 0.5) / n)
        z_noise = norm.ppf((rng.random(n).argsort().argsort() + 0.5) / n)
        score = rho * z_signal + np.sqrt(1.0 - rho ** 2) * z_noise
        ratios = np.sort(ratios)[(-score).argsort().argsort()]

    for row, ratio in zip(archive, ratios):
        row['solar_forecast_p10_kw'] = float(ratio * float(row.get('solar_forecast_kw', 0.0) or 0.0))


def summarize_actions(actions: list[dict]) -> dict:
    """Cycling, grid-charge totals and worst-day cost from recorded actions."""
    df = pd.DataFrame(actions)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    per_day = df.groupby(df['timestamp'].dt.date)['cost_eur'].sum()
    worst_day, worst_cost = per_day.idxmax(), float(per_day.max())
    return {
        'cycled_kwh': float(df['discharge_load_kwh'].sum() + df['discharge_export_kwh'].sum()),
        'grid_charge_kwh': float(df['charge_grid_kwh'].sum()),
        'worst_day': str(worst_day),
        'worst_day_cost_eur': worst_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--alphas', type=str, default='1.0,0.85,0.7')
    parser.add_argument('--rhos', type=str, default='0.0',
                        help='Rank correlation between synthetic p10 and realized shortfall')
    parser.add_argument('--no-inject', action='store_true',
                        help='Do not inject synthetic p10; use the fixture\'s own '
                             'solar_forecast_p10_kw (for real post-Aug-2026 fixtures)')
    parser.add_argument('--fixtures', type=str, default='',
                        help='Comma-separated fixture name filter (e.g. "aug" or "jan,jul")')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rebuild-ratio-model', action='store_true')
    parser.add_argument('--db', type=str, default='/workspace/state/hepo.db')
    args = parser.parse_args()

    if args.rebuild_ratio_model:
        model = build_ratio_model(args.db)
        RATIO_MODEL_PATH.write_text(json.dumps(model, indent=2))
        print(f'Wrote {RATIO_MODEL_PATH} (n={model["n_pairs_day_ahead"]} day-ahead pairs)')
        print('  global p10/p50 quantiles:', [round(v, 3) for v in model['global']])
        return

    model = json.loads(RATIO_MODEL_PATH.read_text())
    alphas = [float(a) for a in args.alphas.split(',')]
    rhos = [float(x) for x in args.rhos.split(',')]
    if args.no_inject:
        rhos = [0.0]  # rho only shapes synthetic p10; meaningless otherwise

    from battery_planners import NemotronLinprogPlanner
    from tests.battery_planner_replay import BatteryReplaySimulator, load_fixture, get_fixtures

    fixture_filter = [x.strip() for x in args.fixtures.split(',') if x.strip()]
    fixture_paths = [p for p in get_fixtures()
                     if not fixture_filter or Path(p).stem in fixture_filter]
    if not fixture_paths:
        print('No fixtures matched.')
        return

    results = []
    for fixture_path in fixture_paths:
        name = Path(fixture_path).stem
        for rho in rhos:
            for alpha in alphas:
                fixture = load_fixture(fixture_path)
                # Same seed for every (rho, alpha): paired comparison.
                if not args.no_inject:
                    inject_synthetic_p10(fixture, model, seed=args.seed, rho=rho)
                os.environ['BATTERY_SOLAR_HEDGE_ALPHA'] = str(alpha)
                sim = BatteryReplaySimulator(fixture)
                n_intervals = len(sim.measurements_df) if sim.measurements_df is not None else 96
                t0 = time.time()
                r = sim.simulate_battery_control(
                    NemotronLinprogPlanner(), 'nemotron-linprog', max_planks=n_intervals)
                elapsed = time.time() - t0
                if not r['success']:
                    print(f'  rho={rho:.2f} alpha={alpha:.2f} {name}: FAILED: {r.get("error")}')
                    continue
                extra = summarize_actions(r['battery_actions'])
                results.append({'alpha': alpha, 'rho': rho, 'fixture': name, **r, **extra, 'elapsed': elapsed})
                print(f"  rho={rho:.2f} alpha={alpha:.2f} {name:5s}  cost={r['cost_with_battery_eur']:7.3f}  "
                      f"savings={r['savings_pct']:6.1f}%  viol={r['soc_violations']}  "
                      f"cycled={extra['cycled_kwh']:6.1f} kWh  grid-chg={extra['grid_charge_kwh']:6.1f} kWh  "
                      f"worst-day={extra['worst_day']} {extra['worst_day_cost_eur']:.3f} EUR  ({elapsed:.0f}s)")

    print()
    print(f"{'rho':>5s} {'alpha':>6s}  {'Δcost vs alpha=1.0 per fixture (EUR/week)':50s} {'avg':>8s}")
    print('-' * 76)
    base_cost = {(r['fixture'], r['rho']): r['cost_with_battery_eur'] for r in results if r['alpha'] == 1.0}
    for rho in rhos:
        for alpha in alphas:
            rows = [r for r in results if r['alpha'] == alpha and r['rho'] == rho]
            deltas = [r['cost_with_battery_eur'] - base_cost[(r['fixture'], rho)] for r in rows]
            fixtures_str = '  '.join(f"{r['fixture']}:{d:+6.3f}" for r, d in zip(rows, deltas))
            print(f"{rho:5.2f} {alpha:6.2f}  {fixtures_str:50s} {np.mean(deltas):+8.3f}")


if __name__ == '__main__':
    main()
