#!/usr/bin/env python3
"""Benchmark LP planner horizon/discount combinations against all fixtures."""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from battery_planners import BatteryPlannerFactory, NemotronLinprogPlanner
from tests.battery_planner_replay import BatteryReplaySimulator, load_fixture, get_fixtures

combos = [
    ("6h/γ0.98 (current)",  24,  0.98),
    ("12h/γ0.990",          48,  0.990),
    ("24h/γ0.995",          96,  0.995),
]

results = []
for label, horizon, discount in combos:
    for fixture_path in get_fixtures():
        name = fixture_path.split("/")[-1].replace(".pkl", "")
        os.environ["BATTERY_LP_HORIZON"] = str(horizon)
        os.environ["BATTERY_LP_DISCOUNT"] = str(discount)

        fixture = load_fixture(fixture_path)
        sim = BatteryReplaySimulator(fixture)
        planner = NemotronLinprogPlanner()

        t0 = time.time()
        r = sim.simulate_battery_control(planner, "nemotron-linprog", max_planks=96)
        elapsed = time.time() - t0

        print(f"  {label:20s} {name:5s}  savings={r['savings_pct']:6.1f}%  soc={r['final_soc_pct']:5.1f}%  "
              f"cost={r['cost_with_battery_eur']:.3f}  base={r['cost_no_battery_eur']:.3f}  "
              f"viol={r['soc_violations']}  ({elapsed:.1f}s)")

        results.append({
            "label": label, "fixture": name,
            "horizon": horizon, "discount": discount,
            "savings_pct": r['savings_pct'],
            "cost": r['cost_with_battery_eur'],
            "base": r['cost_no_battery_eur'],
            "final_soc_pct": r['final_soc_pct'],
            "elapsed": elapsed,
        })

print()
print(f"{'Config':20s} {'Fixtures':40s} {'Avg savings':>12s}")
print("-" * 75)
for label, _, _ in combos:
    savings = [r['savings_pct'] for r in results if r['label'] == label]
    avg = np.mean(savings)
    fixtures_str = "  ".join(f"{r['fixture']}:{r['savings_pct']:+.1f}%" for r in results if r['label'] == label)
    print(f"{label:20s} {fixtures_str:40s}  avg={avg:+.1f}%")
