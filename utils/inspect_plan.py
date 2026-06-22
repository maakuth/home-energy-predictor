from __future__ import annotations
"""
Plan inspection utility: read optimization_plan.json and display
formatted tables, summaries, and filtered views.
"""

import json
import argparse
from datetime import datetime, timezone
from collections import Counter
from typing import Any


def load_plan(path: str = "optimization_plan.json") -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def parse_iso(s: str) -> datetime:
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)


def format_ts(ts_str: str) -> str:
    """Convert plan timestamp (ISO-8601 with offset) to local-timezone string."""
    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    local = dt.astimezone()
    return local.isoformat()


def filter_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.start:
        t = parse_iso(args.start)
        entries = [e for e in entries if parse_iso(e['timestamp']) >= t]
    if args.end:
        t = parse_iso(args.end)
        entries = [e for e in entries if parse_iso(e['timestamp']) <= t]
    if args.actions:
        allowed = set(args.actions.split(','))
        entries = [e for e in entries if e.get('battery_action') in allowed]
    return entries


def show_summary(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not entries:
        print("No entries match filters.")
        return

    n = len(entries)

    action_counts = Counter(e.get('battery_action', 'unknown') for e in entries)
    total_charge_grid = sum(e.get('charge_from_grid_kwh', 0.0) for e in entries)
    total_charge_solar = sum(e.get('charge_from_solar_kwh', 0.0) for e in entries)
    total_discharge_load = sum(e.get('discharge_to_load_kwh', 0.0) for e in entries)
    total_discharge_export = sum(e.get('discharge_to_export_kwh', 0.0) for e in entries)
    total_grid_import = sum(e.get('grid_import_kwh', 0.0) for e in entries)
    total_grid_export = sum(e.get('grid_export_kwh', 0.0) for e in entries)

    socs = [e.get('soc_pct', 0.0) for e in entries if e.get('soc_pct') is not None]
    soc_min = min(socs) if socs else 0
    soc_max = max(socs) if socs else 0

    print(f"Plan entries: {n}")
    print(f"Time range:  {format_ts(entries[0]['timestamp'])}  →  {format_ts(entries[-1]['timestamp'])}")
    print()
    print("Action counts:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:25s}  {count:4d}  ({100*count/n:5.1f}%)")
    print()
    print(f"SoC range:     {soc_min:.1f}%  →  {soc_max:.1f}%")
    print(f"Grid import:   {total_grid_import:8.3f} kWh")
    print(f"Grid export:   {total_grid_export:8.3f} kWh")
    print(f"Charge grid:   {total_charge_grid:8.3f} kWh")
    print(f"Charge solar:  {total_charge_solar:8.3f} kWh")
    print(f"Dischg load:   {total_discharge_load:8.3f} kWh")
    print(f"Dischg export: {total_discharge_export:8.3f} kWh")


def show_detail(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not entries:
        print("No entries match filters.")
        return

    if args.n:
        entries = entries[:args.n]

    fmt = "{:<24s} {:<18s} {:>6s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}"
    print(fmt.format("Timestamp", "Action", "SoC%", "ImpPrc", "Load_kW", "Solar_kW",
                     "ChgGrd", "ChgSol", "DisLd", "DisEx"))
    print("-" * 110)

    for e in entries:
        ts = format_ts(e['timestamp'])
        action = e.get('battery_action', '')
        soc = e.get('soc_pct', 0)
        imp = e.get('import_unit_price', 0)
        load = e.get('predicted_usage_kw', 0)
        solar = e.get('solar_forecast_kw', 0)
        cg = e.get('charge_from_grid_kwh', 0)
        cs = e.get('charge_from_solar_kwh', 0)
        dl = e.get('discharge_to_load_kwh', 0)
        de = e.get('discharge_to_export_kwh', 0)
        print(fmt.format(
            ts, action,
            f"{soc:.1f}", f"{imp:.3f}",
            f"{load:.2f}", f"{solar:.2f}",
            f"{cg:.2f}", f"{cs:.2f}",
            f"{dl:.2f}", f"{de:.2f}",
        ))


def show_charging(entries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Show charging analysis: only charge_solar/charge_grid actions."""
    entries = [e for e in entries if e.get('battery_action') in ('charge_solar', 'charge_grid')]
    if not entries:
        print("No charging entries match filters.")
        return
    if args.n:
        entries = entries[:args.n]

    fmt = "{:<24s} {:>5s} {:>5s} {:>5s} {:>6s} {:>6s} {:>7s} {:>6s} {:>6s} {:>6s}"
    print(fmt.format("Timestamp", "Action", "SoC%", "kW_p", "ImpPrc", "Solar", "Net_kWh",
                     "ChgSol", "ChgGrd", "SurpkW"))
    print("-" * 90)

    for e in entries:
        ts = format_ts(e['timestamp'])
        action = e.get('battery_action', '')
        soc = e.get('soc_pct', 0)
        pwr = e.get('battery_power_kw', 0)
        imp = e.get('import_unit_price', 0)
        solar = e.get('solar_forecast_kw', 0)
        load = e.get('predicted_usage_kw', 0)
        net = e.get('net_load_without_battery_kwh', 0)
        cs = e.get('charge_from_solar_kwh', 0)
        cg = e.get('charge_from_grid_kwh', 0)
        surplus_kw = max(0.0, solar - load)
        a = action[:5]
        print(fmt.format(
            ts, a,
            f"{soc:.1f}", f"{pwr:.1f}",
            f"{imp:.3f}", f"{solar:.1f}",
            f"{net:.3f}",
            f"{cs:.3f}", f"{cg:.3f}",
            f"{surplus_kw:.1f}",
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect battery plan entries")
    parser.add_argument('--file', default='optimization_plan.json', help='Plan JSON file path')
    parser.add_argument('--summary', action='store_true', help='Show summary only (default)')
    parser.add_argument('--detail', action='store_true', help='Show detail table')
    parser.add_argument('--charging', action='store_true', help='Show charging analysis')
    parser.add_argument('--start', help='Filter: start timestamp (ISO-8601)')
    parser.add_argument('--end', help='Filter: end timestamp (ISO-8601)')
    parser.add_argument('--actions', help='Filter: comma-separated action types')
    parser.add_argument('--n', type=int, default=None, help='Max rows to show')
    args = parser.parse_args()

    entries = load_plan(args.file)
    entries = filter_entries(entries, args)

    if args.charging:
        show_charging(entries, args)
    elif args.detail:
        show_detail(entries, args)
    else:
        show_summary(entries, args)


if __name__ == '__main__':
    main()
