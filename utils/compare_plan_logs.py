from __future__ import annotations
"""
Compare optimization plans (from run-frequent logs) with actual
battery/GSHP/solar behavior (from run-often logs).

Supports arbitrary syslog-style log files and date ranges.
"""

import re
import argparse
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional


def parse_syslog_ts(line: str) -> Optional[str]:
    """Extract 'MMM DD HH:MM:SS' from a syslog line, return normalized date."""
    m = re.match(r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})", line)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# 1. Parse run-often logs (recentlog.txt) — actual measurements every ~20s
# ---------------------------------------------------------------------------

class OftenLogParser:
    """Parses run-often.py logs into time-series of actual sensor readings."""

    SENSOR_PATTERNS: dict[str, re.Pattern] = {
        'soc':       re.compile(r"Battery SoC:\s*([\d.]+)%"),
        'batt_power': re.compile(r"Battery Power:\s*([\d-]+)W\s*\((.*?)\)"),
        'grid_power': re.compile(r"Grid Power:\s*([\d-]+)W"),
        'solar':     re.compile(r"Solar:\s*([\d.]+)kW"),
        'gshp':      re.compile(r"GSHP:\s*([\d.]+)kW"),
        'leaf':      re.compile(r"Leaf:\s*([\d.]+)kW"),
    }

    def __init__(self, path: str):
        self.path = path
        self.entries: list[dict[str, object]] = []
        self._parse()

    def _parse(self):
        with open(self.path) as f:
            for line in f:
                ts = parse_syslog_ts(line)
                if not ts:
                    continue
                entry: dict[str, object] = {'_ts': ts}
                found = False
                for key, pat in self.SENSOR_PATTERNS.items():
                    m = pat.search(line)
                    if m:
                        if key == 'batt_power':
                            entry['batt_power_w'] = float(m.group(1))
                            entry['batt_mode'] = m.group(2).strip()
                        else:
                            entry[key] = float(m.group(1))
                        found = True
                if found:
                    self.entries.append(entry)

    _MONTH_NUM = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    }

    @staticmethod
    def _sortable_key(key: str) -> str:
        """Convert 'Mon DD HH:MM' to 'MM-DD HH:MM' for chronological sorting."""
        parts = key.split()
        if len(parts) >= 3:
            month_num = OftenLogParser._MONTH_NUM.get(parts[0], 0)
            day = parts[1].zfill(2)
            return f"{month_num:02d}-{day} {parts[2]}"
        return key

    def resample(self, freq_minutes: int = 15) -> dict[str, list]:
        """Resample to regular intervals (e.g. 15-min), return dict of lists."""
        if not self.entries:
            return {}
        buckets: dict[str, list[dict]] = defaultdict(list)
        for e in self.entries:
            parts = e['_ts'].split()
            time_part = parts[-1]
            h, m, _ = time_part.split(':')
            block_min = (int(m) // freq_minutes) * freq_minutes
            key = f"{parts[0]} {parts[1]} {h}:{block_min:02d}"
            buckets[key].append(e)

        result: dict[str, list] = {
            'time': [], 'soc': [], 'batt_power_w': [], 'grid_power_w': [],
            'solar_kw': [], 'gshp_kw': [], 'leaf_kw': [],
        }
        for key in sorted(buckets, key=lambda k: self._sortable_key(k)):
            group = buckets[key]
            result['time'].append(key)
            result['soc'].append(_median([e.get('soc') for e in group if 'soc' in e]))
            result['batt_power_w'].append(_median([e.get('batt_power_w') for e in group if 'batt_power_w' in e]))
            result['grid_power_w'].append(_median([e.get('grid_power') for e in group if 'grid_power' in e]))
            result['solar_kw'].append(_median([e.get('solar') for e in group if 'solar' in e]))
            result['gshp_kw'].append(_median([e.get('gshp') for e in group if 'gshp' in e]))
            result['leaf_kw'].append(_median([e.get('leaf') for e in group if 'leaf' in e]))
        return result

    def hourly_avg(self) -> dict[str, list]:
        """Return hourly averages as dict of lists."""
        buckets: dict[str, list[dict]] = defaultdict(list)
        for e in self.entries:
            parts = e['_ts'].split()
            h = parts[-1].split(':')[0]
            key = f"{parts[0]} {parts[1]} {h}:00"
            buckets[key].append(e)
        result: dict[str, list] = {
            'time': [], 'soc': [], 'batt_power_w': [], 'grid_power_w': [],
            'solar_kw': [], 'gshp_kw': [],
        }
        for key in sorted(buckets, key=lambda k: self._sortable_key(k)):
            group = buckets[key]
            result['time'].append(key)
            result['soc'].append(_median([e.get('soc') for e in group if 'soc' in e]))
            result['batt_power_w'].append(_median([e.get('batt_power_w') for e in group if 'batt_power_w' in e]))
            result['grid_power_w'].append(_median([e.get('grid_power') for e in group if 'grid_power' in e]))
            result['solar_kw'].append(_median([e.get('solar') for e in group if 'solar' in e]))
            result['gshp_kw'].append(_median([e.get('gshp') for e in group if 'gshp' in e]))
        return result


# ---------------------------------------------------------------------------
# 2. Parse run-frequent logs (recentlog-frequent.txt) — optimization plans
# ---------------------------------------------------------------------------

_PLAN_ENTRY_RE = re.compile(r"(\d{2})-(\d{2})\s+(\d{2}:\d{2})")


def _parse_plan_table(lines: list[str], start_idx: int) -> Optional[list[dict]]:
    """Parse one optimisation plan table from the log lines starting at start_idx.
    
    Returns list of {time, baseload, gshp, grid, solar, soc} or None.
    """
    table_started = False
    entries = []
    for i in range(start_idx, min(start_idx + 300, len(lines))):
        line = lines[i]
        if "Time" in line and "SOC%" in line and "|" in line:
            table_started = True
            continue
        if not table_started:
            continue
        if not line.strip():
            continue
        # Detect end of plan: a line that is present in the log but is not a
        # plan entry (no pipe, or no MM-DD HH:MM | pattern)
        if "|" not in line:
            break
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            break
        # Skip separator lines (dashed lines after header)
        if "---" in parts[0] and "---" in parts[1]:
            continue
        m = _PLAN_ENTRY_RE.search(parts[0])
        if not m:
            break
        try:
            entry = {
                'time': f"{m.group(1)}-{m.group(2)} {m.group(3)}",
                'baseload': float(parts[1]) if parts[1] else None,
                'gshp_kw': float(parts[2]) if parts[2] else None,
                'grid_kw': float(parts[3]) if parts[3] else None,
                'solar_kw': float(parts[4]) if parts[4] else None,
                'soc': float(parts[5]) if parts[5] else None,
            }
            intent_parts = parts[6].split() if len(parts) > 6 else []
            entry['battery_intent'] = intent_parts[0] if intent_parts else ''
            entry['gshp_intent'] = parts[7].strip() if len(parts) > 7 else ''
            entries.append(entry)
        except (ValueError, IndexError):
            continue
    return entries if entries else None


class FrequentLogParser:
    """Parses run-frequent.sh logs into a list of optimisation plans."""

    def __init__(self, path: str):
        self.path = path
        self.plans: list[dict] = []
        self.predictions: list[dict] = []
        self.extraction_info: list[dict] = []
        self._parse()

    def _parse(self):
        with open(self.path) as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect optimisation plan start
            if "Optimization Plan from" in line:
                plan_ts = None
                for j in range(max(0, i - 10), i):
                    tm = re.match(r"(\w{3}\s+\d{1,2}\s+(\d{2}:\d{2}:\d{2}))", lines[j])
                    if tm:
                        plan_ts = tm.group(1)
                table = _parse_plan_table(lines, i + 1)
                if table:
                    self.plans.append({
                        'timestamp': plan_ts,
                        'entries': table,
                    })

            # Battery live SoC
            m = re.search(r"Battery live SoC from HA:\s*([\d.]+)%", line)
            if m:
                self.predictions.append({
                    'type': 'live_soc',
                    'value': float(m.group(1)),
                })

            # Blend info
            m = re.search(r"Blending XGBoost with SARIMA \((\d+)/(\d+) weight\)", line)
            if m:
                self.predictions.append({
                    'type': 'blend_weights',
                    'xgb': int(m.group(1)),
                    'sarima': int(m.group(2)),
                })

            # Extract prediction count
            m = re.search(r"Generated (\d+) predictions at (\d+)-minute resolution", line)
            if m:
                self.predictions.append({
                    'type': 'generated',
                    'count': int(m.group(1)),
                    'resolution': int(m.group(2)),
                })

            # Market prices
            m = re.search(r"Import price:\s*([\d.]+)\s*EUR/kWh\s*\|\s*Export price:\s*([\d.]+)\s*EUR/kWh", line)
            if m:
                self.predictions.append({
                    'type': 'market_prices',
                    'import': float(m.group(1)),
                    'export': float(m.group(2)),
                })

            i += 1


# ---------------------------------------------------------------------------
# 3. Comparison engine
# ---------------------------------------------------------------------------

def _median(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def build_comparison(
    often: OftenLogParser,
    frequent: FrequentLogParser,
    freq_minutes: int = 15,
    plan_index: int = 0,
    day_offset: int = 0,
) -> list[dict]:
    """Compare predictions from one plan against actual measurements.
    
    Args:
        often: Parsed run-often log
        frequent: Parsed run-frequent log
        freq_minutes: Resample interval for actuals
        plan_index: Which plan to use as baseline (0 = first)
        day_offset: If >0, match plan entries N days ahead to actuals
    
    Returns a list of comparison rows with planned vs actual SoC,
    GSHP, solar, and grid for each plan entry time.
    """
    actual_resampled = often.resample(freq_minutes=freq_minutes)
    if not actual_resampled:
        return []

    # Index actuals by time key
    actual_by_time: dict[str, dict] = {}
    for i, t in enumerate(actual_resampled['time']):
        actual_by_time[t] = {
            'soc': actual_resampled['soc'][i],
            'gshp_kw': actual_resampled['gshp_kw'][i],
            'solar_kw': actual_resampled['solar_kw'][i],
            'grid_w': actual_resampled['grid_power_w'][i],
            'batt_power_w': actual_resampled['batt_power_w'][i],
        }

    results = []

    if not frequent.plans:
        return results

    if plan_index >= len(frequent.plans):
        print(f"Warning: plan index {plan_index} out of range (0-{len(frequent.plans)-1}), using plan 0")
        plan_index = 0

    # Build month-independent actual index: { "MM-DD HH:MM": data }
    actual_num_index: dict[str, dict] = {}
    for key, data in actual_by_time.items():
        parts = key.split()
        if len(parts) >= 3:
            month_num = OftenLogParser._MONTH_NUM.get(parts[0], 0)
            day = parts[1].zfill(2)
            num_key = f"{month_num:02d}-{day} {parts[2]}"
            actual_num_index[num_key] = data

    plan = frequent.plans[plan_index]
    for entry in plan['entries']:
        entry_key = entry['time']  # Already "MM-DD HH:MM"
        # Adjust day for multi-day plans
        date_part = entry['time'][:5]
        day_int = int(date_part[3:])
        adj_day = day_int + day_offset
        adj_key = f"{date_part[:3]}{adj_day:02d} {entry['time'][6:]}"

        # Try exact match first, then day-adjusted match
        actual = actual_num_index.get(entry_key) or actual_num_index.get(adj_key)

        if actual and actual['soc'] is not None:
            diff = actual['soc'] - entry['soc'] if entry['soc'] is not None else None
            gshp_diff = (actual['gshp_kw'] - entry['gshp_kw']) if (actual['gshp_kw'] is not None and entry['gshp_kw'] is not None) else None
            solar_diff = (actual['solar_kw'] - entry['solar_kw']) if (actual['solar_kw'] is not None and entry['solar_kw'] is not None) else None
        else:
            diff = None
            gshp_diff = None
            solar_diff = None

        results.append({
            'time': entry['time'],
            'plan_soc': entry['soc'],
            'actual_soc': actual['soc'] if actual else None,
            'soc_diff': diff,
            'plan_gshp': entry['gshp_kw'],
            'actual_gshp': actual['gshp_kw'] if actual else None,
            'gshp_diff': gshp_diff,
            'plan_solar': entry['solar_kw'],
            'actual_solar': actual['solar_kw'] if actual else None,
            'solar_diff': solar_diff,
            'plan_grid': entry['grid_kw'],
            'plan_baseload': entry['baseload'],
            'battery_intent': entry.get('battery_intent', ''),
        })

    return results


# ---------------------------------------------------------------------------
# 4. Output formatters
# ---------------------------------------------------------------------------

def print_summary(comparison: list[dict]) -> None:
    """Print high-level summary statistics."""
    if not comparison:
        print("No comparison data available.")
        return

    soc_diffs = [r['soc_diff'] for r in comparison if r['soc_diff'] is not None]
    if not soc_diffs:
        print("No SoC data to compare.")
        return

    mae = sum(abs(d) for d in soc_diffs) / len(soc_diffs)
    bias = sum(soc_diffs) / len(soc_diffs)
    max_dev = max(abs(d) for d in soc_diffs)
    min_soc = min(r['plan_soc'] for r in comparison if r['plan_soc'] is not None)
    max_soc = max(r['plan_soc'] for r in comparison if r['plan_soc'] is not None)
    actual_min = min(r['actual_soc'] for r in comparison if r['actual_soc'] is not None)
    actual_max = max(r['actual_soc'] for r in comparison if r['actual_soc'] is not None)

    # GSHP analysis
    gshp_diffs = [r['gshp_diff'] for r in comparison if r['gshp_diff'] is not None]
    gshp_mae = sum(abs(d) for d in gshp_diffs) / len(gshp_diffs) if gshp_diffs else 0
    gshp_bias = sum(gshp_diffs) / len(gshp_diffs) if gshp_diffs else 0

    # Compute how many times GSHP was predicted ON vs actually ON (>0.5 kW threshold)
    plan_gshp_on = sum(1 for r in comparison if r['plan_gshp'] is not None and r['plan_gshp'] > 0.5)
    actual_gshp_on = sum(1 for r in comparison if r['actual_gshp'] is not None and r['actual_gshp'] > 0.5)

    print(f"{'='*60}")
    print(f"  PLAN vs REALITY COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Comparison points:   {len(comparison)}")
    print(f"  Time range:          {comparison[0]['time']}  →  {comparison[-1]['time']}")
    print()
    print(f"  ── SoC ──")
    print(f"  MAE (SoC %):         {mae:>7.2f}%")
    print(f"  Bias (SoC %):        {bias:>+7.2f}%  ({'over-discharged' if bias < 0 else 'over-charged'})")
    print(f"  Max deviation:       {max_dev:>7.2f}%")
    print(f"  Plan SoC range:      {min_soc:>5.1f}%  →  {max_soc:>5.1f}%")
    print(f"  Actual SoC range:    {actual_min:>5.1f}%  →  {actual_max:>5.1f}%")
    print()
    print(f"  ── GSHP ──")
    print(f"  GSHP MAE:            {gshp_mae:>7.2f} kW")
    print(f"  GSHP Bias:           {gshp_bias:>+7.2f} kW  ({'over-predicted' if gshp_bias > 0 else 'under-predicted'})")
    print(f"  Plan ON intervals:   {plan_gshp_on:>4d}  (>{'>0.5 kW'})")
    print(f"  Actual ON intervals: {actual_gshp_on:>4d}  (>{'>0.5 kW'})")

    # Print hourly breakdown
    print()
    print(f"  ── Hourly SoC Deviation ──")
    print(f"  {'Hour':<10} {'Avg Diff':>10} {'Min Diff':>10} {'Max Diff':>10} {'Samples':>8}")
    print(f"  {'-'*40}")
    hourly: dict[str, list] = defaultdict(list)
    for r in comparison:
        if r['soc_diff'] is not None:
            h = r['time'][6:11]  # "HH:MM" -> "HH:00" approximate
            hourly[r['time'][6:8]].append(r['soc_diff'])
    for h in sorted(hourly):
        vals = hourly[h]
        avg = sum(vals) / len(vals)
        print(f"  {h:>2}:00        {avg:>+8.2f}%  {min(vals):>+8.2f}%  {max(vals):>+8.2f}%  {len(vals):>6d}")


def print_detail(comparison: list[dict], n: Optional[int] = None) -> None:
    """Print detailed time-by-time comparison table."""
    if not comparison:
        print("No comparison data available.")
        return

    rows = comparison[:n] if n else comparison

    fmt = "{:<14s} {:>7s} {:>7s} {:>7s}  {:>6s} {:>6s}  {:>6s} {:>6s}  {:>6s}"
    print(fmt.format("Time", "P_SoC", "A_SoC", "Diff", "P_GSHP", "A_GSHP", "P_Sol", "A_Sol", "Intent"))
    print("-" * 85)

    entry_count = 0
    for r in rows:
        if r['plan_soc'] is None and r['actual_soc'] is None:
            continue
        entry_count += 1
        soc_s = f"{r['plan_soc']:.1f}" if r['plan_soc'] is not None else "-"
        act_s = f"{r['actual_soc']:.1f}" if r['actual_soc'] is not None else "-"
        diff_s = f"{r['soc_diff']:+.1f}" if r['soc_diff'] is not None else "-"
        pg = f"{r['plan_gshp']:.1f}" if r['plan_gshp'] is not None else "-"
        ag = f"{r['actual_gshp']:.1f}" if r['actual_gshp'] is not None else "-"
        ps = f"{r['plan_solar']:.2f}" if r['plan_solar'] is not None else "-"
        as_ = f"{r['actual_solar']:.2f}" if r['actual_solar'] is not None else "-"
        intent = r['battery_intent'][:10] if r['battery_intent'] else "-"
        marker = " <--" if r['soc_diff'] is not None and abs(r['soc_diff']) > 10 else ""
        print(f"{r['time']:<14s} {soc_s:>7s} {act_s:>7s} {diff_s:>7s}  {pg:>6s} {ag:>6s}  {ps:>6s} {as_:>6s}  {intent:<6s}{marker}")

    if n and entry_count >= n:
        print(f"... ({len(comparison) - n} more entries, use --all or increase --n)")


def print_plan_evolution(comparison: list[dict], n: Optional[int] = None) -> None:
    """Show how plans evolved over time by comparing first plan vs later plans."""
    # This is best done by FrequentLogParser which stores all plans
    pass


def print_plans_list(frequent: FrequentLogParser) -> None:
    """List available plans with their timestamps and entry counts."""
    print(f"\nAvailable plans ({len(frequent.plans)} total):")
    print(f"  {'#':<4} {'Timestamp':<22} {'Entries':<8} {'Range'}")
    print(f"  {'-'*60}")
    for idx, plan in enumerate(frequent.plans):
        ts = plan.get('timestamp', 'unknown')
        entries = len(plan['entries'])
        if entries > 0:
            rng = f"{plan['entries'][0]['time']} → {plan['entries'][-1]['time']}"
        else:
            rng = "empty"
        print(f"  {idx:<4} {str(ts):<22} {entries:<8} {rng}")
    print()


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare battery optimisation plans vs actual measurements from logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --summary\n"
            "  %(prog)s --detail --n 24\n"
            "  %(prog)s --often recentlog.txt --frequent recentlog-frequent.txt --summary\n"
            "  %(prog)s --gshp --n 20\n"
            "  %(prog)s --plan 0 --detail\n"
            "  %(prog)s --plans                          # list available plans\n"
            "  %(prog)s --often older-log.txt --frequent older-log-frequent.txt --stats\n"
        )
    )
    parser.add_argument('--often', default='recentlog.txt',
                        help='Path to run-often log (default: recentlog.txt)')
    parser.add_argument('--frequent', default='recentlog-frequent.txt',
                        help='Path to run-frequent log (default: recentlog-frequent.txt)')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics (default)')
    parser.add_argument('--detail', action='store_true',
                        help='Show time-by-time comparison table')
    parser.add_argument('--gshp', action='store_true',
                        help='Show GSHP-focused analysis')
    parser.add_argument('--solar', action='store_true',
                        help='Show solar-focused analysis')
    parser.add_argument('--plan', type=int, default=0,
                        help='Plan index to compare against (default: 0 = first)')
    parser.add_argument('--plans', action='store_true',
                        help='List available plans and exit')
    parser.add_argument('--n', type=int, default=None,
                        help='Max rows to show in detail mode')
    parser.add_argument('--start', help='Filter: start time (HH:MM)')
    parser.add_argument('--end', help='Filter: end time (HH:MM)')
    parser.add_argument('--hourly', action='store_true',
                        help='Show hourly average comparison')
    parser.add_argument('--stats', action='store_true',
                        help='Show detailed statistics (MAE, bias, RMSE)')
    args = parser.parse_args()

    print(f"Parsing run-often log: {args.often}")
    often = OftenLogParser(args.often)
    print(f"  Read {len(often.entries)} measurements")
    print(f"Parsing run-frequent log: {args.frequent}")
    frequent = FrequentLogParser(args.frequent)
    print(f"  Found {len(frequent.plans)} plan(s)")
    if not frequent.plans:
        print("No plans found in frequent log.")
        sys.exit(1)

    if args.plans:
        print_plans_list(frequent)
        return

    comparison = build_comparison(often, frequent, plan_index=args.plan)

    # Filter by time range if requested
    if args.start or args.end:
        filtered = []
        for r in comparison:
            t = r['time'][6:11]  # HH:MM
            if args.start and t < args.start:
                continue
            if args.end and t > args.end:
                continue
            filtered.append(r)
        comparison = filtered

    if args.detail:
        print_detail(comparison, n=args.n)
    elif args.gshp:
        print_gshp_analysis(comparison, args.n)
    elif args.solar:
        print_solar_analysis(comparison, args.n)
    elif args.hourly:
        print_hourly_comparison(comparison)
    elif args.stats:
        print_stats(comparison)
    else:
        print_summary(comparison)


def print_gshp_analysis(comparison: list[dict], n: Optional[int] = None) -> None:
    """Print GSHP-focused analysis."""
    rows = [r for r in comparison if r['plan_gshp'] is not None or r['actual_gshp'] is not None]
    rows = rows[:n] if n else rows

    if not rows:
        print("No GSHP data.")
        return

    fmt = "{:<14s} {:>7s} {:>7s} {:>7s}  {:>7s} {:>7s} {:>7s}"
    print(fmt.format("Time", "P_GSHP", "A_GSHP", "Diff", "P_Solar", "A_Solar", "P_Grid"))
    print("-" * 70)

    for r in rows:
        pg = f"{r['plan_gshp']:.2f}" if r['plan_gshp'] is not None else "-"
        ag = f"{r['actual_gshp']:.2f}" if r['actual_gshp'] is not None else "-"
        gd = f"{r['gshp_diff']:+.2f}" if r['gshp_diff'] is not None else "-"
        ps = f"{r['plan_solar']:.2f}" if r['plan_solar'] is not None else "-"
        as_ = f"{r['actual_solar']:.2f}" if r['actual_solar'] is not None else "-"
        pgrd = f"{r['plan_grid']:.2f}" if r['plan_grid'] is not None else "-"
        print(f"{r['time']:<14s} {pg:>7s} {ag:>7s} {gd:>7s}  {ps:>7s} {as_:>7s}  {pgrd:>7s}")

    # Summary
    plan_on = sum(1 for r in rows if r['plan_gshp'] is not None and r['plan_gshp'] > 0.5)
    actual_on = sum(1 for r in rows if r['actual_gshp'] is not None and r['actual_gshp'] > 0.5)
    plan_kwh = sum((r['plan_gshp'] or 0) * 0.25 for r in rows)
    actual_kwh = sum((r['actual_gshp'] or 0) * 0.25 for r in rows)
    print()
    print(f"Plan GSHP ON intervals:   {plan_on}")
    print(f"Actual GSHP ON intervals: {actual_on}")
    print(f"Plan GSHP energy:         {plan_kwh:.1f} kWh")
    print(f"Actual GSHP energy:       {actual_kwh:.1f} kWh")
    if plan_kwh > 0:
        print(f"Over-prediction factor:   {plan_kwh / max(actual_kwh, 0.01):.1f}x")


def print_solar_analysis(comparison: list[dict], n: Optional[int] = None) -> None:
    """Print solar-focused analysis."""
    rows = [r for r in comparison if r['plan_solar'] is not None or r['actual_solar'] is not None]
    rows = rows[:n] if n else rows

    if not rows:
        print("No solar data.")
        return

    fmt = "{:<14s} {:>7s} {:>7s} {:>7s}  {:>7s} {:>7s}"
    print(fmt.format("Time", "P_Solar", "A_Solar", "Diff", "P_Grid", "A_SoC"))
    print("-" * 60)

    for r in rows:
        ps = f"{r['plan_solar']:.2f}" if r['plan_solar'] is not None else "-"
        as_ = f"{r['actual_solar']:.2f}" if r['actual_solar'] is not None else "-"
        sd = f"{r['solar_diff']:+.2f}" if r['solar_diff'] is not None else "-"
        pgrd = f"{r['plan_grid']:.2f}" if r['plan_grid'] is not None else "-"
        asc = f"{r['actual_soc']:.1f}" if r['actual_soc'] is not None else "-"
        print(f"{r['time']:<14s} {ps:>7s} {as_:>7s} {sd:>7s}  {pgrd:>7s} {asc:>7s}")


def print_hourly_comparison(comparison: list[dict]) -> None:
    """Show hourly average comparison."""
    hourly: dict[str, dict] = defaultdict(lambda: {'soc_diffs': [], 'gshp_diffs': [], 'solar_diffs': [], 'count': 0})
    for r in comparison:
        h = r['time'][6:8]
        if r['soc_diff'] is not None:
            hourly[h]['soc_diffs'].append(r['soc_diff'])
        if r['gshp_diff'] is not None:
            hourly[h]['gshp_diffs'].append(r['gshp_diff'])
        hourly[h]['count'] += 1

    fmt = "{:<6s} {:>8s} {:>10s} {:>10s} {:>10s} {:>8s}"
    print(fmt.format("Hour", "SoC Diff", "GSHP Diff", "Plan SoC", "Actual SoC", "Count"))
    print("-" * 60)

    for h in sorted(hourly):
        d = hourly[h]
        soc_avg = sum(d['soc_diffs']) / len(d['soc_diffs']) if d['soc_diffs'] else 0
        gshp_avg = sum(d['gshp_diffs']) / len(d['gshp_diffs']) if d['gshp_diffs'] else 0
        plan_soc_at = None
        actual_soc_at = None
        for r in comparison:
            if r['time'][6:8] == h:
                plan_soc_at = r['plan_soc']
                actual_soc_at = r['actual_soc']
                break
        print(f"{h:>2}:00  {soc_avg:>+8.2f}% {gshp_avg:>+10.2f}kW {str(plan_soc_at or '-'):>10s} {str(actual_soc_at or '-'):>10s} {d['count']:>6d}")


def print_stats(comparison: list[dict]) -> None:
    """Print detailed statistics."""
    if not comparison:
        return

    soc_diffs = [r['soc_diff'] for r in comparison if r['soc_diff'] is not None]
    gshp_diffs = [r['gshp_diff'] for r in comparison if r['gshp_diff'] is not None]
    solar_diffs = [r['solar_diff'] for r in comparison if r['solar_diff'] is not None]

    if soc_diffs:
        import math
        n = len(soc_diffs)
        mae = sum(abs(d) for d in soc_diffs) / n
        bias = sum(soc_diffs) / n
        rmse = math.sqrt(sum(d * d for d in soc_diffs) / n)
        max_abs = max(abs(d) for d in soc_diffs)

        print("=== SoC Statistics ===")
        print(f"  Count:      {n}")
        print(f"  MAE:        {mae:.2f}%")
        print(f"  RMSE:       {rmse:.2f}%")
        print(f"  Bias:       {bias:+.2f}%")
        print(f"  Max |diff|: {max_abs:.2f}%")
        print(f"  StdDev:     {math.sqrt(sum((d - bias)**2 for d in soc_diffs) / n):.2f}%")

    if gshp_diffs:
        n_g = len(gshp_diffs)
        gshp_mae = sum(abs(d) for d in gshp_diffs) / n_g
        gshp_bias = sum(gshp_diffs) / n_g
        print(f"\n=== GSHP Statistics ===")
        print(f"  Count:      {n_g}")
        print(f"  MAE:        {gshp_mae:.2f} kW")
        print(f"  Bias:       {gshp_bias:+.2f} kW")

    if solar_diffs:
        n_s = len(solar_diffs)
        solar_mae = sum(abs(d) for d in solar_diffs) / n_s
        solar_bias = sum(solar_diffs) / n_s
        print(f"\n=== Solar Statistics ===")
        print(f"  Count:      {n_s}")
        print(f"  MAE:        {solar_mae:.2f} kW")
        print(f"  Bias:       {solar_bias:+.2f} kW")


if __name__ == '__main__':
    main()
