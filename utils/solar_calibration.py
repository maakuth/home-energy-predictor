from __future__ import annotations
"""
Solar forecast calibration analysis.

Solcast publishes three forecast percentiles per interval via ``detailedHourly``:
  - ``pv_estimate``   (50th percentile; the central / most-likely forecast)
  - ``pv_estimate10`` (10th percentile; worst case / very cloudy)
  - ``pv_estimate90`` (90th percentile; best case / clear sky)

This script compares those *forecast* bounds (archived in the ``predictions``
table as ``solar_forecast_kw`` / ``solar_forecast_p10_kw`` /
``solar_forecast_p90_kw``) against the *measured* solar production, to check how
well the percentiles are calibrated.

For a well-calibrated forecaster, actual solar should fall:
  - bottom (10th) pctile: ``actual <= p10`` about 10% of the time
  - top   (90th) pctile: ``actual >= p90`` about 10% of the time
  - symmetric around p50 (median): ``actual <= p50`` about 50% of the time

This tells us how often trusting the central forecast is dangerous, which
motivates Phase 2 (a cheap-window charging hedge keyed off the worst case).

Usage::

    venv/bin/python3 utils/solar_calibration.py [--days N] [--version X.Y.Z]
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from utils.sqlite_utils import get_db_connection, db_exists
from utils.db_utils import fetch_states_history

load_dotenv(override=True)


def load_archived_forecasts() -> pd.DataFrame:
    """Load the most recent archived solar forecast per target interval."""
    if not db_exists():
        print("⚠️ No database found; nothing to analyze.")
        return pd.DataFrame()

    try:
        conn = get_db_connection()
        # Guard against older dbs that predate the p10/p90 columns.
        # PRAGMA table_info rows are (cid, name, type, notnull, dflt, pk);
        # the column *name* is at index 1.
        cols = ["target_timestamp", "generated_at", "solar_forecast_kw"]
        found = [r[1] for r in conn.execute("PRAGMA table_info(predictions)")]
        for candidate in ("solar_forecast_p10_kw", "solar_forecast_p90_kw"):
            if candidate in found:
                cols.append(candidate)
        query = f"""
            SELECT {", ".join(cols)}
            FROM predictions
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return df
        # The same UTC instant is stored under different raw strings (mixed
        # timezone offsets / separators), so ROW_NUMBER-on-string would leave
        # duplicates once parsed. Dedup on the parsed instant instead, keeping
        # the most recently generated forecast.
        df['target_timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
        df = df.sort_values('generated_at')
        df = df.drop_duplicates(subset='target_timestamp', keep='last')
        return df.set_index('target_timestamp').sort_index()
    except Exception as e:
        print(f"Error reading archived predictions: {e}")
        return pd.DataFrame()


def fetch_actual_solar(days: int) -> pd.DataFrame:
    """Fetch measured solar production, resampled to 15-minute intervals."""
    entity = os.getenv('SOLAR_PRODUCTION_ENTITY', 'sensor.solarh_63038_real_power_kw')
    hist = fetch_states_history([entity], hours=days * 24)
    df = hist.get(entity)
    if df is None or df.empty:
        print("⚠️ No solar measurement history available (needs HA connectivity).")
        return pd.DataFrame()
    df = df.rename(columns={'state': 'solar_actual_kw'})
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp')
    return df[['solar_actual_kw']].resample('15min').mean()


def analyze(df: pd.DataFrame) -> dict[str, float]:
    """Compute solar forecast calibration metrics from a forecast/actual frame."""
    out: dict[str, float] = {}

    valid = df[df['solar_forecast_kw'].notna() & (df['solar_forecast_kw'] > 0)]
    if valid.empty:
        return out

    n = float(len(valid))
    out['n_intervals'] = n
    out['median_abs_error_kw'] = float((valid['solar_actual_kw'] - valid['solar_forecast_kw']).abs().median())
    out['mean_error_kw'] = float((valid['solar_actual_kw'] - valid['solar_forecast_kw']).mean())

    # Percentile calibration: how often does actual fall at/below each bound.
    out['actual_le_p50_frac'] = float((valid['solar_actual_kw'] <= valid['solar_forecast_kw']).sum() / n)

    # p10/p90 were archived only after Phase 1 shipped (2026-08-12); rows from
    # before that have NaN bounds and must not dilute the fractions.
    if 'solar_forecast_p10_kw' in valid.columns:
        p10 = valid.dropna(subset=['solar_forecast_p10_kw'])
        if not p10.empty:
            out['n_p10_intervals'] = float(len(p10))
            out['actual_le_p10_frac'] = float((p10['solar_actual_kw'] <= p10['solar_forecast_p10_kw']).sum() / len(p10))
    if 'solar_forecast_p90_kw' in valid.columns:
        p90 = valid.dropna(subset=['solar_forecast_p90_kw'])
        if not p90.empty:
            out['n_p90_intervals'] = float(len(p90))
            out['actual_ge_p90_frac'] = float((p90['solar_actual_kw'] >= p90['solar_forecast_p90_kw']).sum() / len(p90))

    return out


def daily_shortfall_kwh(df: pd.DataFrame) -> pd.Series:
    """Energy the p50 forecast promised but actuals did not deliver, per day.

    Integrates the positive part of (forecast - actual) over the 15-minute
    intervals (0.25 h each). This sizes an Aug-12-style miss in kWh, which is
    the quantity the Phase-2 cheap-window hedge decision needs.
    """
    valid = df[df['solar_forecast_kw'].notna() & (df['solar_forecast_kw'] > 0)]
    if valid.empty:
        return pd.Series(dtype=float)
    short = (valid['solar_forecast_kw'] - valid['solar_actual_kw']).clip(lower=0) * 0.25
    return short.groupby(short.index.date).sum()


def _print_results(metrics: dict[str, float], per_day: pd.DataFrame) -> None:
    print("\n=== Solar Forecast Calibration ===")
    if not metrics:
        print("No forecast/actual pairs to evaluate.")
        return
    print(f"Intervals evaluated: {metrics['n_intervals']:.0f}")
    print(f"Forecast median |error|  : {metrics['median_abs_error_kw']:.3f} kW")
    print(f"Forecast mean error bias : {metrics['mean_error_kw']:+.3f} kW "
          f"({'over-optimistic' if metrics['mean_error_kw'] < 0 else 'over-pessimistic' if metrics['mean_error_kw'] > 0 else 'unbiased'})")
    print(f"actual <= forecast (p50) : {metrics['actual_le_p50_frac']*100:.1f}%  (ideal ~50%)")
    if 'actual_le_p10_frac' in metrics:
        print(f"actual <= worst-case p10  : {metrics['actual_le_p10_frac']*100:.1f}%  (ideal ~10%, n={metrics['n_p10_intervals']:.0f})")
    if 'actual_ge_p90_frac' in metrics:
        print(f"actual >= best-case p90   : {metrics['actual_ge_p90_frac']*100:.1f}%  (ideal ~10%, n={metrics['n_p90_intervals']:.0f})")

    if not per_day.empty:
        print("\n=== Worst mis-forecast days (actual below p50) ===")
        daily = per_day.groupby(per_day.index.date).apply(
            lambda g: (g['solar_actual_kw'] < g['solar_forecast_kw']).mean()
        )
        daily = daily.sort_values(ascending=False).head(5)
        for day, frac in daily.items():
            print(f"  {day}: actual < forecast on {frac*100:.0f}% of intervals")

        shortfall = daily_shortfall_kwh(per_day).sort_values(ascending=False).head(5)
        if not shortfall.empty:
            print("\n=== Largest solar shortfalls vs p50 (energy not delivered) ===")
            for day, kwh in shortfall.items():
                print(f"  {day}: {kwh:.2f} kWh short of forecast")


def main() -> None:
    days = 14
    for arg in sys.argv[1:]:
        if arg.startswith('--days='):
            days = int(arg.split('=', 1)[1])

    forecasts = load_archived_forecasts()
    if forecasts.empty:
        return

    actuals = fetch_actual_solar(days)
    if actuals.empty:
        print("Calibration skipped (no actual solar history available).")
        return

    # Keep only intervals that overlap the fetched measurement window.
    start = actuals.index.min()
    end = actuals.index.max()
    forecasts = forecasts.loc[start:end]

    df = forecasts.join(actuals, how='inner')
    # Drop the most recent partial interval (actual may be incomplete).
    df = df.iloc[:-1]

    metrics = analyze(df)
    _print_results(metrics, df)


if __name__ == '__main__':
    main()
