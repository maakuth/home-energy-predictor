"""Offline heating-regime and resistance-only benchmark diagnostics.

Only reads the supplied CSV. Never loads .env, contacts HA/SQL, trains a model,
or interprets virtual/supply temperature as measured accumulator temperature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def heating_regimes(index: pd.DatetimeIndex) -> pd.Series:
    """User-confirmed equipment timeline; uncertain changeover days are isolated."""
    local = pd.to_datetime(index, utc=True).tz_convert('Europe/Helsinki')
    dates = local.strftime('%Y-%m-%d')
    labels = np.select(
        [dates < '2026-08-20', dates < '2026-10-06', dates < '2026-10-07',
         dates < '2026-10-19', dates < '2026-10-20', dates < '2026-10-21'],
        ['old_pre_failure', 'old_intermittent', 'removal_transition',
         'resistance_only', 'installation_transition', 'integration_transition'],
        default='new_gshp',
    )
    return pd.Series(labels, index=index, name='heating_regime')


def analyze_heating(
    raw: pd.DataFrame,
    interval_minutes: int = 1,
    tail_hours: float = 24.0,
    reservoir_liters: float = 500.0,
    rise_margin_c: float = 0.3,
    tank_max_c: float = 80.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate unfiltered interval-mean shared heating power (W).

    Energy totals cover *observed* intervals, never extrapolated gaps. Fireplace
    candidates are 15-minute tank gains exceeding the maximum electrically
    explainable gain (assuming zero heat draw). They are not wood-heat estimates
    or proof of combustion. Absence of a candidate is not proof of no wood heat.
    """
    if interval_minutes < 1 or 15 % interval_minutes:
        raise ValueError('interval_minutes must divide 15')
    values = [tail_hours, reservoir_liters, rise_margin_c, tank_max_c]
    if not all(np.isfinite(v) for v in values) or tail_hours < 0 or reservoir_liters <= 0 or rise_margin_c < 0 or tank_max_c <= 20:
        raise ValueError('Invalid diagnostic parameters')
    if raw.empty or 'gshp_power' not in raw:
        raise ValueError('Nonempty raw data with shared gshp_power (W) is required')
    data = raw.copy()
    data.index = pd.to_datetime(data.index, utc=True).tz_convert('Europe/Helsinki')
    if data.index.has_duplicates:
        raise ValueError('Rejecting duplicate timestamps; resolve their provenance first')
    step = pd.Timedelta(minutes=interval_minutes)
    utc = data.index.tz_convert('UTC')
    if not (utc == utc.floor(step)).all():
        raise ValueError('Timestamps must be aligned to the declared interval')
    data = data.sort_index()
    start = data.index.min().normalize()
    end = data.index.max().normalize() + pd.DateOffset(days=1)
    index = pd.date_range(start, end, freq=step, inclusive='left')
    data = data.reindex(index)
    out = pd.DataFrame(index=index)
    out.index.name = 'timestamp'
    out['heating_regime'] = heating_regimes(index)
    resistance = out.heating_regime.eq('resistance_only')

    def signal(column: str) -> pd.Series:
        if column not in data:
            return pd.Series(np.nan, index=index, dtype=float)
        return pd.to_numeric(data[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

    power = signal('gshp_power')
    power = power.where(power.ge(0))
    tank = signal('accumulator_temp')
    tank = tank.where(tank.between(20, tank_max_c))
    weather = signal('outside_temp')
    weather = weather.where(weather.between(-50, 50))
    out['power_valid'] = power.notna()
    out['tank_valid'] = tank.notna()
    out['weather_valid'] = weather.notna()
    out['outside_temp'] = weather
    out['accumulator_temp'] = tank
    out['heating_electricity_kwh'] = power / 1000 * (interval_minutes / 60)

    window = 15 // interval_minutes
    capacity = reservoir_liters * 4.18 / 3600
    gain = (tank - tank.shift(window)) * capacity
    electrical = out.heating_electricity_kwh.shift(1).rolling(window).sum()
    continuous = tank.notna().rolling(window + 1).sum().eq(window + 1)
    same_regime = resistance.rolling(window + 1).sum().eq(window + 1)
    candidate = (gain > electrical + rise_margin_c * capacity) & continuous & same_regime
    out['fireplace_candidate'] = candidate
    # Include the warm-up window before detection, then a configurable storage tail.
    onset = candidate.iloc[::-1].rolling(window + 1, min_periods=1).max().iloc[::-1]
    tail = int(np.ceil(tail_hours * 60 / interval_minutes))
    out['fireplace_affected'] = onset.rolling(tail + 1, min_periods=1).max().astype(bool)
    out['thermal_candidate'] = (resistance & out.power_valid & out.tank_valid
                                & out.weather_valid & ~out.fireplace_affected)

    daily = pd.DataFrame(index=out.resample('D').size().index)
    daily.index.name = 'local_date'
    daily['heating_regime'] = out.heating_regime.resample('D').first()
    daily['heating_electricity_kwh'] = out.heating_electricity_kwh.resample('D').sum(min_count=1)
    for column, name in [('power_valid', 'power_coverage'), ('tank_valid', 'tank_coverage'),
                         ('weather_valid', 'weather_coverage'),
                         ('fireplace_affected', 'fireplace_affected_fraction'),
                         ('thermal_candidate', 'thermal_candidate_fraction')]:
        daily[name] = out[column].resample('D').mean()
    daily['outside_mean_c'] = weather.resample('D').mean()
    daily['fireplace_candidate_windows'] = candidate.resample('D').sum()
    # This mask is deliberately stricter than per-interval eligibility. Retain
    # complete days for later fitting; never splice selected hours into a day.
    daily['provisional_benchmark_day'] = daily.thermal_candidate_fraction.eq(1.0)
    return out, daily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Unfiltered raw interval-mean CSV; gshp_power is the shared meter in W')
    parser.add_argument('--output-dir', type=Path, help='Optional NEW directory for interval masks and daily report')
    parser.add_argument('--interval-minutes', type=int, default=1)
    parser.add_argument('--tail-hours', type=float, default=24)
    parser.add_argument('--reservoir-liters', type=float, default=500)
    parser.add_argument('--rise-margin-c', type=float, default=0.3)
    parser.add_argument('--tank-max-c', type=float, default=80)
    args = parser.parse_args()
    raw = pd.read_csv(args.input, index_col=0, low_memory=False)
    # Processed power has already been split/median-filtered, so reject it.
    if 'baseload_power' in raw or 'heating_power' in raw:
        parser.error('Use raw, unfiltered shared-meter data, not processed training data')
    intervals, daily = analyze_heating(raw, args.interval_minutes, args.tail_hours,
                                     args.reservoir_liters, args.rise_margin_c, args.tank_max_c)
    print(daily.to_string())
    print('\nSensitivity: provisional complete resistance-only days by storage-tail duration')
    for tail in sorted({0.0, 12.0, 24.0, 48.0, args.tail_hours}):
        _, sensitivity = analyze_heating(raw, args.interval_minutes, tail,
                                        args.reservoir_liters, args.rise_margin_c, args.tank_max_c)
        print(f'  {tail:g}h: {int(sensitivity.provisional_benchmark_day.sum())} days')
    print('\nPROVISIONAL only: undetected fireplace heat, stratification, DHW, AAHP and sensor freshness remain unquantified. No COP or demand parameters fitted.')
    if not daily.provisional_benchmark_day.any():
        print('No complete benchmark days identified. Do not relax thresholds merely to obtain a fit; review metering and tank dynamics first.')
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        intervals.to_csv(args.output_dir / 'heating_intervals.csv')
        daily.to_csv(args.output_dir / 'heating_daily.csv')


if __name__ == '__main__':
    main()
