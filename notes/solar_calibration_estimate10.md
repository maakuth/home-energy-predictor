# Solar Forecast Calibration & Worst-Case (estimate10) Monitoring

Status: Phase 1 DONE (monitoring). Phase 2 (hedge) is planned but NOT implemented.

## Motivation

Observed on 2026-08-12: the LP battery planner (nemotron-linprog) did not charge
during the cheap morning hours (spot ~0.059-0.068 EUR/kWh) because it trusted the
central Solcast forecast (`pv_estimate`, 50th percentile), which said daytime
solar would fill the battery for free. Actual solar underdelivered, the battery
crawled at ~24% through the cheap window, and the system had to grid-charge in
the afternoon at ~0.079-0.088 instead. The LP's logic was internally correct —
buying grid energy at 0.059 just displaces free solar that would otherwise be
exported at ~0.003 — but it rests entirely on the central solar forecast being
reliable.

Solcast publishes three percentiles per interval in the forecast entities'
`detailedHourly` array:
- `pv_estimate`   — 50th pct (central / most-likely)
- `pv_estimate10` — 10th pct (worst case / very cloudy)
- `pv_estimate90` — 90th pct (best case / clear)

Before this work only `pv_estimate` was used; 10/90 were dropped.

## What was implemented (Phase 1 — monitoring only)

No behavior change, no VERSION bump. The goal is to collect enough data to decide
whether and how to hedge.

1. **predict_future.py**
   - `generate_inference_data` reads `pv_estimate10`/`pv_estimate90` per interval
     (falling back to the nominal value when the field is missing / NaN).
   - `solar_forecast_p10` / `solar_forecast_p90` added to each inference row and
     to the `future_predictions.json` results.
   - `predictions` table: new columns `solar_forecast_p10_kw`, `solar_forecast_p90_kw`
     (CREATE TABLE + guarded ALTER migration, same pattern as existing columns).

2. **optimize_plan.py**
   - `load_predictions` returns the p10/p90 solar series (fallback to p50 when the
     JSON lacks the fields, so old files keep working).
   - Plan entries carry `solar_forecast_p10_kw` / `solar_forecast_p90_kw`.
   - `predictions` archive: new columns populated.

3. **utils/solar_calibration.py** (new)
   - Compares archived forecasts (p50/p10/p90) against measured solar (HA history,
     resampled to 15 min).
   - Reports: fraction of intervals where actual <= p10 (ideal ~10%), actual >= p90
     (ideal ~10%), actual <= p50 (ideal ~50%), plus bias and median |error|, and the
     worst "over-optimistic" days.
   - Usage: `venv/bin/python3 utils/solar_calibration.py [--days=N]`
   - Requires HA connectivity for actuals; degrades gracefully without it.

## How to read the calibration output

- A well-calibrated Solcast should land roughly: `actual <= p10` ≈ 10%, `actual >= p90` ≈ 10%.
- If `actual <= p10` is well above 10% (say 20-30%), the central forecast is
  systematically optimistic in worst-case weather → the cheap-window miss is a real,
  recurring risk and a hedge is warranted.
- Negative `mean_error` = over-optimistic forecast (actuals tend below p50).

## Phase 2 — Cheap-window charging hedge (PLANNED, NOT implemented)

Goal: don't blindly trust p50 when filling the battery during a cheap window.

Options (evaluate with a few weeks of calibration data first):
- **Blend** fed to the battery LP only: `solar_hedged = alpha*p50 + (1-alpha)*p10`,
  tunable `BATTERY_SOLAR_HEDGE_ALPHA` (default 1.0 = current behaviour). Keep the
  XGBoost feature and GSHP planning on p50 (they are trained/calibrated on it).
- **Conditional insurance**: when entering a cheap window, if worst-case solar over
  the horizon < required fill AND price is cheap (below threshold), buy grid energy.
- **Two-solution LP**: solve with p50 and p10, adopt max(grid-charge) in the cheap
  window.

Caveats:
- Hedge cost = (price − export price) per displaced kWh + extra cycling. In summer
  with reliable solar this recurs most days; the calibration data decides whether it's
  worth it.
- Backtest is possible offline before enabling: the raw fixtures in
  `java-battery-planner/fixtures/{jan,jul,may,oct}.json` already contain
  `pv_estimate10`/`pv_estimate90` plus measured solar. Use the replay harness
  (`tests/battery_planner_replay.py`).
- Enabling a hedge changes inference/planning logic → MINOR version bump per AGENTS.md.

## Files touched (Phase 1)

- predict_future.py
- optimize_plan.py
- utils/solar_calibration.py (new)
- tests/test_predict_future.py
- tests/test_optimize_plan.py
