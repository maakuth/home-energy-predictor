# Solar Forecast Calibration & Worst-Case (estimate10) Monitoring

Status: Phase 1 DONE (monitoring). Phase 2 (hedge) is planned but NOT implemented.

## 2026-08-24 calibration snapshot (14 days, 931 intervals)

From `solar.txt` (run on murrikka):

- `actual <= p50`: **49.4%** — the central forecast is essentially perfectly
  calibrated at the median; no systematic over-forecasting at p50.
- Mean bias **-0.280 kW** (over-optimistic), median |error| 0.593 kW → errors
  are asymmetric: usually fine, occasionally a large over-forecast on cloudy
  days (worst: 2026-08-24 with 68% of intervals below forecast; the original
  2026-08-12 incident day at 42%).
- The p10/p90 lines were **missing** from the first output despite 12 days of
  archived p10/p90 data (since 2026-08-12): a bug in
  `utils/solar_calibration.py` read `r[0]` (integer cid) instead of `r[1]`
  (column name) from `PRAGMA table_info`, so the percentile columns were
  silently never selected. Fixed 2026-08-24; the same change also computes
  p10/p90 fractions only over non-NaN rows (pre-Aug-12 rows no longer dilute
  them) and adds a per-day energy-shortfall report (kWh promised by p50 but
  not delivered — the quantity the Phase-2 decision actually needs).

## 2026-08-24 regenerated numbers (after the fix, n_p10=779)

- `actual <= p10`: **20.8%** (ideal ~10%) — the worst-case bound is violated
  ~2x too often. **The note's hedge criterion ("well above 10%, say 20-30%")
  is MET.** The Aug-12-style cheap-window miss is a recurring risk, not a
  one-off.
- `actual >= p90`: **24.6%** (ideal ~10%) — the best-case bound is also
  violated ~2.5x too often. The p10–p90 band is far too narrow overall:
  actuals fall outside it ~45% of the time instead of ~20%. Solcast
  underestimates its own uncertainty at this site — and these are the
  *shortest-horizon* (keep-last) forecasts, so the multi-hour-ahead numbers
  the planner acts on are likely worse.
- Daily shortfalls vs p50: worst day 16.77 kWh (2026-08-12, the incident
  day), with 5 days ≥ 9.6 kWh in two weeks — shortfalls of the same order as
  (or larger than) usable battery capacity, i.e. all-day underdelivery events.

**Conclusion:** calibration says a hedge is warranted *on risk grounds*.
Whether it is worth its recurring cost (a wrong hedge burns
price−export ≈ 0.057 €/kWh per displaced kWh, and with p50 well-calibrated
solar still beats the central forecast ~half the time) is an economics
question → proceed to the offline fixture backtest (Phase 2 evaluation below)
before enabling anything.

Caveat to keep in mind when reading the numbers: the calibration dedups
archived forecasts with keep-`last` per target interval, i.e. it evaluates the
*shortest-horizon* forecast. The planner acts on multi-hour-ahead forecasts
(e.g. the 04:00 cheap-window decision), whose errors are larger — so these
stats flatter the forecast relative to what the LP actually experiences.

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
