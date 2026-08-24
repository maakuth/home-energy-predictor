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
- Enabling a hedge changes inference/planning logic → MINOR version bump per AGENTS.md.

## Phase 2 backtest — round 1 (2026-08-24): fixtures can't answer the question

Implemented the blend behind `BATTERY_SOLAR_HEDGE_ALPHA` (default 1.0 = off;
LP-only: `solar_hedged = alpha*p50 + (1-alpha)*p10` inside nemotron-linprog;
XGBoost/GSHP keep p50; p10 flows via `BatteryPlannerContext['solar_p10_kwh']`,
also wired through the replay harness). Then ran full-length seasonal replays
(jan/jul/may/oct) with synthetic p10 sampled from the empirical day-ahead
p10/p50 distribution (45k real pairs from the production archive, Aug 2026).

**Key finding: the seasonal fixtures cannot evaluate the hedge.** Their
archived "forecasts" equal the measured actuals *exactly* on every solar
interval (actual < p50 on 0.0% of daytime intervals in all four) — they are
perfect-foresight fixtures. The Aug-12 failure mode (trusting an optimistic
forecast) therefore never occurs in replay, and any forecast hedge can only
add cost. The note's earlier claim that the fixtures contain
`pv_estimate10`/`pv_estimate90` was also wrong — they carry p50 only.

Measured anyway (pure **cost side** of the hedge, i.e. the insurance premium
when forecasts are never wrong): alpha=0.85 costs **+1.0 EUR/week**,
alpha=0.70 **+1.85 EUR/week**, uniform across seasons; worst-day cost also
slightly *rises* (the blunt every-day blend buys unneeded morning energy even
on days that later underdeliver). Zero SoC violations; cycling slightly down.

The **benefit side** (avoided expensive purchases on real underdelivery days,
when p10 was informatively low) is not measurable with perfect-foresight
fixtures. A rho-parameterized synthetic-correlation mode exists in
`bench_solar_hedge.py`, but with zero realized shortfall in the fixtures it
degenerates to noise.

**Decisive next step:** a *real* fixture for the incident window —
`dump_battery_data.py` now exports `solar_forecast_p10_kw/p90_kw`, so on
murrikka:

    venv/bin/python3 dump_battery_data.py --start "2026-08-12" --end "2026-08-23" \
        --output tests/fixtures/aug.pkl --verbose

then `venv/bin/python3 bench_solar_hedge.py --alphas 1.0,0.85,0.7 --fixtures aug
--no-inject` (the replay harness picks up real p10 from the archive
automatically; `--no-inject` keeps the fixture's real p10 instead of the
synthetic model). This replays the actual Aug 12 incident with real
forecasts, real p10 and real prices.

Extra stat worth knowing (production archive, Aug 2026): the **day-ahead**
(20-28h lag) p10/p50 ratio has median **0.31** [p5 0.10, p95 0.69] — much
wider than the keep-last view (median 0.53) that `solar_calibration.py`
evaluates. The uncertainty the planner faces at decision time is roughly
"worst case ≈ ⅓ of central", and the day-ahead distribution is what
`bench_solar_hedge.py`'s ratio model
(`tests/fixtures/solar_p10_ratio_model.json`) captures.

## Phase 2 backtest — round 2 (2026-08-24): real August window

Dumped a real fixture for the incident window (`tests/fixtures/aug.pkl`) and
merged real p10/p90 into it from the production archive. **Two lessons:**

1. **HA recorder retention (~10 days) bites**: dumped on Aug 24 with
   `--start 2026-08-12`, but measurements only go back to Aug 13 11:15 — the
   Aug-12 incident day itself was already purged. Future incident fixtures
   must be dumped *within ~10 days* of the event.
2. The dump's price query returned one row per forecast generation (~279x
   duplication, 298k rows) — fixed with GROUP BY in `dump_battery_data.py`;
   the replay harness now also dedupes price timestamps and normalizes
   horizon array lengths (the first fixture with a non-empty market_prices
   table exposed that padding bug).

**Result** (real forecasts, real p10, real prices, Aug 13-22 window, paired
runs, 0 SoC violations in all arms):

| alpha | total cost | delta vs off | grid-charged |
|-------|-----------|--------------|--------------|
| 1.0 (off) | -99.06 EUR | — | 1401.6 kWh |
| 0.85 | -98.96 EUR | **+0.11 EUR / 10 days** | 1401.7 kWh |
| 0.70 | -98.84 EUR | **+0.22 EUR / 10 days** | 1401.3 kWh |

With *real, informative* p10 the hedge is still net-negative, but ~10x less
so than with noise p10 (+1.0 EUR/week at alpha=0.85) — p10's informativeness
recovers most of the insurance premium, just not all of it. In this window
the stakes are tiny: August import prices are ~0.05-0.09 EUR/kWh with small
spreads, and export revenue dominates (system earns ~8 EUR/day net). The
hedge barely changes dispatch at all (grid charge moves <1 kWh over 10
days).

**Verdict: keep `BATTERY_SOLAR_HEDGE_ALPHA=1.0` (off).** The blunt always-on
blend does not pay for itself in summer conditions. Open questions for a
revisit when autumn/winter price spreads arrive (and p10/p90 archiving has
covered genuinely bad solar weeks):
- *conditional insurance* (hedge only when worst-case solar < required fill
  AND price below threshold) would pay the premium only on threatening days
  instead of every day — the more promising design, but it needs real
  bad-weather data to evaluate;
- whether winter spreads (0.10+ EUR/kWh) flip the sign of the hedge benefit.

The infrastructure (context key, LP blend, harness passthrough, bench
tooling) stays in place, default-off, for that revisit.

## Files touched (Phase 1)

- predict_future.py
- optimize_plan.py
- utils/solar_calibration.py (new)
- tests/test_predict_future.py
- tests/test_optimize_plan.py
