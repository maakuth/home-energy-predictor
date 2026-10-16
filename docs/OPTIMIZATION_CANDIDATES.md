# Battery LP Planner Optimization Candidates

Captured after evaluating `BATTERY_SOLAR_HEDGE_ALPHA` on full-length fixture replays
(all four seasonal fixtures, `rho = 0.0/0.3/0.7/1.0`, five alpha values). Result of that
evaluation: **no alpha below 1.0 improved realised cost**; keep `BATTERY_SOLAR_HEDGE_ALPHA=1.0`.

## High impact
- [x] Enable/evaluate end-of-horizon terminal value on full-length fixtures, not 24 h slices. Opportunity-cost terminal value improved the four-fixture portfolio by EUR 0.0478 with no regression or safety issue. Keep opt-in; see [Terminal-value evaluation](#terminal-value-evaluation).
- [x] Make terminal value opportunity-cost based: value final SoC at the best future price minus next cheapest recharge opportunity. A follow-up POC replaces the misleading percentile enable gate with explicit `BATTERY_TERMINAL_VALUE_MODE=opportunity_cost`; it is not yet integrated.
- [x] Bias-correct load forecast in LP-only inputs using tracked `bias_kw` before planning. Implemented as an opt-in LP input correction; no suitable fixture regime exists yet to quantify its economic effect.
- [ ] Default/elevate `JointLinprogPlanner` over decoupled battery LP when heating or shiftable loads (GSHP, resistive, Leaf) are active; validate with `bench_cooptimization_poc.py`.
- [x] Trial weaker `BATTERY_LP_DISCOUNT` or apply discount only to slack/terminal terms. `0.998` was the best tested portfolio value, but has a July regression; slack-only mode should be rejected. See [Discount evaluation](#discount-evaluation).

## Model fidelity
- [ ] Add multi-scenario/robust solar constraints (p10/p50/p90) instead of a single alpha blend.
- [x] Add a battery switching/dwell cost. `BATTERY_SWITCHING_COST_EUR=0.0001` improved every fixture by EUR 1.0343 in total and is deterministic/safe. It only modestly reduces transitions, so it is a tie-breaker rather than a complete dwell solution. A focused tuning sweep remains in progress.
- [ ] Replace crude fuse model (`fuse × 3 × 0.230`) with per-phase limits/slack if phase data exists.
- [x] Feed capacity/power from BMS SoH: POC scales nameplate capacity and power using static `BATTERY_SOH_PCT`. It is a resilience/correctness feature, not a demonstrated economic improvement, and still needs a live BMS feed.
- [x] Add battery power ramp-rate constraints directly to the LP. POC rejected/rework needed: default `3 kW/min` permits a 45 kW change per 15-minute slot and therefore does not bind for a ±10 kW battery.

## Existing-hook evaluations
- [ ] Re-evaluate `BATTERY_LP_HEADROOM_COST_TOLERANCE_EUR` for real-time resilience vs. planned cost.
- [ ] Apply `IMPORT_PRICE_FLOOR`-style symmetry to export prices below zero.
- [ ] Verify `BATTERY_RAMP_RATE_KW_PER_MIN` plan-vs-actual mismatch on full replay.

## Process / evaluation
- [ ] Record battery cost/savings KPIs tagged by `get_model_version()` for historical comparison.
- [ ] Separate `bias_kw` in reporting so forecast bias is not mistaken for optimizer fault.
- [ ] Build an ablation harness (terminal value, discount, degradation cost, hedge) on full fixtures using the benchmark script pattern.
- [ ] Keep using full-length replay fixtures for any horizon/discount/threshold tuning; short replays only for crashes/constraints.

## Evaluation status (2026-10-16)

All economic results below use complete, safety-passing, full-length replays of
the four included quality fixtures (`jan.pkl`, `may.pkl`, `jul.pkl`, and
`oct.pkl`). Lower EUR cost is better. These are candidate evaluations, not an
approved-reference update or authorization to enable a changed default.

### Switching-cost evaluation

`BATTERY_SWITCHING_COST_EUR=0.000025` is currently the strongest evaluated
candidate. The optimum is a stable `0.000020`–`0.000030` plateau; 0.000025 was
chosen as its central value. Independent runs at 0, 0.0001, and 0.000025 were
byte-identical within each setting.

| Setting | Portfolio cost (EUR) | Change vs 0 | Fixture result | Safety |
|---|---:|---:|---|---|
| `0` | -81.642387 | 0.000000 | baseline | pass |
| `0.000025` | -82.860427 | -1.218040 | improves all four | pass |

The setting reduces direction reversals from 384 to 382 and net-flow variation
from 2487.809 to 2487.145 kWh; action transitions change from 1467 to 1466.
It is therefore a low-cost LP tie-breaker, not a replacement for a real
minimum-runtime/dwell constraint. See [Switching-cost sweep](switching_cost_sweep.md)
for the complete evaluated range and per-fixture metrics.

### Discount evaluation

See [Battery LP Discount Analysis](discount_lp_analysis.md) for per-fixture
flows, the narrow sweep, and the July timing analysis. Summary:

| Setting | Portfolio cost (EUR) | Change vs 0.995 | Verdict |
|---|---:|---:|---|
| `BATTERY_LP_DISCOUNT=0.995` | -81.6424 | 0.0000 | baseline |
| `0.997` | -82.1928 | -0.5504 | conservative: improves every fixture |
| `0.998` | -82.8932 | -1.2508 | best tested portfolio result; July regresses |
| `0.9985` | -82.7803 | -1.1379 | July regresses by EUR 0.0985 |
| slack-only mode | -81.9191 | -0.2767 | reject: July regresses by EUR 0.4296 |

Do not change the production default without a reviewed choice between the
portfolio optimum (`0.998`) and the all-fixture option (`0.997`), plus an
approved quality-reference/tolerance update if accepting a regression.

### Terminal-value evaluation

The opportunity-cost terminal value has a small but consistent benefit:

| Setting | Portfolio cost (EUR) | Change vs disabled | Safety |
|---|---:|---:|---|
| disabled | -81.6424 | 0.0000 | pass |
| opportunity-cost | -81.6902 | -0.0478 | pass |

January, May, and July improve; October is unchanged. A refinement POC adds
explicit `BATTERY_TERMINAL_VALUE_MODE` values (`disabled`,
`opportunity_cost`, `percentile`) while retaining compatibility with the
legacy positive-percentile gate. Keep the behavior opt-in unless more
representative fixtures show a material benefit.
