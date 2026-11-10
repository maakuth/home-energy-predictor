# Archive evaluation and diagnostic consumers

Archive mode is explicit and read-only for **measurement history**. No archive
error retries PostgreSQL or a processed CSV. Compatibility defaults are unchanged.
The common options are `--archive PATH --cutoff ISO_WITH_TIMEZONE`, plus
`--sources`, `--measurement-policy`, `--allow-legacy`, `--legacy-policy`, and
repeatable `--preferred-import`. Cutoff must be a whole UTC minute.

## Consumer inventory

| Consumer | Explicit archive route | Remaining boundary |
| --- | --- | --- |
| `analyze_performance.py` | `fetch_actuals(days=7, *, history=None)`, `analyze(..., history=None)`, common CLI | Existing HA metric publication and performance SQLite writes are retained; this is **not** an offline/dry-run CLI. |
| `analyze_evolution.py` | Common CLI and `history` on analysis/prediction reader | Prediction rows share the actuals cutoff and half-open window; generation after cutoff is excluded. |
| `compare_models.py` | Common CLI and `compare(..., history=None)` | Archive main-model predictions are total load; SARIMA is baseload. Archive blend is refused because targets differ. Compatibility reporting is unchanged. |
| `utils/solar_calibration.py` | Common CLI and `fetch_actual_solar(..., history=None)` | Archived forecasts remain SQLite; generation after cutoff is excluded before deduplication. Unknown actual solar is not a calibration sample. |
| `dump_battery_data.py` | Common CLI and optional history on export, measurement, price and prediction readers | Archive export omits current HA snapshots/current forecast JSON and never synthesizes predictions. Prices come only from existing SQLite predictions, not an unaudited archive price mapping or PG fallback. |
| `utils/heating_benchmark.py` | Common CLI with explicit positive `--days`, instead of `--input` | Reads original shared heating, tank and weather measurements and masks policy-ineligible bins; quality columns survive interval export. |
| `estimate_fireplace_savings.py` | Common CLI and bounded `load_diagnostic_features(..., history=None)` | Archive savings calculation is explicitly blocked: the shared feature contract does not provide audited tank/fireplace evidence. No CSV fallback, inferred zero fireplace activity, or tank-temperature substitution. |

Prediction archives continue to be read from existing SQLite helpers. Archive
cutoff filters forecast generation in performance, evolution, comparison, solar
calibration and battery export. These are retrospective evaluations, not a claim
that every retained forecast was available before its target time. Battery
replay retains `generated_at` for its separate planning-time availability filter.
No predictions/measurements were imported or published to production by this work.

### Historian audit exclusions

* `utils/db_utils.py` is the compatibility historian adapter, not a migrated
  consumer. `extract_data.py` is the existing CSV extraction route; archive
  ingestion/storage has separate ownership.
* `utils/measurement_recorder.py` is the explicit archive recorder adapter.
* `predict_future.py`, `optimize_plan.py`, `utils/uncontrolled_heating.py` and
  SARIMA/training consumers have separate owners; no edits here.
* `utils/explore_db.py`, `utils/verify_access.py`, and `find_wind_sensor.py`
  intentionally inspect recorder connectivity/schema/entity metadata, not model
  history. An archive flag would not perform their stated task; they remain
  explicit operational utilities and were not run.
* `bench_solar_hedge.py` reads forecast percentiles from an explicit SQLite
  archive/fixture, not processed historical measurements or PostgreSQL. Existing
  replay fixtures and other JSON/SQLite plan/log inspection tools are not
  silently rewritten as archive measurements.
* `utils/measurement_shadow.py` deliberately compares CSV with archive inputs;
  removing its CSV side would remove the comparison. It has separate ownership.

## Actuals contract

`utils/archive_evaluation.py` aggregates a fixed half-open minute window. The
original five logical meters are `total_power`, `solar_actual`, `battery_power`,
`gshp_power`, and `leaf_power`. Units remain grid/solar kW and the other meters W.

* Home usage = grid + solar - signed battery / 1000.
* Baseload = home usage - shared heating / 1000 - Leaf / 1000.
* The shared heating meter already includes compressor and elements; nominal
  relay power is not added a second time.
* All fifteen aligned minute bins must be eligible for a quarter to be known.
  Missing/outage/invalid/conflicted bins and partial edge quarters remain NaN,
  never zero. Negative **minute** baseload invalidates its quarter; averaging
  cannot hide it. Baseload uses `baseload_targets` before quarter aggregation.
* `history.eligible` controls policy eligibility. `_source_verified` and
  `baseload_source_verified` remain strict recorder-verification labels;
  complete legacy acceptance does not upgrade provenance. Policy, bounds and
  archive metadata are retained in frame attributes and battery export metadata.

XGBoost hindsight accepts either `frame=` or `history=`. Explicit history loads
shared archive features (including preceding-day lag context), never processed
CSV, and uses `select_training_rows` and complete-quarter predictions. Hindsight
is baseload only; real-time archived main-model predictions represent total load.

## Equipment evidence and uncertainty

User-confirmed correction: old equipment removal occurred **October 7, 2026,
approximately 08:00 Europe/Helsinki**, not October 6. There is no evidence for an
exact minute boundary. October 7 is conservatively labelled
`removal_transition` in its entirety; October 6 remains `old_intermittent` and
complete resistance-only diagnostic days begin October 8. This intentionally
excludes the uncertain partial day, rather than pretending an approximate
equipment event is an exact meter-source switch.

Existing August 20 failure-era and October 19/20/21 installation/integration-era
labels are retained, not newly verified. Exact commissioning, hydraulic
changeover, controller integration, tank sensor placement/freshness, fireplace
operation and historical source-unit evidence remain open. No source YAML
boundaries were changed to infer any of these facts. Benchmark candidates remain
provisional, not COP estimates or measured wood-heat savings.

## Validation and integration

Tests use mock history frames and isolated SQLite prediction fixtures, never
live HA/PostgreSQL. Added regression coverage includes full-bin outage handling,
legacy eligibility versus verification, minute-negative targets, fixed forecast
cutoff, no fallback on archive failure, bounded fireplace requests, solar and
battery diagnostics, and corrected local equipment-era boundaries. Existing
heating/process-data fixture dates move to the first complete resistance-only
day without changing their assertions about filtering/energy.

Validation on this branch: non-slow suite **763 passed, 55 deselected**, with
four passing subtests; focused evaluation/diagnostic/process tests **65 passed**;
repository-wide Pyright **zero errors/warnings**. A full-suite attempt exceeded
its initial 120-second cap. The longer retry remained in the slow planner replay
section and was stopped after several minutes without reported failures; the
full slow suite is **not** claimed to pass. No tuning or fixture truncation was
performed to make it finish.

The integration owner must apply the single shared model-version bump. This
branch deliberately does not edit `VERSION`, shared archive/feature/history
implementations, pipeline shell scripts, or `docs/SCHEDULES.md`. Pruner was
reported unavailable; context was inspected directly. No live pipeline or
operational export was executed.
