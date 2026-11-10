# Staging migration run — 2026-11-10

This records authorized staging work, not a live cutover. Original CSVs and the
original `state/measurement_history.staging.sqlite` remain retained. Active
models, plans, schedules, Home Assistant and PostgreSQL were not changed.

## Artifacts and imports

Run directory: `state/measurement-migration-20261110T152856Z/` (host repository
path differs from the container's `/workspace` prefix). The new archive is
`measurements.sqlite`, UUID `6bcbaf7b-0aba-4dfd-b248-b2e8561ff227`.

* A verified SQLite-aware clone preserved existing October CSV imports.
* Two stable copied CSVs added 1,055,522 rows as unverified legacy resampled data;
  validity flags and overlap conflicts remain intact. No universal precedence
  was selected. See `inventory.json` and `import-results.json`.
* The first host recorder run reported 957,076 events in 341 successful chunks.
* Historical host backfill reported 17,942,365 events in 9,641 successful chunks,
  with no failures, against a fixed November 10 15:28 UTC cutoff. Query bounds
  begin October 2024; actual earliest retained observations are November 2024.
  Empty queried chunks are not zero measurements. Predecessors explain the
  difference between event-report counts and stored observation counts.
* Archive integrity and foreign-key checks passed after both recorder runs.
  See `post-historical-verification.json` and `historical-source-quality.json`.

Recorder access ran only on the host, enforcing read-only PostgreSQL, bounded
pages/day chunks, statement/lock timeouts and resumable committed coverage.
The operator confirmed no ID-reusing recorder changes; the archive is bound to
`homeassistant-7537286243348692366-generation-1`.

## Operator-confirmed interpretation

* Battery is charging-positive and rated at ±10 kW. The operator approved a 10%
  measurement tolerance: inclusive ±11,000 W. Physical bounds are configured
  explicitly, independently of live planner caps.
* Thirty-seven observations exceed that tolerance. The requested zero
  replacements are recorded in `battery-zero-corrections-tolerant.csv` with
  original recorder IDs/values and invalid/source-unverified flags. They do not
  establish zero-battery training truth. The raw archive is unchanged; bounded
  policy interpretation makes affected training intervals missing.
* All seven required measurements use the approved availability-based policy:
  carry unchanged valid states only within source/query boundaries; stop on
  unknown/unavailable states. No arbitrary age threshold is inferred. This does
  not prove detection of every silent outage.
* Old/current solar aliases measure the same total PV plant. The old source's
  final numeric zero on July 18 at 18:38:17.844429 UTC became stale. Its source
  period ends there; the interval to the current source's first retained event
  on August 3 at 12:37:51.874835 UTC remains missing. These are analysis validity
  boundaries, not inferred installation dates. The sparse intermediate alias
  is not used to manufacture continuity.

Current staging configuration is `historical-audited-consumer-sources.yaml`
with `confirmed-tolerant-baseload-policy.yaml`. Earlier candidate source files,
the ±10 kW proposal, and `historical-confirmed-eligibility.json` are superseded
diagnostics, not approved historical training inputs. There is no verified
battery history before May 12; pre-installation zero is not inferred.

## Isolated candidate evaluation

Before historical backfill, model 1.47.0 was fitted in
`candidate-20261110T191747Z/` using 15.76 eligible sample-days before the internal
chronological split. The final seven days were externally held out, with a
seven-minute boundary embargo for existing centered weather denoising.

On 476 complete matched holdout quarters, MAE was 1.415 kW for the candidate,
0.877 for one-hour persistence, 0.940 for the training median and 1.565 for
previous-day persistence. Candidate bias (predicted minus actual) was +0.561 kW.
The candidate was **not promoted**. This is observed-weather/observed-lag
hindsight, not forecast-vintage or recursive multi-day replay; it does not
establish that removing the solar predictor caused the result. Do not tune on
this already-inspected week.

## Remaining validation

`historical-audited-eligibility.json` is the corrected recorder-only target/lag
audit for the observed-battery period: 13,713 eligible targets and 8,816 eligible
training quarters out of 17,485, with zero unverified training rows. This is about
91.83 eligible sample-days at quarter-hour resolution, not a native-minute fit
coverage guarantee.

The isolated expanded-history candidate in `candidate-historical-20261110T195827Z/`
uses unchanged fitting defaults, the corrected policy/source files and normal
coverage gates. Its predeclared outer holdout is September 1–October 1 UTC,
with training ending seven minutes before September 1. The previously inspected
November week is not reused as an independent holdout. September tests the old
heating regime and does not by itself validate the October equipment changeover.
Training passed the unchanged gate with 43.33 eligible native-minute sample-days
before the internal split. The outer holdout had 29,622 eligible minutes and
1,445 complete matched quarters. Quarter MAE: candidate 1.269 kW, one-hour
persistence 1.310, previous-day persistence 1.941, training median 1.368.
Minute MAE was 1.407 for the candidate versus 1.369 for the training median;
the improvement is not universal across metrics/resolutions. Peak RSS was
about 292 MiB and the isolated run took about 274 seconds.

The same frozen model receives a separate post-changeover diagnostic for
October 21–November 3 15:15 UTC in `new-regime-evaluation.json`; there is no
refit or parameter adjustment, and the already-inspected November week is
excluded. Neither this diagnostic nor September hindsight substitutes for
non-actuating forward validation with forecast vintages.
That diagnostic completed on 642 matched quarters: candidate MAE 1.427 kW,
one-hour persistence 1.055, previous-day persistence 1.631 and training median
1.578. Candidate bias was -0.267 kW (predicted minus actual). The frozen model
was not promoted: September's modest improvement does not generalize to the
newer regime against one-hour persistence. This does not establish a causal
effect of the equipment change, a production-model ranking, or an operational
multi-day forecast ranking. Keep these results fixed; do not tune to either
holdout.
Expanded-history evaluation and non-actuating shadows remain separate from live approval. No
active artifact replacement, schedule switch or source-file deletion is approved.

## Code validation

Physical numeric-bound support follows test-driven development: failing tests
preceded implementation. The policy suite passed 45 tests. Full repository
validation on the correction branch passed **962 tests**, with one skipped and
four passing subtests, including slow tests. Repository-wide Pyright was clean.
These isolated tests do not constitute live-rollout approval.

## Forward shadow preparation

The user approved non-actuating forward shadow validation. The frozen
pre-September candidate is pinned in `forward-shadow/protocol.json`, together
with its source/policy bytes and archive identity. `host-forward-shadow.sh`
performs explicit HA forecast reads, staging-only recorder refresh, generation
and later scoring in unique shadow directories. No active artifact replacement,
planner execution or publication is included. See
[forward shadow contracts](FORWARD_SHADOW.md). Preparation/dry-run is not a
claim that a live forecast vintage has already been captured.
