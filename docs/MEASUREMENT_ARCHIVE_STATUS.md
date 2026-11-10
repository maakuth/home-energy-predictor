# Measurement archive migration status — 2026-11-10

Implementation branch: `task/measurement-archive-finish` (extends
`task/measurement-archive-reader`, preserving all prior archive work). Nothing
has been merged into `master` or enabled in the live pipeline. Model version:
1.47.0. No production models were retrained.

## Available for explicit offline use

* Read-only schema/import/source inventory, bounded coverage/conflict reports,
  CSV inventories, content-hash precedence manifests and dry-run backfill plans:
  [audit documentation](MEASUREMENT_ARCHIVE_AUDIT.md).
* A distinct, versioned legacy training policy, original source verification
  preserved, five-meter baseload accounting and exact lag/Leaf-energy gates:
  [training policy](ARCHIVE_TRAINING_POLICY.md).
* Archive XGBoost training with policy/import provenance in model metadata.
* User-approved removal of the misdimensioned solar-forecast ML predictor;
  18-feature contract v2, embedded booster/metadata compatibility checks and
  strict archive-mode rejection of old/mixed artifacts. Measured PV accounting
  and live planner solar outputs remain intact. Explicit retraining is required.
* SARIMA archive training and benchmark prediction, materializing and releasing
  a read snapshot before fitting: [SARIMA documentation](SARIMA_ARCHIVE.md).
* Matched-window, non-actuating CSV/archive processing and repeat-yesterday
  comparisons: [shadow documentation](MEASUREMENT_SHADOW.md).
* Explicit archive arguments in `run_slow.sh`, skipping the CSV copy and retaining
  parameter synchronization. No-argument CSV compatibility remains the default;
  archive failure does not trigger CSV fallback. Isolated shell fixtures cover
  missing CSVs, argument forwarding, failure propagation and rollback.
* SQL-LIMIT/keyset-bounded recorder result pages (including equal-time IDs).
* Transactional legacy import validation of timestamps/alignment, headers,
  duplicate timestamps, malformed rows, empty files and parsed-byte checksums.
* Fixed-cutoff prediction anchors, Leaf/sauna history, autonomous shared-heating
  persistence and optimizer thermal observations, without replacing current HA
  telemetry: [consumer contracts](ARCHIVE_PREDICTION_OPTIMIZER.md).
* Performance/evolution actuals and diagnostic consumers using the same original
  five-meter target semantics: [evaluation inventory](ARCHIVE_EVALUATION.md).
* Explicit category/presence and strict counter-delta policies, unit/sign/
  attribute validation and configurable expiry: [registry](MEASUREMENT_POLICIES.md).
* Transactional v1→v2 migrations, archive UUID/application identity, recorder
  namespace checks, proven predecessors, bounded retry/resume, through-now
  ingestion and verified SQLite backup/restore: [storage](MEASUREMENT_ARCHIVE_STORAGE.md).
* Frequent/weekly archive routes with fixed cutoffs, incremental ingestion,
  local training, failure/sidecar/config-hash checks, locks and explicit rollback:
  [pipeline contracts](ARCHIVE_PIPELINES.md). Execution also requires rollout
  approval because active plan writes can actuate without the HA publisher.
* Single-snapshot feature streaming and early projected training frames. The
  final fit matrix remains O(rows), not constant-memory model fitting.

## Evidence, not readiness

Read-only inventory found current CSV data extending to November 9/10, but no
snapshot newer than October 8 under `old-state`. The newer files have sparse
heating/temperature history and explicit invalid flags. Logical CSV names are
not evidence of source entity/unit identity. **No newer CSVs were imported.**

The staging archive was not modified. A shared-policy audit of October 5 at
15-minute resolution found 2 eligible legacy targets in 96 quarters, zero
source-verified targets and zero eligible XGBoost training rows. This small
window diagnoses missingness; it is not a full-history model evaluation.

The full 733-day two-signal raw profile and real matched legacy shadow are
recorded in [validation evidence](MEASUREMENT_ARCHIVE_VALIDATION.md). The profile
processed 1,055,520 rows with bounded day chunks and indexed queries. The real
72-hour shadow had 459 eligible legacy actuals, zero verified actuals and only
33 matched previous-day forecast/actual pairs. Agreement of imported CSV bytes
is not proof of original sensor fidelity or a model ranking.

## Open evidence/authorization questions

1. **Recorder access and identity:** read-only HA/PostgreSQL metadata/history
   inspection must be run on the operator's host: the container cannot reach
   PostgreSQL. A [standalone evidence collector](MEASUREMENT_EVIDENCE.md) is
   provided; no live inspection was attempted here. Confirm the stable recorder namespace and
   whether a database rebuild changed state-ID identity before first binding.
2. **Measurement dimensions/sign/freshness:** historical wind units and the
   solar forecast contract are now partially resolved by the supplied report:
   sampled wind is km/h and the current tomorrow-forecast state is **daily kWh**,
   confirming a training/inference mismatch with hourly `pv_estimate` power.
   The user selected predictor removal, now implemented; historical hourly
   forecasts are no longer a baseload training gate. See
   [evidence review](MEASUREMENT_EVIDENCE_REVIEW.md). Daily energy is not
   relabelled kW and no nowcast proxy is substituted.
   Supply/verify correct per-period entity/attribute/unit associations, battery
   sign and evidence-based expiry limits. No thresholds are guessed to obtain
   coverage. The example registry is intentionally incomplete.
3. **Recovery imports/backfills:** newer CSVs remain unimported pending source/
   unit/flag provenance review and explicit staging-write authorization. Existing
   conflicts stay excluded; no universal file precedence has been selected.
4. **Physical instrumentation:** confirm active accumulator/tank instrumentation
   and equipment changeover evidence. Fireplace savings are explicitly blocked
   in archive mode until audited tank/fireplace features exist, rather than
   manufacturing zero activity or claiming measured wood heat.
5. **Representative evaluation and deployment:** authorized verified recorder
   data, full model-fit resource/quality evaluation, operational WAL/writer
   profiling, backup scheduling and rollout approval remain operational gates.
   No timer/service was installed or switched; scheduled defaults remain CSV.

The remaining work depends on these answers/data, not another implicit fallback
or fabricated eligibility rule. Merge and operational rollout are distinct.

Integrated validation including predictor removal: **897 non-slow tests passed**, 55 slow tests
deselected, four passing subtests; repository-wide Pyright and shell syntax
checks passed. Full-suite validation: **951 passed, 1 skipped**, four passing
subtests (including slow tests). These are isolated test results, not production
retraining or representative model-quality evaluation. All task commits remain
unmerged; only the intentional `.pruner` workspace aid is left untracked.

**The system is not migration-ready.** Do not delete source CSVs, snapshots or
the staging archive, and do not switch schedules as part of code validation.
