# Measurement archive migration status — 2026-11-10

Implementation branch: `task/measurement-archive-migration-next` (extends
`task/measurement-archive-reader`, preserving all prior archive work). Nothing
has been merged into `master` or enabled in the live pipeline. Model version:
1.45.0. No production models were retrained.

## Available for explicit offline use

* Read-only schema/import/source inventory, bounded coverage/conflict reports,
  CSV inventories, content-hash precedence manifests and dry-run backfill plans:
  [audit documentation](MEASUREMENT_ARCHIVE_AUDIT.md).
* A distinct, versioned legacy training policy, original source verification
  preserved, five-meter baseload accounting and exact lag/Leaf-energy gates:
  [training policy](ARCHIVE_TRAINING_POLICY.md).
* Archive XGBoost training with policy/import provenance in model metadata.
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

## Evidence, not readiness

Read-only inventory found current CSV data extending to November 9/10, but no
snapshot newer than October 8 under `old-state`. The newer files have sparse
heating/temperature history and explicit invalid flags. Logical CSV names are
not evidence of source entity/unit identity. **No newer CSVs were imported.**

The staging archive was not modified. A shared-policy audit of October 5 at
15-minute resolution found 2 eligible legacy targets in 96 quarters, zero
source-verified targets and zero eligible XGBoost training rows. This small
window diagnoses missingness; it is not a full-history model evaluation.

## Still required before migration readiness

1. Establish newer CSV source/unit/flag provenance and any approved conflict
   precedence; only then explicitly import/backfill staging history.
2. Migrate prediction anchors, sauna/Leaf histories, uncontrolled heating,
   optimizer thermal observations and performance/evolution actuals. Do not
   replace immediate-control telemetry with archive history.
3. Implement energy-counter/category policies, validated unit/sign conversions
   and evidence-based per-measurement freshness policies.
4. Finish predecessor ingestion, recorder identity/schema migrations,
   through-now checkpoints/retries, SQLite-aware backup/restore and large-window
   memory/query profiling. Keyset paging alone does not solve these concerns.
5. Run representative full-window shadow comparisons separating verified and
   legacy-eligible evidence; the new synthetic tests are not model rankings.
6. Integrate frequent/weekly pipelines with fixed cutoffs, ingestion, locking,
   failure handling and explicit CSV rollback. They remain CSV-backed today.
7. Obtain operational rollout approval separately from merging code. Never
   infer it from this branch or from an audit report.

**The system is not migration-ready.** Do not delete source CSVs, snapshots or
the staging archive, and do not switch schedules as part of code validation.
