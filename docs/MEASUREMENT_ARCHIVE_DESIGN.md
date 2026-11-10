# Remaining migration contracts

This implementation extends the existing opt-in archive route. It does not
authorize an operational import, a recorder connection, or control publication.

## Shared consumer boundary

`utils.archive_history.ArchiveHistory` is an immutable, explicit request context:
archive path, fixed UTC epoch `cutoff`, optional source/policy paths, legacy read
opt-in, versioned training eligibility policy and import precedence. Consumers
take keyword `history: ArchiveHistory | None = None`. None preserves the existing
compatibility route; an archive error never retries CSV or PostgreSQL.

`history.raw(start, end, measurements=..., interval_seconds=60)` returns original
resampled values and quality metadata. `history.features(...)` returns the shared
feature/eligibility contract. Bounds are half-open, aligned, and never later than
the context cutoff. No consumer chooses a second wall clock for archive history.
Snapshots are materialized and closed before model fitting or optimization.
Current HA state/forecasts remain live telemetry, not historical archive reads.
Missing historical values remain missing: only explicitly labelled operational
fallbacks may estimate a value, never relabel it as measured truth.

Common history CLI: `--archive PATH --cutoff ISO`, optional `--sources`,
`--measurement-policy`, `--allow-legacy`, `--legacy-policy`, `--preferred-import`.
Training/SARIMA retain their existing explicit `--from/--to` window interfaces.
Pipeline cutoff is captured once, rounded down to a whole UTC minute, and passed
unchanged to ingestion, consumers and evaluation windows.

## Ownership and dependencies

* Storage owner: archive/reader schema, recorder predecessor reads, updater and
  ingest CLI, identity checks, retry/run diagnostics, SQLite backup/restore.
* Measurement owner: policy registry and pure numeric/category/counter transforms,
  resampling and feature-loader policy plumbing. Source YAML mappings remain
  unchanged; conversion/freshness metadata uses a separate explicit registry.
* Prediction/optimizer owner: `predict_future.py`, `optimize_plan.py`,
  `utils/uncontrolled_heating.py`, tests. Uses the shared context only.
* Evaluation owner: performance/evolution and offline diagnostic consumers,
  tests. Uses shared actual meter definitions and policy labels.
* Pipeline owner (after interfaces exist): frequent/weekly/slow orchestration,
  cutoff/lock/failure/rollback integration, docs and isolated execution tests.

No two workers edit a shared worktree. Each branch starts from the contract
commit; integration is through Git, never copying files between worktrees.

## Measurement/storage policies

Registry entries declare kind, canonical/input units, sign and optional expiry.
No freshness threshold is invented to improve apparent coverage. Categories use
explicit state mappings, not category averages. Counter deltas require valid
boundary observations, reject resets and gaps by default, never interpolate
unobserved energy or infer equipment absence. Legacy interval provenance remains
separate; unknown source units and counter provenance cannot become verified.

Storage migration is transactional/versioned and refuses unrelated SQLite DBs.
An archive binds an explicit recorder namespace; a changed recorder namespace
fails closed instead of mixing IDs after a rebuild. Missing namespace evidence
cannot be inferred from credentials. Predecessor evidence is tied to successful
query boundaries; it does not claim the older span was queried or carry across
source switches, outages or explicit expiry. Backups use SQLite's backup API,
verify integrity/foreign keys/identity, and never overwrite an existing restore
destination implicitly. No real staging schema migration occurs during tests.

## Rollout boundary

Code provides explicit archive and CSV modes; scheduled defaults remain CSV.
Archive frequent/weekly publishing requires an additional explicit rollout
opt-in, and tests use stubbed subprocesses/isolated state. A dry-run must not
open recorder connections or mutate measurement/control/model state. Missing
coverage, provenance approval and performance evidence remain visible blockers,
not reasons to fill zeroes or silently change data routes.
