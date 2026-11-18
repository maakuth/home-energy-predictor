# CSV pipeline retirement

Updated 2026-11-18. This is a phased plan, not authorization to change live
services, models, or stored data. `/workspace` is production.

## Current status

- The frequent production pipeline uses the measurement archive for historical
  inputs and an isolated archive-trained model bundle. Live telemetry remains
  live. Its no-argument CSV route remains a rollback option.
- The last verified scheduled weekly job used CSV. The explicit archive weekly
  route exists but writes the default model bundle, not the isolated bundle used
  by frequent inference. The scheduled SARIMA benchmark is disabled.
- Preserve historical CSVs, snapshots, archives, offline import/replay tools,
  and explained legacy comparison results. Do not repair the CSV tail merely
  to force parity.

## Phase 1: separate shared processing from CSV I/O — implemented on task branch

- Move `process_frame` and coverage helpers into a format-independent module;
  switch archive feature and shadow imports to it.
- Preserve `process_data.py` as a compatible CSV wrapper, including imports,
  arguments, defaults, and behavior. Leave `extract_data.py` unchanged.
- Characterize values, dtypes, timestamps, lags, eligibility and missing data;
  check that inputs are not mutated. Exercise archive paths without legacy CSVs
  and the compatibility wrapper using isolated fixtures.
- No change to model behavior or `VERSION`, production state, service schedules,
  or the CSV rollback route.
- `utils/feature_processing.py` now owns the shared implementation; the CSV
  wrapper re-exports the old names. Archive features and shadow use the new
  module directly. The moved processing functions have identical syntax trees
  to the originals. Focused regression tests passed (100 tests); the non-slow
  suite passed (1,204 passed, 54 deselected, 8 subtests passed), the full suite
  passed (1,257 passed, 1 skipped, 8 subtests passed), and Pyright reported no
  errors on changed Python files. Deployment and merge are separate decisions.

## Phase 2: weekly archive cutover — pending separate approval

- Audit approved mappings, policy and verified eligibility for every required
  weekly measurement and lag; report actual usable coverage. A 730-day window
  does not imply 730 days of usable samples. Winter recovery is separate.
- Preview the existing archive route and validate training and both evaluation
  steps with isolated outputs and a fixed cutoff; do not publish to HA or write
  plans. Operational `--execute --approve-publish` is not a dry run.
- Keep the live frequent model pinned. Weekly archive training currently writes
  the default model bundle and rejects a separate `--model-dir`. Identify all
  consumers of that bundle, then either separately approve writing it or add
  tested isolated weekly outputs. Successful training must not silently promote
  a model into live inference.
- With explicit operator approval, switch only weekly, then verify an actual
  scheduled archive-backed fit and both evaluations, provenance, cutoff, absence
  of newly generated pipeline CSVs, no plan write, and unchanged frequent model
  selection. Identify any accuracy metrics published to HA.

## Phase 3: replace rollback — pending

- Observe frequent and collector failures, degraded plans and weekly eligibility
  through an agreed period including at least one successful scheduled weekly
  run. Preserve CSV rollback meanwhile.
- Document and fixture-test recovery to a known-good archive-capable revision,
  service configuration and compatible model bundle; verify collector recovery
  and backup restore separately. Code rollback cannot fix an unavailable
  archive.
- Obtain explicit approval to retire CSV rollback. Afterwards missing archive
  configuration must fail clearly, without silent input fallback.

## Phase 4: remove retired production CSV code — pending

- Remove frequent/weekly CSV branches and, after checking all supported callers,
  `extract_data.py` and the `process_data.py` wrapper.
- Audit training, evaluation, optimizer and heating diagnostics, standalone
  SARIMA/comparison and fireplace tools, and `pull-from-murrikka.sh`. Remove
  obsolete production defaults; retain or clearly label supported offline CSV
  uses. Do not remove generic CSV support indiscriminately.
- Update `docs/SCHEDULES.md`, `docs/ARCHIVE_PIPELINES.md` and the migration
  handoff with actual deployment status. Preserve data, snapshots and reports.
- Add regression coverage proving archive scheduled routes run with no legacy
  CSVs and cannot call retired extractor/processor. Run focused tests, the
  non-slow suite, and the full suite before merging to master. Use a dedicated
  temporary worktree and isolated test fixtures.
