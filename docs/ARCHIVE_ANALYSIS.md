# Isolated read-only SQLite analysis

This is the next storage-migration increment after raw collection: measurement
coverage and quality reports from SQLite, with **no active consumer cutover**.
It does not replace model performance analysis or claim training readiness.

Validation: **1,050 passed, 1 skipped, four passing subtests** in the full
repository suite; repository-wide Pyright and diff checks passed. Tests cover
read-only enforcement, identity/hash checks, concurrent collection snapshots,
mapping/query gaps, invalid values, legacy exclusion and non-overwriting output.
The host example's shell/Python syntax was checked without running it. No live
archive analysis or host deployment is implied by these isolated tests.

`utils.archive_analysis` reuses the existing measurement audit/resampler. All
queries share **one read-only SQLite transaction**, including archive identity,
successful-query gaps and per-measurement statistics. The collector may continue
writing concurrently under WAL; the analyzer does not take its writer lock.
The reader closes before report output is written. Do not use `immutable=1` or
copy only the main database file while the collector is writing.

## Contract

- Explicit existing v2 archive, expected archive UUID and recorder namespace.
- Explicit audited source mappings and complete measurement policy, both pinned
  by SHA256. It uses the selected bytes, not environment-dependent aliases.
- Seven required measurements: outside temperature, wind speed, grid, PV,
  battery, GSHP and Leaf power.
- Fixed inclusive `--from` and exclusive `--to`, with timezone and aligned
  boundaries. Resolution is 60 or 900 seconds. The default is 900 seconds;
  these results are interval-duration diagnostics, not a new training target.
- Bounded work: at most 10,080 output bins by default; audit frames are chunked.
  Start with a day or a week. Increasing `--max-bins` is explicit, and longer
  read snapshots can hold back WAL checkpointing. This is not a tuning replay.
- Only a **new** `--output-dir` is accepted. Its parent must already exist.
  No existing plan, model, prediction or report is overwritten.

No PostgreSQL/Home Assistant access, credential loading, ingestion, CSV-file
fallback, feature processing, model fitting, promotion, control publication,
database migration, backup mutation or schedule changes occur. Model version is
unchanged. Missing archives, identity/hash mismatches and incomplete source or
policy definitions fail without publishing a success report. Errors have nonzero
exit status; inspect stderr. An output I/O failure can leave a partial directory;
inspect it and choose a new run path rather than overwriting it.

## Host invocation

Use the existing checkout, shared `.venv` and `requirements.txt`. The following
example uses an explicit completed 24-hour window; change the dates deliberately
for later runs. It reads collector configuration for the approved archive/source
identities but does not read the referenced credential file or run the collector.

```bash
cd /home/markuz/src/home-energy-predictor
STAGE="$PWD/state/measurement-migration-20261110T152856Z"
mkdir -p "$STAGE/analysis"
OUT="$STAGE/analysis/$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python3 - "$STAGE" "$OUT" <<'PY'
import json
from pathlib import Path
import sys
from utils.archive_analysis import main

stage, output = map(Path, sys.argv[1:])
collector = json.loads((stage / 'collector/config.json').read_text())
raise SystemExit(main([
    '--archive', collector['archive'],
    '--archive-uuid', collector['archive_uuid'],
    '--recorder-namespace', collector['recorder_namespace'],
    '--sources', collector['sources'],
    '--sources-sha256', collector['sources_sha256'],
    '--measurement-policy', str(stage / 'confirmed-tolerant-baseload-policy.yaml'),
    '--policy-sha256', 'ae3428a5fd0e034feda323c0767cd809421064d5b48f2b84655e62c6259bf7f2',
    '--from', '2026-11-13T18:00:00Z', '--to', '2026-11-14T18:00:00Z',
    '--interval-seconds', '900', '--output-dir', str(output),
]))
PY
```

The policy hash above identifies the already operator-approved policy: charging-
positive battery with inclusive ±11,000 W tolerance, audited units/signs and
availability-based carry without arbitrary expiry. If the hash differs, stop
and review; do not simply rehash a changed policy to get past the check.
Source validity ends at invalid observations, mapping boundaries and queried
coverage. No first-battery absence or historical PV-gap rule is invented here.

The CLI prints the path of `<output-dir>/report.json` on success. Each report
records its run ID, UTC generation time, fixed bounds, archive identity, source
and policy byte hashes, and analyzer module hash. The module hash is diagnostic,
not full dependency provenance; retain the deployed Git commit/environment when
comparing historical reports.

## Interpretation

- `unmapped_ranges`: periods with no approved source alias. These remain unknown.
- `unqueried_mapped_ranges`: mapped periods not covered by successful recorder
  queries. These are diagnostic only; the analyzer does not fetch anything.
- `available_bins`: finite resampled values under the selected policy.
- `full_coverage_bins`: complete valid measurement duration, not just a successful
  query. A zero-event query can still support valid carry; it is not proof of it.
- `source_verified_bins` / `unit_verified_bins`: evidence flags, separately
  counted. `provenance_bins` and bounded conflict examples explain origins.
- `measurement_eligible_bins`: complete, finite, conflict-free and source-verified
  bins under fixed **verified-only** eligibility. They are measurement-level
  prerequisites, not counts of usable training samples.

SQLite legacy records can be inspected for availability/conflicts only where
the explicit policy allows them; policies requiring recorded units may reject
them altogether. They never become verified measurement truth. Invalid states,
physical-bound violations, conflicts and missing periods do not become zeros.
The raw archive and correction records are not modified.

`target_and_lag_eligibility` remains `not_evaluated` and `migration_ready` remains
false. Baseload/lag contracts, usable full-year history, model-quality evaluation
and active cutover approvals are separate work. This command does not move the
existing `analyze_performance.py` or `analyze_evolution.py` scheduled outputs.

No new timer is installed. Review manual reports before deciding whether to
schedule this read-only job. Retain existing CSVs and snapshots throughout.
