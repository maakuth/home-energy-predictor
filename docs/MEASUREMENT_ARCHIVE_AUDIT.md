# Read-only migration audit

This command reports evidence, **not migration approval**. It never opens the
archive writable, imports data, loads recorder credentials, calls HA/PostgreSQL,
or changes pipelines. `migration_ready` remains false: source/unit evidence,
consumer integration and shadow validation are separate
requirements. No model behavior or version is changed.

## Commands

Run from the repository with its `venv`:

```sh
# Cheap schema/import/hash/date/source/query-range inventory. No cell scan.
venv/bin/python3 -m utils.measurement_audit_cli archive \
  --archive /workspace/state/measurement_history.staging.sqlite

# Bounded numeric availability, full coverage, verification and conflicts.
venv/bin/python3 -m utils.measurement_audit_cli archive \
  --archive /workspace/state/measurement_history.staging.sqlite \
  --sources config/measurement_sources.yaml \
  --measurement gshp_power --measurement outside_temp \
  --from 2026-10-04T21:00:00Z --to 2026-10-04T23:04:00Z \
  --interval-seconds 60 --legacy-policy complete-legacy

# Only plan missing successful-query ranges; do not read measurement cells.
venv/bin/python3 -m utils.measurement_audit_cli archive \
  --archive /workspace/state/measurement_history.staging.sqlite \
  --sources config/measurement_sources.yaml --measurement gshp_power \
  --from 2026-10-08T00:00:00Z --to 2026-11-10T00:00:00Z \
  --dry-run-backfill

# Streaming byte hashes and row/header/date/flag/column availability.
# Optional --archive matches hashes for an idempotent dry-run import plan.
venv/bin/python3 -m utils.measurement_audit_cli inventory \
  --csv /workspace/state/raw_data.csv \
  --csv /workspace/state/raw_data_weekly.csv \
  --archive /workspace/state/measurement_history.staging.sqlite
```

JSON goes to stdout. Missing archives fail without creating a database. Invalid
arguments/schema/manifests fail closed. Source YAML is explicit; unlike the
feature loader the numeric audit does not merge environment-dependent source defaults.
For additional measurements supply an audited YAML containing those mappings.

### Interpreting coverage and policy

The numeric audit always includes legacy **availability**, independent of
eligibility policy. `available_bins` counts finite resampled values;
`full_coverage_bins` counts complete valid duration; `source_verified_bins`
requires original recorder validity. `conflict_bins` includes conflicts even
when explicit precedence selects a finite value. Provenance counts and up to ten
conflict examples per measurement preserve contributing content hashes.

`measurement_eligible_bins` is deliberately only a per-measurement prerequisite:
complete finite values with no conflict, plus source verification under default
`verified-only`. Explicit `complete-legacy` permits unverified legacy values
meeting that prerequisite. Legacy `source_valid` is **never** changed. Invalid
source flags and known recorder outages remain unavailable. All conflicts stay
excluded, including explicitly resolved conflicts. This does **not** calculate
five-meter baseload targets, weather gates, time-aware lag eligibility, contiguous
training windows or training sample counts. `target_and_lag_eligibility` is
reported as `not_evaluated` unless `--training-eligibility` is also selected.
The integrated training audit uses the feature loader's runtime source defaults
with the explicitly supplied dated YAML overrides.

No unit conversions, relay ratings, category/counter rules, freshness thresholds
or pre-installation zeros are inferred. Known counter/category/relay logical
names are rejected by the numeric audit rather than averaged incorrectly.

Audit bounds must be interval-aligned. Default limit is 10,080 output bins per
measurement; narrow the window or consciously raise `--max-bins`. Resampling is
chunked into at most 1,440 output bins and shares one read snapshot. Queries bind
each legacy import ID and timestamp range; metadata uses indexed endpoints,
not full staging-table aggregation. Full staging history is never scanned by
default. Close audits promptly so read snapshots do not pin WAL indefinitely.

Dry-run backfill plans subtract `fetch_ranges` from `mapped_ranges`, preserving
source switches, attributes and mapping gaps. Successful empty queries count as
queried, not as available measurement data. Legacy imports do not suppress a
recorder backfill. Plans do not fetch, refresh recent overlap or claim that old
recorder history is still retained. Import plans likewise only identify existing
content and pending review; `ready_to_import` is conservatively false.

## Portable, explicit overlap precedence

Optionally provide `--precedence-manifest manifest.json` on a scoped audit:

```json
{"preferred_sha256": ["19d2db0dc6c270d2cf9c9ea20fdb7b63dc9db1526aba1e07c4e9f33d1614be34"]}
```

Hashes are ordered, highest priority first, and refer to exact original CSV
bytes. Unknown/duplicate/invalid hashes and unknown fields are rejected. Hashes
resolve to import IDs only inside the current read snapshot. First preferred
present import wins even when its cell is missing. Conflict flags and all
contributing hashes remain visible; policy excludes those bins. This example
illustrates syntax only: **no real overlap precedence has been approved**.
The manifest is not persisted and does not silently configure other consumers.
`resolve_precedence` is available for future explicit consumer integration.

## CSV structural checks and limitations

Inventory streams byte hashing and CSV parsing in separate passes; size, inode
and nanosecond mtime must remain stable. It reports exact header, row count,
date bounds, finite-numeric and nonempty counts per column, source-flag
true/false counts, and numeric date bounds (epoch seconds) per column. Flags
remain metadata, not verified recorder evidence. Malformed widths, invalid/naive
timestamps, declared-resolution misalignment, duplicate headers, and
nonincreasing timestamps block `structurally_importable`. Numeric column counts
also include malformed rows; use structural checks before interpreting them.

Memory does not grow with CSV rows. Consequently duplicate detection is exact
for sorted input via nonincreasing timestamps; unsorted input is blocked rather
than maintaining a huge exact timestamp set. Empty/non-numeric cells remain
missing. Structural validity is **not** source identity, unit correctness or
import authorization. The importer now transactionally rejects duplicate
timestamps/headers, empty headers/files, malformed row widths and timestamps
unaligned to the declared positive integer resolution. It preserves source-flag
cells separately as metadata and verifies that parsed bytes match the recorded
hash, rolling back on content changes. Always review the dry-run inventory
first. A stable stat is not a filesystem
snapshot; inventory static snapshot copies for stronger consistency.

`--root /workspace/old-state` recursively streams all `raw_data*.csv`; this is
explicit opt-in and may read many GB. Prefer selected newer snapshots initially.

## Read-only findings, 2026-11-10

All four selected files below were fully streamed for hashes, rows, headers,
date bounds and column/flag availability. All had valid structure, aligned
one-minute timestamps, stable stats and no nonincreasing timestamps. No import
or staging mutation was performed.

| File | Rows | UTC inclusive range | SHA256 | In staging |
|---|---:|---|---|---|
| `old-state/2026-10-08_00.00.01--1y/raw_data.csv` | 4,321 | 2026-10-04 21:00 – 2026-10-07 21:00 | `19d2db0dc6c270d2cf9c9ea20fdb7b63dc9db1526aba1e07c4e9f33d1614be34` | yes |
| `old-state/2026-10-08_00.00.01--1y/raw_data_weekly.csv` | 1,051,204 | 2024-10-04 23:00 – 2026-10-04 23:03 | `6fba6205a16f757159b7a69fce01808d584a59f436c98a731aec212416b77d79` | yes |
| `state/raw_data.csv` | 4,320 | 2026-11-07 07:00 – 2026-11-10 06:59 | `fa09bd5c2bc8f9e59defe5a4f1ed661f7aaad8d5cdc23077ab5fd687f7b59628` | no |
| `state/raw_data_weekly.csv` | 1,051,202 | 2024-11-09 00:00 – 2026-11-09 00:01 | `c7c60daa2c4882dadfa316fed393d52e100a4dfa50f237f731aa7b5cb6b40514` | no |

Directory discovery found 52 raw CSV paths: 2 current, 2 in the October 8
snapshot, and 48 in September 20 snapshots. **No later snapshot was found** under
`/workspace/old-state`. Earlier September contents were not fully scanned:
repeated ~134 MB weekly files cannot extend October/November history merely by
their snapshot dates. They remain available for separate recovery review.

The October imports have no source-validity flags. Both current files have
`gshp_power_source_valid`, `leaf_power_source_valid`,
`total_power_source_valid`, `solar_actual_source_valid`, and
`battery_power_source_valid`. Presence alone is not validity.

| Flag base name | Short true / false / blank | Weekly true / false / blank |
|---|---|---|
| gshp_power | 4,320 / 0 / 0 | 28,290 / 183 / 1,022,729 |
| leaf_power | 4,276 / 0 / 44 | 900,360 / 148,382 / 2,460 |
| total_power | 4,320 / 0 / 0 | 1,032,612 / 18,589 / 1 |
| solar_actual | 4,310 / 0 / 10 | 138,824 / 963 / 911,415 |
| battery_power | 4,320 / 0 / 0 | 231,485 / 28,442 / 791,275 |

Blank flags are not affirmative validity evidence. The reader conservatively
excludes legacy cells when an explicit flag column has a blank/false value.

| Logical column | Current short numeric rows | Current weekly numeric rows |
|---|---:|---:|
| outside_temp | 4,320 | 28,412 |
| gshp_power | 4,320 | 28,488 |
| total_power | 4,320 | 1,051,202 |
| solar_actual | 4,319 | 139,834 |
| battery_power | 4,320 | 259,957 |
| leaf_power | 4,291 | 1,048,414 |
| solar_forecast | 4,242 | 802,693 |
| wind_speed | 4,225 | 1,051,184 |

The current short file also has supply/return temperatures, not the old
accumulator temperature. Current weekly heating/outdoor history is sparse,
despite having two years of timestamps: do not promise two usable training
years. Logical CSV names cannot prove which renamed entity, unit, extraction
configuration or historical fill produced each value. Verify these associations
against dated extraction/source evidence before importing. In particular,
current sources do not retroactively restore old-entity heating/temperature
history. The confirmed heating meter includes compressor and resistive load;
do not subtract nominal relay loads or infer component truth from these CSVs.

Per-column weekly numeric endpoints confirm actual extension, not just extra
timestamp rows: outdoor temperature begins October 20 05:27 UTC and ends
November 8 23:57; shared-heating values end November 8 23:59; grid and battery
values reach November 9 00:01; solar and Leaf values reach November 8 23:59.
These endpoints do not imply continuity or eligibility between them. The short
file carries all six of these measurements through November 10 06:59 UTC.

The bounded staging overlap check (124 minute bins, October 4 21:00–23:04 UTC
exclusive) reproduces 121 available shared-heating samples with 3 conflicts and
116 available temperature samples with 8 conflicts. Both have **zero verified
bins**. Under `complete-legacy` those agreeing counts are per-measurement
eligible prerequisites only, not baseload or lag training rows. Under default
`verified-only`, legacy measurement eligibility is zero.

Remaining blockers: review newer CSV source/unit/flag evidence and overlap
precedence; complete remaining consumers, representative shadow comparisons
and operational approval. Nothing here
authorizes a live cutover.
## Integrated target/lag audit

Add `--training-eligibility` to a scoped `archive` audit to evaluate the shared
five-meter target policy, predictor inputs, exact 1h/24h lags and full-day Leaf
energy. `--measurement` is optional in this mode. This uses a separate single
read snapshot including the preceding day, identified in the returned metadata;
it does not fit a model or require feature imputation to handle missing history.

```sh
venv/bin/python3 -m utils.measurement_audit_cli archive \
  --archive /workspace/state/measurement_history.staging.sqlite \
  --sources config/measurement_sources.yaml \
  --from 2026-10-05T00:00:00Z --to 2026-10-06T00:00:00Z \
  --interval-seconds 900 --legacy-policy complete-legacy --training-eligibility
```

Read-only execution on 2026-11-10: 96 quarters, **2 eligible targets, 0 verified
targets, 0 eligible XGBoost training rows**. Leaf's complete previous-day energy
had zero eligible windows; solar and Leaf availability excluded 94 targets.
This one-day check diagnoses missingness, not model quality or full-history
training coverage. No precedence was selected, imports performed, or live
pipeline changed. The archive remains **not migration-ready**.
