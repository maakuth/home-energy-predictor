# Explicit archive pipelines and CSV rollback

The scheduled no-argument shell jobs still use CSV. No systemd service/timer has
been changed. The archive implementation is not a declaration that production
data, units, freshness or model quality have been approved.

## Preview first

`run_frequent.sh --archive ...` and `run_weekly.sh --archive ...` dispatch to
`utils.archive_pipeline`. It captures one UTC-minute cutoff (or accepts an
explicit `--cutoff`) and prints its plan without taking locks, connecting to the
recorder, creating an archive, fitting models, resetting metering or publishing.

From a development virtualenv the same preview is:

```sh
venv/bin/python3 -m utils.archive_pipeline --mode weekly \
  --archive /path/to/measurements.sqlite \
  --bootstrap-from 2024-11-10T00:00:00Z \
  --cutoff 2026-11-10T00:00:00Z \
  --recorder-namespace CONFIRMED-RECORDER-IDENTITY \
  --sources config/measurement_sources.yaml \
  --measurement-policy /path/to/audited-policy.yaml
```

The bootstrap date is an explicit query floor, not proof of retained history.
Successful covered ranges are not fetched again except the recent 900-second
overlap. Ingestion merges runtime source defaults with dated YAML overrides;
the inventory includes the eight baseload inputs, sauna temperature and the
physical accumulator only when enabled. Disable obsolete instrumentation or
provide correct mappings; a missing required recorder entity fails the run.

The example policy registry is deliberately incomplete and is **not** a usable
production configuration. Actual execution requires explicit source and policy
files and a policy entry for every requested measurement. It does not choose
unproven wind/forecast units or freshness limits on the operator's behalf.

## Operational execution requires two approvals

Only after deployment/data/model review may an operator append both
`--execute --approve-publish`. These are not development-test flags.

**Writing `optimization_plan.json` can actuate through `run_often.py` even if
`push_to_ha.py` is skipped.** Therefore execution without rollout approval is
refused, rather than misleadingly described as a shadow run. Use the read-only
[shadow command](MEASUREMENT_SHADOW.md) for non-actuating comparisons.

Execution takes the existing `/tmp/hepo.lock` (nonblocking frequent, waiting
weekly). Frequent execution preserves quarter-boundary net-metering reset
under that lock. Ingestion is incremental with fixed bounds and explicit
recorder namespace. Namespace mismatch fails closed before mixing rebuilt
recorder IDs. All subprocess failures terminate the job; there is no CSV retry.

Weekly execution trains XGBoost from projected day-sized feature chunks and
SARIMA from its explicit 30-day archive window. It never re-extracts 730 days to
CSV. The default XGBoost training window remains 730 local-archive days;
`--training-days`, `--sarima-days` and `--analysis-days` select explicit windows,
not empirically tuned model parameters. Both trainers use `--require-success`:
retaining an older model after a failed quality gate is not pipeline success.

Before advancing, the runner validates model training metadata and the
success-only prediction/plan sidecars against archive path, fixed cutoff and
legacy policy. Available source/policy/precedence fields are checked as well.
Configuration byte hashes must remain unchanged between subprocesses. Missing,
stale, empty or mismatched artifacts stop before subsequent optimization or
publication. Feedback-database persistence is a separate boundary: sidecars
attest to the complete JSON artifacts, not optional SQLite feedback writes.

## Rollback

Keep CSVs, snapshots and previous model/plan backups. Return the shell service
invocation to **no arguments** for the existing CSV compatibility pipeline.
Archive failure never selects this mode automatically. `run_slow.sh` likewise
retains its copied-CSV default and explicit archive window option, with model
parameter synchronization retained. No rollback command deletes the archive.

The shell dispatch, pipeline planning and subprocess/failure tests run against
isolated fixtures/stubs with CSV files absent. They do not execute real HA,
PostgreSQL, production fits or control publishing.
