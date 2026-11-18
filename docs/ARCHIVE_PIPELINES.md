# Explicit archive pipelines and CSV rollback

The scheduled no-argument shell jobs still use CSV. No systemd service/timer has
been changed. The archive implementation is not a declaration that production
data, units, freshness or model quality have been approved.

## Preview first

`run_frequent.sh --archive ...` and `run_weekly.sh --archive ...` dispatch to
`utils.archive_pipeline`. It captures one UTC-minute cutoff (or accepts an
explicit `--cutoff`) and prints its plan without taking locks, connecting to the
recorder, creating an archive, fitting models, resetting metering or publishing.
When a measurement policy is supplied, the preview lists required and missing
policy entries. This uses the current process configuration, not the `.env`
loaded only during execution; entry presence does not prove policy approval,
source coverage or migration readiness.

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
The approval flag also authorizes active model writes in weekly mode; weekly
mode trains and evaluates but does not run prediction, optimize or publish a
plan. Existing analysis may still publish accuracy metrics.

**Writing `optimization_plan.json` can actuate through `run_often.py` even if
`push_to_ha.py` is skipped.** Therefore execution without rollout approval is
refused, rather than misleadingly described as a shadow run. Use the read-only
[shadow command](MEASUREMENT_SHADOW.md) for non-actuating comparisons.

Execution takes the existing `/tmp/hepo.lock` (nonblocking frequent, waiting
weekly), and requires `--writer-lock` with the absolute path to the same shared
archive writer lock used by the collector. This second lock is nonblocking and
held only for ingestion, not model fitting or inference. A busy writer lock or
failed ingestion in frequent mode skips ingestion and attempts prediction from
eligible existing archive data; weekly mode stops. Configure the existing shared
lock, not a new independent file. Frequent execution preserves quarter-boundary
net-metering reset under the pipeline lock. There is no CSV retry.

Weekly execution trains XGBoost from projected day-sized feature chunks; SARIMA
training is disabled. It never re-extracts 730 days to CSV. The default XGBoost
training window remains 730 local-archive days; `--training-days` and
`--analysis-days` select explicit windows, not empirically tuned model parameters.
The trainer uses `--require-success`:
retaining an older model after a failed quality gate is not pipeline success.

Frequent archive mode optionally accepts `--model-dir /absolute/path/to/state`
to use a separate archive-trained model bundle. This prevents the unchanged CSV
weekly job from replacing that bundle. Weekly archive mode rejects this option:
its trainer still writes the normal state directory, so it must not appear to
train one bundle and predict with another. Bundle selection is not a promotion
or approval of the candidate.

Before advancing, the runner validates model training metadata and the
success-only prediction/plan sidecars against archive path, fixed cutoff and
legacy policy. Available source/policy/precedence fields are checked as well.
Configuration byte hashes must remain unchanged between subprocesses. Missing,
stale, empty or mismatched artifacts stop normal optimization and publication.
Feedback-database persistence is a separate boundary: sidecars
attest to the complete JSON artifacts, not optional SQLite feedback writes.

If frequent prediction or optimization fails, including artifact verification,
the approved runner writes a **local degraded** single-slot plan under the
pipeline lock. Its entry and sidecar identify the degradation and failed step;
it requests idle battery, STOP/OFF for managed loads, and contains no invented
usage, solar, SoC or price. Unknown effective cost is null to avoid interpreting
it as a cheap price. No safe price-aware planning is attempted: the fallback has
no independently validated current prices, load or battery state. Live control
still applies manual overrides, thermal recovery, phase limits, ramping and other
guards; therefore idle is an intent, not a guarantee of instantaneous zero power.
Manual Leaf charging remains protected. The current-slot plan expires at the
next interval; on a missing current slot the existing controller takes its safe
outputs *before* checking the battery manual override. Long runs and missed
schedules can thus leave no current degraded entry.

The degraded plan is not sent through `push_to_ha.py`: that publisher requires
numeric usage and defaults missing prices to zero, which would fabricate
telemetry and cheap-price signals. HA plan sensors may consequently remain stale;
other independent invocations of the publisher are not guarded. Plan replacement
is atomic, with unique temporary files; metadata is replaced first and is not a
transaction with the plan. Readers should trust the label in the plan itself.
A failed plan replacement can leave old plan plus new metadata, and other
writers must cooperate with the same pipeline lock. A locked pipeline, invalid
configuration, write failure or process crash can still prevent a fresh plan;
those failures are not disguised as successful fallback.

## Rollback

Keep CSVs, snapshots and previous model/plan backups. Return the shell service
invocation to **no arguments** for the existing CSV compatibility pipeline.
Archive failure never selects this mode automatically. `run_slow.sh` is disabled
in both cases. No rollback command deletes the archive or SARIMA artifacts.

The shell dispatch, pipeline planning and subprocess/failure tests run against
isolated fixtures/stubs with CSV files absent. They do not execute real HA,
PostgreSQL, production fits or control publishing.
