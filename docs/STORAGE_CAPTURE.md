# Fresh storage-comparison captures with frozen CSV sidecars

This manual host workflow issues a non-actuating shadow forecast and compares
CSV/SQLite history under the same frozen model, captured weather/PV, actual
generation time, cutoff, targets and inference contract. It does not train,
ingest, change schedules, write active artifacts or publish controls.

Use the existing project checkout and shared `.venv`. No new checkout or
environment is needed. The old `host-forward-shadow.sh` remains pinned to its
original code/protocol and is **not** modified or used for this new workflow.
The original protocol supplies frozen model/source/policy/archive identities;
its old `required_code_commit` applies only to that old runner. This workflow
uses the newly tested integrated code and records its commit separately.

Initial sidecar implementation validation: **1,069 passed, 1 skipped, four passing subtests** in the full
repository suite; repository-wide Pyright and diff checks passed. Focused tests
cover pre-capture freezing, full context, malformed/changed CSVs, cutoff rollover,
exclusive outputs and sidecar hash checks during generation/replay. Host command
syntax was checked without making live requests. These tests do not constitute
a real forecast capture or operational cutover approval.

The subsequent bounded-wait fix passed **57 focused tests** with Pyright clean;
its integration was approved without waiting for the full-suite rerun. The
full-suite count above describes the initial sidecar implementation, not that fix.

## 1. Capture on the host, waiting for a normal CSV refresh

Run as `markuz` on murrikka, **not in the development container**. This step
explicitly performs the existing four Home Assistant state GETs and read-only
`weather.get_forecasts` request. It does not contact PostgreSQL or call controls.

The CSV updates every 15 minutes. The command below waits up to **900 seconds**
for that normal refresh, checking every two seconds and recomputing the real
current-minute cutoff on each attempt. You can start it between refreshes: no
CSV sidecar is written and no HA requests occur until complete, stable timestamp
context is available. It does not request another extraction or shift the cutoff
to an older CSV. Do not manually run the full frequent pipeline to obtain a CSV:
that pipeline actuates controls.

Without the optional wait flag, capture still fails immediately on stale context.
Malformed schemas/timestamps, existing outputs and permission errors are not
retried. The maximum permitted wait is 1,800 seconds. A timeout reports the
missing minute count and required/available endpoints; it leaves no sidecar or
input snapshot. If a minute rolls over after successful freezing but before HA
capture starts, validation still fails rather than backdating; use a new directory.

```bash
set -euo pipefail
cd /home/markuz/src/home-energy-predictor
STAGE="$PWD/state/measurement-migration-20261110T152856Z"
mkdir -p "$STAGE/storage-captures"
RUN="$STAGE/storage-captures/run-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -m 0700 "$RUN"
export PYTHON_DOTENV_DISABLED=1
git rev-parse HEAD > "$RUN/code-commit.txt"
.venv/bin/python3 -m utils.forward_shadow capture \
  --env-file "$PWD/.env" --csv "$PWD/state/raw_data.csv" \
  --wait-for-csv-seconds 900 \
  --csv-output "$RUN/shadow-history.csv" --output "$RUN/shadow-inputs.json"
```

CSV bytes are frozen **before** HA capture begins. Source inode/size/mtime are
checked for concurrent replacement. A 16 MiB size bound avoids accidentally
loading the full multi-year weekly extract. Timestamp order, uniqueness,
timezone, minute alignment, all five meters/validity flags and the full preceding
1,440 timestamps are required. Interior gaps are rejected, not filled.

The snapshot binds the sidecar's hash, path, observed source mtime, freeze time,
history cutoff, row range and declared-valid counts. CSV flags still do **not**
establish recorder or original unit verification. Missing/invalid measurement
values remain unchanged; full timestamp coverage is not full measurement validity.

Outputs must be new and distinct. If capture fails, inspect the directory;
a frozen CSV may remain without a completed input snapshot. Keep the evidence
and choose a new run directory. Do not replace it with a later CSV extract.
In particular, the earlier "CSV lacks the full preceding 1440 minutes" error
can mean an otherwise complete CSV simply ends at the previous quarter-hour.
Use the wait-enabled command in a new run directory instead of editing data or
reusing an old cutoff. Failed directories containing only `code-commit.txt` are
safe to retain; there is no issued forecast in them.

## 2. Let the collector catch up, then generate and compare

Keep `STAGE` and `RUN` from step 1. Wait for the next existing collector tick
(:02/:07/:12/...), normally about two minutes after a quarter-hour CSV refresh.
The generator retains the real generation time and only emits strictly future
targets. The captured forecasts must still be no more than 15 minutes old.
If that guard expires, capture again rather than overriding timestamps.

The following checks the frozen experiment identities and successful query
coverage before generation. It never refreshes the archive itself. Missing
coverage means retry after collection, within the freshness limit; invalid
anchor measurements remain failures and must not be defaulted to zero.

```bash
.venv/bin/python3 - "$STAGE" "$RUN" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from utils.archive_history import ArchiveHistory
from utils.archive_quality import METERS
from utils.forward_shadow import _provenance, generate_shadow, write_new
from utils.measurement_audit import backfill_plan
from utils.measurement_reader import MeasurementReader
from utils.measurement_sources import load_sources
from utils.storage_replay import compare

stage, run = map(Path, sys.argv[1:])
root = stage / 'forward-shadow'
protocol = json.loads((root / 'protocol.json').read_text())
snapshot = json.loads((run / 'shadow-inputs.json').read_text())
model = stage / protocol['candidate_relative_path']
history = ArchiveHistory(stage / 'measurements.sqlite', snapshot['history_cutoff'],
    sources_path=root / 'shadow-sources.yaml', policy_path=root / 'shadow-policy.yaml')
expected = {key: protocol[key] for key in
    ('archive_uuid', 'recorder_namespace', 'sources_sha256', 'policy_sha256')}
def check_controls():
    if _provenance(history) != expected:
        raise ValueError('Frozen archive/source/policy identity changed; stop for review')
    for name, digest in protocol['model_files_sha256'].items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen model changed; stop for review')
check_controls()
with MeasurementReader(history.path) as reader:
    gaps = backfill_plan(reader, load_sources(history.sources_path),
        history.cutoff - 86400, history.cutoff, list(METERS))
if gaps:
    raise ValueError('Collector has not covered the full history window yet; retry after its next tick')
vintage = generate_shadow(history, model, snapshot)
check_controls()
write_new(run / 'shadow-forecast.json', vintage)
report = compare(csv=run / 'shadow-history.csv', history=history, model_dir=model,
    snapshot=snapshot, vintage=vintage)
check_controls()
write_new(run / 'storage-replay.json', report)
print(json.dumps({'run_directory': str(run), 'status': report['status'],
    'origin': vintage['generated_at'], 'targets': len(vintage['forecasts']),
    'predictions': None if report['predictions'] is None else
        {k: v for k, v in report['predictions'].items() if k != 'rows_detail'}}, indent=2))
PY
```

Generation and replay both verify the sidecar hash and its pre-capture timing.
A different later CSV is rejected for these new snapshots, even if it would fill
an inconvenient PV tail gap. Old captures without sidecar metadata retain their
explicitly retrospective replay semantics. Archive readers still use retained
event-time history; this does not freeze the whole SQLite database as-of-issue,
so late arrivals can change a later replay. Inspect saved-vintage reproduction.

A blocked CSV route does not invalidate the separately issued archive vintage.
If generation succeeds but comparison fails, preserve the vintage; do not rerun
generation into the same path. Use the standalone replay command to diagnose
that capture with its original sidecar. No result is fed to the active plan.

## 3. Repeat and later score

Capture several origins across different times/load conditions. Keep each
vintage and its sidecar separate; overlapping horizons are not independent
evidence or permission to tune/promote a model. Review anchor/feature/prediction
differences and explained CSV validity failures, not just an equality count.

After observations settle, use the existing `utils.forward_shadow score` and
`utils.storage_replay --actuals-cutoff ...` commands with **new output paths**.
Generation/replay use the original snapshot cutoff; later scoring uses an
explicit later actuals cutoff. Only completed, valid target quarters are scored.
See [forward shadow](FORWARD_SHADOW.md) and [storage replay](STORAGE_REPLAY.md).
No timer, new collector, optimizer replay or active cutover is installed here.
