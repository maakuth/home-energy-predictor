# Data-only SQLite collector

Status: **prepared, not deployed**. This is the first incremental rollout step,
not an archive consumer cutover. No existing service, CSV job, model artifact,
prediction, optimization plan or control setting is changed.

Validation: **1,026 passed, 1 skipped, four passing subtests** in the full
repository suite; repository-wide Pyright and diff checks passed. Rendered
service/timer templates passed `systemd-analyze verify`, and the calendar
expression was checked. These are isolated checks, not a live recorder test
or evidence that the host timer is deployed.

`utils.measurement_collector` reads recorder observations into the existing v2
archive. It preserves raw values (including invalid states), queries uncovered
ranges plus a 15-minute trailing overlap, and commits bounded chunks. Successful
chunks survive failures; the next run retries uncovered ranges. Source validity
boundaries remain authoritative. Collection does not establish training eligibility.

The default command is a dry-run: local identity/mapping checks only, with no
PostgreSQL connection, lock, report or archive writes. `--execute` acquires the
shared staging lock, checks the pinned archive UUID/namespace and source-file
SHA256, confirms a read-only PostgreSQL session, then opens the writable archive.
It never creates or upgrades an archive and has no CSV fallback.

## Host preparation (operator-run)

Run on **murrikka**, as `markuz`, not in the development container. Review these
paths before executing. Use the existing `~/src/home-energy-predictor` checkout
with the tested collector code integrated into `master`. No additional code
checkout or worktree is created: updating the main checkout also updates collector code
on subsequent runs. Stop the collector timer/service before changing code or
dependencies, validate the update, then resume it. The archive UUID and source
mapping hash remain pinned independently; code is not pinned to a separate release.
No live database calls occur in this preparation block.

```bash
set -euo pipefail
REPO=/home/markuz/src/home-energy-predictor
CODE="$REPO"
PYTHON="$REPO/.venv/bin/python3"
STAGE="$REPO/state/measurement-migration-20261110T152856Z"
CONFIG="$STAGE/collector/config.json"
export REPO CODE PYTHON STAGE CONFIG

test -x "$PYTHON"
"$PYTHON" -m pip install -r "$CODE/requirements.txt"
```

Use the existing project's `.venv` and `requirements.txt`, shared with the live
jobs. Do not create another environment. Coordinate dependency installation
with all scheduled jobs using this environment so they do not start mid-update.
Configuration, reports, frozen mappings and the archive remain under the existing staging directory.
`code-commit.txt` records the preparation-time commit only, not a release pin.

### Existing CSV job dependencies

The live `run_frequent.sh` and collector both use the project's **`.venv`**.
Updating source code does not install new dependencies there.
`PyYAML>=6.0` is already declared in `requirements.txt`, but archive modules are
imported during CSV prediction startup too. A stale live environment can therefore
fail with `ModuleNotFoundError: No module named 'yaml'` even without SQLite enabled.

On the host, refresh the live environment from the project's dependency file:

```bash
cd /home/markuz/src/home-energy-predictor
.venv/bin/python3 -m pip install -r requirements.txt
.venv/bin/python3 -c "import yaml; import predict_future"
```

Coordinate installation with the existing scheduled jobs so they do not start
mid-update. The import check does not run prediction, connect to the recorder or
publish controls. Inspect the next normal 15-minute run in the journal; do not
manually execute the full live pipeline merely to test imports. If the preparation
step above already refreshed this environment, only the import check is needed here.

The following renders the exact host configuration and units without installing
or starting them. It freezes the **exact audited mapping bytes**, including all
seven measurements, and pins the approved recorder identity. Outputs use
exclusive creation: stop and inspect any existing `collector/` directory first.

```bash
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

repo, code, stage = (Path(os.environ[k]) for k in ('REPO', 'CODE', 'STAGE'))
out = stage / 'collector'
out.mkdir(mode=0o700)
sources = out / 'sources.yaml'
contents = (stage / 'historical-audited-consumer-sources.yaml').read_bytes()
with sources.open('xb') as stream:
    stream.write(contents)
config = json.loads((code / 'config/measurement_collector.example.json').read_text())
config.update(
    archive=str(stage / 'measurements.sqlite'),
    archive_uuid='6bcbaf7b-0aba-4dfd-b248-b2e8561ff227',
    recorder_namespace='homeassistant-7537286243348692366-generation-1',
    sources=str(sources), sources_sha256=hashlib.sha256(contents).hexdigest(),
    env_file=str(repo / '.env'), lock_file=str(stage / 'historical-backfill.lock'),
    reports_dir=str(out / 'reports'),
)
config_path = Path(os.environ['CONFIG'])
with config_path.open('x') as stream:
    stream.write(json.dumps(config, indent=2) + '\n')
values = {'USER': 'markuz', 'CODE_DIR': str(code), 'PYTHON': os.environ['PYTHON'],
          'CONFIG': str(config_path), 'STAGE_DIR': str(stage),
          'SOURCES': str(sources), 'ENV_FILE': str(repo / '.env')}
service = (code / 'deploy/systemd/hepo-measurement-collector.service.in').read_text()
for key, value in values.items():
    service = service.replace('@' + key + '@', value)
with (out / 'hepo-measurement-collector.service').open('x') as stream:
    stream.write(service)
with (out / 'hepo-measurement-collector.timer').open('x') as stream:
    stream.write((code / 'deploy/systemd/hepo-measurement-collector.timer').read_text())
with (out / 'code-commit.txt').open('x') as stream:
    import subprocess
    stream.write(subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True))
print('Prepared:', config_path)
print('Sources SHA256:', config['sources_sha256'])
PY
cd "$CODE"
"$PYTHON" -m utils.measurement_collector --config "$CONFIG"
```

Expect `status: dry_run`, the approved archive identity and seven measurements.
The bootstrap floor is `2026-11-10T15:28:00Z`: older history was backfilled;
this catches up temperature and weather as well as the five meters. The mapping
hash is derived from the selected bytes, not inferred from a filename. Any later
mapping change must be reviewed and explicitly re-pinned.

The explicit dotenv file supplies DB_HOST, DB_NAME, DB_USER, optional DB_PASSWORD
and DB_PORT. No ambient `.env` search is used. Prefer an existing SELECT-only
recorder account; do not change recorder roles as part of this deployment.
The client also enforces read-only transactions and statement/lock timeouts.
The namespace is operator-confirmed, not independently discoverable from SQL;
a recorder rebuild/restore requires stopping and reviewing identity first.

## Install, first run, then enable

These commands explicitly perform host installation and the first live read.
Review rendered units first. Do not run existing frequent/weekly archive
pipelines for this step: writing the active plan can actuate control.

```bash
systemd-analyze verify "$STAGE/collector/hepo-measurement-collector.service" "$STAGE/collector/hepo-measurement-collector.timer"
sudo install -m 0644 "$STAGE/collector/hepo-measurement-collector.service" /etc/systemd/system/hepo-measurement-collector.service
sudo install -m 0644 "$STAGE/collector/hepo-measurement-collector.timer" /etc/systemd/system/hepo-measurement-collector.timer
sudo systemctl daemon-reload
sudo systemctl start hepo-measurement-collector.service
sudo journalctl -u hepo-measurement-collector.service -n 60 --no-pager
```

Inspect the latest `collector/reports/collector-*.json` before enabling the timer:
require `success`, exit code 0, empty `failures`, all seven requested measurements,
and successful per-source chunks through the reported fixed UTC-minute cutoff.
Chunk event counts are diagnostic fetch counts, not counts of newly unique rows.
A lock-busy skip is not evidence of successful ingestion; retry after the manual
backfill/shadow finishes. Do not remove the shared lock file.

Only after that verification:

```bash
sudo systemctl enable --now hepo-measurement-collector.timer
systemctl list-timers hepo-measurement-collector.timer --all
```

The schedule is at :02/:07/:12/... each hour, with missed-run catch-up. The
service writes only inside staging; code, mappings, config and credentials are
read-only under systemd. First catch-up can exceed five minutes: systemd does
not start a second instance of an already-running oneshot. A 20-minute timeout
interrupts long catch-ups; completed chunks remain for the next run.

## Reports and stopping

Each execution creates a unique report, initially `running`, then atomically
replaced with the final result. Exit codes: 0 success, 1 failure, 75 shared lock
busy (expected skip), 130 graceful interruption. Invalid configuration can fail
before a report exists; inspect the journal too. Abrupt termination or power loss
may leave a `running` report: check the service before treating it as active.
Reports and ingestion diagnostics may contain upstream error text; keep staging
private. There is no automatic report pruning or backup installation. Retain
the existing snapshots and CSVs, monitor disk usage and repeated failures, and
use SQLite-aware backups rather than copying an open WAL database alone.

To stop collection without affecting the live CSV/control jobs:

```bash
sudo systemctl disable --now hepo-measurement-collector.timer
sudo systemctl stop hepo-measurement-collector.service
```

Do not delete the archive, CSVs, lock, reports or snapshots. Updating collector
code/config and moving any consumers to SQLite are separate rollout decisions.
