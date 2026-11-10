# Archive storage and ingestion operations

These commands are opt-in, not an operational cutover. No services or backup
jobs are installed. Never run a writable command on the real staging archive
without separate approval. `hepo.db`, recorder and Home Assistant are not
measurement archive migration targets.

## Identity and migration

Opening `MeasurementArchive` creates schema v2 on a new path or transactionally
migrates a structurally recognized v1 archive. Application ID `0x4845504d`, an
application identity and a persistent UUID distinguish v2 archives. Unknown
versions, foreign applications and unexpected schemas fail before journal-mode
changes. The migration and namespace binding commit together; failure rolls
them back. `MeasurementReader` supports v1 and v2 read-only and never migrates.

Operational ingestion requires an explicit `--recorder-namespace`, a stable
operator-managed installation/database-generation identifier, not a password or
an identity inferred from connection credentials. First binding of an old
archive is an explicit operator assertion that existing IDs belong to that
recorder generation. Once bound, a mismatch fails closed. A rebuilt recorder
requires a separate NEW archive, not a renamed namespace on an existing archive.
Low-level Python APIs retain optional namespace/predecessor arguments for
fixtures and compatibility; operational callers should always bind identity.

## Incremental updates

```sh
venv/bin/python3 -m utils.measurement_ingest_cli \
  --archive /local/path/measurements.sqlite \
  --sources /local/path/audited-sources.yaml \
  --from 2026-10-08T00:00:00+03:00 --through-now \
  --recorder-namespace household-recorder-generation-1 \
  --overlap-seconds 900 --chunk-seconds 86400 --retries 2 \
  --report /local/path/ingestion-run.json
```

`--through-now` is mutually exclusive with `--to`; it captures the clock once
and floors the cutoff to a UTC minute. Pipelines should instead supply their
already-fixed `--to` cutoff to every stage. `--from` is an explicit bootstrap
floor: repeat it on subsequent runs to fill old gaps as well as append history.
When omitted, it is inferred from existing queried coverage, and every selected
source must already have coverage. Empty archives/sources require explicit
bootstrap bounds. Source switches clip requests before any fetch.

`--include-default-sources` is allowed only with `--sources`: it merges the
runtime default inventory with the dated YAML overrides. Repeated `--measurement`
selects required names. It does not prove that defaults match installed meters;
operators must audit the complete resolved mapping. Missing recorder entities
fail/report independently rather than acquiring empty successful coverage.

Add `--dry-run` to print request bounds without recorder connections, writes,
archive creation, migrations or report-file creation. With explicit bounds it
does not even open the archive. It reports candidate requests, not a claim that
all requested history will be refetched: actual updates fetch gaps plus recent
overlap only. Omitted bounds require read-only inspection of existing coverage.

Recorder SQL uses bounded keyset pages (default 2,000 rows), stable ordering by
timestamp and recorder state ID, no OFFSET or unbounded buffered SQL result.
Time chunks default to one day. Each successful chunk commits independently;
restart fills the unfinished gaps. Retries are finite and configured, with
PostgreSQL rollback after a failed attempt. One source's failure does not stop
other sources. `ingestion_runs` records chunk bounds, attempt, outcome, observed
event count and error; `--report` exports this invocation's diagnostics and
failures as JSON to a new explicit output path (no overwrite). Failed event
counts are attempted, not committed counts.
An abrupt process death can leave a committed chunk without its following
diagnostic row; coverage, not diagnostics, is the resume authority.

Recent overlap captures late recorder rows and duplicate reads are idempotent.
Rows arriving older than the overlap require explicit backfill against a NEW
archive or an intentional bounded overlap large enough to include the affected
tail; the tool does not silently requery all covered history. A changed row with
an existing recorder ID is an error, never an overwrite.

## Predecessor and coverage semantics

The operational CLI fetches the latest recorder event strictly before each
chunk's left boundary, including the highest state ID at equal timestamps.
Predecessor observations and boundary evidence commit with the successful
range; generator/query failure rolls all three back. A predecessor does not
extend queried coverage backwards. `MeasurementReader.carry_supported(source_id,
event_timestamp, coverage_start)` accepts an event at/after that boundary or
committed evidence at that exact boundary matching the event timestamp. V1
archives have no predecessor evidence. Source mapping, invalid-state and expiry
rules remain the resampler's responsibility; evidence must never override a
mapping switch or an outage. Queried coverage is not measurement availability.

## Backup, restore and proposed schedule

```sh
venv/bin/python3 -m utils.measurement_backup backup \
  --source /local/path/measurements.sqlite \
  --destination /local/backups/measurements-2026-11-10.sqlite
venv/bin/python3 -m utils.measurement_backup verify \
  --source /local/backups/measurements-2026-11-10.sqlite
venv/bin/python3 -m utils.measurement_backup restore \
  --source /local/backups/measurements-2026-11-10.sqlite \
  --destination /local/restore-test/measurements.sqlite
```

Backup uses SQLite's online backup API from one read snapshot, so committed WAL
contents are included. Do not copy only the main file of a live WAL database.
Source and copy undergo full integrity, foreign-key and archive-identity checks.
Restore uses the same verified copy procedure and requires a new destination;
existing paths are never overwritten. V1 backups stay v1. URI/network connection
strings are rejected; paths must refer to local files. Backup checks scan the
archive and hold a read snapshot; allow disk headroom for WAL growth during a
large backup. A failed owned destination is removed; interruptions may leave an
incomplete destination, which must be verified and reviewed before removal.

Proposed operator-installed schedule: daily uniquely dated verified backup after
ingestion, an additional backup before any schema migration/import, and weekly
restore-to-new-path verification. Keep several generations and an independently
protected off-host copy under an explicit retention policy. This document does
not install timers, prune backups or modify `docs/SCHEDULES.md`'s live schedule.

## Evidence and remaining rollout gates

Fixture tests cover migration rollback, v1 read-only access, namespace mismatch,
unrelated DB rejection, predecessor rollback, random gap/backfill planning,
late records, chunk failure/resume, finite retry, fixed cutoff, dry-run and WAL
backup/new-path restore. No live recorder calls or staging migrations were used.
Bounded SQL and time chunks constrain ingestion buffers, but production-scale
throughput, recorder query plans and disk/WAL growth still require approved
representative profiling. Backup integrity checking is intentionally full-scan.
Full training-window frame memory is outside this storage implementation.
