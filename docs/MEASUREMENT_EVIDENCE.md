# Operator-run measurement evidence collector

The container cannot reach the recorder. `utils/measurement_evidence.py` is a
standalone, read-only collector for the operator's host. It does not require
checking out/merging the archive migration, importing application/control code,
or opening a measurement archive. Dependencies are the existing `psycopg2` and
`python-dotenv`; PyYAML is needed only with a custom `--sources` file.

Run from the host application's directory with its normal virtualenv activated.
The script reads **that working directory's** `.env`, preserving already-exported
environment settings. Credentials are not copied into its report.

To extract just the collector from the task branch without switching branches:

```sh
script=$(mktemp /tmp/hepo-evidence.XXXXXX)
git show task/measurement-archive-finish:utils/measurement_evidence.py > "$script"
python3 "$script" --ha --output measurement-evidence.json
```

Or, when running from the implementation checkout:

```sh
python3 -m utils.measurement_evidence --ha --output measurement-evidence.json
```

`--ha` is optional: omit it if only PostgreSQL is reachable. Output must be a
**new** non-operational `.json` file and is created with mode 0600. Without
`--output`, JSON is printed to stdout. Review before sharing: the report contains
entity/database names, retention dates and bounded numeric summaries, but not
passwords/tokens, raw attribute payloads, GPS coordinates or full forecast curves.

## What it does

* Requires the server to confirm `transaction_read_only=on`, never commits, and
  sets per-query statement/lock timeouts. Only metadata/history SELECTs and
  transaction-local read settings are issued. It never ingests, migrates,
  backs up, trains, changes schedules, writes active JSON or calls HA services.
* Collects PostgreSQL database/server/table identity evidence. Cluster identity
  may require privileges; denial is reported without changing privileges or
  preventing other inspection. An operator still confirms namespace continuity:
  restore/truncate/rebuild can reuse identity evidence.
* Uses endpoint `LIMIT 1` queries to inspect retained date bounds and samples
  at most 200 recent events per selected source by default. `--sample-limit`
  permits an explicit 1..10000 limit. Attribute payloads are capped at 65,536
  characters per observation; truncated/malformed JSON is not trusted.
* Default sampling window is the preceding seven days, captured once. Explicit
  `--from/--to` must be timezone-aware and span at most 31 days. Sampling is
  clipped to each declared source period and preserves mapping gaps. It is not
  a full coverage audit or evidence that a bounded sample contains all changes.
* Preserves observed `unit_of_measurement`, `state_class`, `device_class`, wind
  and temperature unit variants without converting energy to power. Numeric
  sign counts do **not** establish battery charging/discharging semantics.
  Update gaps do **not** automatically choose a freshness threshold.
* Optional HA inspection sends GET requests only to selected `/api/states/...`
  endpoints. Redirects are refused, responses are bounded and only selected
  metadata is retained. Current units cannot retrospectively verify historical
  ones. No POST, forecast service or control endpoint is called.

The embedded inventory matches the known sensor families and includes the
user-confirmed dated outdoor/shared-heating switches. Existing entity environment
settings apply to other runtime defaults. A custom dated YAML can replace it;
add `--include-default-sources` to merge the runtime inventory with that file.
Unknown fields, duplicate keys, overlaps and naive bounds are rejected.

For historical switch evidence, run a separate bounded inspection:

```sh
python3 "$script" --from 2026-10-01T00:00:00Z --to 2026-10-22T00:00:00Z \
  --measurement outside_temp --measurement gshp_power \
  --measurement solar_forecast --measurement wind_speed \
  --output measurement-evidence-october.json
```

No sample/metadata result is automatically turned into a production registry,
source-verification flag, namespace binding, import precedence or rollout
approval. Return the report(s) to resolve those remaining design/evidence gates.
