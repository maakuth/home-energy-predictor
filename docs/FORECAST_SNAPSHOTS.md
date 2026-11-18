# Experimental forecast snapshot collection

This is an independent, non-actuating capture primitive, **not a deployed
service or a validated NIBE replay pipeline**. It calls only Home Assistant
state reads and the read-only `weather.get_forecasts` service. No live call,
timer installation or source change is authorized by the existence of this
code. Get explicit permission before attempting a live capture.

The CLI needs a dedicated SQLite archive path in an existing directory, a scheduled
timezone-aware origin, HA URL, weather entity, price entity, zone and an
`HA_TOKEN` environment variable. For example, after approval:

```
python -m utils.forecast_snapshots --archive /path/to/forecast-archive.sqlite \
  --origin 2026-11-18T12:00:00Z --host https://example.invalid \
  --weather-entity weather.home --price-entity sensor.example \
  --price-zone FI
```

The raw weather service response and weather state are saved unchanged,
including any supplied units and issue times. The raw price entity state is
likewise retained with the configured entity and zone; its classification is
**unverified**. The CLI accepts Nordpool `raw_today`/`raw_tomorrow` objects
with explicit `start`, `end`, `value`, or ENTSO-E
`prices_today`/`prices_tomorrow` (or `prices`) objects with `time`, `price`.
Timestamps must be timezone-aware, prices finite numbers, and intervals or
point timestamps must not overlap or duplicate. At least one Nordpool
interval must end after the scheduled origin, or one ENTSO-E point must be at
or after it. Nordpool tomorrow data is ignored for validation when
`tomorrow_valid` is false. Untimestamped numeric arrays, mixed schemas and
other variants are not supported. A validation failure records a price error;
the weather capture remains independent.

This checks only the presence and basic integrity of a window, not its
continuity, horizon, freshness at completion, actual publication time, units,
or zone. ENTSO-E points have no inferred duration and cannot yet be mapped as
intervals. The configured zone is provenance, not a verified provider zone.
No issue or publication time is invented. A successful capture is not proof
that prices are raw spot instead of a tariff, fallback or prediction, nor that
they match NIBE Smart Price Adaption. Inspect and document the actual source
before replay. HEPO's `utils/price_utils.py` distinguishes several
sources and derives import prices; those tariffs must not be relabelled spot.

Every invocation stores a separate attempt row with source start/completion
times, status and error type; exception messages are not stored because they
can include credentials. Timestamped arrays of point objects are split into
`forecast_series`, ordered `forecast_members` and deduplicated
`forecast_points`. The points have indexed original target strings and columns
for temperature, wind speed, wind bearing and price `value`; extra fields
remain JSON. The non-point response envelope remains JSON in `payloads`.
Identical complete payloads and identical individual points are shared using
SHA-256. All point fields, ordering, nested envelopes and source-provided
timestamp strings can be reconstructed; original JSON whitespace and object
key ordering are not retained.

Extraction recognizes nonempty arrays whose objects each have a string
`datetime`, `start`, `period_start`, `timestamp`, `target` or `time` field. Other
source shapes remain in the envelope, rather than being guessed or dropped.
The target index holds raw provider strings, not normalized instants: adapters
must validate their timezone and interval semantics before time-range queries
or replay. No native compression is applied. The database has an indexed UTC
origin and is independent of `hepo.db` and the measurement archive. Existing
databases without the forecast archive identifier are refused, not modified.
Schema version 1 forecast databases are also refused; no automatic migration is
provided.
Writes use a single SQLite transaction, including quota checks and payload
insertion. A partial source failure exits nonzero but preserves the successful
source.

Missing invocations cannot create their own failure record. Compare expected
scheduled origins to attempt rows using `vintage_coverage()` below. Quotas
(`--max-attempts`, `--max-bytes`) fail closed without deleting old data;
monitor disk and arrange reviewed archival/retention before the limit is
reached. The byte quota limits SQLite's database pages at commit, not
temporary rollback journals, filesystem usage or external writes. The default
100,000 attempts covers under three years at a 15-minute cadence; choose limits
for the intended retention and actual payload sizes. Never point it at an active
artifact database. The earlier experimental JSON-file format is not migrated
automatically; preserve any such files separately until a reviewed importer
exists.

`select()` returns the latest same-attempt successful pair whose scheduled
origin and both completion times are no later than the cutoff. Optional
`exact_origin` requires that scheduled origin, without falling back to an older
attempt; `max_age` is a nonnegative `datetime.timedelta` measured from the
scheduled origin to the cutoff, inclusive. The two restrictions can be combined.
They do not change the completion-time cutoff. Timestamps must be timezone-aware.
The default remains unrestricted latest-as-of selection. For example:

```python
from datetime import timedelta
from utils.forecast_snapshots import select, vintage_coverage

pair = select(archive, cutoff, exact_origin=origin, max_age=timedelta(minutes=30))
coverage = vintage_coverage(archive, expected_origins, cutoff)
```

`vintage_coverage` returns one row per supplied origin in input order, with UTC
`origin`, `status` and `attempts` (the number of attempts whose origin and
both source completions are no later than cutoff). Status is `complete` if any
such attempt has both sources successful, `failed` if there are visible attempts
but none complete, `not_available` if attempts exist but none were visible by
cutoff, and `missing` if no attempt exists. This is a **retrospective diagnostic**:
the distinction between `missing` and `not_available` uses rows after cutoff,
and must not be used as an as-of model feature. Expected origins and cutoff
must be explicitly supplied; the report does not infer schedules. It reports
attempt availability, not the validity or target coverage of payloads.

Selection does not validate the semantics or coverage of raw payloads (and
callers of `capture()` can supply arbitrary payloads). `map_intervals()` maps
only fully covered quarter-hour intervals, rejecting overlaps and never
extrapolating. Provider-specific adapters must establish explicit interval
semantics; a timestamp alone does not prove an hourly validity interval.

## Work required before rollout or model claims

* Validate the price source and publication timing and establish whether it
  matches NIBE Smart Price Adaption. Identify fallback and missing periods.
* Build validated weather/price adapters, gap/coverage reporting and an
  isolated dry-run integration with source-valid GSHP and priority history.
  Missing priority must not become a DHW label; a power-state proxy must be
  explicitly labelled as such.
* Implement and test sequential scoring, cutoff-safe persistence and model
  features, lead-time and coverage reporting, MAE, bias and energy error.
  The archive primitive alone does not provide a scoring pipeline.
* Review data retention and timer proposal separately: cadence and expected
  origins, source latency, timeouts, overlap, capacity alerts, failure and gap
  monitoring, and rollback. Collection must not run in or block live
  optimization. No service or timer is included or enabled.

## Optional nonblocking collection

Forecast snapshots can be collected by an **independent** systemd user service.
They are never run from the optimization path of `run_frequent.sh`, and the
archive pipeline dry-run does not invoke them. `run_frequent.sh` may optionally
trigger the configured service with `systemctl --user start --no-block` when
`FORECAST_SNAPSHOT_SYSTEMD_UNIT` is set; this returns immediately and cannot
delay or fail optimization. By default the variable is empty and no trigger is
attempted.

Configuration is explicit. Copy `config/forecast_snapshot.example.json`, replace
every placeholder with an absolute local path, and point `env_file` at the
repository `.env` containing `HA_HOST` and `HA_TOKEN`. The archive must be a
dedicated SQLite file, never `hepo.db` or an active artifact database.

Dry-run validation:

```bash
python -m utils.forecast_snapshots --config /path/to/forecast-snapshot.json
```

Execution:

```bash
python -m utils.forecast_snapshots --config /path/to/forecast-snapshot.json --execute
```

The origin is the current 15-minute UTC boundary. Collection holds a nonblocking
lock; a busy lock skips the run with exit status 75. Source failures are recorded
as attempt rows and do not stop the successful source.

## Deployment

Render and install the user service/timer like the measurement collector, but
do not enable it without review. The service runs `utils.forecast_snapshots` with
the configured JSON. It uses the shared project `.venv`; no separate environment
is created.

```bash
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
systemd-analyze --user verify /path/to/hepo-forecast-snapshot.service /path/to/hepo-forecast-snapshot.timer
install -m 0644 /path/to/hepo-forecast-snapshot.service "$UNIT_DIR/hepo-forecast-snapshot.service"
install -m 0644 /path/to/hepo-forecast-snapshot.timer "$UNIT_DIR/hepo-forecast-snapshot.timer"
systemctl --user daemon-reload
systemctl --user start hepo-forecast-snapshot.service
```

Only after reviewing the report and journal:

```bash
systemctl --user enable --now hepo-forecast-snapshot.timer
```

## Blocker for automatic execution

The existing frequent job is a systemd `oneshot` service. A background
subprocess started from it can be terminated when that service's cgroup is
cleaned up, so this implementation deliberately does **not** start capture in
the background from `run_frequent.sh`. Automatic execution therefore requires
the separate timer above (or an equivalent persistent unit). Without that
systemd change, capture remains a manual or externally triggered operation;
this is intentional and avoids unsafe fire-and-forget behavior. Optimization is
never blocked or failed by capture.
