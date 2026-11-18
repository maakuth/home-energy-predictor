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
likewise retained; its classification is **unverified**. No issue or
publication time is invented. A successful HTTP/state read is not proof that
the selected entity contains a forecast window, nor that it is raw spot
instead of a tariff, fallback or prediction. Inspect and document its schema,
units, zone and relationship to NIBE before building a price adapter or
using it in a replay. HEPO's `utils/price_utils.py` distinguishes several
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
`datetime`, `start`, `period_start`, `timestamp` or `target` field. Other
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

Missing invocations must be detected by comparing scheduled origins to
attempt rows; they cannot create their own failure record. Quotas
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
origin and both completion times are no later than the cutoff. It does not
validate the semantics or coverage of raw payloads. `map_intervals()` maps
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
