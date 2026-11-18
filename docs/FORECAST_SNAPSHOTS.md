# Experimental forecast snapshot collection

This is an independent, non-actuating capture primitive, **not a deployed
service or a validated NIBE replay pipeline**. It calls only Home Assistant
state reads and the read-only `weather.get_forecasts` service. No live call,
timer installation or source change is authorized by the existence of this
code. Get explicit permission before attempting a live capture.

The CLI needs an existing, dedicated archive directory, a scheduled
timezone-aware origin, HA URL, weather entity, price entity, zone and an
`HA_TOKEN` environment variable. For example, after approval:

```
python -m utils.forecast_snapshots --directory /path/to/dedicated-archive \
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

Every invocation stores an immutable uniquely named JSON attempt, with
separate source start/completion times and error type; exception messages are
not stored because they can include credentials. A partial failure exits
nonzero but preserves the successful source. Missing invocations must be
detected by comparing scheduled origins to files; they cannot create their own
failure record. Quotas (`--max-files`, `--max-bytes`) fail closed without
deleting old files; monitor disk and arrange reviewed archival/retention before
the limit is reached. The quota applies to cooperating processes in this
directory, not other writers. Never point it at an active artifact directory.

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
