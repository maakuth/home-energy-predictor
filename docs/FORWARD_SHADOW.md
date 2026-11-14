# Non-actuating forward shadow validation

`utils.forward_shadow` captures forecast vintages, predicts into a new explicit
JSON file and scores them against later recorder actuals. It never calls the
operational `predict()`, optimizer, publisher, price/cache pipeline or feedback
SQLite writer. There is no default output, model directory or history archive.
New output paths are required; active artifact filenames are rejected. No
timer/service is installed and existing schedules remain unchanged.

For fresh common-contract CSV/SQLite comparisons, use the
[origin-frozen CSV sidecar workflow](STORAGE_CAPTURE.md). Capture accepts paired
`--csv` / `--csv-output` arguments; they are optional for existing shadow users.
The sidecar is frozen before HA requests, and its byte hash/cutoff/timing are
bound into the input snapshot and verified before generation/replay.
`capture --wait-for-csv-seconds 900` can wait for the next normal CSV refresh
before freezing or contacting HA; it never requests ingestion or backdates capture.

## Three explicit operations

```sh
# Host only: four HA state GETs and the read-only weather.get_forecasts request.
venv/bin/python3 -m utils.forward_shadow capture \
  --env-file /absolute/path/to/.env --output /new/run/shadow-inputs.json

# Refresh the staging recorder through the snapshot's history_cutoff using the
# existing ingestion CLI before generation. Do not use a later cutoff here.
venv/bin/python3 -m utils.forward_shadow generate \
  --archive /staging/measurements.sqlite --cutoff SNAPSHOT_CUTOFF_ISO \
  --sources /frozen/shadow-sources.yaml --measurement-policy /frozen/shadow-policy.yaml \
  --model-dir /isolated/candidate/state --snapshot /new/run/shadow-inputs.json \
  --output /new/run/shadow-forecast.json

# After future observations arrive, refresh staging and score with a later cutoff.
venv/bin/python3 -m utils.forward_shadow score \
  --archive /staging/measurements.sqlite --cutoff LATER_ACTUALS_CUTOFF_ISO \
  --sources /frozen/shadow-sources.yaml --measurement-policy /frozen/shadow-policy.yaml \
  --forecast /new/run/shadow-forecast.json --output /new/run/shadow-score.json
```

The snapshot stores real request start/completion times and a whole-minute
historical cutoff captured before requests. HA operations are limited to
Solcast today/tomorrow, `sensor.ulkolampotila`, `weather.home`, and its hourly
forecast read. The weather request uses POST transport but does not actuate.
Both direct and modern `service_response` envelopes are supported. Capture
never returns credentials in its data. Offline operations do not contact HA or
PostgreSQL; the reused predictor helpers are pure inference/anchor functions.
SQLite read-only WAL access may create shared-memory/empty WAL sidecars, but
does not mutate main archive observations or perform feedback writes.

## Honest origins, inputs and baselines

* Explicit source and policy files and recorder-only `verified-only` history
  are required. Archive UUID, recorder namespace and exact config hashes are
  retained. Actuals scoring refuses changed interpretation or another archive.
* The existing 18-feature model contract, embedded booster and feature names
  are validated. Metadata must show training before the forecast origin.
* Snapshots must match the historical cutoff and precede generation. Capture
  must be at most 15 minutes old; an overly long refresh requires a new capture,
  not a backdated issue time. This is a shadow-vintage freshness guard, not a
  new sensor expiry policy.
* Temperature/wind units must match C/km/h. Required forecast fields must be
  finite; conflicting duplicate timestamps, negative PV/wind and unsupported
  forecast gaps fail rather than inventing weather or solar profiles.
* Existing model feature construction is reused, but elapsed/current partial
  quarters are omitted. Every emitted 15-minute target starts strictly after
  the recorded generation time. No prices, thermal dispatch or active plan is
  needed to test baseload predictions.
* Calendar features use **UTC**, matching `process_frame` training. The existing
  operational predictor instead uses Helsinki calendar fields; this mismatch
  was discovered during shadow preparation and remains unchanged in live code.
  Each shadow artifact declares its UTC convention. This is a candidate
  experiment, not a bit-identical replay of the operational inference route.
  Resolving and testing the live calendar contract is a separate rollout gate.
* One-hour persistence is fixed at the historical forecast origin for the whole
  horizon. It does **not** read each target's future one-hour-lag actual.
  Previous-day profiles use only complete quarters already observed by origin;
  unavailable/beyond-history profile values remain null.
* Optional `generate --reference-forecast /captured/reference.json` reads only
  `predicted_baseload` from an explicitly captured JSON list. Unsupported total
  load fields are not silently reinterpreted. The host runner retains reference
  observation time and file mtime, without pretending these establish its
  original model-generation time. The active reference file is never changed.

## Scoring and operational limits

Only completed target quarters at or before the actuals cutoff are scored.
Five aligned meter definitions, invalid-state handling and physical bounds are
unchanged. Incomplete/invalid actuals are excluded, not filled with zero. Each
baseline comparison reports candidate and baseline on the **same timestamps**.
Scores include lead-time groups, unsettled/invalid counts, MAE, RMSE and bias
(prediction minus actual). No matched rows means null errors, not zero errors.
Every vintage remains separate; there is no best-vintage or latest-after-target
selection.

During the November staging migration, `host-forward-shadow.sh` in the migration
run directory orchestrates capture, read-only recorder refresh, generation and
scoring. `forward-shadow/protocol.json` pins the code revision, candidate file
hashes, source/policy hashes and archive identity. Each invocation uses a new
directory under `forward-shadow/`; `--score-only` skips HA requests and only
refreshes/scores existing vintages. `--dry-run` checks the frozen inputs without
recorder/HA calls, archive mutation or forecast generation. Do not run ingestion
concurrently. The runner is not an OS sandbox, so use only the pinned reviewed
code and never replace it with the operational prediction pipeline.

This is data collection for a forward evaluation, not approval to promote the
candidate. Let actuals arrive and review representative horizons, missingness,
both recent heating regimes where available and matched baseline errors before
requesting a separate live decision. A single issue/score run is not enough to
establish model superiority or migration readiness.

Validation: 981 repository tests passed, one skipped and four passing subtests,
including slow tests; repository-wide Pyright was clean. Focused tests cover
non-actuation, no feedback/active artifact writes, honest origins/baselines,
UTC feature alignment, unit/contract errors, HA response envelopes and invalid
future actuals. A synthetic-input smoke test read the frozen real candidate and
staging archive without network calls; it was not a real captured forecast.
