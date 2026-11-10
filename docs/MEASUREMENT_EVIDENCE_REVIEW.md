# Supplied evidence review — 2026-11-10

Input: operator-produced `measurement-evidence.json`, generated at
2026-11-10 09:52:17 UTC, SHA256
`954b2b61a70006d555e7025f609aece1e2849a3dc52a36e8380b56026b067ea2`.
Inspection window: November 3 09:52 through November 10 09:52 UTC. PostgreSQL
confirmed read-only operation; no query errors were reported. HA results are
current metadata, not historical verification. The original report is not
modified, imported into the archive, or automatically promoted to a policy.

## Resolved observations

| Measurement | Observed recorder source | Sampled unit |
| --- | --- | --- |
| Outdoor temperature | `sensor.ulkolampotila` | °C |
| Shared heating | `sensor.instantaneous_used_power_32167` | W |
| Grid | `sensor.sahkokauppa_nyt` | kW |
| Solar actual | `sensor.solar_plant_real_power_kw_2` | kW |
| Battery | `sensor.be_stat_batt_power` | W |
| Leaf | `sensor.tasmota_energy_power_3` | W |
| Wind attribute | `weather.home.wind_speed` | km/h via `wind_speed_unit` |
| Tomorrow solar forecast **state** | `sensor.solcast_pv_forecast_forecast_tomorrow` | **kWh, device_class energy** |
| Supply / return | `sensor.nibesupply_line_bt2_30006` / `sensor.nibe_return_line_bt3_30008` | °C |
| Energy counter | `sensor.mummun_energy` | kWh, total_increasing |

Normal wind units were present in 152 of 163 events; 11 had missing metadata and
nonnumeric/unavailable values. Leaf had three invalid/nonnumeric events and
solar actual one. These do not become verified zeroes. No negative battery
values were in the 200-event sample, so it does not independently establish the
charging/discharging sign convention. Database/cluster/table identity evidence
is available, but namespace continuity still needs operator confirmation.

## Confirmed training/inference mismatch

All 42 inspected tomorrow-forecast recorder events declared daily **energy**
(`kWh`, `device_class: energy`, `state_class: total`). Legacy extraction reads
that entity's scalar state into `solar_forecast`; old XGBoost models include it.
Inference fills `solar_forecast` with target-hour `pv_estimate` power
from the live `detailedHourly` forecast. Before the approved repair, the archive
default mapping repeated the legacy scalar source with a kW consumer convention.

This is a genuine input-contract mismatch, not a reason to change baseload
meter truth or optimizer thresholds. Do not relabel kWh as kW, divide a daily
total by 24 to invent a profile, difference successive forecast revisions as an
energy counter, or replace it with observed solar under an unchanged
"forecast" label. Quality targets are captured before compatibility solar
substitution; the report does not establish that all current targets are bad.

The sampled recorder report contains no recognized forecast-entry keys, whereas
the current HA result contains `period_start` and `pv_estimate*`. The first
collector did not distinguish absent arrays from empty arrays. It therefore
does **not** prove that all historical hourly payloads are unavailable. The
follow-up collector now reports those attribute shapes explicitly.

### Considered repair choices

1. Discover a retained power forecast or detailed historical payload. Verify its
   target interval, issue/availability time, unit history and coverage before
   choosing a new source. A state called `power_now` is not automatically the
   same as a future-target hourly forecast.
2. Existing prediction archives store `solar_forecast_kw` with both
   `target_timestamp` and `generated_at`, potentially preserving the exact input
   series. A proper adapter must retain both timestamps and select a vintage
   available at the defined forecast origin **for each target**, with explicit
   minute/quarter-hour alignment. Latest-per-target or a global cutoff is not
   sufficient; retrospective/backfilled predictions must be excluded.
3. Explicitly remove the XGBoost predictor and retrain with a new feature/model
   contract. This also changes eligibility/required inputs, not just the feature
   list. Keep planner solar forecasts independently; they are not the same
   dependency as this ML predictor.

The user explicitly selected **option 3: remove the predictor**. Model version
1.47.0 implements an 18-feature baseload contract (version 2), removing forecast
source/eligibility/ingestion dependencies. Measured PV remains mandatory for
baseload truth and live solar forecasts remain in planner output. Optional raw
archive access labels the legacy daily state kWh, not kW. Archive inference and
hindsight reject old/mixed model artifacts; explicit retraining and validation
are still required. CSV compatibility remains available for rollback. No
production model was retrained and no accuracy improvement is claimed.

## Retention and instrumentation cautions

* The new temperature entity's earliest retained event is October 20 06:41 UTC,
  after the user-declared October 19 00:00 Helsinki mapping boundary. Preserve
  the resulting unknown interval; do not silently shift the mapping or carry
  the old sensor across it.
* The current shared-power entity has records beginning October 20 05:27 UTC,
  just before its declared 08:30 Helsinki activation boundary. This does not
  redefine the source boundary or the approximate equipment-removal time.
* The current solar source begins August 3; battery begins May 12. Earliest
  retained events are **not installation dates** or proof of earlier zero load.
  Older aliases overlap newer histories; no switch boundary is inferred below.
* The reported EV SoC/position entities have no events after July 15 and current
  HA requests returned HTTP errors (status not recorded). They are not usable
  fresh observations; no replacement entity is guessed.
* The old pump-temperature source has only one sampled event, November 6, at
  97.66 °C. No claim of current physical tank temperature follows from it.
  Supply/return readings are circuit temperatures, not an audited accumulator.
* Up to ~16–19 hour update gaps in solar/Leaf samples can reflect unchanged
  night/idle states. They cannot by themselves set expiry limits or prove an
  outage. No freshness threshold is inferred.

## Local forecast archive inspection

Read-only metadata inspection found the workspace-root `hepo.db` empty (no
tables); no schema was created. The October 8 snapshot has a `predictions` table
with both timestamps and nominal solar kW. Its first **text-indexed** target
endpoint is April 15 with a May 30 generation timestamp: direct evidence that
some rows are retrospective. Its final indexed target is October 8. These are
indexed endpoint observations, not a full UTC/date/count or forecast-origin
coverage audit. No SQLite data or WAL was modified. Current host archive
coverage is still unknown.

## Follow-up report reviewed

The supplied `measurement-source-followup.json`, generated November 10 at
11:54:19 UTC, discovered 38 entities without prefix truncation or PostgreSQL
errors. Retained `power_now`, `power_in_30_minutes` and `power_in_1_hour` states
report W and begin April 30, 2025. These are not automatically equivalent to
future-target hourly forecast vintages. `forecast_this_hour`/`forecast_next_hour`
report Wh; daily forecast states report kWh.

Sampled recorder Solcast events have absent `detailedHourly`/`detailedForecast`
arrays; current HA today/tomorrow entities have populated arrays. This does not
prove absence throughout all historical events. No historical curves are
reconstructed and no additional broad discovery report is needed for removal.

Current solar aliases report W (`sensor.solar_plant_real_power_2`) and kW
(`sensor.solar_plant_real_power_kw_2`). Older `solarh_63038` histories end around
August 7 and overlap newer histories. Endpoint dates do not establish a valid
source-switch boundary or equipment installation date. Namespace continuity,
battery sign, evidence-based freshness limits and authorized staging writes
remain operational gates; the reports are not automatically promoted to policy.
