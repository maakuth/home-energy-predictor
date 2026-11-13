# Controlled CSV/SQLite history replay

`utils.storage_replay` compares historical data paths under one common inference
contract. It never fits a model, calls HA/PostgreSQL, writes a feedback database,
executes the optimizer, changes control or emits a new forward forecast vintage.

It holds fixed the compatible model and feature order, captured weather/PV
inputs, original issue time, target grid, UTC calendar and canonical physical
bounds. Both routes derive exact cutoff-relative anchors and preceding-day
Leaf energy, without filling missing meters or treating corrected zeros as
verified truth. Autonomous heating is compared using the same preceding-day
observed-mean assumption, with valid-minute counts reported.

```sh
venv/bin/python3 -m utils.storage_replay \
  --archive /staging/measurements.sqlite --cutoff 2026-11-12T19:41:00Z \
  --sources /frozen/shadow-sources.yaml --measurement-policy /frozen/shadow-policy.yaml \
  --csv /static/raw-data-snapshot.csv --model-dir /isolated/candidate/state \
  --snapshot /capture/shadow-inputs.json --vintage /capture/shadow-forecast.json \
  --actuals-cutoff 2026-11-13T20:42:00Z --output /new/replay.json
```

## Interpretation and safeguards

* Full CSV history coverage for the preceding 24 hours is mandatory. Sorted,
  unique, timezone-aware minute timestamps and explicit five-meter validity
  flags are required. Incomplete windows are not shortened silently.
* CSV flags establish declared CSV validity, **not recorder verification**.
  CSV values must already use the documented canonical units; their original
  event units, observation ages and event-level outlier durations cannot be
  recovered from resampled values. Physical bounds are rechecked on CSV bin
  means; recorder bounds were applied before duration integration.
* Snapshot, model, archive identity and interpretation hashes must match the
  captured experiment. Model artifacts and source/policy controls are checked
  for changes. Recorder history and optional actuals share one read-only
  snapshot, closed before model inference. No model parameters change.
* Both data paths use the same captured weather, never their own observed
  weather values. Differences are reported per raw measurement, anchor and
  fitted feature (with appropriate units), plus predictions and heating/load
  proxies. A route with missing anchors is blocked, not defaulted to zero.
* Optional retrospective scores compare both routes against identical completed,
  valid recorder quarters. No actuals beyond the supplied cutoff are read.
  These are replay diagnostics, not independent forward model-selection scores.
* The archive-to-saved-vintage comparison checks whether currently retained
  history reproduces the original forecast. This is **not** guaranteed as-of-issue
  history: late-arriving records can alter an event-time replay.

This isolates normalized historical data inputs; it is **not a bit-identical
replay of the old serving code**. In particular, old online anchors read
PostgreSQL events rather than CSV, and some old fallback/aggregation semantics
differ intentionally. Known minute-target/quarter-output and frozen-origin lag
approximations are held fixed, not repaired or tuned here.

## November 13 controlled results

Artifacts: `state/measurement-migration-20261110T152856Z/` under
`storage-replay-20261113T211016Z/`, using a stable copied 4,320-row raw CSV spanning
November 10 21:00–November 13 20:59 UTC. The same frozen candidate and three
previously captured forecast origins were selected.

| Origin | Controlled comparison result |
| --- | --- |
| November 11 07:29 UTC | Blocked: copied CSV lacks the full previous-day context. |
| November 12 19:41 UTC | 102 model predictions **identical** across CSV/SQLite; archive replay also exactly reproduces the saved vintage. |
| November 13 20:42 UTC | CSV's one-hour baseload anchor unavailable because its PV value/flag is missing. SQLite has valid recorded zero PV; no replacement was invented to force equality. |

In the fully comparable replay, 24-hour heating means were 1.191758333 kW (CSV)
and 1.191689877 kW (SQLite): a difference of about **0.0685 W**. The two routes
also had identical retrospective MAE (1.203592 kW) on the same 99 completed
quarters, because their predictions were identical. This isolates the data path
in that experiment; it does not rank different models or establish universal
equivalence.

Raw minute values are not generally identical. `aggregation-evidence.json`
independently recomputes the maximum-disagreement minute for each of the five
meters: the CSV matches the arithmetic mean of event samples, while SQLite
matches elapsed-time integration including the preceding state. All five
checks agree with their respective declared aggregation. Rapidly changing
loads can therefore have large single-minute differences without corruption.

`pv-tail-gap-explanation.json` traces the blocked November 13 anchor: the last
recorded PV state was 0.0 kW at 12:48:12 UTC. The CSV validity series ends at
12:48, and its value ends at 13:03 after limited interpolation. Per-source CSV
resampling stops at the last source event before the global merge; the merged
tail stays missing. The archive instead carries the recorded zero inside
successful query coverage under the operator-approved availability policy.
This is an explained input-availability difference, not evidence of a SQLite
data-loss regression. It does not prove the old online PostgreSQL anchor path
would have failed in the same way.

`summary-v2.json` and `*-replay-v2.json` are the current reports. Earlier replay
reports remain retained but were superseded to label feature-difference units
explicitly (for example Leaf energy is kWh, not kW).

## Full plan replay remains a separate gate

The existing captures lack a common frozen set of import/export prices, battery
SOC/constraints, thermal/relay/EV state, dispatch configuration/parameters and
current-slot latches. `optimizer_replay.status` therefore remains
`not_evaluated`. A baseload-plus-heating proxy is not an optimizer plan.
Capture those inputs explicitly before a non-actuating full plan comparison;
do not substitute today's live values into an old origin or invent SOC/prices.
No migration-readiness or live-cutover approval follows from this report alone.

Validation: targeted replay/shadow/calendar/consumer tests passed 74 cases.
The full repository suite passed **1,007 tests**, with one skipped and four
passing subtests; repository-wide Pyright was clean. Test-driven regressions
cover equal and differing anchors, incomplete/invalid CSVs, physical bounds,
fixed model/snapshot controls, completed-only actuals and non-actuation.
