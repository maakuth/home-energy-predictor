# Archive validation evidence — 2026-11-10

All operational-data checks below were read-only. No staging schema migration,
new CSV import, recorder/HA call, production fit, service switch or control
publication was performed.

## Full-length raw-window profile

`utils.measurement_profile` was run over the staging archive from
2024-10-05 00:00 UTC to 2026-10-08 00:00 UTC: **733 days, 1,055,520 minute rows**.
The two measurements were shared heating and outdoor temperature, with explicit
legacy reading and one-day chunks. Each chunk snapshot closes before the next;
this is a throughput/memory diagnostic, not an atomic historical comparison.

* 733 chunks; maximum 1,440 rows and 621,184 deep DataFrame bytes per chunk.
* Elapsed 830.54 seconds with tracemalloc enabled, about 1,271 rows/second.
* Python traced peak 4,632,360 bytes; process RSS high-water 87,478,272 bytes.
* SQLite plans used import-ID/time primary-key searches and
  `observations_time(source_id, timestamp, recorder_id)`, not full-table scans.

These figures are specific to two legacy measurements on this machine.
Tracemalloc excludes some native allocations; RSS includes interpreter/import
costs. They do not certify PostgreSQL throughput, writer latency under live WAL
load, an eight-signal feature window, or XGBoost/SARIMA fitting memory.

The full run began before final registry metadata integration. A final-code
seven-day repeat over October 1–8 processed 10,080 rows in 8.03 seconds, with
624,064 bytes maximum chunk frame size (including added unit-verification
columns), 4,654,243 traced peak bytes and 87,187,456 bytes process RSS high-water.
Index plans were unchanged. Neither run fitted a model or wrote staging data.

## Feature construction memory and snapshots

`iter_archive_features` constructs chunks under **one** consistent read snapshot,
including exact preceding-day lag context and seven-bin centered-median halos.
Exhaust or explicitly close its generator; do not fit while holding it open.
Invalid chunks remain ineligible rows rather than being filled or dropped.
Archive processing never substitutes forecast solar for a verified zero meter.
CSV compatibility retains its original substitution configuration.

`load_archive_training_features` projects required features and eligibility
columns before concatenation, discarding repeated per-measurement tuple/string
metadata while preserving source/import/policy identity and window quality
summaries in attributes. Pipeline XGBoost uses this path via
`--feature-chunk-seconds 86400`. **The final fit matrix remains O(rows)**; this
is bounded feature construction, not a constant-memory XGBoost implementation.
Full production-window fit resource budgeting remains an operational check.

Tests compare streamed and whole-window features, retain entirely invalid
chunks, verify closure before fitting, and commit a concurrent WAL writer while
asserting that later feature chunks still see the original snapshot.

## Real, non-actuating matched shadow

Compared the already-imported October snapshot's `raw_data.csv` against staging
over its complete 72-hour window, 2026-10-04 21:00 UTC to October 7 21:00 UTC,
at one-minute resolution. Both raw and feature archive reads shared one snapshot.
No import preference or explicit measurement registry was supplied; these remain
legacy canonical-compatibility values, not unit-verified recorder truth.

* 4,320 timestamps; none missing from the selected CSV timeline.
* 459 eligible legacy baseload actuals; **zero verified actuals**; 3,861 excluded.
* All matched finite raw values agreed within numerical tolerance. Conflicts
  remained visible: temperature 8, shared heating 3, grid 3, forecast 16, Leaf 17.
* Only 33 timestamps had honest matched actuals and exact previous-day anchors.
  Repeat-yesterday MAE was 1.06427 kW on both routes; forecast differences were
  below 4e-16 kW mean absolute error.

This is expected consistency for CSV bytes already imported into the archive,
**not independent validation of recorder fidelity or a model ranking**. The
short window was not used to tune any horizon, threshold or discount. Verified
versus legacy results remain separate, and scores with no samples remain null.
CSV shadow reads are streamed, retaining only the requested window plus context;
duplicate/unordered input is rejected rather than silently sorted/folded.

Representative verified-recorder shadows, post-replacement coverage and full
model evaluation still require authorized recorder data and audited metadata.
