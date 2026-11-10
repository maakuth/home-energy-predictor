# Read-only CSV/archive shadow comparison

This diagnostic implements G1/G2 only. It does not ingest, fit models, contact
Home Assistant/PostgreSQL, write predictions, publish controls, or change any
scheduled pipeline. It is not a migration-readiness certificate or SARIMA/model
ranking. Use an explicitly selected, audited measurement archive and CSV.

```sh
venv/bin/python3 -m utils.measurement_shadow \
  --csv /path/to/raw_data.csv --csv-mode raw \
  --archive /path/to/measurement_history.sqlite \
  --sources config/measurement_sources.yaml \
  --from 2026-10-21T00:00:00+03:00 --to 2026-10-22T00:00:00+03:00 \
  --interval-seconds 60
```

JSON goes to stdout; feature-processing progress goes to stderr. No output file
is opened by the command. No default operational CSV/archive is consulted and
missing explicit files fail, rather than triggering a compatibility fallback.
Source YAML supplies dated overrides to the loader's runtime source defaults;
its existing unit/sign assumptions still require auditing before cutover.

## Window and provenance contract

- Both bounds are required, timezone-aware, interval-aligned, and interpreted as
  the same UTC **[from, to)** window. Resolutions must be whole minutes dividing
  24 hours. CSV timestamps require explicit offsets/timezones; duplicate or
  misaligned timestamps fail. Different offsets representing the same instant
  align correctly, including across DST.
- The CSV must span the entire requested window; the command refuses a shorter
  intersection. Interior absent timestamps are reindexed as missing, reported,
  and never filled for actuals. Archive output must match the exact grid.
- Include the preceding 24 hours in the CSV for repeat-yesterday evaluation.
  Without that context the report still compares the window, but baseline
  matched-row counts decrease (possibly to zero). Archive features load their
  own preceding-day context. A day means 24 elapsed UTC hours, not necessarily
  yesterday's local wall-clock hour on DST transitions.
- `--csv-mode raw` treats the file as already resampled legacy measurements.
  Feature processing runs in memory; five-meter actuals are captured **before**
  median filters, fills, or solar substitution. Finite/physical checks alone do
  not prove original recorder fidelity. Absent equipment is never assumed zero.
- `--csv-mode processed` compares existing feature columns; there is no raw
  measurement comparison. Baseline actuals require `baseload_target_valid` and
  `baseload_quality_provenance` (`source_flags` or
  `legacy_finite_checks_only`). Without these, feature differences remain
  available but actual-based scores are unavailable, not zero. Provided
  processed lag columns are **not** trusted as honest baseline observations:
  repeat-yesterday is recomputed from valid actuals at the exact prior instant.

The default training policy is `verified-only`. To explicitly evaluate complete
legacy bins, add **both** `--allow-legacy --legacy-policy complete-legacy`.
Allowing legacy reads alone does not change eligibility. Optional repeated
`--preferred-import ID` arguments select explicit reader precedence, not a claim
that conflicts disappeared. Source-valid flags are never promoted by shadow
evaluation. The report preserves the feature loader's `archive_training`
metadata, policy, archive/import provenance, and window/context bounds.

## Reading the report

- `raw` / `features`: union of comparable feature names, missing/present counts,
  matched finite rows, differing rows, signed archive-minus-CSV mean and absolute
  differences. Raw reports also include mean coverage fraction, source-verified
  counts, provenance counts, and conflict counts. Value differences are measured
  in each column's existing unit; they are not all kW.
- `eligibility`: verified targets, eligible-but-unverified legacy/mixed targets,
  excluded archive rows, and usable CSV actual rows. The categories are disjoint.
  Source verification, physical completeness, conflicts, and policy eligibility
  remain distinct; an archive's queried range alone does not prove availability.
- `baseline`: separate verified-target and legacy-eligible-target repeat-yesterday
  error summaries (MAE/RMSE/bias in kW). Both routes use **identical timestamps**
  with valid actuals and exact eligible lag observations in both routes, each
  scored against its own actuals. Bias is prediction minus actual. The group is
  determined by the archive **target** verification, while lag eligibility follows
  the declared policy; a verified target can have a legacy-eligible lag under
  `complete-legacy`. This is not a verified-only forecast-history claim.
  `prediction_difference` and `actual_difference` use the same error-field
  names for archive-minus-CSV differences on those matched rows, exposing
  prediction disagreement separately from changes to the evaluation actuals.
- Zero matched rows yield JSON `null` errors, never artificial zero errors.
  Feature differences include diagnostic filled features, but those fills are
  never used to invent actuals for baseline scoring.

Recorder resampling uses **time-weighted complete-bin means**; legacy extraction
used **event/sample means**, often followed by interpolation or forward filling.
An event arriving near the end of a bin contributes little duration but may have
equal event-mean weight. Thus even matching sources/windows can legitimately
differ. Legacy imported values do not become recorder events or source-verified
because they match numerically.

The raw and feature loaders each hold one read-only snapshot, not a single shared
snapshot across both calls. Use an offline archive for reproducible comparisons;
concurrent ingestion could change their views. Frames are materialized in
memory (the CSV is read in full before trimming); choose bounded diagnostic
inputs/windows. Missing mandatory archive coverage
can make existing feature coverage checks fail; this is an explicit diagnostic
failure, never grounds for silently switching to CSV. This command depends on
the archive loader's explicit legacy eligibility contract and makes no rollout
or operational-cutover changes.
