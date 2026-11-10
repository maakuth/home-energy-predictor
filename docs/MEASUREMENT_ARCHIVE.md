# Historical measurement source configuration

The archive migration is under development. Existing scheduled CSV pipelines
are unchanged; configuring sources does not switch them to the archive.

`config/measurement_sources.yaml` maps logical measurements to recorder entities.
Install `requirements.txt` to include the YAML parser dependency, PyYAML.

## Periods and units

- Quote timezone-aware ISO timestamps in `from` and `to`.
- `from` is inclusive, `to` exclusive.
- Without `from`, the source starts at the requested backfill boundary.
- Without `to`, the source ends at the command's fixed `--to` cutoff.
- Overlapping sources are rejected. Gaps remain unqueried, not zero-filled.
- Unknown fields and duplicate YAML keys are rejected before opening databases.
- The optional `attribute` field identifies an attribute of an entity; original
  state text and raw attributes are retained without numeric conversion.

The checked-in outdoor-temperature mapping switches at October 19, 2026,
00:00 Helsinki time. Shared heating power switches from `sensor.mlp_teho` to
`sensor.instantaneous_used_power_32167` at October 20, 2026, 08:30 Helsinki time.
Both heating sensors measure watts including resistive elements. The old meter
also recorded resistive-only heating after removal of the old compressor around
October 7 at 08:00. That equipment event does not terminate its measurement
mapping. It must remain separate regime metadata for thermal analysis.

For the old equipment, the compressor reportedly drew about 3–4.5 kW and each
of the two resistive elements approximately 6 kW (possibly nearer 5.9 kW);
simultaneous operation of both elements was rare, not impossible. These are
approximate operating characteristics, not calibrated measurements. In
particular, applying modulo 5.9 or 6 kW to the shared meter is not a reliable
compressor measurement: a slightly lower element draw would be misclassified.
The archive retains the full original shared reading. Any future component
decomposition must retain uncertainty and use relay histories where available;
it must not replace measured shared power in baseload accounting.

## Explicit ingestion

These are operational examples, not commands to run as part of development.
They contact the PostgreSQL recorder configured in `.env` and write the archive.
Back up runtime state and verify configuration before using them.

```sh
venv/bin/python3 -m utils.measurement_ingest_cli \
  --archive state/measurement_history.sqlite \
  --recorder-namespace CONFIRMED-RECORDER-IDENTITY \
  --sources config/measurement_sources.yaml \
  --measurement gshp_power \
  --from '2026-10-01T00:00:00+03:00' \
  --to '2026-11-01T00:00:00+02:00'
```

Omit `--measurement` to ingest all measurements in the mapping file. Repeat it
to select multiple measurements. `--entity` remains available for an explicit
single-source fetch, but cannot be combined with `--sources`.

The command intersects each mapping period with the requested window before
fetching. The example queries the old heating meter only before the October 20
boundary and the new one only at/after it. Repeating a request uses per-source
coverage and an overlap (default 900 seconds), rather than fetching the whole
window again. An old source failure does not prevent trying the new source,
but any failure makes the command exit nonzero. Missing recorder metadata is a
failure, not a successful empty historical query.

These two dated mappings override the full runtime source inventory. The
ingestion CLI's explicit `--include-default-sources` merges both; it does not
establish historical units or freshness. Source/registry evidence and rollout
approval are still required before scheduled pipelines can switch.

## Read-only interval access

`MeasurementReader` opens an existing archive in SQLite read-only mode and holds
a consistent snapshot until the context exits. It cannot create missing files,
initialize a schema, or write data. Release the context before model fitting.

`reader.legacy_numeric(measurement, start, end, interval_seconds=60)` reads
native-resolution numeric legacy intervals. Bounds must be UTC-epoch aligned.
Equivalent numeric representations agree; different values or missing-vs-valid
values are conflicts and return NaN by default. `preferred_imports=[id, ...]`
is an explicit per-read precedence order, not a persisted rewrite. The first
listed import present at a timestamp wins even when its value is missing, so
selection cannot silently fill its outage from another import. Unknown IDs are
rejected. Conflict flags and contributing IDs remain visible after selection.
Legacy source-validity flags, when present, can invalidate filled numeric values.
No legacy reading is ever promoted to source-verified truth.

`utils.measurement_resampling.resample_numeric` integrates raw state durations
for numeric power/temperature or explicit relay-state mappings. It applies the
dated source configuration before integration and reports:

- `value`: time-weighted mean, NaN unless the full output interval is valid;
- `coverage_fraction`: fraction of elapsed time with usable readings;
- `source_valid`: true only for complete original-recorder coverage;
- `provenance`, `conflict`, contributing `import_ids` and `entities`.

Recorder values take precedence within successful fetch coverage. Explicitly
unavailable values and queried intervals without a known state block legacy
fallback. States do not carry across mapping boundaries or unqueried gaps.
An optional positive `max_age_seconds` bounds unchanged-state carry. Relay
integration preserves short on/off runs instead of treating absent data as off.
All weighting uses elapsed UTC time, including on Helsinki DST-change days.

Legacy fallback is disabled unless `allow_legacy=True`. Its native resolution
must divide the requested output interval. Legacy interval means are integrated
over their declared duration, not treated as original state-change events.
Partial coverage is reported, never interpolated into a complete measurement.
Without an explicit registry, the numeric compatibility resampler does not infer
cumulative-energy reset/delta handling or
categorical feature policies; those require separate measurement definitions.

## October snapshot overlap audit

The staging archive contains two legacy imports from the October 8 snapshot.
Their common interval is October 4, 2026, 21:00–23:03 UTC inclusive (124 minute
samples). Across 16 common measurement columns, the import audit found 113
different non-null cells and 145 cells missing in only one import.

The differences are not all harmless rounding: for example, the resistive-heater
column disagrees at 15 timestamps (6000 W vs 0 W). Shared heating power differs
at three final timestamps, while outdoor temperature differs at eight samples.
Several differences occur near extraction boundaries, but the source snapshots
alone do not establish which values are correct. No blanket newer-file or
non-null-wins rule has been applied.

A read-only audit using the reader/resampler retained 121 agreeing shared-heating
samples and marked three unresolved; for outdoor temperature, 116 agreed and
eight remained unresolved. All were still labelled legacy/unverified. Bounded
primary-key queries avoid scanning the entire 1.59 GB archive for this small
window. The staging database and source CSVs were not changed by the audit.

These reader APIs do not switch the scheduled pipelines away from CSV.

## In-memory feature windows and opt-in XGBoost training

`process_data.process_frame(raw)` generates the existing feature set without
writing files or mutating the input. The CSV entry point uses the same function.

`utils.archive_features.load_archive_raw` reads bounded time chunks under a
single read-only snapshot. It returns values plus source-validity, coverage,
provenance and conflict columns. Default entity definitions match the existing
extractor and are evaluated at call time; the dated YAML overrides those defaults.
The initial default selection is the eight baseload inputs: outdoor temperature,
wind, solar forecast, grid, solar production, battery, shared heating and Leaf.
Numeric optional sensors and heater relays can be explicitly selected. Category
and cumulative-energy measurements require an explicit
[interpretation registry](MEASUREMENT_POLICIES.md), rather than being silently
interpreted as numeric power.

`load_archive_features` includes the preceding 24 hours for lag/rolling context,
then returns only the requested window. Feature generation uses the existing
imputation and filtering rules, while archive quality metadata is kept separate
and restored unchanged. Missing required observations can therefore prevent
feature generation, and invalid meter truth remains ineligible for training.
Legacy opt-in does not relax source verification. A separate explicit training
eligibility policy is documented in [Archive training policy](ARCHIVE_TRAINING_POLICY.md).
The default one-minute
aggregation of original recorder events is time-weighted, unlike legacy event
means; the completed consumer/type/pipeline changes are associated with model version 1.46.0.

XGBoost can now train without reading or producing raw/processed CSV files:

```sh
venv/bin/python3 train_model.py \
  --archive state/measurement_history.sqlite \
  --sources config/measurement_sources.yaml \
  --from '2026-10-21T00:00:00+03:00' \
  --to '2026-11-09T00:00:00+02:00'
```

This is an operational example, not a development command: it writes model
artifacts under `state/` if the unchanged coverage gate passes. Missing archive
files, malformed windows and missing sources never silently fall back to CSV.
`--allow-legacy` and repeatable `--preferred-import` enable explicit legacy
reading/selection, but alone do not make unverified legacy targets eligible for fitting.
`--legacy-policy complete-legacy --allow-legacy` opts into complete conflict-free
legacy targets without changing original-source flags. Missing mandatory meters
and incomplete lag/Leaf-energy windows remain excluded.
The existing CSV training arguments remain supported and are still the scheduled
default. No models have been trained against staging data as part of development.

SARIMA now accepts explicit archive windows for both training and benchmark
prediction; see [SARIMA archive mode](SARIMA_ARCHIVE.md). Read-only
[inventory/coverage audits](MEASUREMENT_ARCHIVE_AUDIT.md) are also available.
Prediction, optimization, evaluation and diagnostic consumers now accept an
explicit [fixed-cutoff history context](MEASUREMENT_ARCHIVE_DESIGN.md).
[Pipeline archive routes](ARCHIVE_PIPELINES.md), strict measurement types and
[verified backup/restore](MEASUREMENT_ARCHIVE_STORAGE.md) are implemented.
[Validation evidence](MEASUREMENT_ARCHIVE_VALIDATION.md) separates synthetic
tests, full-length profiling and legacy-only real shadows. Source/unit/freshness
evidence, authorized backfills, verified-recorder shadows and deployment approval
remain outstanding. **The system is not migration-ready for live cutover.**
