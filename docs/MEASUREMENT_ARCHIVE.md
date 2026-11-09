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

These two mappings are not yet a complete ingestion inventory for all training
and inference features. Archive-backed resampling, legacy conflict resolution,
and consumer migration must be completed before scheduled pipelines can switch.
