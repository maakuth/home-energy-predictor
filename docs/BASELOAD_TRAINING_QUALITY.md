# Baseload training quality (model 1.39.0)

## Automatic changes

`process_data.py` computes training truth from sorted, aligned **raw** samples,
before median filtering, zero filling or solar fallback:

`baseload_power = total_power + solar_actual - (battery_power + gshp_power + leaf_power) / 1000`

Grid and solar are kW; battery, shared heating and Leaf meters are W. Battery
power is positive on charge, negative on discharge. The heating meter includes
both elements, so their relay-derived estimates are not subtracted again.

Every meter is required. Missing, nonfinite or negative component readings are
unknown (grid/battery may legitimately be negative). Negative calculated
baseload is invalid, not clipped into a zero training label. No equipment
installation date is guessed. Forecast solar is never training ground truth,
even when `SOLAR_FALLBACK_TO_FORECAST` remains enabled for other operational
calculations. Existing diagnostic columns may still contain legacy smoothing
or fallback values; only the quality-tagged target is training truth.

Output metadata:

- `baseload_target_valid`: numerical/source-state eligibility of the target.
- `baseload_exclusion_reason`: missing/invalid meter names or negative residual.
- `baseload_quality_provenance`: `source_flags` for new extracts with validity
  metadata, otherwise `legacy_finite_checks_only`.

New extraction carries `*_source_valid` for each accounting meter. These flags
preserve invalid/unavailable state updates even when numeric resampling fills
the old value. Sensors that report only on change can carry a known valid
state; this **does not establish sensor freshness or heartbeat health**. Legacy
CSV interpolation cannot be undone or distinguished from measurements. Neither
provenance label guarantees accurate calibration, correct entity mappings,
physical power bounds or synchronized original sensor updates.

The 1h/24h baseload and 1h Leaf lags use clock-time lookup without forward
filling. Invalid source targets produce missing lag features. The Leaf 24h
energy feature uses only preceding measurements and requires full native
sample coverage. Quality reasons remain on the processed timeline for audit;
existing all-null required-signal diagnostics still reject unusable extracts.

## Training and prediction

XGBoost automatically selects valid targets with available 1h/24h target lags.
Equipment regime and fireplace uncertainty are **not** blanket exclusions.
The training log reports exclusions and eligible regimes. Older processed CSVs
without quality metadata are rejected without overwriting the existing model;
regenerate them using the new processor.

Training additionally requires 14 eligible **sample-days**, computed from the
input's median native interval (at most 15 minutes), after quality/lag/holdout
selection. `--min-coverage-days` controls this safety gate; it is not a model
tuning parameter. This minimum is necessary but does not ensure good seasonal
or regime coverage. An initial read-only audit of the legacy two-year weekly
extract found 8,114 valid targets but only **340** with valid 1h/24h source lags.
The default gate therefore retains the current model until history is repaired
or sufficient new, correctly mapped sensor history accumulates.

The feature schema in `utils/baseload_quality.py` omits old accumulator/pump
temperatures, running state, tank rate-of-change and inferred fireplace flags.
Supply/return temperatures are not automatically added. All schema features
are provided by existing inference generation; the saved feature list selects
them. Existing saved models retain their original schema until retraining.

SARIMA respects quality flags when present, leaves missing/partial quarters
missing rather than forward-filling, and requires enough nonmissing quarters
for parameter training. Legacy CSV loading remains supported for SARIMA but
does not retroactively establish quality. Its regular-grid state-space model
can handle missing observations; this is not permission to compare partial
quarter energy as complete-quarter truth.

## Rollout and remaining work

Merge/redeploy the processor and trainers together, then regenerate training
data before retraining. No live retraining or deployment is part of this change.
Model version is bumped because target and predictor behaviour intentionally
change. Evaluate chronological holdouts by season and heating regime against
the existing model before deploying the replacement.

Still outstanding: validated battery/PV installation history to recover older
seasons, heartbeat/staleness policy, physical meter-bound checks, historical
replay of inference anchors, persisted audit reports, candidate-model promotion
and atomic artifact bundles. Those validation/publication changes are milestone
2; current training still saves directly to the existing model paths.

Thermal benchmark diagnostics remain separate. No COP, heat-loss coefficient,
Home Assistant entity or operational setting is changed automatically.
