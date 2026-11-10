# Archive training eligibility

**Approved solar predictor removal (model 1.47.0):** new XGBoost models exclude
`solar_forecast`. Historical daily energy (kWh) was incompatible with inference's
hourly forecast power; no replacement forecast history is invented. Measured
`solar_actual` remains mandatory for target accounting, and live solar forecasts
remain available to planning. SARIMA target semantics are unchanged. See the
[evidence review](MEASUREMENT_EVIDENCE_REVIEW.md).

Archive training is opt-in. Scheduled CSV jobs are unchanged; this is not a
pipeline cutover. Original source verification is never relaxed.

`load_archive_features(..., legacy_policy=...)` supports:

* `verified-only` (default): complete, source-verified recorder bins only.
* `complete-legacy`: also complete finite legacy/mixed bins, only when
  `allow_legacy=True`. This accepts the limitations of previously resampled CSV
  history; it does not establish recorder fidelity or freshness.

Both policies reject all conflicts (including explicitly selected conflicting
imports), missing values, incomplete bins, invalid source readings and negative
unsigned power. All five baseload meters must be present: grid, solar, battery,
shared heating and Leaf. Missing pre-installation meters are not assumed zero.
An explicit invalid legacy source flag is already removed by the reader and
cannot become eligible through this policy.

`baseload_target_valid` remains source-verified target validity. The separate
`baseload_training_eligible` column expresses eligibility under the declared
policy. Target values use original aligned meter means before median filters,
solar substitution or zero fills. Exact 1h/24h lags inherit eligibility from
their historical timestamps; previous-day Leaf energy requires a complete
window. Archive XGBoost selection also requires eligible temperature and wind
rather than admitting imputed weather values. No solar-forecast source,
eligibility column or pipeline ingestion is required for baseload training.

Each measurement retains source validity, coverage fraction, provenance,
conflict status, contributing import IDs and mapped entities. DataFrame
`attrs['archive_training']` includes a versioned policy contract, requested and
context windows, resolution and import hashes/paths. The loader closes its
single read snapshot before returning for fitting. XGBoost persists this in
`state/model_training_metadata.json`, alongside eligible/verified target counts
and eligible date bounds. Metadata lists imports in the archive snapshot;
per-row import IDs identify actual contributors. No hash precedence is implied.

## Model compatibility

The eligibility policy remains contract version 1. The independent baseload
feature contract is version 2, with 18 ordered inputs. Training writes it to
`state/model_training_metadata.json` and embeds it in the booster as
`baseload_feature_contract`; `state/model_features.json` stores the ordered list.
Archive prediction and hindsight evaluation require matching metadata, embedded
contract, ordered feature list and actual booster feature names. Old or mixed
artifacts fail closed, with no implicit CSV fallback or fabricated features.
Pipeline training checks also require the current metadata and feature list.

Existing models are not migrated by editing JSON. Explicit retraining and
representative validation are required before archive-mode use. No-argument CSV
inference still supports existing persisted feature lists for rollback; newly
trained CSV models also omit the predictor. Compatibility does not repair an
old model's dimensional mismatch. No production retraining or quality gain is
claimed by the code/tests alone.

```sh
venv/bin/python3 train_model.py --archive /path/to/staging.sqlite \
  --from 2026-10-01T00:00:00Z --to 2026-11-01T00:00:00Z \
  --allow-legacy --legacy-policy complete-legacy
```

This is an explicit model-writing command, not an audit. It still enforces the
normal coverage gate. The imported staging history may have no eligible periods
because solar/battery/Leaf meters are absent. Use read-only audit/shadow tools
before choosing training windows. No implicit CSV fallback occurs on failure.
