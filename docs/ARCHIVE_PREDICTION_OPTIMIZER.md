# Prediction and optimizer archive inputs

Both CLIs accept the common explicit history flags:

```text
python predict_future.py --archive PATH --cutoff 2026-11-01T12:07:00Z
python optimize_plan.py --archive PATH --cutoff 2026-11-01T12:07:00Z
```

Optional `--sources`, `--measurement-policy`, `--allow-legacy`,
`--legacy-policy`, and repeated `--preferred-import` are passed through the
immutable `ArchiveHistory` context. Omitting archive options preserves the
compatibility route. Archive errors never retry PostgreSQL or CSV.

Archive prediction may also take `--model-dir /absolute/candidate/state` to
select a separate model bundle rather than `state/`. It reads the model, feature
list and training metadata from that directory, checks their compatibility and
hashes, and refuses output if the bundle changes during inference. The success
sidecar records the directory and hashes. The option requires archive history;
the CSV route retains its existing `state/` model. This does not itself promote
the candidate or change schedules.

## Success contract for the pipeline

Archive-mode success writes `state/future_predictions_metadata.json` or
`state/optimization_plan_metadata.json`, **after** writing the corresponding
complete output artifact. With test/output path overrides, sidecars live beside
the output artifact and retain these fixed basenames. Each sidecar is replaced
atomically and includes:

* `history_cutoff`: exclusive UTC epoch seconds, the exact supplied cutoff;
* `archive_path`: resolved archive path;
* `sources_path`, `measurement_policy_path` (null means default registry);
* `legacy_policy`, `allow_legacy`, `preferred_imports`;
* `model_version`, `generated_at`, `current_telemetry_and_forecasts: live`.

Prediction metadata also includes `model_training` when the existing
`state/model_training_metadata.json` is present. No prediction DB schema changes
are required. Existing optional SQLite feedback-archive failures remain warnings;
the success marker attests to the JSON artifact, not feedback DB persistence.

The runner must require **both a zero process exit status and a matching sidecar
cutoff** before advancing. Old files can remain after failure: neither their
existence nor an older success marker authorizes publishing them. Missing solar,
model/features, historical anchors, empty horizons/predictions, nonfinite model
outputs, missing optimizer prices, and missing autonomous heating observations
raise in archive mode. CLI exceptions exit nonzero. Sidecar write failures also
propagate; artifact/sidecar replacement is not a multi-file transaction.

## Historical semantics

Prediction anchors use unfilled, aligned minute bins of all five shared meter
definitions and `apply_training_policy`, at cutoff minus one and 24 hours. Leaf
energy uses the same labelled, causal estimate as archive training: at most
120 seconds of missing coverage in at most three partially or fully incomplete
minute bins. Each incomplete bin is estimated in full from an earlier eligible
reading, not treated as verified energy. Conflicts and longer or unanchored gaps
still fail. The sidecar records the input policy and estimate diagnostics.
A synthetic cutoff
query row is missing data, never a future raw read or fabricated measurement.
Shared heating remains the raw shared meter: no relay subtraction or forecast
solar substitution is used to make an anchor appear valid. Missing anchors fail
closed, including when optional accumulator instrumentation is disabled.

Sauna historical reads use eligible archive actuals. Yesterday evening is
selected by the full Europe/Helsinki date (including month/year and DST), and
warming compares current live temperature with the exact cutoff-minus-30-minute
bin. Missing sauna evidence disables that heuristic; it does not claim measured
cold. Forecast generation and near-term heuristic windows use the fixed cutoff
in Helsinki, independent of the machine's timezone or clock advancing mid-run.

Optimizer fireplace evidence uses adjacent final eligible accumulator and shared
heating bins. Disabled/missing accumulator evidence means no fireplace detection.
Archive temperatures never replace the current live accumulator, compressor,
relay, battery or EV state. Current-slot load latching still uses the live clock.

With all heating dispatch disabled, the autonomous electrical-load persistence
forecast uses eligible raw shared-meter observations in the day preceding the
cutoff, not the rounded-down first plan slot. It logs eligible/1440 minute
coverage. Partial observations yield an explicitly labelled observed mean, not a
full-day energy claim; no eligible observations or archive read errors abort
optimization instead of silently assuming zero. It does not command a heater.

This route changes no schedules, rollout settings, raw archive schema, or model
version. Live telemetry/forecast reads remain operational calls; offline tests
must mock them. No rollout or live recorder/control access is implied.
