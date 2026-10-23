# Heating history and resistance-only benchmark

## Equipment timeline

Dates use **Europe/Helsinki**, not UTC. The owner confirmed the old GSHP was
considered broken on 2026-08-20 but continued intermittently until removal on
2026-10-06. Its replacement was installed on 2026-10-19; integration completed
on 2026-10-20. Exact intraday changeover times are unknown.

| Local dates | `heating_regime` |
|---|---|
| Before 2026-08-20 | `old_pre_failure` (not a guarantee of healthy operation) |
| 2026-08-20 through 2026-10-05 | `old_intermittent` |
| 2026-10-06 | `removal_transition` |
| 2026-10-07 through 2026-10-18 | `resistance_only` |
| 2026-10-19 | `installation_transition` |
| 2026-10-20 | `integration_transition` |
| From 2026-10-21 | `new_gshp` (not a guarantee commissioning is complete) |

Processing adds this column as metadata. XGBoost's explicit feature list does
not include it. No rows are removed because of the regime. Preserve raw history;
invalid metering, not equipment identity alone, determines baseload suitability.
The thermal masks below must **not** be used as blanket baseload exclusions.

## Offline report

Run from the repository root, against a local raw extract only:

```sh
venv/bin/python3 utils/heating_benchmark.py \
  --input /path/to/raw_data_weekly.csv \
  --output-dir /tmp/opencode/heating-benchmark
```

The output directory must not exist. Without it, the command only prints a
daily report and storage-tail sensitivity. No network, database, `.env`, model
or runtime-state access occurs. There is no automatic retraining or parameter
update. For a smaller report, supply a raw extract of the period of interest.

An initial read-only check of the saved 7–18 October 2026 extract found
approximately **953.28 kWh** of heating electricity, but **no provisional
complete days**, even with a zero-hour storage tail. Unexplained tank rises
occur on every day under the default assumptions. This does **not** establish
daily fireplace use: the owner recalls only about two firing days. It indicates
that single-sensor tank dynamics/meter alignment need investigation before
these diagnostics can isolate wood input. Do not loosen thresholds just to
obtain a calibration set. These results deliberately prevent automatic fitting.

Input timestamps identify the start of interval-mean samples. The default
interval is one minute; use `--interval-minutes 5` or `15` if appropriate.
`gshp_power` must be the **shared heating electrical meter in watts**, including
both elements. Do not supply compressor-only power or the median-filtered,
split processed data. Supply/return and virtual temperatures are never used as
substitutes for measured `accumulator_temp`.

Energy is integrated as interval mean power times interval duration. There is
no smoothing, interpolation, extrapolation or forward filling. Missing and
negative power is unknown, not zero. Duplicate/off-grid timestamps are rejected.
Reports include whole local days, including missing edges of partial days and
23/25-hour DST days. Daily energy is the **observed portion only**: consult
coverage before comparing days. Existing raw extracts can themselves contain
interpolated/stale readings; this report cannot recover original update ages.

## Fireplace uncertainty

The owner recalls approximately two firing days during expensive electricity,
but not their dates. The fireplace feeds the same water circuit. Therefore:

`heat draw + storage losses + change in storage ≈ element electricity + wood heat`

Electricity-to-heat conversion is approximately 1:1 at the elements, **not** a
complete measurement of heat demand. Shared-meter parasitics also require audit.

The report flags 15-minute measured tank rises exceeding the maximum gain
explainable by shared electrical input, assuming zero heat draw. It requires
continuous valid tank/power observations entirely within resistance-only dates.
The diagnostic uses `--reservoir-liters` (default 500) and `--rise-margin-c`
(default 0.3); these are uncertainty assumptions, not fitted physical constants.
Tank readings outside 20°C to `--tank-max-c` (default 80°C) are unknown, not
clipped into plausible temperatures. Verify the historical 97.2°C reading on
15 October before changing this bound.

Candidates may reflect stratification, circulation or sensor faults, not fire.
Wood heat can also maintain a steady tank temperature and escape detection.
The old `is_fireplace_active` heuristic is not used as ground truth; in particular
its heater list omits the lower element. Missing relay records do not imply
electrical heating was off.

Each candidate masks its preceding 15-minute window and the following storage
tail (`--tail-hours`, default 24). The report compares 0/12/24/48-hour tails.
`fireplace_candidate_windows` counts overlapping windows, **not firing events**.

`thermal_candidate` means resistance-only, valid power/weather/tank, and outside
that mask. `provisional_benchmark_day` additionally requires a complete eligible
day. Neither means confirmed fireplace-free. Use these to review candidates,
not automatically fit COP or heat-loss coefficients. Do not select the two
lowest-electricity days or exclude expensive hours: that would introduce bias.

## Subsequent calibration

Audit complete candidate days for tank stratification/storage change, DHW,
indoor comfort, other heat sources and AAHP coverage. Missing measurements on
17–18 October need recovery before weather-linked fitting. Preserve independent
complete blocks for validation and report sensitivity to ambiguous fireplace
days and storage assumptions. This short autumn window cannot validate winter
extrapolation. It does not identify wood energy or COP uniquely.

Only after those checks should existing `GSHP_HEAT_LOSS_K` and
`GSHP_BASELINE_DEMAND_KW` settings be reconsidered. Evaluate replacement-GSHP
electricity separately by equipment regime **and model version**. Do not use
the virtual accumulator to validate COP: it already assumes COP. Supply/return
temperatures without water flow do not measure thermal power.

Separate follow-up work remains necessary for battery-installation-aware
historical coverage, legacy feature/supply-temperature mismatch and optional
training features after sensor removal. This report intentionally does not
silently change those model behaviours.
