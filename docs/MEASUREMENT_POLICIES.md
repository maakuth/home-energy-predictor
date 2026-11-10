# Measurement interpretation policies

Source YAML is unchanged. A separate registry is selected explicitly with
`load_archive_raw(..., policy_path=path)`, also forwarded by `ArchiveHistory`.
`config/measurement_policy.example.yaml` is illustrative and **not operationally
ready**. It deliberately lacks entries for all default requested measurements.
No policy file is loaded implicitly; configuring one does not authorize imports,
recorder calls, rollout, or any live writes.

## Public API and provenance

* `load_policies(path) -> dict[str, MeasurementPolicy]` validates YAML, rejecting
  duplicate keys, unknown fields, invalid types, unsupported kinds/units,
  incompatible dimensions, non-finite mappings, invalid signs and expiry values.
* `parse_policies(mapping)` accepts raw dictionaries or parsed policies.
* `require_policy_entries(registry, measurements)` rejects missing entries. Use
  this for pipeline preflight after `load_policies`; loading a file alone is not
  a completeness or readiness check.
* `load_archive_raw(..., require_policies=True)` performs the same completeness
  check on its requested measurements (`BASELOAD_INPUTS` when omitted), before
  opening the archive. `policies=registry` is an alternative to `policy_path`,
  not an additional override. Partial registries are allowed only when this
  completeness option is false, preserving numeric compatibility.

Raw `attrs['archive']` includes `policy_path`, exact-file `policy_sha256` (or
`None` for in-memory/default policies), `policy_semantic_sha256` of the validated,
default-expanded sorted JSON registry, full `policies`, and
`implicit_policy_measurements`. File parsing and hashing use the same bytes.
Feature `attrs['archive_training']` retains these fields for model metadata.
An empty/partial registry must not be called migration-ready. No loader iterator
was added: the API remains a materialized frame under one read-only snapshot;
`chunk_seconds` bounds the internal numeric queries, not final-frame memory.

## Fields and observation validation

Every entry requires `kind`, `input_unit`, `canonical_unit`, and integer `sign`
(`1` or `-1`). Optional fields are `require_unit` (false), `attribute` (null),
`unit_attribute` (`unit_of_measurement`), `max_age_seconds` (null),
`nonnegative` (false), `state_values`, and `aggregation` (`duration`).
The attribute must exactly match every source mapping for the measurement.
Weather's `wind_speed` must explicitly name `unit_attribute: wind_speed_unit`:
the entity's generic unit does not verify an attribute's unit.

The actual recorder attribute named by `unit_attribute` is checked against the
declared input unit on **every observation**. A conflicting, unknown or malformed
unit invalidates it even if units are optional. Absent units invalidate it when
`require_unit: true`; otherwise values remain usable but `*_unit_verified` is
false. Missing attributes, malformed JSON, unknown/unavailable states, invalid
numbers and unmapped categories remain missing. Required units cannot be
verified by a legacy CSV and therefore block legacy fallback. `*_source_valid`
retains its recorder-coverage meaning; it does not by itself verify units.

Supported numeric conversions: W/kW, Wh/kWh, C/°C/°F, m/s/km/h, percent (`%`),
and dimensionless (`'1'`). Numeric conversion must preserve dimension. Energy
becomes power only through an explicit counter delta. Sign applies after
numeric conversion or state mapping. `nonnegative` rejects negatives rather
than clipping them. No freshness threshold is inferred. When supplied, expiry
uses original observation timestamps, including predecessor observations.

## Compatibility conventions and unresolved evidence

Without a policy, numeric behavior is unchanged, **unverified for units**, and
has no invented expiry. Processor canonical conventions are:

| Measurements | Canonical convention |
| --- | --- |
| total_power | kW, import positive/export negative |
| solar_actual / solar_forecast | kW, production positive |
| battery_power | W, charging positive/discharging negative |
| gshp_power | W, full shared meter including resistive elements |
| leaf_power, AAHP and relay powers | W, consumption positive |
| temperature measurements | C |
| ev_soc | % |
| mummun_power | kW from explicit counter policy only |

Explicit entries for these names must output their canonical units. These are
consumer conventions, not proof of each historical sensor's unit or sign.
In particular, the forecast-tomorrow entity may represent energy: its actual
dimension must be established; a kWh value cannot simply be relabelled kW.
Wind compatibility values pass through unchanged: the repository's extraction
reads `weather.home.wind_speed` but does not establish its historical unit.
An explicit audited wind policy must choose input/output units; km/h in the
example is not an operational finding. Unit/sign evidence is needed for every
entity period, as are justified freshness limits where operationally required.
A single registry entry applies to all mapped periods: differing input units or
attributes across a switch require separate logical measurements/policies until
per-period policy support exists; conflicting metadata fails closed.

Existing relay defaults integrate on/off duration using configured nominal
`RESISTIVE_HEATER_POWER_KW` / `BULK_HEATER_POWER_KW`, not measured element power.
Explicit `state` entries instead supply canonical state values themselves.
No configuration or registry can infer pre-installation zeros.
Feature loading masks the compatibility processor's numeric fills wherever the
original resampled measurement was missing, including pre-installation periods.
Existing numeric denoising of present observations remains unchanged.

## Categories and presence

Kinds `state` and `presence` require `input_unit: state` and an explicit finite,
lower-case `state_values` mapping. Values are canonical outputs. `duration`
integrates mapped durations (for example on/off rated watts, or occupancy
fraction), never numeric encodings of unspecified categories. `presence` outputs
dimensionless values. `aggregation: any_home` additionally requires `home: 1`
and explicit other-state mappings to 0/1; a fully covered bin returns 1 if home
occupied any positive duration, otherwise 0. It is not a category average.
Unmapped `not_home`, zones and other states remain missing; they are not assumed
equivalent to away. Even any-home bins require complete valid duration coverage.
`ev_position` requires an explicit presence policy. Feature loading preserves
these interpreted values and missingness rather than the CSV processor's
string comparison, median smoothing or away/zero fills.

## Counters

Kind `counter` requires Wh/kWh input, W/kW output and sign 1. For each output bin:

1. One source must cover the entire bin. A same-bin source switch is rejected,
   even if counters happen to have numerically matching values.
2. Successful fetch coverage must continuously cover the bin, with no gaps.
3. Actual observations must exist at **both exact endpoints**. A right endpoint
   at the old source's exclusive end is usable for that old source's delta.
4. All intervening observations must be finite, nonnegative, unit-valid and
   monotonically nondecreasing. Resets and unavailable/unknown events invalidate
   the bin; resets are never clipped to zero. If declared, `max_age_seconds`
   limits every successive observation gap, including the endpoint gap.
5. Power is the boundary energy delta divided by the elapsed interval. No
   interpolation, extrapolation or partial-bin estimate is made. Counter
   coverage is strict: 1 for an accepted delta, 0 otherwise.

Use logical `mummun_power` with a counter policy; if no dedicated source exists,
it reads the unchanged `mummun_energy` source mapping. A direct `mummun_energy`
counter request also returns the policy's declared power unit, not energy;
prefer the power name to avoid ambiguity. Both names require counter policies.
Legacy `mummun_energy` consists of resampled means, not boundary observations;
it is never differenced as original truth. Counter legacy fallback is disabled,
including legacy derived `mummun_power` (which has different semantics). Other
numeric legacy intervals retain canonical compatibility values and always remain
unverified, with import/conflict provenance intact.

## Predecessor coverage

Numeric/state integration uses `reader.carry_supported(source_id, timestamp,
coverage_start)` when available, while still enforcing source-period start and
explicit expiry. Without that reader API (v1), only observations at/after the
successful coverage start may carry. No resampler manufactures coverage across
an unqueried gap. Counter boundaries never use carry-forward states.
