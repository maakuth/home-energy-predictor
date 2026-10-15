# Model and Planner Quality Gates — implementation checklist

- [x] Add manifest, digest validation, and fixture inventory support.
- [x] Add deterministic structured replay reports and full safety accounting.
- [x] Add reference/candidate comparison with per-fixture and portfolio gates.
- [x] Wire quality marker, report CLI, and repository artifacts/documentation.
- [x] Add tests and run fast plus manifest-quality validation.

## Decisions retained for review

- `tests/fixtures/quality_reference.json` is the approved deterministic
  `1.34.0` / `nemotron-linprog` reference. Two independent reports matched
  byte-for-byte; fixture and portfolio tolerances are an approved 0.0 EUR.
- `aug.pkl` remains catalogued but excluded from the default portfolio because
  of its size; adding it requires a manifest review.
- Forecast lead-time comparisons require a candidate forecast archive. Planner
  replays using archived forecasts report forecast evaluation as unavailable.
