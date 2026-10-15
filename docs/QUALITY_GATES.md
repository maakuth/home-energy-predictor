# Offline Model and Planner Quality Gates

Quality gates evaluate immutable historical fixtures only. They never contact
Home Assistant, PostgreSQL, production SQLite, or the operational `.env`.

## Fixture contract

`tests/fixtures/quality_manifest.json` is the immutable portfolio definition.
It records every fixture digest, collection period, source model version,
interval count, selected planner, regime rationale, and review tolerances.
The report runner validates selected SHA-256 digests before unpickling a
fixture. `aug.pkl` is inventoried but intentionally excluded until its runtime
and value as a separate regime have been reviewed.

## Run a report

```bash
venv/bin/python3 utils/run_quality_gate.py \
  --planner nemotron-linprog \
  --output /tmp/opencode/quality-candidate.json
```

This is a full-length replay and may be slow. It emits deterministic JSON with
fixture identity, evaluator and model versions, completion/safety status,
realized and no-battery EUR costs, absolute EUR delta, imported/exported
energy, final SoC, and action counts. Forecast metrics are explicitly marked
unavailable for planner replays because their archived forecasts are fixed
inputs; a forecast-model change must instead supply candidate predictions at
the same historical issue times.

## Establish and use a reference

Run the command on the explicitly approved revision, inspect every seasonal
regime, then commit its compact JSON output as the approved reference. The
repository's initial `quality_reference.json` is the approved deterministic
`1.34.0` / `nemotron-linprog` reference. Its zero-EUR fixture and portfolio
tolerances are intentional: reproducibility was verified with two identical
full reports. Future tolerance changes require a documented review.

```bash
venv/bin/python3 utils/run_quality_gate.py \
  --planner nemotron-linprog \
  --reference tests/fixtures/quality_reference.json \
  --output /tmp/opencode/quality-candidate.json
```

With `--reference`, a mismatch in fixture digest, an incomplete/safety-failed
replay, a per-fixture EUR regression beyond tolerance, or a portfolio
regression returns a non-zero status. A portfolio improvement never overrides
a hard fixture failure. Changes that intentionally improve model, inference, or
optimization behavior require the `VERSION` bump described in `AGENTS.md` and
a reviewed reference update/trade-off summary.

## Test policy

Fast safety checks remain:

```bash
venv/bin/python3 -m pytest -k 'not slow'
```

Full replay invariants are marked both `slow` and `quality`; scheduled CI and
intentional model/planner change review should run `pytest -m quality` and
publish the structured report as an artifact. Until a reviewed reference and
tolerances are committed, reports are report-only rather than economic blockers.
