# SARIMA archive input (opt-in)

Scheduled SARIMA training and prediction are disabled in this code. The
standalone commands below remain available for explicitly isolated offline
analysis; they do not authorize a production schedule or artifact change.

CSV remains the default compatibility route. No scheduled pipeline has been
switched by this implementation. Both entry points accept the same explicit,
timezone-aware, inclusive `--from` and exclusive `--to` archive window:

```bash
venv/bin/python3 train_sarima.py --archive /path/to/history.sqlite \
  --from 2026-10-01T00:00:00+03:00 --to 2026-10-15T00:00:00+03:00 \
  --sources config/measurement_sources.yaml --params-path /path/to/params.pkl

venv/bin/python3 sarimax_predictor.py --archive /path/to/history.sqlite \
  --from 2026-10-12T00:00:00+03:00 --to 2026-10-15T00:00:00+03:00 \
  --params-path /path/to/params.pkl --output /path/to/forecast.json \
  --no-archive-results
```

These commands fit models; use explicit scratch output paths for evaluation.
The predictor requires an existing parameter file. `--no-archive-results`
prevents the usual benchmark-result write to SQLite; it does not suppress the
JSON output. These commands do not ingest data or contact Home Assistant or
PostgreSQL.

The default policy is `verified-only`. Legacy training requires both
`--allow-legacy --legacy-policy complete-legacy`; reading legacy values alone
does not authorize training on them. Optional repeated `--preferred-import ID`
selects explicit import precedence and also requires `--allow-legacy`. Source
verification flags remain unchanged. Archive-derived features must provide
`baseload_training_eligible`; missing metadata or archive failures never cause
a CSV fallback.

`--interval-seconds` defaults to 60 and must be whole minutes dividing 900.
SARIMA uses actual, unsmoothed baseload targets on a regular UTC 15-minute grid.
Every constituent interval must be eligible and finite; incomplete quarters and
gaps stay missing. Negative baseload is not a target. Archive mode uses the
entire explicit window (not the CSV lookback). The feature loader closes its
consistent read-only snapshot before fitting starts.

The parameter pickle retains `params`, `order`, `seasonal_order`, and
`last_index`, adding `training_metadata` with archive request/policy, feature
metadata, provenance counts, and eligible/missing quarter counts. Old parameter
files remain usable. Forecast warm-start retry and physical safety caps remain
in place.

Explicit CSV rollback examples:

```bash
venv/bin/python3 train_sarima.py --data-path /path/to/processed.csv --days 14
venv/bin/python3 sarimax_predictor.py --data-path /path/to/processed.csv
```

Existing `SLOW_DATA_PATH`, `SLOW_PARAMS_PATH`, and `TEST_SARIMA_PARAMS`
compatibility overrides remain supported by their existing entry points.
# Slow-task wrapper and rollback

`run_slow.sh` now exits without running a forecast and rejects arguments.
No timer configuration is changed; after deployment the existing hourly
invocation becomes a no-op. Existing artifacts are retained.
