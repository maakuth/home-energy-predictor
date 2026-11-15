# Offline optimizer candidate comparison

`utils.optimizer_shadow` compares two **candidate** plans from one explicitly
supplied JSON bundle. It does not capture inputs, read a measurement archive,
run inference, contact Home Assistant or PostgreSQL, publish a plan, or write
`hepo.db`. Only the three built-in battery planners and the no-battery path are
accepted. Configuration is scoped to the calculation, not applied to
`os.environ`. This is an application contract, not an OS sandbox.

```sh
PYTHON_DOTENV_DISABLED=1 venv/bin/python3 -m utils.optimizer_shadow \
  --input /path/to/frozen-bundle.json \
  --output-dir /path/to/new-shadow-directory
```

The output directory must not already exist. It receives only
`optimizer-shadow-comparison.json`, containing both candidate plans, counts of
different intents, signed and absolute numeric differences, and the SHA256 of
the canonical JSON input. Numeric summaries include a matched count because
some planner-specific fields may be absent; they are not an exhaustive
equivalence check. The hash is an identity for the provided bundle, not
proof of origin or authenticity. Do not point a controller at this output.

## Bundle contract

`tests/test_optimizer_shadow.py` contains a complete minimal synthetic example.
The required fields are:

* `format_version`: `1`; `origin`: timezone-aware issue time; `interval_minutes`:
  positive integer; `model_sha256`: lowercase 64-character SHA256 identity;
  `output_timezone`: IANA timezone; and `tomorrow_valid`: boolean.
* `config`: the complete captured mapping of optimizer and built-in planner
  configuration names to **strings**, preserving absent keys. Defaults apply to
  absent keys, not values from the host environment. `PLAN_INTERVAL_MINUTES`,
  if present, must equal `interval_minutes`. Do not include credentials.
* `runtime`: explicit `ev_soc_state` (object or null), `allow_export`,
  `battery_soc_pct` (number or null), `battery_enabled`, `model_version`,
  `gshp_failed`, `current_acc_temp`, `compressor_running`, `resistive_running`,
  and `fireplace_on`. Flags are booleans. An enabled battery requires SOC.
  EV state must be null, explicitly `unknown`/`unavailable`, or a finite SOC in
  0–100; null and unavailable states invoke the existing fixed-duration fallback.
* `horizon`: nonempty, consecutive, aligned timestamps strictly after `origin`.
  Each row contains `timestamp`, `outside_temp`, `is_sauna_active`,
  `ev_position`, `solar_kw`, `solar_p10_kw`, `solar_p90_kw`, `market_price`,
  `import_price`, `export_price`, `fallback_price`, `sarima_lower`, and
  `sarima_upper`. Flags are integer 0/1; SARIMA bounds may be null. Power is kW,
  prices EUR/kWh, and timestamps must be timezone-aware.
* `routes` has exactly `compatibility` and `archive`; each has finite
  nonnegative `baseload_kw` and `uncontrolled_heating_kw` arrays aligned to the horizon.
  These are the only inputs allowed to differ between routes.

Captured arrays are required: missing values, timestamps, prices, or SOC are
not filled with today's live state. Tariff prices must already be resolved.
The comparison preserves the optimizer's input-timestamp Leaf schedule and
uses the requested output timezone, including its daylight-saving rules.

## Interpretation and remaining gates

These are **pre-latch candidate plans**. The operational publisher applies the
current-slot latch using a previous plan and a later wall-clock time; that is
not reproduced here. No cost-optimality, storage parity, as-of-issue history,
model identity, provenance, optimizer equivalence or migration readiness is
proved merely by successfully processing a bundle. In particular, the tool
does not verify that the stated model hash produced the supplied forecasts.

The existing November captures lack common prices, SOC, thermal/relay/EV state
and configuration. Do not populate them retrospectively from live state. A new,
separately authorized host capture is needed before a real comparison. CSV
availability differences remain documented legacy limitations and are not a
reason to fabricate matching inputs. Active cutover and model promotion each
still require separate approval.
