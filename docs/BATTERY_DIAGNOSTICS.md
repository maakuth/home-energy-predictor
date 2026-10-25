# Battery control diagnostics

Every completed `run_often.py` cycle emits one line to stdout/journal:

```text
HEPO_BATTERY_CONTROL {"battery_power_w":-1000.0,...,"schema_version":1,...}
```

The example is abbreviated; actual lines contain valid compact JSON, with sorted
keys, no spaces around separators, and no embedded newlines. The fixed prefix
works with `grep`, including when journal/syslog adds its own prefix. Missing or
non-finite numbers are JSON `null`, **not zero**, `NaN`, or `Infinity`. Existing
human-readable logs remain unchanged. No separate log file or live mode query
is added; retain/export the journal with the rest of the run-often logs.

## Grep and parse

```bash
# All structured battery records, stripping any journal/syslog prefix:
grep -F 'HEPO_BATTERY_CONTROL ' recentlog.txt | sed 's/^.*HEPO_BATTERY_CONTROL //' > battery.jsonl

# Native mode selected and the Home Assistant service accepted the request:
jq -c 'select(.mode_requested == "self_consumption" and .command_status == "accepted")' battery.jsonl

# Exact reason codes are grep-friendly as well:
grep -F '"observed_wrong_direction"' battery.jsonl
grep -F '"command_status":"failed"' battery.jsonl

# Counts by fallback/guard reason:
jq -r '.decision_reasons[]' battery.jsonl | sort | uniq -c

# Compact timeline; signed battery watts, signed grid watts, and decision:
jq -r '[.timestamp,.planned_action,.effective_action,.previous_requested_mode,.mode_requested,.command_status,.battery_power_w,.grid_power_w,(.decision_reasons|join(","))] | @tsv' battery.jsonl
```

Python consumers can use
`utils.battery_diagnostics.parse_battery_diagnostic(line)` on raw journal/syslog
lines. It returns a dictionary or `None` for other/malformed lines. Consumers
should inspect `schema_version`; version 1 uses the fields below. Additional
fields may be added without changing the meaning of existing ones.

## Core fields and units

| Fields | Meaning |
|---|---|
| `timestamp`, `logged_at`, `model_version` | Timezone-aware cycle-start and log-emission timestamps, and the semantic model version. |
| `plan_slot_id`, `plan_mtime` | Selected plan entry's timestamp and plan file modification time (epoch seconds). Null without a current entry/file. |
| `planned_action`, `planned_power_w` | Original plan action and battery power before manual overrides or smoothing. |
| `effective_action`, `override_mode`, `control_action` | Action after manual/cheap-export overrides; manual select value or `auto`; final controller action label (`net_metering` where applicable). |
| `battery_power_w`, `grid_power_w`, `solar_power_w` | Measured battery power (**positive charging**), grid power (**positive importing**), and solar generation. All in watts. |
| `battery_last_updated`, `grid_last_updated` | Original HA sensor timestamps, when supplied, for identifying stale/asynchronous readings. |
| `soc_pct`, `phase_currents_a`, `import_meter_kwh`, `export_meter_kwh` | SoC, per-phase current readings, and cumulative energy meters. Missing readings remain null. |
| `solar_used_w` | Solar value used by control; may be forecast fallback rather than a measurement. |
| `native_estimate_w`, `native_clamped_estimate_w` | `battery_power_w - grid_power_w`, before and after physical inverter power limits. **Positive charging**; these are estimates, not commands. |
| `mode_requested` | Final intended `self_consumption` or `fixed`, **after all guards**. Null if no command was attempted. Check `command_status` for acceptance/skips. |
| `previous_requested_mode`, `previous_request_action`, `previous_request_slot_id`, `previous_request_at`, `previous_command_status` | Previous cached controller request, including its recorded result/time when available. Old cache files may have null metadata. This is not observed inverter mode. |
| `was_native_for_hysteresis`, `native_selected_before_guards` | Scoped hysteresis input and initial native eligibility. The latter may be true even when a later safety guard selects fixed mode. |
| `guarded_target_w`, `fixed_setpoint_w` | Target after all guards, and the actual integer fixed-power setpoint, **positive charging**. `fixed_setpoint_w` is null in native mode: no numeric command was sent. |
| `bridge_set_power_w` | Signed bridge command, **positive discharging** (opposite the other battery-power fields). Null in native mode. |
| `command_status`, `command_error`, `command_completed_at` | Result of the request, optional failure category, and completion time. See below. |
| `inverter_mode_verified` | Always false: neither a successful HA request nor the bridge's retained last-requested mode verifies physical inverter mode. |

### Limits and interventions

`max_charge_w`, `max_discharge_w`, `soc_floor_pct`, `main_fuse_a`, and
`deadband_w` record applicable configuration. `directional_planned_power_w`
records the unsmoothed plan/manual target; `smoothed_planned_power_w` records
the smoothing output. `discharge_cap_w` and `discharge_reentry_cap_w` record the
directional discharge limit and hysteresis threshold.

`discharge_budget_kwh` records the plan budget even when manual override bypasses
it. `discharge_budget_active` identifies whether the budget check ran;
`discharge_used_kwh` and `discharge_remaining_kwh` record its accounting.
`ramp_before_w`/`ramp_after_w`, `budget_before_w`/`budget_after_w`, and
`fuse_before_w`/`fuse_after_w` show the successive power adjustments, positive
charging. Values are null when that stage did not run. Fuse safety can override
a budget/direction cap; the final `guarded_target_w` records the outcome.

`decision_reasons` is a list of machine-readable codes; multiple constraints may
apply. An empty list means native eligibility passed with no subsequent guard.
`decision_detail` retains the human-readable controller explanation, if present.

| Reason codes | Meaning |
|---|---|
| `observed_wrong_direction`, `expected_wrong_direction` | Measured battery direction or estimated native response contradicts the directional action beyond the deadband. |
| `observed_discharge_cap`, `expected_discharge_cap` | Actual or estimated discharge exceeds the plan cap. |
| `direction_hysteresis`, `power_cap_hysteresis`, `zero_discharge_cap` | Native re-entry is withheld near zero/near the power cap, or no discharge is permitted. |
| `battery_telemetry_missing`, `grid_telemetry_missing`, `soc_telemetry_missing` | Required reading is absent, unavailable, or non-finite. |
| `discharge_budget_cap`, `discharge_budget_exhausted` | The interval budget reduced the setpoint. |
| `phase_current_cap`, `soc_floor` | Fuse-current or SoC-floor intervention. |
| `resistive_heater_shed`, `bulk_heater_shed` | Last-resort heating-load shedding after the safe battery response. |
| `net_metering`, `fixed_action` | Normal non-native control path, not a native tracking failure. |
| `no_plan`, `no_current_plan_entry` | No readable/nonempty plan (no battery command, preserving existing behavior), or no current slot (fixed zero command). |

## Interpreting results

- `accepted` means the **Home Assistant service accepted the request**, not that
  the inverter executed it. `failed` means the service failed (`service_failed`)
  or raised (`exception`). `skipped_unavailable` means no service was called;
  the legacy boolean return still treats this as degradation-mode success.
  `not_attempted` applies to missing-plan cycles. `unknown` is reserved for a
  caller/test double that does not supply request-result diagnostics.
- Measurements are assembled **before the command in the same record**; they
  describe the response to earlier requests, not the newly selected mode. Use
  prior accepted requests, timestamps, `previous_*` metadata and sensor update
  times to attribute measurements. Reject stale readings and large sample gaps.
- The controller cache is scoped differently for hysteresis and diagnostics:
  `previous_*` can refer to another slot/action, while `was_native_for_hysteresis`
  is true only in the matching slot/action/manual context. Skipped requests may
  also update the legacy cache; inspect `previous_command_status`.
- Compare residual grid import/export and mode-switch frequency during native
  operation. Integrate signed battery power over actual elapsed sample times to
  estimate wrong-direction energy and cap overshoot; separate safety/budget-
  constrained periods from native tracking failures. Cumulative meters provide
  net-energy context. Twenty-second sampling cannot measure all brief reversals
  or sub-cycle response times.

This is logging only: no model version bump or control-policy change.
