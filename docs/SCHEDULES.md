# HEPO Script Schedules

This app runs as a number of scripts scheduled as systemd units:

| Script | Frequency | Purpose |
|--------|-----------|---------|
| `run_often.py` | Every 20 seconds | Reads current HA state and plan; requests native battery load following or a fixed power setpoint |
| `run_frequent.sh` | Every 15 minutes (at spot pricing period break) | Extracts 3 days, predicts from "now", optimizes battery/GSHP/EV, pushes to HA |
| `run_slow.sh` | At :57 of every hour | SARIMA benchmark prediction (does not affect active plan) |
| `run_weekly.sh` | 02:00 every Monday | Extracts 730 days, retrains XGBoost + SARIMA, runs analysis for last 14 days |

## Battery runtime control

The plan's `follow` action (including a manual `follow` override) publishes
`self_consumption` to `hoymiles/battery/set_mode` via Home Assistant's
`mqtt.publish` service. The inverter then follows the load natively; HEPO does
not send numeric power commands, apply software ramping, or run the net-metering
PI controller while native mode is active. The command is reasserted on each
run; the retained `hoymiles/battery/mode_state` reports the last successfully
requested mode, not independently verified inverter state.

`charge_solar` and `discharge_load` also prefer native mode when both measured
battery power and the estimated native response agree with the requested
direction. The estimated response is `battery_power - grid_power` (positive
means charging). A clearly wrong measured direction forces fixed control even
when that estimate has already returned to the correct direction.

- `charge_solar` falls back to charge-only fixed power, or zero without surplus.
  It no longer automatically changes to `discharge_load` when solar falls short.
- `discharge_load` falls back to discharge-only fixed power capped by the
  current plan's unsmoothed discharge power and interval discharge budget.
- A 200 W deadband avoids mode flapping near zero: native mode is entered on a
  clear allowed-direction signal, retained through small zero crossings, and
  left on a clear reversal. Re-entry after a discharge-power cap requires
  200 W of headroom below the cap. Very small planned discharges use fixed power.
- Hysteresis records the last successful control request in
  `state/native_follow_control.json` (`HEPO_NATIVE_FOLLOW_STATE_FILE` overrides
  the path), scoped to the plan slot, action, and manual override mode. It does
  not treat that record as confirmation of actual inverter state.

These three actions bypass the net-metering PI controller, including while
using fixed fallback. Directional fallback also bypasses ramping so a stop or
power cap takes effect immediately. Missing/non-finite battery or grid
telemetry disables native mode and requests zero power, subject to fuse safety.

Other actions and fixed fallbacks retain the Home Assistant `number.set_value` path to
the bridge's signed `hoymiles/battery/set_power` command. Every successful power
command switches the bridge to `fixed`, including zero watts for `idle`; no
separate `fixed` mode command should follow a nonzero power command, because it
would reset power to zero.

HEPO estimates native power from current grid/battery readings for its existing
discharge-budget and phase-fuse checks. When either check requires a cap, it
uses fixed power instead. At/below the configured SoC floor (or without SoC
telemetry), native mode is also disabled to avoid uncontrolled discharge.
Normal native following resumes when those constraints no longer apply.
Inverter/BMS hardware limits still apply; native mode cannot enforce HEPO's
software follow-power limit or ramp rate between the 20-second checks.
Direction restrictions are likewise best-effort: brief wrong-direction flow
can occur between checks, and fuse protection may override a directional cap.

Each completed cycle also emits a `HEPO_BATTERY_CONTROL` compact JSON record
with signed telemetry, plan context, exact decision reasons, final requested
mode and command result. See [battery diagnostics](BATTERY_DIAGNOSTICS.md) for
the schema, grep/jq examples, and measurement/request timing caveats.

## systemd Timer Examples

### run_often (every 20 seconds)

```
# /etc/systemd/system/hepo-often.timer
[Unit]
Description=HEPO often timer

[Timer]
OnBootSec=30s
OnUnitActiveSec=20s

[Install]
WantedBy=timers.target
```

### run_frequent (every 15 minutes)

```
# /etc/systemd/system/hepo-frequent.timer
[Unit]
Description=HEPO frequent timer

[Timer]
OnCalendar=*:0/15
Persistent=true

[Install]
WantedBy=timers.target
```

### run_slow (at :57 every hour)

```
# /etc/systemd/system/hepo-slow.timer
[Unit]
Description=HEPO slow timer

[Timer]
OnCalendar=*:57
Persistent=true

[Install]
WantedBy=timers.target
```

### run_weekly (Monday 02:00)

```
# /etc/systemd/system/hepo-weekly.timer
[Unit]
Description=HEPO weekly timer

[Timer]
OnCalendar=Mon *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```
