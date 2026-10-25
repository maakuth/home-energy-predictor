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

Other actions retain the existing Home Assistant `number.set_value` path to
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
