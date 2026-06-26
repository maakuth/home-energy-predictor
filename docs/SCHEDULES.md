# HEPO Script Schedules

This app runs as a number of scripts scheduled as systemd units:

| Script | Frequency | Purpose |
|--------|-----------|---------|
| `run_often.py` | Every 20 seconds | Load-following: reads current HA state, computes battery setpoint from existing plan, pushes single power command |
| `run_frequent.sh` | Every 15 minutes (at spot pricing period break) | Extracts 3 days, predicts from "now", optimizes battery/GSHP/EV, pushes to HA |
| `run_slow.sh` | At :57 of every hour | SARIMA benchmark prediction (does not affect active plan) |
| `run_weekly.sh` | 02:00 every Monday | Extracts 730 days, retrains XGBoost + SARIMA, runs analysis for last 14 days |

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
