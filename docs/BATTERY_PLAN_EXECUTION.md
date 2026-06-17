# Battery Plan Execution

This document describes the different battery action modes used in the HEPO system, what they mean, and their permitted limits at each stage of the pipeline.

## Overview

Battery actions flow through two stages:

1. **Planning** — the battery planner (heuristic or LP) assigns an action per interval
2. **Real-time execution** — `compute_load_following_setpoint` in `utils/battery_utils.py` may adjust the setpoint based on live sensor readings

### The idle / follow distinction

| Action | Planned dispatch | Real-time behaviour |
|--------|-----------------|---------------------|
| `idle` | None. Battery power = 0 kW. | Battery does nothing regardless of grid conditions. Setpoint forced to 0 kW. |
| `follow` | None. Battery power = 0 kW. | Load-following active — battery adjusts to zero out the grid meter, capped at `BATTERY_FOLLOW_MAX_KW` (default 2 kW). |

The planners decide per interval whether to emit `idle` or `follow` via `should_idle_interval()` in `battery_planners/base.py`. The function uses several criteria:

1. **Futile check**: if net load > max battery power, load-following can't zero the grid → `idle`
2. **Negligible flow**: if net load < 0.2 kW, not worth cycling → `idle`
3. **No degradation cost**: if `BATTERY_DEGRADATION_COST_EUR_PER_KWH` is 0 → `follow` (legacy default)
4. **Cost-benefit**: cycling cost (degradation + efficiency loss) vs grid benefit → whichever is cheaper

## Battery Actions

### idle

| Layer | Behaviour |
|-------|-----------|
| **Planning** | No dispatch. Battery power = 0 kW. |
| **Real-time** | Battery power forced to 0 kW regardless of grid state. No load-following. |

**Use case**: when load-following would do more harm than good (e.g., EV charging at night would drain the battery, or solar surplus too small to justify cycling).

**Permitted limits**: 0 kW (hard forced).

---

### follow

| Layer | Behaviour |
|-------|-----------|
| **Planning** | No dispatch. Battery power = 0 kW. |
| **Real-time** | Proportional control adjusts battery to drive net grid flow toward zero, with a 200 W deadband. Capped at `BATTERY_FOLLOW_MAX_KW` (default 2 kW) to prevent overriding the planner's decision. |

**Use case**: intervals where the planner has no profitable dispatch but small grid fluctuations can still be smoothed.

**Permitted limits**: `[-max_follow_kw, max_follow_kw]` ∩ `[-max_battery_kw, max_battery_kw]` ∩ phase caps.

---

### charge_solar

| Layer | Behaviour |
|-------|-----------|
| **Planning** | Battery charges using only solar energy. `charge_from_solar_kwh > 0`, `charge_from_grid_kwh ≈ 0`. |
| **Real-time** | Setpoint is capped to the actual solar surplus (`solar_kw - actual_load_kw`). If surplus is zero or negative, setpoint is clamped to 0 (no charging from grid). |

**Permitted limits**: `[0, min(planned_kw, surplus_kw)]` — never charges from grid.

---

### charge_grid

| Layer | Behaviour |
|-------|-----------|
| **Planning** | Battery charges from the grid (price arbitrage — buy low). `charge_from_grid_kwh > 0`, `charge_from_solar_kwh` may be zero or non-zero (if also solar-charging). |
| **Real-time** | **No load-following adjustment.** Planned setpoint is passed through unchanged. |

**Permitted limits**: `[0, max_battery_kw]` — only phase current capping applies.

---

### charge_mixed

| Layer | Behaviour |
|-------|-----------|
| **Planning** | Battery charges from both solar and grid simultaneously. Only used by the heuristic planner. |
| **Real-time** | **No load-following adjustment.** Pass-through. |

**Permitted limits**: `[0, max_battery_kw]`.

---

### discharge_load

| Layer | Behaviour |
|-------|-----------|
| **Planning** | Battery discharges to cover house load. `discharge_to_load_kwh > 0`, `discharge_to_export_kwh ≈ 0`. |
| **Real-time** | Discharge is capped to the actual house load (`actual_load_kw`). Never exports to grid. |

**Permitted limits**: `[-min(abs(planned_kw), actual_load_kw), 0]` — never exports.

---

### discharge_export

| Layer | Behaviour |
|-------|-----------|
| **Planning** | Battery discharges and exports excess to the grid (price arbitrage — sell high). `discharge_to_export_kwh > 0`, `discharge_to_load_kwh` may be zero or non-zero. |
| **Real-time** | **No load-following adjustment.** Pass-through. |

**Permitted limits**: `[-max_battery_kw, 0]`.

---

### discharge_mixed

| Layer | Behaviour |
|-------|-----------|
| **Planning** | Battery discharges to both load and export simultaneously. Only used by the heuristic planner. |
| **Real-time** | **No load-following adjustment.** Pass-through. |

**Permitted limits**: `[-max_battery_kw, 0]`.

---

## Universal Limits (All Modes)

Regardless of action, the final setpoint is always clamped to:

### Physical Battery Limits

```
adjusted_kw = max(-max_battery_kw, min(max_battery_kw, adjusted_kw))
```

Where `max_battery_kw` is typically 10 kW (configurable via `MAX_BATTERY_KW` env var).

### Phase Current Capping

If three-phase current sensors are available, the setpoint is further constrained to prevent exceeding the main fuse rating on any phase:

- **Import limit**: each phase current `Ip` must stay below `MAIN_FUSE_SIZE_A` (default 25A) when charging
- **Export limit**: each phase current `Ip` must stay above `-MAIN_FUSE_SIZE_A` when discharging

The per-phase power constraints are:

```
P_max_phase = battery_w + (fuse - Ip) × 3 × 230   # charging headroom
P_min_phase = battery_w + (-fuse - Ip) × 3 × 230   # discharging headroom
```

The combined constraint is the intersection across all three phases:

```
combined_min_kw = max(P_min_phase_A, P_min_phase_B, P_min_phase_C) / 1000
combined_max_kw = min(P_max_phase_A, P_max_phase_B, P_max_phase_C) / 1000
adjusted_kw = clamp(adjusted_kw, combined_min_kw, combined_max_kw)
```

### Follow Cap

When `planned_action == 'follow'`, the setpoint is additionally capped:

```
if abs(target_kw) > BATTERY_FOLLOW_MAX_KW:
    adjusted_kw = battery_kw   # maintain current setpoint, don't over-correct
```

Default: `BATTERY_FOLLOW_MAX_KW = 2.0`. Prevents load-following from draining the battery on large, sustained grid flows (e.g., EV charging).

## Summary Table

| Action | Planner Source | Load-Following Adjustment | Charge from Grid? | Export to Grid? | Real-Time Limit |
|--------|---------------|---------------------------|-------------------|-----------------|-----------------|
| `idle` | Both | No (forced 0) | Never | Never | 0 kW |
| `follow` | Both | Yes (capped at `FOLLOW_MAX_KW`) | Opportunistic | Opportunistic | ±follow_max_kw, phase caps |
| `charge_solar` | Both | Yes (capped to surplus) | Never | N/A | surplus_kw cap |
| `charge_grid` | Both | No | Yes | N/A | ±max_kw, phase caps |
| `charge_mixed` | Heuristic only | No | Yes | N/A | ±max_kw, phase caps |
| `discharge_load` | Both | Yes (capped to load) | N/A | Never | actual_load cap |
| `discharge_export` | Both | No | N/A | Yes | ±max_kw, phase caps |
| `discharge_mixed` | Heuristic only | No | N/A | Yes | ±max_kw, phase caps |

## Config Env Vars

| Variable | Default | Description |
|----------|---------|-------------|
| `BATTERY_FOLLOW_MAX_KW` | 2.0 | Max battery power (kW) for load-following corrections |
| `BATTERY_DEGRADATION_COST_EUR_PER_KWH` | 0.0 | Cycling cost driving the idle vs follow decision |
