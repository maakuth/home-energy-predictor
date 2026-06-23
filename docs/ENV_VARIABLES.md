# Environment Variables Reference

This document describes every configuration knob defined in `.env.template`, verified against the Python source code. Copy the template to `.env` and adjust these values for your local setup.

---

## Database Configuration (PostgreSQL)

Used by `extract_data.py`, `find_wind_sensor.py`, `utils/db_utils.py`, `utils/verify_access.py` and `utils/explore_db.py` to connect to the Home Assistant PostgreSQL database.

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_HOST` | PostgreSQL server hostname or IP. | `localhost` |
| `DB_PORT` | PostgreSQL server port. | `5432` |
| `DB_NAME` | Name of the Home Assistant database. | `homeassistant` |
| `DB_USER` | Username for the ML agent. | `ml_agent` |
| `DB_PASSWORD` | Password for the database user. | *(secret)* |

---

## Home Assistant Configuration

Used by `utils/ha_utils.py` (REST API client), `utils/verify_access.py`, `utils/inspect_nordpool.py`, `utils/inspect_solcast.py` and `utils/inspect_solcast_today.py`.

| Variable | Description | Example |
|----------|-------------|---------|
| `HA_HOST` | Base URL of your Home Assistant instance. The code automatically prepends `http://` if the scheme is missing. | `http://192.168.0.1:8123` |
| `HA_TOKEN` | Long-lived access token for the Home Assistant REST API. Sent as a `Bearer` token in the `Authorization` header. | *(secret)* |

---

## Electricity Pricing and Tariffs

All monetary values are in **EUR per kWh** unless noted otherwise. These are consumed by `build_tariff_prices()` in `optimize_plan.py`.

| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `GRID_FEES_EUR_PER_KWH` | Single combined value for all grid transfer fees, taxes, VAT, and fixed adders. | `0.06` | When the fetched market prices are **already inclusive** of fees, this is subtracted to estimate export revenue. When prices are **raw spot prices**, this is added to compute the import price. |

---

## Battery Optimization Settings

Consumed by `plan_battery_dispatch()` in `optimize_plan.py`.

The dispatch logic uses a **profit-only** strategy with **marginal opportunity cost ranking**:
1. **Solar surplus** is captured (`charge_solar`) **only when storing is more valuable than exporting**. If the current export price exceeds `opportunity_cost × round_trip_efficiency`, the solar is exported to grid instead. This enables grid-arbitrage strategies (export solar at peak, charge cheap grid later).
2. **Discharge to load** happens when the current import price is at least as good as the marginal opportunity cost. The battery only discharges the *excess* energy not needed for strictly better future peaks, creating a gradual ramp rather than binary on/off.
3. **Discharge to export** happens when the current export price is the best in the remaining horizon or when it meets the arbitrage threshold. Like load discharge, it is limited to the excess energy beyond what is reserved for better opportunities.
4. **Grid charging** only happens when there is a profitable price delta to exploit, the current interval is the cheapest before that profitable discharge, and expected PV surplus over the next 24 h is not enough to fill the remaining battery room.

| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `BATTERY_CAPACITY_KWH` | Total usable battery capacity. | `50.0` | **Set to `0` to disable battery simulation entirely.** When `0`, the optimizer outputs an all-`idle` battery plan. |
| `BATTERY_MIN_SOC_PCT` | Minimum allowed State of Charge during normal operation. | `10.0` | Hard floor used by the dispatch logic. |
| `BATTERY_MAX_SOC_PCT` | Maximum allowed State of Charge during normal operation. | `90.0` | Prevents the optimizer from over-charging. |
| `BATTERY_RESERVE_SOC_PCT` | Emergency reserve SOC. | `10.0` | Defaults to the value of `BATTERY_MIN_SOC_PCT` in code if omitted. The effective floor is `max(MIN, RESERVE)`. |
| `BATTERY_INITIAL_SOC_PCT` | Fallback starting SOC. | `50` | Used only when `sensor.be_soc` is unavailable in Home Assistant. At runtime the code prefers the live HA sensor value. |
| `BATTERY_MAX_CHARGE_KW` | Maximum charge power in kW. | `10.0` | Limited by inverter / BMS. |
| `BATTERY_MAX_DISCHARGE_KW` | Maximum discharge power in kW. | `10.0` | Limited by inverter / BMS. |
| `BATTERY_CHARGE_EFFICIENCY` | Charge conversion efficiency. | `0.95` | Clamped to `[0.01, 1.0]` in code. |
| `BATTERY_DISCHARGE_EFFICIENCY` | Discharge conversion efficiency. | `0.95` | Clamped to `[0.01, 1.0]` in code. |
| `BATTERY_ALLOW_EXPORT` | Whether stored energy may be sold back to the grid. | `true` | When `false`, `discharge_to_export` is blocked and only self-consumption discharge is allowed. |
| `BATTERY_ALLOW_EXPORT_ENTITY` | Home Assistant boolean entity ID that overrides `BATTERY_ALLOW_EXPORT` at runtime. | `input_boolean.battery_allow_export` | The switch state (`on`/`off`) takes precedence over the env variable. If the entity is unavailable, the env value is used as fallback. |
| `BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH` | Minimum price margin (EUR/kWh) to trigger near-term arbitrage grid charging. | `0.005` | Set to `0.0` for aggressive local arbitrage (capture all profitable 15-min spreads). Set to `0.005` (default) to avoid cycling for micro-gains. Only affects grid charging; discharge decisions are unaffected. |
| `BATTERY_NET_METERING` | When true, use cumulative energy meters to match quarterly average instead of instantaneous load following. | `true` | With net metering, the grid meter only cares about net energy per 15-minute interval. This mode reads cumulative import/export sensors and adjusts battery power to match the planned quarterly average, preventing solar leakage. |
| `BATTERY_RAMP_RATE_KW_PER_MIN` | Max battery power change rate in kW per minute. | `3.0` | Prevents sudden power swings from net metering end-of-interval corrections (or any other source of step changes). The ramp is applied relative to the actual battery sensor reading, not the previous setpoint. Set to `0.0` to disable. |
| `BATTERY_LP_HORIZON` | Planning horizon for the LP planner (`nemotron-linprog`) in 15-min intervals. Used when tomorrow's spot prices are known. | `192` | Ignored by the heuristic planner. 192 = 48 hours. |
| `BATTERY_LP_HORIZON_FALLBACK` | Fallback horizon used when tomorrow's spot prices aren't published yet (context `tomorrow_valid=False`). | `96` | 96 = 24 hours. The planner reads `tomorrow_valid` from `BatteryPlannerContext` to decide which horizon to use. |
| `BATTERY_LP_DISCOUNT` | Per-interval discount factor (γ) applied to future costs in the LP planner. | `0.995` | γ < 1 prevents over-optimistic planning in receding-horizon mode. Set to 1.0 for no discount. |
| `BATTERY_LP_PARALLEL` | Enable HiGHS parallel solver. | `0` | Usually slower for small LP problems (< 100 variables). Set to `1` to enable. |
| `BATTERY_TERMINAL_VALUE_PERCENTILE` | Price percentile used to assign a terminal value to the final SoC in the LP planner. | `0.0` | 0 = no terminal value (planner only optimises within its horizon). A value of 50 uses the median price. |
| `BATTERY_DEGRADATION_COST_EUR_PER_KWH` | Battery wear cost per kWh cycled in the LP planner. | `0.0` | Adds a linear cost to charging and discharging. 0 = disabled. |

---

## GSHP (Ground Source Heat Pump) Settings

Consumed by `plan_gshp_dispatch()` in `optimize_plan.py`.

| Variable | Description | Example | Unit / Notes |
|----------|-------------|---------|--------------|
| `GSHP_INITIAL_TEMP` | Starting accumulator / buffer temperature. | `50.0` | °C. Fallback when `sensor.mlp_varaajan_lampotila` is unavailable in HA. |
| `GSHP_MIN_TEMP` | Hard minimum accumulator temperature. | `42.0` | °C. If the temperature drops to or below this, the heat pump **must** start regardless of price. |
| `GSHP_MAX_TEMP` | Hard maximum accumulator temperature. | `55.0` | °C. If the temperature reaches this, the heat pump **must** stop. |
| `GSHP_IS_RUNNING` | Initial on/off state of the heat pump. | `false` | bool. Fallback when `sensor.mlp_teho` is unavailable in HA. |
| `GSHP_ELECTRIC_POWER_KW` | Fallback nominal electrical power. | `4.0` | kW. **Only used when both `GSHP_POWER_MIN_KW` and `GSHP_POWER_MAX_KW` are absent.** In that case both min and max are set to this value. |
| `GSHP_COP` | Coefficient of Performance. | `3.5` | —. `thermal_kw = electric_kw * COP`. |
| `GSHP_HEAT_LOSS_K` | House heat-loss coefficient. | `0.1` | kW/°C. Used in the demand formula: `max(0, (20.0 - outside_temp) * heat_loss_k)`. Code default is `0.135`. |
| `GSHP_BASELINE_DEMAND_KW` | Baseline thermal demand. | `1.0` | kW. Represents DHW, circulation and standby heat loss. Added to the weather-driven demand. |
| `GSHP_POWER_MIN_KW` | Minimum compressor / inverter power. | `3.4` | kW. Used at `min_temp` in a linear power ramp. |
| `GSHP_POWER_MAX_KW` | Maximum compressor / inverter power. | `4.2` | kW. Used at `max_temp` in a linear power ramp. |
| `GSHP_STRATEGIC_STOP_DIFF_EUR` | Spot-price difference that triggers a strategic stop. | `0.05` | €/kWh. If the current effective price is `>=` (cheapest price in the safe lookahead window + this value), the pump stops to wait for cheaper hours. |

---

## EV Charging Settings

Consumed by `optimize()` in `optimize_plan.py`.

| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `EV_TARGET_SOC_PCT` | Desired final State of Charge after charging. | `80.0` | % |
| `EV_CAPACITY_KWH` | Total EV battery capacity. | `60.0` | kWh |
| `EV_CHARGE_POWER_KW` | Charger power level. | `3.5` | kW |
| `EV_CHARGE_HOURS` | Fallback charging duration. | `4.0` | hours. **Only used when the current EV SoC (`sensor.xpz_491_battery_level`) is unknown.** When the SoC is known, the optimizer calculates the exact number of slots needed from the deficit. |

---

## Planning Configuration

| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `PLAN_INTERVAL_MINUTES` | Time granularity of the optimization horizon. | `15` | minutes. Affects battery kWh calculations, GSHP temperature simulation, and EV slot counting. |
| `LOW_COST_PERCENTILE` | Percentile threshold for the low cost signal. | `30.0` | Intervals with effective cost below this percentile are marked as low cost. Used by `push_to_ha.py` to generate the `sensor.hepo_low_cost_signal` boolean. |

---

## Knobs Used in Code but Missing from `.env.template`

The following variables are read by `optimize_plan.py` (and other modules) but are **not present** in the current `.env.template`. You can add them manually if your deployment needs them.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAIN_FUSE_SIZE_A` | `25.0` | Main fuse rating per phase (Amps). Maximum grid import is clamped to `fuse × 3 × 0.230` kW. Prevents battery grid-charging from overloading the connection. |
| `GSHP_RESERVOIR_LITERS` | `500` | Buffer tank volume in litres. Used to compute `kWh_per_degree = (litres × 4.18) / 3600`. |
| `GSHP_HEATING_EFFICIENCY` | `1.0` | Efficiency multiplier applied to thermal output. |
| `GSHP_INITIAL_TEMP_DROP` | `3.0` | Temperature drop (°C) applied to the accumulator when the heat pump starts. Simulates thermal layering (mixing of cold return water). |
| `SAUNA_HOT_WATER_DEMAND_KW` | `6.0` | Extra thermal demand added when the sauna is predicted to be active. |
| `LEAF_BACKUP_HOURS` | `4.0` | Night backup duration for the Leaf EV strategy. |
| `LEAF_DAILY_TARGET_KWH` | `10.0` | Daily energy target for the Leaf EV strategy. |
| `HEPO_DISABLE_BATTERY` | `false` | Set to `true` to force-disable battery optimization at runtime (testing / degradation mode). |

---

## Notes

- The `.env` file is **git-ignored**; never commit credentials or local overrides.
- All `get_env_float()` calls fall back to sensible defaults if a variable is missing, so the system will not crash when a knob is omitted.
