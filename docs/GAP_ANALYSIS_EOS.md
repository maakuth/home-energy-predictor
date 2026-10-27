# Gap Analysis: Akkudoktor EOS vs. Home Energy Predictor (HEPO)

Date: 2026-10-27
Source reviewed: https://github.com/akkudoktor-eos/eos (cloned to /tmp/EOS)
Compared against: HEPO repository at /workspace (VERSION 1.41.0)

## 1. Purpose and scope

This document captures a feature-by-feature gap analysis between the Akkudoktor Energy
Optimization System (EOS) and the Home Energy Predictor (HEPO). The question answered is:

> What can EOS do that HEPO cannot?

It is an analysis document only. No code or behavior changes are implied by it.

## 2. Project profiles

### 2.1 Akkudoktor EOS
- General-purpose, device-simulation + genetic-algorithm energy optimizer.
- Exposed as a REST API server with a web dashboard (EOSdash).
- Distributed as a Home Assistant add-on, Docker image, or from source.
- Pluggable prediction providers for PV, price, load, weather, feed-in tariff, and fees.
- Measurement framework with typed channels, quality metadata, and battery capacity estimation.
- Integration adapters for Home Assistant and Node-RED.

### 2.2 Home Energy Predictor (HEPO)
- Custom, ML-heavy household optimizer built around Home Assistant + PostgreSQL.
- XGBoost + SARIMAX load forecasting, Solcast PV, Nordpool prices.
- MPC-based battery dispatch (heuristic + Nemotron-Linprog planners) plus GSHP, resistive, and
  bulk-heater co-optimization and EV charging.
- 20-second load-following battery setpoint control (`run_often.py`).
- Deep performance/evolution analytics in SQLite (`hepo.db`) with model versioning.
- Replay testing framework for battery planners.

## 3. Gap table: capabilities EOS has that HEPO lacks

### 3.1 Controllable devices and loads

| Capability | EOS | HEPO |
|---|---|---|
| Shiftable home appliances (dishwasher, washing machine, dryer) with cycle counts, per-cycle time windows, earliest start, deadlines (strict/best-effort), min gap between cycles, load profiles | Yes | No |
| Flexible household loads beyond fixed baseload | Yes | No |
| EV departure targets (reach a SoC by a deadline or within a max duration) | Yes | No (target SoC only) |
| Inverter as a first-class simulated device | Yes | No |
| Battery-to-grid export as discrete optimized rates | Yes (configurable rates) | Partial (`BATTERY_ALLOW_EXPORT` boolean) |
| Heat pump as generic temperature-modelled device with COP | Yes | Yes, but more specialized (GSHP + resistive + bulk) |

### 3.2 Optimization approach

| Capability | EOS | HEPO |
|---|---|---|
| Genetic-algorithm scenario search over a 48 h horizon | Yes | No (MPC + LP/heuristic) |
| Graphical / PDF simulation results and plan visualization | Yes | No (text/summary via `inspect_plan.py`) |
| Self-consumption interpolator tuning for low-memory devices | Yes | No |

### 3.3 Prediction ecosystem

| Capability | EOS | HEPO |
|---|---|---|
| PV forecast providers (Forecast.solar, pvlib, PVnode, Solcast, VRM, HA, Akkudoktor, import) | Many | Solcast only |
| Electricity price providers (Tibber, Energy Charts, SMARD, fixed, import) | Many | Nordpool + prediction feed |
| Feed-in tariff as a separate first-class prediction | Yes | Estimated export prices only |
| Electricity fee prediction (grid fees) as a provider | Yes | Combined into `GRID_FEES_EUR_PER_KWH` |
| Load prediction providers (Akkudoktor profile, VRM, import) | Yes | XGBoost + SARIMA |
| Weather providers (OpenMeteo, BrightSky, ClearOutside, import) | Many | HA outside-temp sensor |
| Adjusted predictions refined by real measurements | Yes | No (performance analysis only) |
| Dynamic prediction import via REST `PUT` endpoint | Yes | No (extract from HA/PostgreSQL) |
| Interval interpolation utilities | Yes | Partial |

### 3.4 Measurement framework

| Capability | EOS | HEPO |
|---|---|---|
| Typed measurement channels (power/cumulative/interval energy, units, integration, max gap, timestamp ref) | Yes | No (raw HA/PostgreSQL extraction) |
| Sample quality metadata (measured/estimated/invalid/unavailable, resets, generations) | Yes | No |
| Battery capacity estimation from independent SoC anchors + DC power | Yes | No (configured capacity only) |
| Household energy balance aggregation from meter readings | Yes | No |
| Measurement-driven prediction refinement | Yes | No |
| Retention manager for stored data | Yes | No (manual archiving) |

### 3.5 Server, integration, and deployment

| Capability | EOS | HEPO |
|---|---|---|
| REST API server with Swagger/OpenAPI docs | Yes | No |
| Web dashboard (EOSdash) | Yes | No |
| Home Assistant add-on packaging | Yes | No (scripts/systemd) |
| Docker images | Yes | No |
| Node-RED adapter | Yes | No (push to HA only) |
| Dynamic configuration via API | Yes | No (`.env` / scripts) |
| Small-device tuning documentation | Yes | No |

### 3.6 Operational infrastructure

| Capability | EOS | HEPO |
|---|---|---|
| Config migration utilities and versioned config scopes (GENETIC vs GENETIC0) | Yes | No |
| Lazy imports / memory tuning for constrained hardware | Yes | No |

## 4. Where HEPO is stronger (for balance)

- ML load forecasting: XGBoost + SARIMAX benchmark with feature engineering vs. EOS's standard load profile + adjustments.
- Heating-specific optimization: GSHP + top resistive + bulk heater co-optimization with virtual accumulator, COP ramp, strategic stop thresholds, health monitoring.
- Performance/evolution analytics: MAE/bias by window, battery ROI, plan stability, action flips, lead-time analysis, model versioning.
- Price modeling: price-pegging, kulutusvaikutus, import/export floors, grid-charge margins, solar hedge (p10/p50).
- Load-following runtime: 20-second battery setpoint control with ramp-rate limiting and net-metering correction.
- Replay testing framework for battery planners.

## 5. Recommended direction: port EOS advantages into HEPO

Porting selected EOS capabilities into HEPO is the easier and lower-risk path:

1. You own HEPO end-to-end: no upstream contracts, release cadence, or review process.
2. HEPO's optimization core is already extensible (pluggable battery planners, coordinated multi-device plan).
3. HEPO's specialized strengths (heating, price modeling, analytics, load-following) do not fit EOS's generic device/genetic abstractions well; porting them into EOS would be a larger, riskier rewrite or a divergent fork.
4. Provider additions and EV deadlines are bounded, localized work in HEPO.

Porting HEPO's strengths into EOS is harder because it would require conforming to a large generic framework, and much of HEPO's value is household-specific (Finnish price-pegging, GSHP virtual accumulator) that upstream is unlikely to accept.

### Caveat
Some EOS advantages are infrastructure-heavy (REST server, dashboard, HA add-on, Docker). If those are wanted, consider a thin FastAPI wrapper around HEPO or adopting EOS's server scaffolding as a reference, rather than building a full dashboard from scratch.

## 6. Suggested adoption priorities (EOS -> HEPO)

1. EV departure deadlines (min SoC by time / max duration).
2. Shiftable home-appliance scheduling (cycles, windows, deadlines, load profiles).
3. Additional PV/price/weather providers plus a prediction import endpoint.
4. Measurement quality metadata and battery capacity estimation.
5. Thin REST API for observability.
6. Node-RED adapter and Docker packaging.

## 7. Conclusion

The dominant gap is not battery optimization (HEPO is strong there) but **device breadth (appliances, flexible loads, EV deadlines), prediction-provider breadth, measurement quality, and server/deployment ergonomics**. Closing these by porting selected EOS capabilities into HEPO is the recommended path; porting HEPO into EOS would be a larger and less natural effort.