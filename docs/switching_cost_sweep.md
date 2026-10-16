# Offline Battery Switching-Cost Sweep

Evaluated 2026-10-16 using the four included immutable quality fixtures
(`jan.pkl`, `may.pkl`, `jul.pkl`, and `oct.pkl`; 2,880 intervals total) with
`nemotron-linprog`. No production, Home Assistant, PostgreSQL, or database
state was used. Lower cost is better.

Every run completed with `safety.ok=true` and zero SoC violations.

| `BATTERY_SWITCHING_COST_EUR` | Total cost EUR | Change vs 0 | Direction reversals | Action transitions | Net-flow variation kWh |
|---:|---:|---:|---:|---:|---:|
| 0.000000 | -81.642387 | 0.000000 | 384 | 1467 | 2487.809 |
| 0.000015 | -82.823198 | -1.180811 | 384 | 1469 | 2487.809 |
| 0.000020 | -82.860427 | -1.218040 | 382 | 1466 | 2487.145 |
| 0.000025 | -82.860427 | -1.218040 | 382 | 1466 | 2487.145 |
| 0.000030 | -82.860427 | -1.218040 | 382 | 1466 | 2487.145 |
| 0.000040 | -82.845346 | -1.202959 | 382 | 1467 | 2487.145 |
| 0.000050 | -82.689368 | -1.046981 | 381 | 1465 | 2474.917 |
| 0.000075 | -82.712173 | -1.069786 | 380 | 1464 | 2473.290 |
| 0.000100 | -82.676653 | -1.034267 | 378 | 1458 | 2461.064 |
| 0.000150 | -82.557501 | -0.915114 | 375 | 1450 | 2434.173 |
| 0.000200 | -82.587887 | -0.945500 | 376 | 1447 | 2425.291 |
| 0.000250 | -82.418509 | -0.776122 | 372 | 1435 | 2395.790 |

The best result is a stable `0.000020`–`0.000030` plateau. `0.000025` is the
central, reproducibly rerun default candidate. Its per-fixture cost is
`-22.354654`, `-20.508245`, `-32.464359`, and `-7.533168` EUR for January,
May, July, and October respectively.

The term delivers only modest smoothing; it should be treated as an economic
LP tie-breaker, not a substitute for a dwell or minimum-runtime constraint.
Values above about `0.000040` increasingly give up realized cost for diminishing
additional smoothing. This remains an offline candidate until normal production
change review approves it.
