# Forecast model hand-off (baseload discrepancy)

During planning, HEPO intentionally switches from the XGBoost future
predictions in `state/future_predictions.json` to the SARIMA series in
`state/sarimax_predictions.json` when that file exists.  SARIMA is used as the
primary model and XGBoost only fills intervals beyond SARIMA's 24 h horizon.

Consequences:

- `future_predictions.json` can start near 1.7 kW while `optimization_plan.json`
  reports a ~6.7 kW predicted baseload for the same run.  This is a model/source
  hand-off, not a dispatch arithmetic error.
- Battery economics must therefore be judged on the SARIMA-driven plan and
  realised/replayed outcomes, not on the XGBoost series alone.

Do not compare the two series as if they were the same forecast.
