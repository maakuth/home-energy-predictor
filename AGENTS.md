# home-energy-predictor notes for agents
- This is a python app that tries to predict home energy usage by an ML model
- Implement changes using test-driven development: first add failing test, do changes, observe test passing
- Use python virtualenv to run tests: .venv/bin/python3 -m pytest
- DON'T do any changes to database or home assistant without explicit permission. The machine running agent probably doesn't even have access to these.
- Offer to save your work to git frequently
- Don't do heredoc hacks or other shenanigans to modify files. If there's something preventing file modification, say so and the user will help.
- The development likely isn't running on the machine that has connectivity to the HA and psql, so don't bother trying to run it against them 
- You can get the up to date situation of from the original system by executing pull-from-murrikka.sh

### Strategic Adaptation (Feedback Loop)
- The system stores historical performance metrics in the `performance_analysis` table in `hepo.db`.
- **Metrics available:** `mae_kw`, `bias_kw` (over/under prediction), and battery ROI metrics (`battery_planned_savings_eur`, `planned_spread`).
- **Agent Guidance:** Future improvements should query this table to identify systemic issues. For example:
  - If `bias_kw` is consistently positive, the optimizer is being too conservative with battery storage.
  - If `planned_spread` is decreasing over time, the discharge thresholds in `.env` may need adaptive adjustment based on market volatility.
