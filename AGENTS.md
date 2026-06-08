# home-energy-predictor notes for agents

## General Development Guidelines
- This is a python app that tries to predict home energy usage by an ML model
- Implement changes using test-driven development: first add failing test, do changes, observe test passing
- Use python virtualenv to run tests: `venv/bin/python3 -m pytest`
- If the venv breaks (e.g., after a system Python upgrade), recreate it with:
  ```bash
  rm -rf venv
  python3 -m venv venv
  venv/bin/pip install -r requirements.txt
  ```
- Offer to save your work to git frequently
- Don't do heredoc hacks or other shenanigans to modify files. If there's something preventing file modification, say so and the user will help.
- There are a lot of tunables in .env.template, documented in ENV_VARIABLES.md. If the user asks for some model behaviour change, see if there's a tunable that could be used to implement it.

## The environment
- DON'T do any changes to database or home assistant without explicit permission. The machine running agent probably doesn't even have access to these.
- MAKE SURE new tests don't interfere with existing data, in case development happens in the same working directory with the production
- The development likely isn't running on the machine that has connectivity to the HA and psql, so don't bother trying to run it against them
- You can get the up to date situation of from the original system by executing pull-from-murrikka.sh, though you might already be running at murrikka. Better check the timestamps of hepo.db, etc.


## Model Versioning (IMPORTANT)

### Overview
- Model versions use **semantic versioning** (MAJOR.MINOR.PATCH) stored in `VERSION` file
- **NOT** tied to git commits - only intentional training logic changes increment the version
- This allows developers to iterate on code without polluting the model version history
- All predictions and analysis are tagged with the model version, enabling performance tracking

### When to Update VERSION
Update the version when you make changes that affect **model training or inference logic**:

**MINOR bump** (e.g., 1.0.0 → 1.1.0):
- ✅ Changed feature engineering in `process_data.py`
- ✅ Modified XGBoost training parameters in `train_model.py`
- ✅ Adjusted SARIMA window or parameters in `train_sarima.py`
- ✅ Changed prediction blending weights or ensemble logic
- ✅ Added/removed sensors or input features
- ✅ Modified battery optimization constraints in `optimize_plan.py`

**PATCH bump** (e.g., 1.0.0 → 1.0.1):
- ✅ Bug fixes in prediction or optimization logic
- ✅ Corrected incorrect feature calculations
- ✅ Fixed off-by-one errors or unit conversions

**DO NOT bump for:**
- ❌ Documentation or code comments
- ❌ Refactoring without logic change
- ❌ Changes to `run_daily.sh`, `run_frequent.sh`, analysis scripts
- ❌ Output formatting or logging changes
- ❌ Git commits that don't affect model behavior

### How to Update VERSION
```bash
# Edit VERSION file
echo "1.1.0" > VERSION

# Commit with clear message
git add VERSION <modified-model-files>
git commit -m "Bump version to 1.1.0: [brief description of model change]

Details of what changed in the training/inference logic..."
```

### Example Commits
✅ **Good - Includes version bump:**
```
Bump to 1.1.0: Add GSHP temperature as feature

- Added mlp_pumpun_lampotla to feature engineering
- Improves model sensitivity to heating system state
```

❌ **Bad - Missing version bump:**
```
Add GSHP temperature as feature
```

❌ **Bad - Unnecessary bump:**
```
Bump to 2.0.0: Update documentation
```

### How It Works
- `utils/git_utils.py::get_model_version()` reads from `VERSION` file
- All training code calls `get_model_version()` to tag predictions/analysis
- Database queries can filter by model version to track which code version generated predictions
- Performance metrics (MAE, bias) are tracked per model version

### Querying by Version
```sql
-- See performance for each model version
SELECT DISTINCT model_version, mae_kw, bias_kw 
FROM performance_analysis 
ORDER BY analysis_timestamp DESC;

-- Compare accuracy across versions
SELECT version, COUNT(*) as samples, AVG(ABS(error)) as mae
FROM predictions
GROUP BY version
ORDER BY version DESC;
```

## Strategic Adaptation (Feedback Loop)
- The system stores historical performance metrics in the `performance_analysis` table in `hepo.db`.
- **Metrics available:** `mae_kw`, `bias_kw` (over/under prediction), and battery ROI metrics (`battery_planned_savings_eur`, `planned_spread`).
- **Agent Guidance:** Future improvements should query this table to identify systemic issues. For example:
  - If `bias_kw` is consistently positive, the optimizer is being too conservative with battery storage.
  - If `planned_spread` is decreasing over time, the discharge thresholds in `.env` may need adaptive adjustment based on market volatility.
  - Use model versioning to identify which changes actually improved performance

## Running tests
$ venv/bin/python3 -m pytest

Normally we want to go fast, the only slow ones are SARIMA tests. If you didn't touch SARIMA stuff, you don't need to run those every time.
Run them before commit though.

$ venv/bin/python -m pytest -k 'not sarima'
