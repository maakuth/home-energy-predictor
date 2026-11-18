# Home Energy Predictor Agent Guide

## Development and Safety
- Use test-driven development for behavior changes: add a failing test, implement the change, then confirm it passes.
  - When a feature is ready and tests are passing, review that there's enough tests to cover regressions, but not too much redundancy
- Run tests with `venv/bin/python3 -m pytest`. Do not use `.venv`; it belongs to the host system.
- Fix Pyright errors rather than suppressing them.
- Before changing model behavior, check `.env.template` and `docs/ENV_VARIABLES.md` for a suitable configuration option.
- Pytest isolates runtime files automatically. Do not bypass that isolation or run operational scripts against live state.
- Do not change Home Assistant, PostgreSQL, or `hepo.db` without explicit permission. Do not make live Home Assistant or PostgreSQL calls without permission.

## Runtime
- `docs/SCHEDULES.md` is the source of truth for systemd schedules and pipeline details.
- `run_often.py` controls the live battery setpoint every 20 seconds; `run_frequent.sh` re-optimizes every 15 minutes; `run_slow.sh` runs the hourly SARIMA benchmark; `run_weekly.sh` retrains models and runs analysis.

## Model Versioning
- `VERSION` stores the semantic model version independently of Git history.
- Bump MINOR for intentional training, inference, or optimization behavior changes. Bump PATCH for fixes to that behavior. Do not bump for documentation, logging, or no-behavior refactors.
- `get_model_version()` tags prediction, optimization-plan, and performance archives.
- Compare historical metrics by model version before tuning. `bias_kw` is predicted minus actual usage, so a positive value means over-prediction; do not infer an optimizer fault from it alone.

## Testing and Evaluation
- Normally run `venv/bin/python3 -m pytest -k 'not slow'`; run the full suite before merging to master branch. Slow tests are marked `@pytest.mark.slow`.
- Do not tune planner horizons, discounts, or thresholds with short replay windows. Use full-length fixture replays for tuning; short replays only detect crashes and constraint violations.

## Inspection
- Inspect the current plan with `venv/bin/python3 utils/inspect_plan.py --summary`; it defaults to `state/optimization_plan.json`.
- Compare plans with observed behavior using `venv/bin/python3 utils/compare_plan_logs.py --summary`. Use `--help` for focused views.

## Temporary Workspace
- Run all test-driven development cycles in a dedicated temporary Git worktree; never edit application code in `/workspace` directly. Multiple temporary worktrees may be used concurrently for separate tasks.
- Before using a temporary worktree, run `git worktree list` from `/workspace` and check the selected worktree with `git -C <worktree> status --short`. If it is dirty or belongs to an active task, stop and ask rather than deleting or overwriting it. A new, uniquely named worktree may instead be created for the task.
- Create a named task branch from the intended base when creating a new worktree, or use `git switch -c <branch-name> <base-branch>` in a clean, inactive temporary worktree. Preserve existing branches and other tasks' work.
- Ensure the selected worktree's `venv` is a symlink to `/workspace/venv`. This is a local workspace aid: do not stage it.
- Run tests from the selected temporary worktree using its `venv` symlink. Return work through the named Git branch; never copy changed files between workspaces manually.
- The temporary workspace can vanish between prompts due to container restart, so commit and push your work to a branch in main workspace often
