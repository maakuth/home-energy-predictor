# TODO

## Completed
- [x] **Environment Setup**: Verified access to PostgreSQL and Home Assistant API.
- [x] **Data Extraction**: Implemented `extract_data.py` to pull 365 days of history for relevant entities.
- [x] **Data Processing**: Implemented `process_data.py` for denoising, fireplace logic, and feature engineering.
- [x] **Model Training**: Implemented `train_model.py` using XGBoost. Trained model with MAE ~1.01 kWh.
- [x] **Prediction Script**: Implemented `predict_tomorrow.py` to fetch current state/forecast from HA and generate 24h predictions.
- [x] **Optimization Logic**: Implemented `optimize_plan.py` to compare predictions against spot prices and generate EV/Heating strategies.
- [x] **Home Assistant Integration**: Implemented `push_to_ha.py` to push the optimization plan to `sensor.hepo_optimization_plan`.

## In Progress
- [x] **Orchestration**: Create a main script (e.g., `run_daily.sh` or a Python coordinator) to run the full pipeline.

## Future Work
- [ ] **Feedback Loop**: Implement the daily accuracy check and automatic retraining trigger.
- [ ] **Refinement**: Improve feature handling (e.g., actual weather forecast for tomorrow instead of current temp).
