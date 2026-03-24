Technical Specification: Home Energy Prediction & Optimization (HEPO)
1. System Overview

The goal is to predict the next 24 hours of household power consumption (with a focus on heating and EV charging) to optimize usage against tomorrow's electricity spot prices. The system will run on a local PC with access to a Home Assistant (PostgreSQL) database.
2. Data Engineering & Cleaning

Input: states and state_attributes tables from PostgreSQL.
Target Entities: * sensor.outside_temperature

    sensor.mlp_teho (Ground Source heat pump power)

    Air to air heat pump power: sensor.saikaan_olohuone_current_power sensor.mokkimokin_ilp_power 
    
    Air to air heat pump energy consumption (power can be derived): sensor.mummun_energy

    sensor.mlp_varaajan_lampotila (accumulator top temperature)
    
    sensor.xpz_491_battery_level device_tracker.xpz_491_position (EV SoC and whether it's at home)

    sensor.sahkokauppa_nyt (total household power)

Pipeline Steps:

    Extraction: SQL query to pull the last 365 days of state history for the above entities.

    Denoising: * Apply a Rolling Median Filter (window=3) to remove "insane" spikes.

        Clip values based on physical limits (e.g., Power>0, Temp<50°C).

    Resampling: Aggregate all data to 1-hour intervals.

        Continuous values (Temp/Power): mean()

        State values (EV Home): max() (if home at any point in the hour, consider home).

    Gap Filling: Linear interpolation for gaps <2 hours; drop/ignore larger gaps to avoid synthetic bias.

3. Feature Engineering: The Fireplace Logic

To isolate the heating system's relationship with outside temperature, the agent must infer "Fireplace Assistance."

    Logic: * Calculate Rate of Change (RoC)=ΔtTacc​(t)−Tacc​(t−1)​.

        Identify "Fireplace Hours": If RoC>0.3°C/hr AND (GSHP_Power+AAHP_Power)<Threshold, then is_fireplace_active = True.

    Heating Demand Factor: Create a feature Pheat_norm​ which represents the power used when the fireplace is off. This helps the model learn the true heat-loss coefficient of the building.

4. Model Specification

The agent should implement a Gradient Boosted Regressor (XGBoost or LightGBM) for its ability to handle non-linear efficiencies (COP) of heat pumps.
Model Features (X):

    Weather: outside_temp, solar_forecast_kwh (external API).

    Thermal State: accumulator_temp, is_fireplace_active (lagged).

    EV State: is_home, soc_needed (Target SOC - Current SOC).

    Temporal: hour_of_day, day_of_week, month (to proxy ground temp for GSHP).

Target (Y):

    total_power_usage_kwh (hourly).

5. Deployment & Execution Loop

The CLI agent will orchestrate the following script cycle at 18:00 daily.
Step 1: Data Refresh

Sync the latest 24 hours of HA data and the 24-hour solar/weather forecast.
Step 2: Inference

Generate a 24-hour vector [P1​,P2​,...P24​] representing predicted power usage for tomorrow.
Step 3: Optimization Logic

Compare prediction against tomorrow’s spot prices (€/kWh):

    EV Strategy: Identify the N cheapest hours while ev_is_home == True to reach target SOC.

    Thermal Strategy: If spot price is in the bottom 20th percentile, increase GSHP setpoint by 2°C to charge the 500L accumulator (thermal buffering).

Step 4: HA Integration

The agent will push the "Optimization Plan" back to Home Assistant via REST API or MQTT to trigger automations.
6. Feedback & Retraining

    Accuracy Check: Every day at 17:55, calculate the Mean Absolute Error (MAE) of the previous day's forecast.

    Automatic Retraining: If the 7-day rolling MAE increases by >20%, trigger a full model retrain on the most recent 90 days of data.

Implementation Stack for Agent:

    Language: Python 3.10+

    DB Connector: psycopg2 or SQLAlchemy

    ML Libraries: scikit-learn, xgboost, pandas

    API: Home Assistant REST API
