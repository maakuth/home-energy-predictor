import json
import sqlite3
import os
from utils.ha_utils import push_ha_state
from utils.sqlite_utils import get_db_connection, db_exists

def push_accuracy():
    """Reads the latest performance metrics from hepo.db and pushes to HA."""
    if not db_exists():
        return

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_analysis'")
        if not cur.fetchone():
            conn.close()
            return

        # Get the latest analysis
        cur.execute("SELECT mae_kw, bias_kw, model_version, period_days FROM performance_analysis ORDER BY analysis_timestamp DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()

        if row:
            mae, bias, version, days = row
            if mae is not None:
                push_ha_state('sensor.hepo_accuracy', f"{mae:.3f}", {
                    'friendly_name': f'HEPO Real-time MAE ({days}d window)',
                    'unit_of_measurement': 'kW',
                    'bias': float(bias) if bias is not None else 0.0,
                    'model_version': version,
                    'period_days': days
                })
                print(f'✅ Accuracy metrics pushed: MAE={mae:.3f}, Version={version}')
    except Exception as e:
        print(f"⚠️ Error pushing accuracy metrics: {e}")

def push_plan():
    print('Loading optimization plan...')
    try:
        with open('optimization_plan.json', 'r') as f:
            plan = json.load(f)
    except FileNotFoundError:
        print('Error: optimization_plan.json not found.')
        return

    # State: Total predicted energy from current hour to end of tomorrow (sum of hourly predictions)
    total_energy = sum(p['predicted_usage_kwh'] for p in plan)
    
    # Calculate 24h estimate (first 24 hours of the plan)
    # Each interval is 15 mins. 24h = 96 intervals.
    intervals_in_24h = 96 
    usage_24h = sum(p['predicted_usage_kwh'] for p in plan[:intervals_in_24h])
    
    # Push the full optimization plan
    attributes = {
        'friendly_name': 'HEPO Optimization Plan',
        'plan': plan,
        'unit_of_measurement': 'kWh',
        'device_class': 'energy',
        'predicted_24h_usage': round(usage_24h, 2)
    }
    
    print(f'Pushing optimization plan to Home Assistant...')
    if push_ha_state('sensor.hepo_optimization_plan', f"{total_energy:.2f}", attributes):
        print('✅ Plan successfully pushed!')

    # Push current GSHP intent
    current_gshp_intent = plan[0].get('gshp_intent', 'STOP')
    attributes_gshp = {
        'friendly_name': 'HEPO GSHP Intent',
        'simulated_temp': plan[0].get('gshp_temp_simulated')
    }
    push_ha_state('sensor.hepo_gshp_intent', current_gshp_intent, attributes_gshp)
    print(f'✅ GSHP Intent pushed: {current_gshp_intent}')

    # Push Leaf charging intent
    current_leaf_intent = plan[0].get('leaf_intent', 'OFF')
    push_ha_state('sensor.hepo_leaf_charging_intent', current_leaf_intent, {
        'friendly_name': 'HEPO Leaf Charging Intent'
    })
    print(f'✅ Leaf Intent pushed: {current_leaf_intent}')

    # Also push 24h usage as a standalone sensor for easier history tracking
    attributes_24h = {
        'friendly_name': 'HEPO Predicted 24h Consumption',
        'unit_of_measurement': 'kWh',
        'device_class': 'energy'
    }
    push_ha_state('sensor.hepo_predicted_24h_usage', f"{usage_24h:.2f}", attributes_24h)

if __name__ == '__main__':
    push_plan()
    push_accuracy()
