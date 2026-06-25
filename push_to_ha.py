from __future__ import annotations
import json
import os
from typing import Any, Optional
from utils.ha_utils import push_ha_state
from utils.battery_utils import get_current_plan_entry
from utils.sqlite_utils import get_db_connection, db_exists

def push_accuracy() -> None:
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

def push_plan() -> None:
    print('Loading optimization plan...')
    # Support environment variable override for testing
    plan_file = os.getenv('TEST_PLAN_FILE', 'state/optimization_plan.json')
    try:
        with open(plan_file, 'r') as f:
            plan = json.load(f)
    except FileNotFoundError:
        print(f'Error: {plan_file} not found.')
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

    # Find the current plan entry (not just plan[0], which may be a future interval)
    current = get_current_plan_entry(plan)
    if current is None:
        current = plan[0]

    # Push current GSHP intent
    current_gshp_intent = current.get('gshp_intent', 'STOP')
    attributes_gshp = {
        'friendly_name': 'HEPO GSHP Intent',
        'simulated_temp': current.get('gshp_temp_simulated')
    }
    push_ha_state('sensor.hepo_gshp_intent', current_gshp_intent, attributes_gshp)
    print(f'✅ GSHP Intent pushed: {current_gshp_intent}')

    # Push Leaf charging intent
    current_leaf_intent = current.get('leaf_intent', 'OFF')
    push_ha_state('sensor.hepo_leaf_charging_intent', current_leaf_intent, {
        'friendly_name': 'HEPO Leaf Charging Intent'
    })
    print(f'✅ Leaf Intent pushed: {current_leaf_intent}')

    # Push effective cost signal
    current_effective_cost = current.get('effective_cost', 0.0)
    push_ha_state('sensor.hepo_effective_cost', f"{current_effective_cost:.4f}", {
        'friendly_name': 'HEPO Effective Cost',
        'unit_of_measurement': 'EUR/kWh',
        'icon': 'mdi:cash'
    })
    print(f'✅ Effective Cost pushed: {current_effective_cost:.4f} EUR/kWh')

    # Push low cost signal (boolean)
    low_cost_percentile = float(os.getenv('LOW_COST_PERCENTILE', '30.0'))
    effective_costs = [p.get('effective_cost', 0.0) for p in plan]
    if len(effective_costs) > 0:
        import numpy as np
        threshold = np.percentile(effective_costs, low_cost_percentile)
        low_cost_signal = 'ON' if current_effective_cost <= threshold else 'OFF'
    else:
        threshold = 0.0
        low_cost_signal = 'OFF'
    push_ha_state('sensor.hepo_low_cost_signal', low_cost_signal, {
        'friendly_name': 'HEPO Low Cost Signal',
        'icon': 'mdi:flash',
        'low_cost_threshold': round(threshold, 4) if len(effective_costs) > 0 else None,
        'percentile': low_cost_percentile
    })
    print(f'✅ Low Cost Signal pushed: {low_cost_signal} (threshold={threshold:.4f})')

    # Also push 24h usage as a standalone sensor for easier history tracking
    attributes_24h = {
        'friendly_name': 'HEPO Predicted 24h Consumption',
        'unit_of_measurement': 'kWh',
        'device_class': 'energy'
    }
    push_ha_state('sensor.hepo_predicted_24h_usage', f"{usage_24h:.2f}", attributes_24h)

    # Push current period power balance
    current_import = current.get('grid_import_kwh', 0.0)
    current_export = current.get('grid_export_kwh', 0.0)
    current_net = current_import - current_export
    push_ha_state('sensor.hepo_period_balance', f"{current_net:.3f}", {
        'friendly_name': 'HEPO Period Power Balance',
        'unit_of_measurement': 'kWh',
        'import_kwh': round(current_import, 3),
        'export_kwh': round(current_export, 3),
        'net_kw': round(current_net * 4.0, 3),  # kWh to kW for 15-min interval
    })
    print(f'✅ Period Balance pushed: net={current_net:.3f} kWh (import={current_import:.3f}, export={current_export:.3f})')

if __name__ == '__main__':
    push_plan()
    push_accuracy()
