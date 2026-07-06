# Plan: Add Period Power Balance Sensor to run_frequent

## Objective
Make `run_frequent.sh` (via `push_to_ha.py`) publish a new Home Assistant sensor with the current measuring period's power balance, including import and export as attributes.

## Background
- `run_frequent.sh` runs every 15 minutes and calls `push_to_ha.py` at step 4.
- `push_to_ha.py` already pushes several sensors from the optimization plan: effective cost, GSHP intent, Leaf intent, low cost signal, etc.
- The optimization plan (`state/optimization_plan.json`) contains per-interval fields: `grid_import_kwh`, `grid_export_kwh`, `battery_action`, `soc_pct`, etc.
- `run_often.py` (every 20s) computes actual interval import/export from cumulative energy meters, but does not push to HA.

## Open Clarification
**Question for user:** Should the sensor show:
1. **Planned values** from the optimization plan (what the optimizer expects for the current 15-minute period), or
2. **Actual measured values** from cumulative energy meters (what really happened in the just-completed 15-minute period)?

The simpler approach is planned values from the plan, since `push_to_ha.py` already has that data. Actual measured values would require reading HA cumulative sensors inside `push_to_ha.py` and computing deltas against `state/net_metering_state.json`.

## Proposed Implementation (Planned Values)

### 1. Modify `push_to_ha.py`
In `push_plan()`, after pushing the low cost signal, add a new sensor push:

```python
    # Push current period power balance
    current_import = current.get('grid_import_kwh', 0.0)
    current_export = current.get('grid_export_kwh', 0.0)
    current_net = current_import - current_export
    push_ha_state('sensor.hepo_period_balance', f"{current_net:.3f}", {
        'friendly_name': 'HEPO Period Power Balance',
        'unit_of_measurement': 'kWh',
        'import_kwh': round(current_import, 3),
        'export_kwh': round(current_export, 3),
        'net_kw': round(current_net * 4.0, 3),  # 15-min interval -> kW
    })
    print(f'✅ Period Balance pushed: net={current_net:.3f} kWh (import={current_import:.3f}, export={current_export:.3f})')
```

### 2. Add test in `tests/test_push_to_ha.py`
Add `test_push_period_balance` to `TestPushPlan`:

```python
    @patch('push_to_ha.push_ha_state')
    @patch('utils.battery_utils.is_battery_available')
    @patch('utils.battery_utils.call_ha_service')
    def test_push_period_balance(self, mock_service, mock_battery_available, mock_push):
        """Test that period power balance sensor is pushed with import/export attributes."""
        mock_battery_available.return_value = True
        mock_service.return_value = {}
        
        plan_data = [
            {
                'predicted_usage_kwh': 0.5,
                'effective_cost': 0.08,
                'gshp_intent': 'STOP',
                'leaf_intent': 'OFF',
                'battery_power_kw': 0.0,
                'battery_action': 'follow',
                'soc_pct': 50.0,
                'grid_import_kwh': 2.5,
                'grid_export_kwh': 0.8,
            }
        ]
        plan_data.extend([plan_data[0].copy() for _ in range(95)])
        
        with open(self.plan_file, 'w') as f:
            json.dump(plan_data, f)
        
        push_plan()
        
        balance_call = None
        for call in mock_push.call_args_list:
            args, kwargs = call
            if args[0] == 'sensor.hepo_period_balance':
                balance_call = call
                break
        
        self.assertIsNotNone(balance_call, "period balance sensor push not found")
        self.assertEqual(balance_call[0][1], '1.700')
        self.assertEqual(balance_call[0][2]['import_kwh'], 2.5)
        self.assertEqual(balance_call[0][2]['export_kwh'], 0.8)
        self.assertEqual(balance_call[0][2]['net_kw'], 6.8)
```

### 3. Version bump
Per `AGENTS.md`, this is an output/publishing addition, not a model training or inference logic change. **No version bump needed.**

## Verification
- Run `venv/bin/python3 -m pytest tests/test_push_to_ha.py -k 'not slow'`
- Confirm the new test passes.
