
import unittest
from datetime import datetime, timedelta, timezone
import pandas as pd
from predict_future import generate_inference_data

class TestPredictFuture(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        self.start_time = self.now + timedelta(minutes=15)
        self.end_time = self.now + timedelta(hours=2)
        self.interval = 15
        
        # Mock solar data
        solar_indices = [self.now + timedelta(minutes=15*i) for i in range(20)]
        self.df_solar = pd.DataFrame({'pv_estimate': [1.0] * 20}, index=solar_indices)
        
        # Mock current states
        self.current_states = {'temp_val': 10.0, 'acc_val': 45.0, 'soc_val': 80.0}
        self.sauna_states = {'is_sauna_detected': False, 'was_warm_yesterday': False, 'now': self.now}

    def test_forecast_mapping(self):
        # Create a weather forecast with a different temperature
        forecast_ts = self.now + timedelta(hours=1)
        df_weather = pd.DataFrame({'temperature': [20.0]}, index=[forecast_ts])
        
        inference_data, _ = generate_inference_data(
            self.start_time, self.end_time, self.interval, 
            self.df_solar, df_weather, self.current_states, self.sauna_states
        )
        
        # At 1h from now (index 3 if interval is 15m), it should pick up the 20.0C forecast
        # Index 0: now+15, Index 1: now+30, Index 2: now+45, Index 3: now+60
        self.assertEqual(inference_data[3]['outside_temp'], 20.0)
        # At index 0, it should use the fallback (10.0) because now+15 is closer to nothing or too far from 1h? 
        # Wait, index_get_indexer with nearest will pick it if it's the closest.
        # now+15 is 45 mins from now+60. now+0 is 15 mins from now+15. 
        # But we didn't put now+0 in the df_weather.
        
    def test_fallback_when_no_forecast(self):
        df_weather = pd.DataFrame() # Empty
        
        inference_data, _ = generate_inference_data(
            self.start_time, self.end_time, self.interval, 
            self.df_solar, df_weather, self.current_states, self.sauna_states
        )
        
        # Should all be the fallback temperature
        for row in inference_data:
            self.assertEqual(row['outside_temp'], 10.0)

    def test_proximity_limit(self):
        # Current time is 'now'. 
        # Forecast is at 'now + 0h'.
        df_weather = pd.DataFrame({'temperature': [20.0]}, index=[self.now])
        
        # We predict for 'now + 3h'
        start_time = self.now + timedelta(hours=3)
        end_time = self.now + timedelta(hours=3)
        
        inference_data, _ = generate_inference_data(
            start_time, end_time, self.interval, 
            self.df_solar, df_weather, self.current_states, self.sauna_states
        )
        
        # Point at +3h is 3 hours away from nearest forecast (at +0h).
        # Should NOT use 20.0, should fallback to 10.0.
        self.assertEqual(inference_data[0]['outside_temp'], 10.0)

    def test_ev_position_heuristic(self):
        # Test for a workday at 10:00 (car should be away)
        workday_morning = datetime(2024, 5, 22, 10, 0, tzinfo=timezone.utc) # Wednesday
        start_time = workday_morning
        end_time = workday_morning
        
        inference_data, _ = generate_inference_data(
            start_time, end_time, self.interval, 
            self.df_solar, pd.DataFrame(), self.current_states, self.sauna_states
        )
        
        # Currently it's always 1, this should fail if we want it to be 0
        self.assertEqual(inference_data[0]['ev_position'], 0, "Car should be away on a workday morning")

        # Test for a workday at 23:00 (car should be home)
        workday_night = datetime(2024, 5, 22, 23, 0, tzinfo=timezone.utc)
        inference_data_night, _ = generate_inference_data(
            workday_night, workday_night, self.interval, 
            self.df_solar, pd.DataFrame(), self.current_states, self.sauna_states
        )
        self.assertEqual(inference_data_night[0]['ev_position'], 1, "Car should be home at night")

        # Test for a weekend at 10:00 (car should be home)
        weekend_morning = datetime(2024, 5, 25, 10, 0, tzinfo=timezone.utc) # Saturday
        inference_data_weekend, _ = generate_inference_data(
            weekend_morning, weekend_morning, self.interval, 
            self.df_solar, pd.DataFrame(), self.current_states, self.sauna_states
        )
        self.assertEqual(inference_data_weekend[0]['ev_position'], 1, "Car should be home on weekend morning")

    def test_ev_position_near_term_override(self):
        # Current time is 'now'. Car is AWAY currently.
        self.current_states['ev_pos_val'] = 0
        
        # Predicting for 'now + 1h' (should be away due to override)
        start_time = self.now + timedelta(hours=1)
        
        # Make sure 'now' is a weekend night so heuristic would say HOME
        self.sauna_states['now'] = datetime(2024, 5, 25, 23, 0, tzinfo=timezone.utc)
        start_time = self.sauna_states['now'] + timedelta(hours=1)
        
        inference_data, _ = generate_inference_data(
            start_time, start_time, self.interval, 
            self.df_solar, pd.DataFrame(), self.current_states, self.sauna_states
        )
        self.assertEqual(inference_data[0]['ev_position'], 0, "Car should be away due to near-term override")

        # Predicting for 'now + 3h' (should follow heuristic -> HOME)
        start_time_3h = self.sauna_states['now'] + timedelta(hours=3)
        inference_data_3h, _ = generate_inference_data(
            start_time_3h, start_time_3h, self.interval, 
            self.df_solar, pd.DataFrame(), self.current_states, self.sauna_states
        )
        self.assertEqual(inference_data_3h[0]['ev_position'], 1, "Car should be home after 2 hours (heuristic)")

if __name__ == '__main__':
    unittest.main()
