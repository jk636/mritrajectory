import unittest
import numpy as np

# Adjust path for imports
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf import timing # Assuming girf package is accessible

class TestTimingChecks(unittest.TestCase):

    def setUp(self):
        self.dt = 1e-5  # 10 us, for easier time-to-index conversion in tests
        self.num_points = 200
        self.time_vector = np.arange(self.num_points) * self.dt

        # Sample gradient waveform for X-axis
        self.grad_x = np.zeros(self.num_points)
        # Plateau 1: 0.01 T/m from t=0.5ms (idx=50) to t=1.0ms (idx=100, exclusive end)
        self.plateau1_amp = 0.01
        self.plateau1_start_idx = 50
        self.plateau1_end_idx = 100
        self.grad_x[self.plateau1_start_idx : self.plateau1_end_idx] = self.plateau1_amp

        # Plateau 2: 0.02 T/m from t=1.2ms (idx=120) to t=1.5ms (idx=150, exclusive end)
        self.plateau2_amp = 0.02
        self.plateau2_start_idx = 120
        self.plateau2_end_idx = 150
        self.grad_x[self.plateau2_start_idx : self.plateau2_end_idx] = self.plateau2_amp

        self.gradient_waveforms_dict = {'x': self.grad_x, 'y': np.zeros(self.num_points)}

        # Sample actual event data map
        self.actual_event_data = {
            'rf1': {'start_time_s': 0.000495, 'duration_s': 0.0002}, # Starts slightly before grad plateau1
            'grad_x_p1': {'start_time_s': 0.000500, 'duration_s': 0.000500}, # Corresponds to plateau1
            'adc1': {'start_time_s': 0.000502, 'duration_s': 0.000400}
        }

    # --- Tests for check_gradient_event_timing ---
    def test_plateau_amplitude_ok(self):
        event_def = {
            'id': 'test_p1_ok', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': self.plateau1_start_idx * self.dt,
            'expected_duration_s': (self.plateau1_end_idx - self.plateau1_start_idx) * self.dt,
            'expected_amplitude_T_per_m': self.plateau1_amp,
            'amplitude_tolerance_percent': 5.0
        }
        result = timing.check_gradient_event_timing(self.grad_x, self.dt, event_def)
        self.assertTrue(result['amplitude_ok'])
        self.assertAlmostEqual(result['measured_avg_amplitude_T_per_m'], self.plateau1_amp)

    def test_plateau_amplitude_too_low(self):
        event_def = {
            'id': 'test_p1_low', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': self.plateau1_start_idx * self.dt,
            'expected_duration_s': (self.plateau1_end_idx - self.plateau1_start_idx) * self.dt,
            'expected_amplitude_T_per_m': self.plateau1_amp * 1.2, # Expect 20% higher
            'amplitude_tolerance_percent': 5.0
        }
        result = timing.check_gradient_event_timing(self.grad_x, self.dt, event_def)
        self.assertFalse(result['amplitude_ok'])

    def test_plateau_amplitude_too_high(self):
        event_def = {
            'id': 'test_p1_high', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': self.plateau1_start_idx * self.dt,
            'expected_duration_s': (self.plateau1_end_idx - self.plateau1_start_idx) * self.dt,
            'expected_amplitude_T_per_m': self.plateau1_amp * 0.8, # Expect 20% lower
            'amplitude_tolerance_percent': 5.0
        }
        result = timing.check_gradient_event_timing(self.grad_x, self.dt, event_def)
        self.assertFalse(result['amplitude_ok'])

    def test_plateau_timing_window_edge_cases(self):
        # Window at the beginning
        event_def_start = {
            'id': 'test_start_window', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': 0,
            'expected_duration_s': 10 * self.dt, # First 10 points (are zero)
            'expected_amplitude_T_per_m': 0.0,
            'amplitude_tolerance_percent': 1.0
        }
        result_start = timing.check_gradient_event_timing(self.grad_x, self.dt, event_def_start)
        self.assertTrue(result_start['amplitude_ok'])
        self.assertAlmostEqual(result_start['measured_avg_amplitude_T_per_m'], 0.0)

        # Window out of bounds (too late)
        event_def_late = {
            'id': 'test_late_window', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': (self.num_points - 5) * self.dt,
            'expected_duration_s': 10 * self.dt, # Window extends beyond waveform
            'expected_amplitude_T_per_m': 0.0,
            'amplitude_tolerance_percent': 1.0
        }
        result_late = timing.check_gradient_event_timing(self.grad_x, self.dt, event_def_late)
        self.assertFalse(result_late['amplitude_ok']) # Fails due to out of bounds
        self.assertTrue(np.isnan(result_late['measured_avg_amplitude_T_per_m']))
        self.assertIn("out of waveform bounds", result_late['details'])


    def test_plateau_event_not_a_plateau(self):
        # Check a ramp segment
        ramp_grad = np.linspace(0, 0.01, self.num_points)
        event_def_ramp = {
            'id': 'test_ramp', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': (self.num_points // 4) * self.dt,
            'expected_duration_s': (self.num_points // 2) * self.dt,
            'expected_amplitude_T_per_m': 0.005, # Mid-point of the ramp
            'amplitude_tolerance_percent': 1.0 # Very tight tolerance
        }
        result_ramp = timing.check_gradient_event_timing(ramp_grad, self.dt, event_def_ramp)
        # Average might be close, but it's not a plateau. Current check only looks at average.
        # This test will pass if avg is close, demonstrating limitation of current simple check.
        self.assertTrue(result_ramp['amplitude_ok'])
        self.assertAlmostEqual(result_ramp['measured_avg_amplitude_T_per_m'], 0.005, places=3)


    # --- Tests for check_event_synchronization ---
    def test_events_synchronized(self):
        res = timing.check_event_synchronization(0.1, 0.1000005, 1e-6) # 0.5us diff, 1us limit
        self.assertTrue(res['is_synchronized'])
        self.assertAlmostEqual(res['actual_delay_s'], -0.5e-6)

    def test_events_not_synchronized(self):
        res = timing.check_event_synchronization(0.1, 0.100002, 1e-6) # 2us diff, 1us limit
        self.assertFalse(res['is_synchronized'])
        self.assertAlmostEqual(res['actual_delay_s'], -2e-6)

    def test_synchronization_exact_match(self):
        res = timing.check_event_synchronization(0.1, 0.1, 1e-6)
        self.assertTrue(res['is_synchronized'])
        self.assertAlmostEqual(res['actual_delay_s'], 0.0)

    def test_synchronization_at_boundary(self):
        res = timing.check_event_synchronization(0.1, 0.100001, 1e-6) # 1us diff, 1us limit
        self.assertTrue(res['is_synchronized'])
        self.assertAlmostEqual(res['actual_delay_s'], -1e-6)

        res_outside = timing.check_event_synchronization(0.1, 0.100001001, 1e-6) # Just outside
        self.assertFalse(res_outside['is_synchronized'])


    # --- Tests for process_timing_checks ---
    def test_process_gradient_plateau_check_pass_and_fail(self):
        nominal_events = [
            {'id': 'p1_ok', 'type': 'gradient_plateau_check', 'axis': 'x',
             'expected_start_time_s': self.plateau1_start_idx * self.dt,
             'expected_duration_s': (self.plateau1_end_idx - self.plateau1_start_idx) * self.dt,
             'expected_amplitude_T_per_m': self.plateau1_amp, 'amplitude_tolerance_percent': 5.0},
            {'id': 'p2_fail_amp', 'type': 'gradient_plateau_check', 'axis': 'x',
             'expected_start_time_s': self.plateau2_start_idx * self.dt,
             'expected_duration_s': (self.plateau2_end_idx - self.plateau2_start_idx) * self.dt,
             'expected_amplitude_T_per_m': self.plateau2_amp * 1.5, 'amplitude_tolerance_percent': 5.0}
        ]
        results = timing.process_timing_checks(nominal_events,
                                               gradient_waveforms_dict=self.gradient_waveforms_dict,
                                               dt=self.dt)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]['amplitude_ok'])
        self.assertEqual(results[0]['event_id'], 'p1_ok')
        self.assertFalse(results[1]['amplitude_ok'])
        self.assertEqual(results[1]['event_id'], 'p2_fail_amp')

    def test_process_synchronization_check_pass_and_fail(self):
        nominal_events = [
            {'id': 'sync1_ok', 'type': 'synchronization_check',
             'event_A_id': 'rf1', 'event_A_timepoint': 'start_time_s',
             'event_B_id': 'grad_x_p1', 'event_B_timepoint': 'start_time_s',
             'max_allowed_delay_s': 10e-6}, # rf1=0.495ms, grad_x_p1=0.500ms. Diff=5us. OK.
            {'id': 'sync2_fail', 'type': 'synchronization_check',
             'event_A_id': 'rf1', 'event_A_timepoint': 'start_time_s',
             'event_B_id': 'adc1', 'event_B_timepoint': 'start_time_s',
             'max_allowed_delay_s': 1e-6} # rf1=0.495ms, adc1=0.502ms. Diff=7us. Fail.
        ]
        results = timing.process_timing_checks(nominal_events, actual_event_data_map=self.actual_event_data)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]['is_synchronized'])
        self.assertFalse(results[1]['is_synchronized'])

    def test_process_mixed_event_types(self):
        nominal_events = [
            {'id': 'p1_mix_ok', 'type': 'gradient_plateau_check', 'axis': 'x',
             'expected_start_time_s': self.plateau1_start_idx * self.dt,
             'expected_duration_s': (self.plateau1_end_idx - self.plateau1_start_idx) * self.dt,
             'expected_amplitude_T_per_m': self.plateau1_amp, 'amplitude_tolerance_percent': 5.0},
            {'id': 'sync1_mix_ok', 'type': 'synchronization_check',
             'event_A_id': 'rf1', 'event_A_timepoint': 'start_time_s',
             'event_B_id': 'grad_x_p1', 'event_B_timepoint': 'start_time_s',
             'max_allowed_delay_s': 10e-6}
        ]
        results = timing.process_timing_checks(nominal_events,
                                               actual_event_data_map=self.actual_event_data,
                                               gradient_waveforms_dict=self.gradient_waveforms_dict,
                                               dt=self.dt)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]['amplitude_ok'])
        self.assertTrue(results[1]['is_synchronized'])

    def test_process_missing_data_gradient_check(self):
        nominal_events = [{'id': 'p1_no_grad_data', 'type': 'gradient_plateau_check', 'axis': 'z', # Axis z not in dict
                           'expected_start_time_s': 0, 'expected_duration_s': 0.001,
                           'expected_amplitude_T_per_m': 0.01, 'amplitude_tolerance_percent': 5.0}]
        results = timing.process_timing_checks(nominal_events,
                                               gradient_waveforms_dict=self.gradient_waveforms_dict, dt=self.dt)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'SKIPPED')
        self.assertIn("waveform for axis 'z' not provided", results[0]['reason'])

        results_no_dt = timing.process_timing_checks(nominal_events,
                                                 gradient_waveforms_dict=self.gradient_waveforms_dict, dt=None) # No dt
        self.assertEqual(results_no_dt[0]['status'], 'SKIPPED')
        self.assertIn("'dt' not provided", results_no_dt[0]['reason'])


    def test_process_missing_data_sync_check(self):
        nominal_events = [{'id': 'sync_no_actual', 'type': 'synchronization_check',
                           'event_A_id': 'evt_missing', 'event_A_timepoint': 'start_time_s',
                           'event_B_id': 'rf1', 'event_B_timepoint': 'start_time_s',
                           'max_allowed_delay_s': 1e-6}]
        results = timing.process_timing_checks(nominal_events, actual_event_data_map=self.actual_event_data)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['status'], 'SKIPPED')
        self.assertIn("Actual data for 'start_time_s' of event 'evt_missing' not found", results[0]['reason'])


if __name__ == '__main__':
    unittest.main()
