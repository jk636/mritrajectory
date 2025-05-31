import unittest
import numpy as np
from scipy.signal import convolve # For reference if needed, though PNSModel uses it internally

# Adjust path for imports
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf.pns import PNSModel

class TestPNSModel(unittest.TestCase):

    def setUp(self):
        self.dt = 10e-6  # 10 us, different from other tests to specifically test dt influence
        self.default_thresholds = {
            'rheobase_T_per_s': 20.0,
            'chronaxie_ms': 0.36,
            'max_total_pns_normalized': 0.8
        }
        self.pns_model = PNSModel(pns_thresholds=self.default_thresholds, dt=self.dt)

        self.num_points = 200
        self.time_vector = np.arange(self.num_points) * self.dt

        # Sample slew rates
        self.sr_x = np.zeros(self.num_points)
        self.sr_x[10:20] = 100 # T/m/s pulse
        self.sr_y = np.sin(2 * np.pi * 500 * self.time_vector) * 50 # 50 T/m/s @ 500 Hz
        self.sr_z = np.ones(self.num_points) * 10 # Constant slew rate

        self.slew_rates_dict = {'x': self.sr_x, 'y': self.sr_y, 'z': self.sr_z}
        self.slew_rates_array = np.stack([self.sr_x, self.sr_y, self.sr_z], axis=-1)


    def test_01_initialization(self):
        self.assertEqual(self.pns_model.dt, self.dt)
        self.assertEqual(self.pns_model.pns_thresholds['rheobase_T_per_s'], 20.0)
        self.assertAlmostEqual(self.pns_model.chronaxie_s, 0.36e-3)

        # Test init with no thresholds (uses defaults)
        model_default_thresh = PNSModel(dt=self.dt)
        self.assertEqual(model_default_thresh.pns_thresholds['rheobase_T_per_s'], 20.0) # Default value

        with self.assertRaises(ValueError): # dt not provided
            PNSModel()

    def test_02_generate_nerve_response_function(self):
        nerve_filter = self.pns_model._generate_nerve_response_function()
        self.assertIsInstance(nerve_filter, np.ndarray)
        self.assertTrue(len(nerve_filter) > 0)

        # Filter should be positive and decaying
        self.assertTrue(np.all(nerve_filter >= 0))
        if len(nerve_filter) > 1:
            self.assertTrue(nerve_filter[0] > nerve_filter[-1] or len(nerve_filter) == 1) # decaying or single point

        # Test caching: calling again should return same object ID if dt/chronaxie unchanged
        filter_again = self.pns_model._generate_nerve_response_function()
        self.assertIs(nerve_filter, filter_again)

        # Test with very small chronaxie / dt ratio leading to short filter
        short_chronaxie_model = PNSModel(pns_thresholds={'chronaxie_ms': self.dt * 1e3 * 0.5}, dt=self.dt) # Chronaxie = 0.5 * dt
        short_filter = short_chronaxie_model._generate_nerve_response_function(filter_duration_factor=1)
        # Expect short filter, possibly 1 point if factor is small enough
        # The implementation ensures at least 1 point.
        self.assertTrue(len(short_filter) >= 1)


    def test_03_compute_pns_zero_slew(self):
        zero_sr = {'x': np.zeros(self.num_points), 'y': np.zeros(self.num_points)}
        pns_ts = self.pns_model.compute_pns(zero_sr)
        self.assertEqual(len(pns_ts), self.num_points) # mode='same' for convolution
        np.testing.assert_array_almost_equal(pns_ts, 0.0)

    def test_04_compute_pns_simple_pulse(self):
        # Simple pulse on one axis
        pulse_sr = np.zeros(self.num_points)
        pulse_sr[10:20] = self.pns_model.pns_thresholds['rheobase_T_per_s'] # Pulse height = rheobase

        pns_ts = self.pns_model.compute_pns({'x': pulse_sr})

        # The peak of PNS_normalized should be related to how much the convolved pulse
        # (filtered slew rate) matches the rheobase.
        # If the pulse is long enough (>> chronaxie), the convolved output should reach pulse_sr height.
        # (Because sum of filter coeffs * (value) approx value, if filter sum is 1).
        # So peak of pns_ts should be close to 1.0.
        # Duration of pulse: 10 * dt = 10 * 10e-6 = 100 us = 0.1 ms.
        # Chronaxie = 0.36 ms. Pulse is shorter than chronaxie.
        # So, effective dB/dt will be less than rheobase. Peak PNS should be < 1.0.
        self.assertTrue(np.max(pns_ts) < 1.0 and np.max(pns_ts) > 0)
        self.assertEqual(len(pns_ts), self.num_points)


    def test_05_compute_pns_with_weights_and_multiple_axes(self):
        axis_weights = {'x': 0.5, 'y': 1.0, 'z': 0.0} # z-axis is ignored
        pns_ts = self.pns_model.compute_pns(self.slew_rates_dict, axis_weights=axis_weights)
        self.assertEqual(len(pns_ts), self.num_points)

        # Compute PNS for x and y separately with their weights
        pns_x_only_ts = self.pns_model.compute_pns({'x': self.slew_rates_dict['x']}, axis_weights={'x':0.5})
        pns_y_only_ts = self.pns_model.compute_pns({'y': self.slew_rates_dict['y']}, axis_weights={'y':1.0})

        # Expected RSS: sqrt((0.5 * pns_norm_x_activity)^2 + (1.0 * pns_norm_y_activity)^2)
        # This requires access to per_axis_normalized from history.
        # The current returned value is the final RSS.
        # Let's check if it's greater than individual contributions (if both are non-zero)
        # And that z contribution (weight 0) is excluded.

        # Get per-axis normalized activities from history
        last_pns_computation = self.pns_model.pns_values_history[-1]
        pns_norm_x = last_pns_computation['per_axis_normalized']['x']
        pns_norm_y = last_pns_computation['per_axis_normalized']['y']
        pns_norm_z = last_pns_computation['per_axis_normalized']['z'] # Should be small if slew_z is small or zero if filter applied

        expected_rss = np.sqrt((pns_norm_x * 0.5)**2 + (pns_norm_y * 1.0)**2 + (pns_norm_z * 0.0)**2)
        np.testing.assert_array_almost_equal(pns_ts, expected_rss, decimal=5)


    def test_06_compute_pns_with_array_input(self):
        pns_ts_array_in = self.pns_model.compute_pns(self.slew_rates_array)
        pns_ts_dict_in = self.pns_model.compute_pns(self.slew_rates_dict) # Assumes generic axis names for array
        # This comparison is tricky if axis names/order differ.
        # PNSModel with array input uses generic axis_0, axis_1, ...
        # If slew_rates_dict has 'x', 'y', 'z', these match if axis_weights not given.
        # Let's recompute dict with generic names for comparison.
        generic_slew_dict = {f'axis_{i}': self.slew_rates_array[:,i] for i in range(self.slew_rates_array.shape[1])}
        pns_ts_generic_dict = self.pns_model.compute_pns(generic_slew_dict)

        np.testing.assert_array_almost_equal(pns_ts_array_in, pns_ts_generic_dict, decimal=5)


    def test_07_check_limits(self):
        # Case 1: PNS well below limit
        pns_ts_low = np.full(self.num_points, 0.1) # Flat PNS at 0.1
        is_compliant_low, peak_low = self.pns_model.check_limits(pns_ts_low)
        self.assertTrue(is_compliant_low)
        self.assertAlmostEqual(peak_low, 0.1)

        # Case 2: PNS at limit
        pns_ts_at_limit = np.full(self.num_points, self.default_thresholds['max_total_pns_normalized'])
        is_compliant_at, peak_at = self.pns_model.check_limits(pns_ts_at_limit)
        self.assertTrue(is_compliant_at)
        self.assertAlmostEqual(peak_at, self.default_thresholds['max_total_pns_normalized'])

        # Case 3: PNS above limit
        pns_ts_high = np.full(self.num_points, self.default_thresholds['max_total_pns_normalized'] + 0.1)
        is_compliant_high, peak_high = self.pns_model.check_limits(pns_ts_high)
        self.assertFalse(is_compliant_high)
        self.assertAlmostEqual(peak_high, self.default_thresholds['max_total_pns_normalized'] + 0.1)

        # Test with no timeseries arg (uses last from history)
        self.pns_model.compute_pns({'x': np.full(self.num_points, 10.0)}) # Should produce some PNS
        is_compliant_hist, _ = self.pns_model.check_limits() # Uses history
        self.assertIsInstance(is_compliant_hist, bool)

        # Test with empty timeseries
        is_compliant_empty, peak_empty = self.pns_model.check_limits(np.array([]))
        self.assertTrue(is_compliant_empty) # Empty should be compliant
        self.assertEqual(peak_empty, 0.0)


    def test_08_optimize_slew_rate_conceptual(self):
        # Make slew rates that will exceed limits
        high_slew_val = self.pns_model.pns_thresholds['rheobase_T_per_s'] * 5 # Make it high enough
        sr_exceed = {'x': np.full(self.num_points, high_slew_val)}

        # Calculate PNS to populate history for check_limits if optimize internally calls it
        # self.pns_model.compute_pns(sr_exceed)

        target_factor = 0.8 # Target 80% of the model's hard limit (max_total_pns_normalized)
        # This target_factor is applied to self.pns_thresholds['max_total_pns_normalized']
        # So, effective target limit = self.pns_thresholds['max_total_pns_normalized'] * target_factor
        # But optimize_slew_rate currently uses self.pns_thresholds['max_total_pns_normalized'] directly
        # if target_pns_limit_factor is None, or multiplies it if provided.

        opt_result = self.pns_model.optimize_slew_rate(sr_exceed, target_pns_limit_factor=target_factor)

        self.assertIn('scaled_slew_rates', opt_result)
        self.assertIn('status', opt_result)

        original_peak = opt_result['original_peak_pns']
        expected_target_limit = self.pns_model.pns_thresholds['max_total_pns_normalized'] * target_factor
        self.assertAlmostEqual(opt_result['target_pns_limit_used'], expected_target_limit)

        if original_peak > expected_target_limit :
            self.assertTrue(opt_result['scaling_factor_applied'] < 1.0)
            self.assertTrue(opt_result['scaling_factor_applied'] > 0.0) # Should not be zero unless target is zero
            # Check if new peak (if recomputed) would be around target_limit
            # For a single axis linear system like this, scaling slew by X should scale PNS by X
            self.assertAlmostEqual(original_peak * opt_result['scaling_factor_applied'], expected_target_limit, places=3)

            scaled_sr_x = opt_result['scaled_slew_rates']['x']
            np.testing.assert_array_almost_equal(scaled_sr_x, sr_exceed['x'] * opt_result['scaling_factor_applied'])
            self.assertEqual(opt_result['status'], "Exceeds Target, Scaling Suggested")
        else:
            # If it was already compliant with the tighter target_factor
            self.assertAlmostEqual(opt_result['scaling_factor_applied'], 1.0)
            self.assertTrue(opt_result['status'].startswith("Compliant with Target"))

        # Test case where it's already compliant with default limit (target_pns_limit_factor=None)
        compliant_sr = {'x': np.full(self.num_points, 1.0)} # Low slew
        opt_result_compliant = self.pns_model.optimize_slew_rate(compliant_sr) # target_pns_limit_factor is None
        self.assertAlmostEqual(opt_result_compliant['scaling_factor_applied'], 1.0)
        self.assertTrue(opt_result_compliant['status'].startswith("Compliant"))


if __name__ == '__main__':
    unittest.main()
