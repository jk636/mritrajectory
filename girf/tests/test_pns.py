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
        self.dt = 10e-6  # 10 us
        self.default_thresholds_config = {
            'default': {
                'rheobase_T_per_s': 20.0,
                'chronaxie_ms': 0.36,
                'max_total_pns_normalized': 0.8
            },
            'x': {'rheobase_T_per_s': 22.0, 'chronaxie_ms': 0.30}, # X-axis specific
            'y': {'chronaxie_ms': 0.40} # Y-axis uses default rheobase
            # Z-axis will use all default values
        }
        self.pns_model = PNSModel(pns_thresholds_config=self.default_thresholds_config, dt=self.dt)

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
        self.assertEqual(self.pns_model.pns_thresholds['default']['rheobase_T_per_s'], 20.0)
        self.assertEqual(self.pns_model.pns_thresholds['x']['rheobase_T_per_s'], 22.0)
        self.assertAlmostEqual(self.pns_model._get_axis_param('x', 'chronaxie_ms') / 1000.0, 0.30e-3)
        self.assertAlmostEqual(self.pns_model._get_axis_param('y', 'chronaxie_ms') / 1000.0, 0.40e-3)
        self.assertAlmostEqual(self.pns_model._get_axis_param('z', 'chronaxie_ms') / 1000.0, 0.36e-3) # Falls back to default

        # Test init with no thresholds config (uses internal defaults)
        model_no_config = PNSModel(dt=self.dt)
        self.assertEqual(model_no_config.pns_thresholds['default']['rheobase_T_per_s'], 20.0)

        with self.assertRaises(ValueError): # dt not provided
            PNSModel()

    def test_02_generate_nerve_response_function(self):
        # Test for default axis
        filter_default = self.pns_model._generate_nerve_response_function(axis_name='default')
        self.assertIsInstance(filter_default, np.ndarray)
        self.assertTrue(len(filter_default) > 0)
        # Check if uses default chronaxie: 0.36ms
        # Expected filter length: 5 * 0.36e-3 / 10e-6 = 5 * 36 = 180
        chronaxie_s_default = self.pns_model.pns_thresholds['default']['chronaxie_ms'] / 1000.0
        expected_len_default = int(5 * chronaxie_s_default / self.dt)
        self.assertEqual(len(filter_default), expected_len_default)


        # Test for x-axis (specific chronaxie: 0.30ms)
        filter_x = self.pns_model._generate_nerve_response_function(axis_name='x')
        chronaxie_s_x = self.pns_model._get_axis_param('x', 'chronaxie_ms') / 1000.0
        expected_len_x = int(5 * chronaxie_s_x / self.dt) # 5 * 0.30e-3 / 10e-6 = 5 * 30 = 150
        self.assertEqual(len(filter_x), expected_len_x)
        self.assertNotEqual(len(filter_x), len(filter_default)) # Should be different due to different chronaxie

        # Test caching
        filter_x_again = self.pns_model._generate_nerve_response_function(axis_name='x')
        self.assertIs(filter_x, filter_x_again)

        filter_default_again = self.pns_model._generate_nerve_response_function() # axis_name='default'
        self.assertIs(filter_default, filter_default_again)


    def test_03_compute_pns_zero_slew(self):
        zero_sr = {'x': np.zeros(self.num_points), 'y': np.zeros(self.num_points)}
        pns_ts = self.pns_model.compute_pns(zero_sr)
        self.assertEqual(len(pns_ts), self.num_points) # mode='same' for convolution
        np.testing.assert_array_almost_equal(pns_ts, 0.0)

    def test_04_compute_pns_simple_pulse(self):
        # Simple pulse on one axis
        pulse_sr_val = self.pns_model._get_axis_param('x', 'rheobase_T_per_s') # Pulse height = x-axis rheobase
        pulse_sr_x = np.zeros(self.num_points)
        pulse_sr_x[10:20] = pulse_sr_val

        pns_ts = self.pns_model.compute_pns({'x': pulse_sr_x})

        # Chronaxie for x is 0.30ms. Pulse duration 0.1ms. Shorter than chronaxie.
        # So, effective dB/dt will be less than rheobase. Peak PNS (normalized by x-rheobase) should be < 1.0.
        peak_val = np.max(pns_ts) if pns_ts.size > 0 else 0
        self.assertTrue(peak_val < 1.0 and peak_val > 0, f"Peak PNS was {peak_val}, expected < 1.0 and > 0.")
        self.assertEqual(len(pns_ts), self.num_points)


    def test_05_compute_pns_with_weights_and_multiple_axes_specific_thresholds(self):
        axis_weights = {'x': 0.5, 'y': 1.0, 'z': 0.0} # z-axis is ignored
        pns_ts = self.pns_model.compute_pns(self.slew_rates_dict, axis_weights=axis_weights)
        self.assertEqual(len(pns_ts), self.num_points)

        # Compute PNS for x and y separately with their weights and specific thresholds
        # This requires getting the per-axis normalized values from the model's history
        # after a full compute_pns call that includes all axes.

        # Run compute_pns for all axes first
        _ = self.pns_model.compute_pns(self.slew_rates_dict, axis_weights=axis_weights)

        # Get per-axis normalized activities from history (last entry)
        # Note: self.pns_model.pns_values_history[-1] contains the results of the compute_pns call above
        # which already applied the axis_weights in its internal RSS calculation.
        # To verify, we need the unweighted per-axis normalized activities if we want to apply weights here.
        # The stored 'per_axis_normalized' is before weighting for RSS.

        last_pns_calc = self.pns_model.pns_values_history[-1]
        pns_norm_x_unweighted = last_pns_calc['per_axis_normalized']['x']
        pns_norm_y_unweighted = last_pns_calc['per_axis_normalized']['y']
        pns_norm_z_unweighted = last_pns_calc['per_axis_normalized']['z']

        # Apply weights manually
        weighted_pns_x = pns_norm_x_unweighted * axis_weights.get('x', 1.0)
        weighted_pns_y = pns_norm_y_unweighted * axis_weights.get('y', 1.0)
        weighted_pns_z = pns_norm_z_unweighted * axis_weights.get('z', 1.0)

        expected_rss_ts = np.sqrt(weighted_pns_x**2 + weighted_pns_y**2 + weighted_pns_z**2)
        np.testing.assert_array_almost_equal(pns_ts, expected_rss_ts, decimal=5)


    def test_06_compute_pns_with_array_input(self):
        # PNSModel uses utils.standardize_trajectory_format which defaults to x,y,z for 3 dims
        pns_ts_array_in = self.pns_model.compute_pns(self.slew_rates_array)

        # For comparison, create a dict with x,y,z keys from the array
        slew_dict_xyz = {
            'x': self.slew_rates_array[:,0],
            'y': self.slew_rates_array[:,1],
            'z': self.slew_rates_array[:,2]
        }
        pns_ts_dict_xyz = self.pns_model.compute_pns(slew_dict_xyz)
        np.testing.assert_array_almost_equal(pns_ts_array_in, pns_ts_dict_xyz, decimal=5)


    def test_07_check_limits(self):
        default_limit = self.pns_model.pns_thresholds['default']['max_total_pns_normalized']

        pns_ts_low = np.full(self.num_points, default_limit * 0.5)
        is_compliant_low, peak_low = self.pns_model.check_limits(pns_ts_low)
        self.assertTrue(is_compliant_low)
        self.assertAlmostEqual(peak_low, default_limit * 0.5)

        pns_ts_at_limit = np.full(self.num_points, default_limit)
        is_compliant_at, peak_at = self.pns_model.check_limits(pns_ts_at_limit)
        self.assertTrue(is_compliant_at)
        self.assertAlmostEqual(peak_at, default_limit)

        pns_ts_high = np.full(self.num_points, default_limit * 1.1)
        is_compliant_high, peak_high = self.pns_model.check_limits(pns_ts_high)
        self.assertFalse(is_compliant_high)
        self.assertAlmostEqual(peak_high, default_limit * 1.1)

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
