import unittest
import numpy as np

# Adjust path for imports if necessary
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf import sequence_checks

class TestSequenceChecks(unittest.TestCase):

    def setUp(self):
        self.dt = 4e-6  # 4 us
        self.num_points = 512
        self.time_vector = np.arange(self.num_points) * self.dt

        # Sample gradient waveforms for moment checking
        self.grad_x_m0_nulled = np.sin(2 * np.pi * 1e3 * self.time_vector) # Symmetric, M0 should be near zero
        self.grad_y_m0_not_nulled = np.ones(self.num_points) * 0.01 # Unipolar, M0 non-zero
        # For M1 nulling, need something like a symmetric gradient pair
        g_bipolar_segment = np.ones(self.num_points // 2) * 0.01
        self.grad_z_m1_nulled = np.concatenate([g_bipolar_segment, -g_bipolar_segment])
        if len(self.grad_z_m1_nulled) < self.num_points: # ensure correct length if num_points is odd
            self.grad_z_m1_nulled = np.pad(self.grad_z_m1_nulled, (0, self.num_points - len(self.grad_z_m1_nulled)), 'constant')


        self.gradient_waveforms_for_moment_tests = {
            'x': self.grad_x_m0_nulled,
            'y': self.grad_y_m0_not_nulled,
            'z': self.grad_z_m1_nulled
        }

    # --- Tests for EPI Checks ---
    def test_epi_echo_train_length_basic(self):
        res_valid = sequence_checks.check_epi_echo_train_length(etl=64)
        self.assertTrue(res_valid['etl_ok'])
        self.assertEqual(res_valid['etl_value'], 64)

        res_zero = sequence_checks.check_epi_echo_train_length(etl=0)
        self.assertFalse(res_zero['etl_ok'])
        self.assertIn('ETL must be a positive integer', res_zero['details'])

        res_neg = sequence_checks.check_epi_echo_train_length(etl=-10)
        self.assertFalse(res_neg['etl_ok'])

    def test_epi_etl_with_t2_star_check(self):
        # Case 1: ETL duration OK
        res_ok = sequence_checks.check_epi_echo_train_length(
            etl=50, t2_star_ms=40, max_allowed_etl_t2_star_ratio=1.0, time_per_echo_ms=0.7 # ETL dur = 35ms <= 40ms
        )
        self.assertTrue(res_ok['etl_ok']) # Basic check still true
        self.assertTrue(res_ok['etl_duration_vs_t2star_ok'])
        self.assertAlmostEqual(res_ok['total_etl_duration_ms'], 35.0)
        self.assertAlmostEqual(res_ok['t2_star_limit_ms'], 40.0)

        # Case 2: ETL duration too long
        res_long = sequence_checks.check_epi_echo_train_length(
            etl=80, t2_star_ms=40, max_allowed_etl_t2_star_ratio=1.0, time_per_echo_ms=0.7 # ETL dur = 56ms > 40ms
        )
        self.assertTrue(res_long['etl_ok']) # Basic etl > 0 is ok
        self.assertFalse(res_long['etl_duration_vs_t2star_ok'])
        self.assertAlmostEqual(res_long['total_etl_duration_ms'], 56.0)

        # Case 3: Missing params for T2* check
        res_missing_params = sequence_checks.check_epi_echo_train_length(etl=64, t2_star_ms=50) # Missing ratio and time_per_echo
        self.assertTrue(res_missing_params['etl_ok'])
        self.assertNotIn('etl_duration_vs_t2star_ok', res_missing_params) # Check should not run

        # Case 4: Invalid params for T2* check (e.g. zero time_per_echo)
        res_invalid_params = sequence_checks.check_epi_echo_train_length(
            etl=50, t2_star_ms=40, max_allowed_etl_t2_star_ratio=1.0, time_per_echo_ms=0
        )
        self.assertTrue(res_invalid_params['etl_ok'])
        self.assertEqual(res_invalid_params.get('etl_duration_vs_t2star_check'), 'Skipped')


    def test_epi_phase_encoding_blips_placeholder(self):
        # Test the placeholder function runs and returns expected structure
        res = sequence_checks.check_epi_phase_encoding_blips()
        self.assertIn('details', res)
        self.assertIn('Placeholder', res['details'])
        self.assertTrue(res['overall_blip_consistency_ok']) # Default placeholder returns True

        dummy_blips_ky = np.array([0.001, 0.00101, 0.00099])
        res_with_data = sequence_checks.check_epi_phase_encoding_blips(ky_blip_amplitudes=dummy_blips_ky)
        self.assertTrue(res_with_data['blip_consistency_ky_ok']) # Should be consistent


    # --- Tests for Spiral Checks ---
    def test_spiral_readout_duration(self):
        res_ok = sequence_checks.check_spiral_readout_duration(actual_readout_duration_ms=8.0, max_readout_duration_ms=10.0)
        self.assertTrue(res_ok['readout_duration_ok'])

        res_too_long = sequence_checks.check_spiral_readout_duration(actual_readout_duration_ms=12.0, max_readout_duration_ms=10.0)
        self.assertFalse(res_too_long['readout_duration_ok'])

        res_no_max = sequence_checks.check_spiral_readout_duration(actual_readout_duration_ms=8.0) # No max limit
        self.assertTrue(res_no_max['readout_duration_ok'])
        self.assertIsNone(res_no_max['limit_ms'])

        res_invalid_actual = sequence_checks.check_spiral_readout_duration(actual_readout_duration_ms=0)
        self.assertFalse(res_invalid_actual['readout_duration_ok'])


    def test_spiral_gradient_moment_nulling(self):
        # Test M0 (area) for x-axis (sine wave, should be near zero over full cycle(s))
        # The self.grad_x_m0_nulled is one cycle of sine, its integral should be zero.
        res_m0_x = sequence_checks.check_spiral_gradient_moment_nulling(
            {'x': self.grad_x_m0_nulled}, self.dt, moments_to_check=(0,), moment_tolerance=1e-7
        )
        self.assertTrue(res_m0_x['overall_moments_ok'])
        self.assertTrue(res_m0_x['details']['x'][0]['is_nulled'])
        self.assertAlmostEqual(res_m0_x['details']['x'][0]['value'], 0.0, delta=1e-7)

        # Test M0 for y-axis (unipolar, should NOT be nulled)
        res_m0_y = sequence_checks.check_spiral_gradient_moment_nulling(
            {'y': self.grad_y_m0_not_nulled}, self.dt, moments_to_check=(0,), moment_tolerance=1e-7
        )
        self.assertFalse(res_m0_y['overall_moments_ok'])
        self.assertFalse(res_m0_y['details']['y'][0]['is_nulled'])
        self.assertTrue(np.abs(res_m0_y['details']['y'][0]['value']) > 1e-5) # Should be significantly non-zero

        # Test M1 for z-axis (symmetric bipolar, M1 should be near zero)
        res_m1_z = sequence_checks.check_spiral_gradient_moment_nulling(
            {'z': self.grad_z_m1_nulled}, self.dt, moments_to_check=(1,), moment_tolerance=1e-7 # Check M1 (index 1)
        )
        self.assertTrue(res_m1_z['overall_moments_ok'])
        self.assertTrue(res_m1_z['details']['z'][1]['is_nulled'])
        self.assertAlmostEqual(res_m1_z['details']['z'][1]['value'], 0.0, delta=1e-7)

        # Test multiple axes and moments
        res_multi = sequence_checks.check_spiral_gradient_moment_nulling(
            self.gradient_waveforms_for_moment_tests, self.dt, moments_to_check=(0,1), moment_tolerance=1e-7
        )
        self.assertFalse(res_multi['overall_moments_ok']) # Because M0 of Y and M1 of X/Y are likely not nulled
        self.assertTrue(res_multi['details']['x'][0]['is_nulled'])  # M0 of X (sine)
        self.assertFalse(res_multi['details']['y'][0]['is_nulled']) # M0 of Y (unipolar)
        self.assertTrue(res_multi['details']['z'][1]['is_nulled'])  # M1 of Z (bipolar)


    # --- Tests for Dispatcher ---
    def test_dispatch_epi(self):
        epi_params = {'sequence_type': 'epi_3d', 'echo_train_length': 32}
        results = sequence_checks.run_sequence_specific_checks(epi_params)
        self.assertEqual(len(results), 2) # ETL check and Blip check
        self.assertTrue(any('etl_check' in res['check_id'] for res in results))
        self.assertTrue(any('blip_check' in res['check_id'] for res in results))
        self.assertTrue(results[0]['etl_ok']) # Basic check for etl=32

    def test_dispatch_spiral(self):
        spiral_params = {'sequence_type': 'spiral_3d', 'actual_readout_duration_ms': 5.0}
        results = sequence_checks.run_sequence_specific_checks(
            spiral_params, gradient_waveforms_dict=self.gradient_waveforms_for_moment_tests, dt=self.dt
        )
        self.assertEqual(len(results), 2) # Duration check and Moment check
        self.assertTrue(any('duration_check' in res['check_id'] for res in results))
        self.assertTrue(any('moment_check' in res['check_id'] for res in results))
        self.assertTrue(results[0]['readout_duration_ok'])

    def test_dispatch_unknown_type(self):
        unknown_params = {'sequence_type': 'flash_2d'}
        results = sequence_checks.run_sequence_specific_checks(unknown_params)
        self.assertEqual(len(results), 1)
        self.assertIn('error', results[0])
        self.assertIn('Unknown or unsupported sequence_type', results[0]['error'])

    def test_dispatch_missing_data_for_checks(self):
        # Spiral moment check needs gradients and dt
        spiral_params_no_grads = {'sequence_type': 'spiral_3d', 'actual_readout_duration_ms': 5.0}
        results = sequence_checks.run_sequence_specific_checks(spiral_params_no_grads, dt=self.dt) # No gradients

        moment_check_result = next(r for r in results if "moment_check" in r['check_id'])
        self.assertFalse(moment_check_result['moments_ok']) # Or 'SKIPPED'
        self.assertIn('Gradient waveforms or dt not provided', moment_check_result['details'])


if __name__ == '__main__':
    unittest.main()
