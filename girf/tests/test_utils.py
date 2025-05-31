import unittest
import numpy as np
from scipy.interpolate import interp1d # For checking resample_waveform internals if needed
from scipy.signal import convolve as scipy_convolve # For checking convolve_signals

# Assuming 'girf' package is in PYTHONPATH or installed.
# If running tests from parent directory of 'girf':
# from girf import utils
# Or, adjust path:
import sys
import os
# Add the parent directory of 'girf' to sys.path to find the girf package
# This assumes the test script is run from a location where 'girf' is a subdirectory
# or 'girf' is in the PYTHONPATH.
# For robust testing, a proper test runner setup (e.g. using python -m unittest discover)
# from the project root is better.
# If this script is girf/tests/test_utils.py, then parent of girf is two levels up.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf import utils # This should now work if path is set up
from girf.utils import DEFAULT_GAMMA_PROTON # Import constants if needed

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.dt = 4e-6  # 4 us
        self.gamma = DEFAULT_GAMMA_PROTON
        self.test_k_dict = {'x': np.array([0., 10., 20., 15.]),
                            'y': np.array([0., 0., 5., 5.])} # m^-1
        self.test_k_array = utils.standardize_trajectory_format(self.test_k_dict, target_format='array')
        # Expected gradients for test_k_dict (g[0]=k[0]/(dt*g), g[i]=(k[i]-k[i-1])/(dt*g))
        # Gx[0] = 0 / (dt*g) = 0
        # Gx[1] = (10-0)/(dt*g) = 10/(dt*g)
        # Gx[2] = (20-10)/(dt*g) = 10/(dt*g)
        # Gx[3] = (15-20)/(dt*g) = -5/(dt*g)
        # Gy[0] = 0
        # Gy[1] = (0-0)/(dt*g) = 0
        # Gy[2] = (5-0)/(dt*g) = 5/(dt*g)
        # Gy[3] = (5-5)/(dt*g) = 0
        self.expected_g_x = np.array([0., 10., 10., -5.]) / (self.dt * self.gamma)
        self.expected_g_y = np.array([0., 0., 5., 0.]) / (self.dt * self.gamma)
        self.test_g_dict = {'x': self.expected_g_x, 'y': self.expected_g_y}
        self.test_g_array = utils.standardize_trajectory_format(self.test_g_dict, target_format='array')

    def test_standardize_trajectory_format(self):
        # Dict to Array
        array_from_dict = utils.standardize_trajectory_format(self.test_k_dict, target_format='array')
        self.assertIsInstance(array_from_dict, np.ndarray)
        self.assertEqual(array_from_dict.shape, (4, 2))
        np.testing.assert_array_almost_equal(array_from_dict[:, 0], self.test_k_dict['x'])

        # Array to Dict
        dict_from_array = utils.standardize_trajectory_format(self.test_k_array, target_format='dict', default_axis_names=['x', 'y'])
        self.assertIsInstance(dict_from_array, dict)
        self.assertIn('x', dict_from_array)
        np.testing.assert_array_almost_equal(dict_from_array['y'], self.test_k_array[:, 1])

        # Single axis array to dict
        single_axis_arr = np.array([1,2,3,4])[:,np.newaxis]
        single_axis_dict = utils.standardize_trajectory_format(single_axis_arr, target_format='dict')
        self.assertIn('x', single_axis_dict) # Default name for single axis
        self.assertEqual(len(single_axis_dict['x']), 4)

        # Error cases
        with self.assertRaises(ValueError): # Inconsistent lengths in dict
            utils.standardize_trajectory_format({'x': np.array([1,2]), 'y': np.array([1,2,3])}, target_format='array')
        with self.assertRaises(ValueError): # Wrong num_spatial_dims for dict
            utils.standardize_trajectory_format(self.test_k_dict, num_spatial_dims=3, target_format='array')
        with self.assertRaises(ValueError): # Wrong num_spatial_dims for array
            utils.standardize_trajectory_format(self.test_k_array, num_spatial_dims=1, target_format='dict')


    def test_compute_gradient_waveforms(self):
        # Test with dict input, dict output
        g_dict_out = utils.compute_gradient_waveforms(self.test_k_dict, self.gamma, self.dt, output_format='dict')
        self.assertIn('x', g_dict_out)
        np.testing.assert_array_almost_equal(g_dict_out['x'], self.expected_g_x)
        np.testing.assert_array_almost_equal(g_dict_out['y'], self.expected_g_y)

        # Test with array input, array output
        g_array_out = utils.compute_gradient_waveforms(self.test_k_array, self.gamma, self.dt, output_format='array')
        np.testing.assert_array_almost_equal(g_array_out[:,0], self.expected_g_x)
        np.testing.assert_array_almost_equal(g_array_out[:,1], self.expected_g_y)

        with self.assertRaises(ValueError):
            utils.compute_gradient_waveforms(self.test_k_array, self.gamma, 0, output_format='array') # dt=0

    def test_compute_slew_rates(self):
        # Expected slew rates for self.test_g_dict
        # SRx[0] = Gx[0]/dt
        # SRx[1] = (Gx[1]-Gx[0])/dt etc.
        expected_sr_x = np.array([self.expected_g_x[0],
                                  self.expected_g_x[1]-self.expected_g_x[0],
                                  self.expected_g_x[2]-self.expected_g_x[1],
                                  self.expected_g_x[3]-self.expected_g_x[2]]) / self.dt
        expected_sr_y = np.array([self.expected_g_y[0],
                                  self.expected_g_y[1]-self.expected_g_y[0],
                                  self.expected_g_y[2]-self.expected_g_y[1],
                                  self.expected_g_y[3]-self.expected_g_y[2]]) / self.dt

        sr_dict_out = utils.compute_slew_rates(self.test_g_dict, self.dt, output_format='dict')
        np.testing.assert_array_almost_equal(sr_dict_out['x'], expected_sr_x)
        np.testing.assert_array_almost_equal(sr_dict_out['y'], expected_sr_y)

        sr_array_out = utils.compute_slew_rates(self.test_g_array, self.dt, output_format='array')
        np.testing.assert_array_almost_equal(sr_array_out[:,0], expected_sr_x)
        np.testing.assert_array_almost_equal(sr_array_out[:,1], expected_sr_y)

    def test_integrate_trajectory(self):
        # Test if integrating gradients gives back original k-space (with initial_k0=0)
        k_recon_dict = utils.integrate_trajectory(self.test_g_dict, self.gamma, self.dt,
                                                  initial_k0={'x':0., 'y':0.}, output_format='dict')
        np.testing.assert_array_almost_equal(k_recon_dict['x'], self.test_k_dict['x'], decimal=5) # Precision issues with float
        np.testing.assert_array_almost_equal(k_recon_dict['y'], self.test_k_dict['y'], decimal=5)

        # Test with non-zero initial_k0
        initial_offset = {'x': 1.0, 'y': -1.0}
        k_recon_offset = utils.integrate_trajectory(self.test_g_dict, self.gamma, self.dt,
                                                    initial_k0=initial_offset, output_format='dict')
        expected_k_x_offset = self.test_k_dict['x'] + initial_offset['x']
        expected_k_y_offset = self.test_k_dict['y'] + initial_offset['y']
        np.testing.assert_array_almost_equal(k_recon_offset['x'], expected_k_x_offset, decimal=5)
        np.testing.assert_array_almost_equal(k_recon_offset['y'], expected_k_y_offset, decimal=5)

        # Test integration of zero gradients
        zero_grads = {'x': np.zeros(4), 'y': np.zeros(4)}
        k_from_zero_grads = utils.integrate_trajectory(zero_grads, self.gamma, self.dt, initial_k0=0.0)
        np.testing.assert_array_almost_equal(k_from_zero_grads['x'], np.zeros(4))


    def test_check_gradient_strength(self):
        g_ok, max_g, _ = utils.check_gradient_strength(self.test_g_dict, gmax_T_per_m=np.max(np.abs(self.test_g_array)) + 0.001)
        self.assertTrue(g_ok)

        g_fail, max_g_f, details_f = utils.check_gradient_strength(self.test_g_dict, gmax_T_per_m=np.max(np.abs(self.test_g_array)) * 0.5)
        self.assertFalse(g_fail)
        self.assertIn('first_exceeding_value_T_per_m', details_f)
        self.assertAlmostEqual(max_g_f, details_f['first_exceeding_value_T_per_m'])

    def test_check_slew_rate(self):
        sr_array = utils.compute_slew_rates(self.test_g_array, self.dt, output_format='array')
        smax_limit_ok = np.max(np.abs(sr_array)) + 100
        smax_limit_fail = np.max(np.abs(sr_array)) * 0.5

        sr_ok, _, _ = utils.check_slew_rate(sr_array, smax_T_per_m_per_s=smax_limit_ok)
        self.assertTrue(sr_ok)

        sr_fail, _, details_fail_sr = utils.check_slew_rate(sr_array, smax_T_per_m_per_s=smax_limit_fail)
        self.assertFalse(sr_fail)
        self.assertIn('first_exceeding_value_T_per_m_per_s', details_fail_sr)


    def test_resample_waveform(self):
        original_wf = np.array([0., 1., 2., 3., 4.])
        # Upsample
        resampled_up = utils.resample_waveform(original_wf, 9) # (5-1)*2 + 1 = 9 for linear
        self.assertEqual(len(resampled_up), 9)
        np.testing.assert_almost_equal(resampled_up[0], 0.0)
        np.testing.assert_almost_equal(resampled_up[2], 1.0) # Should be original point
        np.testing.assert_almost_equal(resampled_up[1], 0.5) # Interpolated point

        # Downsample
        resampled_down = utils.resample_waveform(original_wf, 3)
        self.assertEqual(len(resampled_down), 3)
        np.testing.assert_almost_equal(resampled_down[0], 0.0)
        np.testing.assert_almost_equal(resampled_down[1], 2.0) # Middle point
        np.testing.assert_almost_equal(resampled_down[2], 4.0)

        # Complex data
        complex_wf = original_wf + 1j * original_wf[::-1] # 0+4j, 1+3j, 2+2j, 3+1j, 4+0j
        resampled_complex = utils.resample_waveform(complex_wf, 9)
        self.assertEqual(len(resampled_complex), 9)
        self.assertTrue(np.iscomplexobj(resampled_complex))
        np.testing.assert_almost_equal(resampled_complex[0], 0+4j)
        np.testing.assert_almost_equal(resampled_complex[1].real, 0.5)
        np.testing.assert_almost_equal(resampled_complex[1].imag, 3.5)

    def test_resample_girf_spectra(self):
        girf_dict = {
            'x': np.array([0+1j, 1+2j, 2+3j]),
            'y': np.array([10., 20., 30., 40.])
        }
        target_len = 5
        resampled_girfs = utils.resample_girf_spectra(girf_dict, target_len)
        self.assertIn('x', resampled_girfs)
        self.assertEqual(len(resampled_girfs['x']), target_len)
        self.assertTrue(np.iscomplexobj(resampled_girfs['x']))
        self.assertEqual(len(resampled_girfs['y']), target_len)


    def test_convolve_signals(self):
        signal = np.array([0.,0.,1.,1.,1.,0.,0.])
        kernel = np.array([1.,1.,1.]) / 3.0
        # Expected: (mode='same')
        # 0, 1/3, 2/3, 1, 2/3, 1/3, 0 (approx)
        # Padded signal for 'same': [0,0,0,0,1,1,1,0,0,0,0] (depends on kernel length)
        # Convolution result should have same length as signal for mode='same'
        convolved = utils.convolve_signals(signal, kernel, mode='same')
        self.assertEqual(len(convolved), len(signal))
        # Check a few points based on manual calculation or known scipy behavior
        # e.g. center of pulse should be smoothed average of 1,1,1 -> 1
        self.assertAlmostEqual(convolved[3], 1.0)
        # Point before pulse: (0*1/3 + 0*1/3 + 1*1/3) = 1/3
        self.assertAlmostEqual(convolved[1], (0+0+1)/3.0)


    def test_absolute_value(self):
        self.assertEqual(utils.absolute_value(-5.5), 5.5)
        np.testing.assert_array_almost_equal(utils.absolute_value(np.array([-1, 2, -3.5])), np.array([1, 2, 3.5]))
        self.assertAlmostEqual(utils.absolute_value(3+4j), 5.0)
        np.testing.assert_array_almost_equal(utils.absolute_value(np.array([3+4j, -6-8j])), np.array([5.0, 10.0]))

if __name__ == '__main__':
    unittest.main()
