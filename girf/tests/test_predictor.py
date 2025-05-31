import unittest
import numpy as np
import json
import tempfile
import os

# Adjust path for imports
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf.predictor import TrajectoryPredictor
from girf.calibrator import GIRFCalibrator # To create dummy GIRF files for loading
from girf.utils import DEFAULT_GAMMA_PROTON

class TestTrajectoryPredictor(unittest.TestCase):

    def setUp(self):
        self.dt = 4e-6  # seconds
        self.gamma = DEFAULT_GAMMA_PROTON
        self.num_points = 128
        self.time_vector = np.arange(self.num_points) * self.dt

        # Create a dummy GIRF file for loading tests
        self.girf_axes = ['x', 'y']
        self.dummy_girf_spectra = {}
        # Simple GIRF: x-axis is identity (pass-through), y-axis attenuates by 0.5 and adds phase
        fft_freqs = np.fft.fftfreq(self.num_points, d=self.dt)
        self.dummy_girf_spectra['x'] = np.ones(self.num_points, dtype=np.complex128)
        self.dummy_girf_spectra['y'] = 0.5 * np.exp(-1j * np.pi/2 * np.sign(fft_freqs)) # 0.5 gain, -90deg phase shift for non-DC
        self.dummy_girf_spectra['y'][0] = 0.5 # DC component phase is 0

        # Save this dummy GIRF to a temporary file
        # Need to mimic GIRFCalibrator's save format (complex numbers as [real, imag] lists)
        girf_to_save = {
            "gradient_axes": self.girf_axes,
            "girf_spectra_complex": {
                axis: [[val.real, val.imag] for val in spectrum]
                for axis, spectrum in self.dummy_girf_spectra.items()
            },
            "waveform_params": {}
        }

        # Create a temporary file that persists until explicitly deleted
        self.temp_girf_file_handle, self.temp_girf_file_path = tempfile.mkstemp(suffix='.json')
        with open(self.temp_girf_file_path, 'w') as f:
            json.dump(girf_to_save, f)

        # Nominal trajectory: simple ramps
        self.nominal_kx = np.linspace(0, 100, self.num_points) # m^-1
        self.nominal_ky = np.linspace(0, 50, self.num_points)  # m^-1
        self.nominal_k_array = np.stack([self.nominal_kx, self.nominal_ky], axis=-1)
        self.nominal_k_dict = {'x': self.nominal_kx, 'y': self.nominal_ky}


    def tearDown(self):
        # Close and delete the temporary file
        os.close(self.temp_girf_file_handle)
        os.remove(self.temp_girf_file_path)

    def test_01_initialization(self):
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        self.assertEqual(predictor.dt, self.dt)
        self.assertEqual(predictor.gamma, self.gamma)
        self.assertEqual(predictor.girf_spectra, {})

        predictor_with_girf = TrajectoryPredictor(girf_spectra=self.dummy_girf_spectra, dt=self.dt)
        self.assertTrue(np.array_equal(predictor_with_girf.girf_spectra['x'], self.dummy_girf_spectra['x']))

        with self.assertRaises(ValueError): # dt not provided (now raises warning, but should error if used)
             # Test that error is raised if dt is None and prediction is attempted.
             # The __init__ itself might not error but warn.
             pred_no_dt = TrajectoryPredictor()
             pred_no_dt.predict_trajectory(self.nominal_k_array)


    def test_02_load_girf_from_dict(self):
        predictor = TrajectoryPredictor(dt=self.dt)
        predictor.load_girf(self.dummy_girf_spectra)
        self.assertIn('x', predictor.girf_spectra)
        np.testing.assert_array_equal(predictor.girf_spectra['y'], self.dummy_girf_spectra['y'])

    def test_03_load_girf_from_file(self):
        predictor = TrajectoryPredictor(dt=self.dt)
        predictor.load_girf(self.temp_girf_file_path) # Uses GIRFCalibrator internally or direct JSON
        self.assertIn('x', predictor.girf_spectra)
        self.assertIn('y', predictor.girf_spectra)
        # Compare magnitudes as phase might have small numerical differences after load/save of complex
        np.testing.assert_array_almost_equal(
            np.abs(predictor.girf_spectra['x']), np.abs(self.dummy_girf_spectra['x'])
        )
        np.testing.assert_array_almost_equal(
            np.abs(predictor.girf_spectra['y']), np.abs(self.dummy_girf_spectra['y'])
        )
        # A more robust check would be np.allclose for complex arrays
        self.assertTrue(np.allclose(predictor.girf_spectra['x'], self.dummy_girf_spectra['x']))
        self.assertTrue(np.allclose(predictor.girf_spectra['y'], self.dummy_girf_spectra['y']))


    def test_04_internal_conversions_kspace_to_gradients(self):
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        # Test with array input
        grads_array = predictor._trajectory_to_gradients(self.nominal_k_array)
        self.assertEqual(grads_array.shape, self.nominal_k_array.shape)

        # Expected grad_x[0] = kx[0]/(gamma*dt)
        # Expected grad_x[1] = (kx[1]-kx[0])/(gamma*dt)
        expected_gx0 = self.nominal_kx[0] / (self.gamma * self.dt)
        expected_gx1 = (self.nominal_kx[1] - self.nominal_kx[0]) / (self.gamma * self.dt)

        # The _trajectory_to_gradients pads with g[0]=(k[0]-k_prev_assumed_zero)/(g*dt)
        # if prepend=k[0:1,:] is used in diff, then diff[0] is k[0]-k[0]=0.
        # The current implementation in predictor is:
        # gradients = np.diff(trajectory_kspace, axis=0) / (self.gamma * self.dt)
        # padding = np.zeros((1, trajectory_kspace.shape[1]), dtype=gradients.dtype)
        # gradients_padded = np.concatenate([padding, gradients], axis=0)
        # This means g[0] is 0, g[1] is (k[1]-k[0])/(g*dt)
        self.assertAlmostEqual(grads_array[0,0], 0.0) # g[0] is 0 due to padding strategy
        self.assertAlmostEqual(grads_array[1,0], expected_gx1)


    def test_05_internal_conversions_gradients_to_kspace(self):
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        # Use gradients from previous test (or recompute)
        # Note: the current _trajectory_to_gradients has a specific padding.
        # For perfect recon, we need gradients whose integral IS the k-space.
        # Let's use gradients computed by utils.compute_gradient_waveforms for this test
        from girf import utils as girf_utils # Assuming utils.py is in the same package path
        true_nominal_grads_arr = girf_utils.compute_gradient_waveforms(self.nominal_k_array, self.gamma, self.dt, output_format='array')

        k_recon_array = predictor._gradients_to_trajectory(true_nominal_grads_arr,
                                                           initial_kspace_point=self.nominal_k_array[0,:])
        self.assertEqual(k_recon_array.shape, self.nominal_k_array.shape)
        np.testing.assert_array_almost_equal(k_recon_array, self.nominal_k_array, decimal=5)

    def test_06_predict_trajectory_ideal_girf(self):
        # No GIRF loaded, so predictor assumes ideal GIRF (output = input)
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        predicted_k = predictor.predict_trajectory(self.nominal_k_array)
        np.testing.assert_array_almost_equal(predicted_k, self.nominal_k_array, decimal=5)

        predicted_k_dict_in = predictor.predict_trajectory(self.nominal_k_dict)
        # predictor converts dict to array internally, output is always array
        np.testing.assert_array_almost_equal(predicted_k_dict_in, self.nominal_k_array, decimal=5)


    def test_07_predict_trajectory_with_simple_girf(self):
        predictor = TrajectoryPredictor(girf_spectra=self.dummy_girf_spectra, dt=self.dt, gamma=self.gamma)
        predicted_k = predictor.predict_trajectory(self.nominal_k_array)

        self.assertEqual(predicted_k.shape, self.nominal_k_array.shape)

        # Axis X has identity GIRF, so kx should be very close to nominal_kx
        np.testing.assert_array_almost_equal(predicted_k[:,0], self.nominal_k_array[:,0], decimal=3)
        # Decimal is lower due to FFT, IFFT, and potential GIRF resizing numerical errors.

        # Axis Y has GIRF = 0.5 * exp(-j*pi/2*sign(f)). This is 0.5 gain and phase shift.
        # The effect on the k-space trajectory (integral of gradients) is more complex than simple scaling/phase shift
        # of the k-space itself. We expect it to be different from nominal_ky.
        self.assertFalse(np.allclose(predicted_k[:,1], self.nominal_k_array[:,1]))
        # A rough check: magnitude might be scaled.
        # This depends on the spectrum of the nominal_ky. If it's mostly low freq, effect of phase shift might be small on magnitude.
        # For a ramp gradient (from ramp k-space), its FFT is broad.
        # The 0.5 gain in GIRF should lead to roughly 0.5 scaling of gradients, thus k-space.
        # This is a qualitative check, precise values are hard without full simulation.
        self.assertTrue(np.mean(np.abs(predicted_k[:,1])) < np.mean(np.abs(self.nominal_k_array[:,1])))


    def test_08_predict_trajectory_girf_resizing(self):
        # Test when GIRF length is different from trajectory FFT length
        short_girf_x = np.ones(self.num_points // 2, dtype=np.complex128)
        girf_short = {'x': short_girf_x, 'y': self.dummy_girf_spectra['y'][:self.num_points//2]}

        predictor_short = TrajectoryPredictor(girf_spectra=girf_short, dt=self.dt, gamma=self.gamma)
        # Prediction should run due to internal GIRF resizing (padding)
        predicted_k_short_girf = predictor_short.predict_trajectory(self.nominal_k_array)
        self.assertEqual(predicted_k_short_girf.shape, self.nominal_k_array.shape)
        # Check x-axis (was padded with 1s) - should still be close to nominal
        np.testing.assert_array_almost_equal(predicted_k_short_girf[:,0], self.nominal_k_array[:,0], decimal=3)


        long_girf_x = np.ones(self.num_points * 2, dtype=np.complex128)
        girf_long = {'x': long_girf_x, 'y': np.pad(self.dummy_girf_spectra['y'],(0,self.num_points),mode='edge')}

        predictor_long = TrajectoryPredictor(girf_spectra=girf_long, dt=self.dt, gamma=self.gamma)
        # Prediction should run due to internal GIRF resizing (truncation)
        predicted_k_long_girf = predictor_long.predict_trajectory(self.nominal_k_array)
        self.assertEqual(predicted_k_long_girf.shape, self.nominal_k_array.shape)
        # Check x-axis (was truncated version of ones) - should still be close to nominal
        np.testing.assert_array_almost_equal(predicted_k_long_girf[:,0], self.nominal_k_array[:,0], decimal=3)


    def test_09_validate_trajectory(self):
        predictor = TrajectoryPredictor(dt=self.dt)
        predictor.nominal_trajectory_kspace = self.nominal_k_array.copy()

        # Case 1: Predicted = Nominal (should pass with high threshold)
        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy()
        is_valid_ideal = predictor.validate_trajectory(validation_threshold=0.01)
        self.assertTrue(is_valid_ideal)

        # Case 2: Predicted is slightly different
        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy() * 1.05 # 5% deviation
        is_valid_small_dev = predictor.validate_trajectory(validation_threshold=0.06) # Pass if avg dev < 6%
        self.assertTrue(is_valid_small_dev)
        is_invalid_small_dev = predictor.validate_trajectory(validation_threshold=0.04) # Fail if avg dev > 4%
        self.assertFalse(is_invalid_small_dev)

        # Case 3: Predicted is significantly different
        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy() * 1.5 # 50% deviation
        is_invalid_large_dev = predictor.validate_trajectory(validation_threshold=0.1) # Fail if avg dev > 10%
        self.assertFalse(is_invalid_large_dev)

        # Case 4: Error conditions for validate
        predictor.predicted_trajectory_kspace = None
        self.assertFalse(predictor.validate_trajectory()) # No predicted
        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy()
        predictor.nominal_trajectory_kspace = None
        self.assertFalse(predictor.validate_trajectory()) # No nominal
        predictor.nominal_trajectory_kspace = self.nominal_k_array[:-1,:] # Shape mismatch
        self.assertFalse(predictor.validate_trajectory())


if __name__ == '__main__':
    unittest.main()
