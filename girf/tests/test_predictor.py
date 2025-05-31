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
from girf.utils import DEFAULT_GAMMA_PROTON, compute_gradient_waveforms # For test comparison

class TestTrajectoryPredictor(unittest.TestCase):

    def setUp(self):
        self.dt = 4e-6  # seconds
        self.gamma = DEFAULT_GAMMA_PROTON
        self.num_points = 256 # Increased for better FFT resolution
        self.time_vector = np.arange(self.num_points) * self.dt

        # Create a dummy GIRF file for loading tests, now including harmonics
        self.girf_axes = ['x', 'y']
        self.dummy_girf_spectra_data = {} # This will hold the {axis: spectrum_array}
        # Simple GIRF: x-axis is identity (pass-through), y-axis attenuates by 0.5 and adds phase
        fft_freqs = np.fft.fftfreq(self.num_points, d=self.dt)
        self.dummy_girf_spectra_data['x'] = np.ones(self.num_points, dtype=np.complex128)
        self.dummy_girf_spectra_data['y'] = 0.5 * np.exp(-1j * np.pi/2 * np.sign(fft_freqs)) # 0.5 gain, -90deg phase shift for non-DC
        self.dummy_girf_spectra_data['y'][0] = 0.5 # DC component phase is 0

        # Dummy harmonic components (as stored by GIRFCalibrator)
        # Xk_residual_peak / N_calibration_points = (A/2)*e^(j*phi)
        self.h_freq_x = 25e3 # 25 kHz
        self.h_amp_x_phys = 0.005 # Physical amplitude of this harmonic component in gradient units (T/m)
        self.h_phase_x = np.pi / 4
        self.h_cv_x = (self.h_amp_x_phys / 2) * np.exp(1j * self.h_phase_x)
        self.dummy_harmonic_components_data = {
            'x': [{'freq_hz': self.h_freq_x, 'complex_value': self.h_cv_x}],
            'y': [] # No harmonics for y-axis
        }

        # Save this dummy GIRF (with harmonics) to a temporary file
        # Mimic GIRFCalibrator's save format
        girf_to_save_enhanced = {
            "gradient_axes": self.girf_axes,
            "default_dt_s": self.dt,
            "girf_spectra_data": { # New nested structure for spectra
                axis: {"spectrum_complex_list": [[val.real, val.imag] for val in spectrum]}
                for axis, spectrum in self.dummy_girf_spectra_data.items()
            },
            "harmonic_components_data": { # New structure for harmonics
                axis: [{'freq_hz': h['freq_hz'],
                        'complex_value_real_imag': [h['complex_value'].real, h['complex_value'].imag]}
                       for h in h_list]
                for axis, h_list in self.dummy_harmonic_components_data.items()
            },
            "input_waveform_params": {}
        }

        self.temp_girf_file_handle, self.temp_girf_file_path = tempfile.mkstemp(suffix='.json')
        with open(self.temp_girf_file_path, 'w') as f:
            json.dump(girf_to_save_enhanced, f)

        # Nominal trajectory: simple ramps
        self.nominal_kx = np.linspace(0, 100, self.num_points) # m^-1
        self.nominal_ky = np.linspace(0, 50, self.num_points)  # m^-1
        self.nominal_k_array = np.stack([self.nominal_kx, self.nominal_ky], axis=-1)
        self.nominal_k_dict = {'x': self.nominal_kx, 'y': self.nominal_ky}


    def tearDown(self):
        os.close(self.temp_girf_file_handle)
        if os.path.exists(self.temp_girf_file_path): # Check existence before removing
            os.remove(self.temp_girf_file_path)

    def test_01_initialization(self):
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        self.assertEqual(predictor.dt, self.dt)
        self.assertEqual(predictor.gamma, self.gamma)
        self.assertEqual(predictor.girf_spectra, {})
        self.assertEqual(predictor.harmonic_components, {}) # New attribute

        predictor_with_data = TrajectoryPredictor(
            girf_spectra=self.dummy_girf_spectra_data,
            harmonic_components=self.dummy_harmonic_components_data,
            dt=self.dt
        )
        self.assertTrue(np.array_equal(predictor_with_data.girf_spectra['x'], self.dummy_girf_spectra_data['x']))
        self.assertEqual(len(predictor_with_data.harmonic_components['x']), 1)


        # Test dt warning / error
        pred_no_dt_init = TrajectoryPredictor() # Should warn on init
        with self.assertRaises(ValueError): # Should error on predict if dt still None
             pred_no_dt_init.predict_trajectory(self.nominal_k_array)


    def test_02_load_girf_from_dict(self):
        predictor = TrajectoryPredictor(dt=self.dt)
        # Create a dict that mimics the structure after GIRFCalibrator.load_calibration or save_calibration
        full_girf_data_dict = {
            'girf_spectra': self.dummy_girf_spectra_data,
            'harmonic_components': self.dummy_harmonic_components_data
        }
        predictor.load_girf(full_girf_data_dict)
        self.assertIn('x', predictor.girf_spectra)
        np.testing.assert_array_equal(predictor.girf_spectra['y'], self.dummy_girf_spectra_data['y'])
        self.assertIn('x', predictor.harmonic_components)
        self.assertEqual(len(predictor.harmonic_components['x']), 1)
        self.assertEqual(predictor.harmonic_components['x'][0]['freq_hz'], self.h_freq_x)

    def test_03_load_girf_from_file(self):
        predictor = TrajectoryPredictor(dt=self.dt)
        predictor.load_girf(self.temp_girf_file_path)
        self.assertIn('x', predictor.girf_spectra)
        self.assertIn('y', predictor.girf_spectra)
        self.assertTrue(np.allclose(predictor.girf_spectra['x'], self.dummy_girf_spectra_data['x']))
        self.assertTrue(np.allclose(predictor.girf_spectra['y'], self.dummy_girf_spectra_data['y']))

        self.assertIn('x', predictor.harmonic_components)
        self.assertEqual(len(predictor.harmonic_components['x']), 1)
        loaded_harmonic_x = predictor.harmonic_components['x'][0]
        self.assertAlmostEqual(loaded_harmonic_x['freq_hz'], self.h_freq_x)
        np.testing.assert_allclose(loaded_harmonic_x['complex_value'], self.h_cv_x)


    def test_04_internal_conversions_kspace_to_gradients(self):
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        # Now uses utils.compute_gradient_waveforms
        grads_array = predictor._trajectory_to_gradients(self.nominal_k_array)
        self.assertEqual(grads_array.shape, self.nominal_k_array.shape)

        expected_gx0 = self.nominal_kx[0] / (self.gamma * self.dt)
        expected_gx1 = (self.nominal_kx[1] - self.nominal_kx[0]) / (self.gamma * self.dt)

        self.assertAlmostEqual(grads_array[0,0], expected_gx0)
        self.assertAlmostEqual(grads_array[1,0], expected_gx1)


    def test_05_internal_conversions_gradients_to_kspace(self):
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        # Now uses utils.integrate_trajectory
        true_nominal_grads_arr = compute_gradient_waveforms( # Use directly from utils for clean test
            self.nominal_k_array, self.gamma, self.dt, output_format='array'
        )
        k_recon_array = predictor._gradients_to_trajectory(true_nominal_grads_arr,
                                                           initial_kspace_point=self.nominal_k_array[0,:])
        self.assertEqual(k_recon_array.shape, self.nominal_k_array.shape)
        np.testing.assert_array_almost_equal(k_recon_array, self.nominal_k_array, decimal=5)

    def test_06_predict_trajectory_ideal_girf(self):
        # No GIRF loaded (empty girf_spectra and harmonic_components)
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        predicted_k = predictor.predict_trajectory(self.nominal_k_array) # No GIRF/harmonics loaded
        np.testing.assert_array_almost_equal(predicted_k, self.nominal_k_array, decimal=5)

        predicted_k_dict_in = predictor.predict_trajectory(self.nominal_k_dict)
        np.testing.assert_array_almost_equal(predicted_k_dict_in, self.nominal_k_array, decimal=5)


    def test_07_predict_trajectory_with_simple_girf_no_harmonics(self):
        # Load only GIRF spectra, no harmonics
        predictor = TrajectoryPredictor(girf_spectra=self.dummy_girf_spectra_data, dt=self.dt, gamma=self.gamma)
        predicted_k = predictor.predict_trajectory(self.nominal_k_array, apply_harmonics=False)

        self.assertEqual(predicted_k.shape, self.nominal_k_array.shape)
        np.testing.assert_array_almost_equal(predicted_k[:,0], self.nominal_k_array[:,0], decimal=3)
        self.assertFalse(np.allclose(predicted_k[:,1], self.nominal_k_array[:,1]))
        self.assertTrue(np.mean(np.abs(predicted_k[:,1])) < np.mean(np.abs(self.nominal_k_array[:,1])))


    def test_08_predict_trajectory_girf_resizing(self):
        short_girf_x = np.ones(self.num_points // 2, dtype=np.complex128)
        girf_data_short = {'x': short_girf_x,
                           'y': self.dummy_girf_spectra_data['y'][:self.num_points//2]}

        predictor_short = TrajectoryPredictor(girf_spectra=girf_data_short, dt=self.dt, gamma=self.gamma)
        predicted_k_short_girf = predictor_short.predict_trajectory(self.nominal_k_array, apply_harmonics=False)
        self.assertEqual(predicted_k_short_girf.shape, self.nominal_k_array.shape)
        np.testing.assert_array_almost_equal(predicted_k_short_girf[:,0], self.nominal_k_array[:,0], decimal=3)

        long_girf_x = np.ones(self.num_points * 2, dtype=np.complex128)
        girf_data_long = {'x': long_girf_x,
                          'y': np.pad(self.dummy_girf_spectra_data['y'],(0,self.num_points),mode='edge')}
        predictor_long = TrajectoryPredictor(girf_spectra=girf_data_long, dt=self.dt, gamma=self.gamma)
        predicted_k_long_girf = predictor_long.predict_trajectory(self.nominal_k_array, apply_harmonics=False)
        self.assertEqual(predicted_k_long_girf.shape, self.nominal_k_array.shape)
        np.testing.assert_array_almost_equal(predicted_k_long_girf[:,0], self.nominal_k_array[:,0], decimal=3)


    def test_09_validate_trajectory(self):
        predictor = TrajectoryPredictor(dt=self.dt) # No dt error here, just init
        # Need to populate nominal_trajectory_kspace for validate to run
        predictor.nominal_trajectory_kspace = self.nominal_k_array.copy()

        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy()
        self.assertTrue(predictor.validate_trajectory(0.01))

        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy() * 1.05
        self.assertTrue(predictor.validate_trajectory(0.06))
        self.assertFalse(predictor.validate_trajectory(0.04))

        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy() * 1.5
        self.assertFalse(predictor.validate_trajectory(0.1))

        predictor.predicted_trajectory_kspace = None
        self.assertFalse(predictor.validate_trajectory())
        predictor.predicted_trajectory_kspace = self.nominal_k_array.copy()
        predictor.nominal_trajectory_kspace = None
        self.assertFalse(predictor.validate_trajectory())
        predictor.nominal_trajectory_kspace = self.nominal_k_array[:-1,:]
        self.assertFalse(predictor.validate_trajectory())


    def test_10_predict_with_apply_harmonics(self):
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        predictor.load_girf({ # Load dict with both GIRF and Harmonics
            'girf_spectra': self.dummy_girf_spectra_data,
            'harmonic_components': self.dummy_harmonic_components_data
        })

        # Predict WITHOUT harmonics
        _ = predictor.predict_trajectory(self.nominal_k_dict, apply_harmonics=False)
        # The predictor stores gradients as self.predicted_gradients_time
        # This is an array (T, N_axes). axes_names stores mapping.
        idx_x = predictor.axes_names.index('x')
        grad_x_no_harm = predictor.predicted_gradients_time[:, idx_x].copy()

        # Predict WITH harmonics
        _ = predictor.predict_trajectory(self.nominal_k_dict, apply_harmonics=True)
        grad_x_with_harm = predictor.predicted_gradients_time[:, idx_x].copy()

        # Calculate the expected time-domain waveform of the single harmonic for x
        expected_harmonic_x_wf = np.zeros(self.num_points, dtype=float)
        if self.dummy_harmonic_components_data['x']:
            h_info = self.dummy_harmonic_components_data['x'][0]
            h_freq = h_info['freq_hz']
            cv_calib = h_info['complex_value'] # This is Xk_peak / N_calib = (A/2)exp(j*phi)

            h_spectrum = np.zeros(self.num_points, dtype=np.complex128)
            freq_bins = np.fft.fftfreq(self.num_points, d=self.dt)

            idx_pos = np.argmin(np.abs(freq_bins - h_freq))
            if np.isclose(freq_bins[idx_pos], h_freq):
                h_spectrum[idx_pos] = cv_calib

            if h_freq != 0: # Avoid double counting DC or Nyquist
                idx_neg = np.argmin(np.abs(freq_bins - (-h_freq)))
                if np.isclose(freq_bins[idx_neg], -h_freq):
                    h_spectrum[idx_neg] = np.conj(cv_calib)

            # ifft(S) * N reconstructs A*cos(2*pi*f*t+phi)
            expected_harmonic_x_wf = (np.fft.ifft(h_spectrum) * self.num_points).real

        # Assert that grad_x_with_harm is approx grad_x_no_harm + expected_harmonic_x_wf
        np.testing.assert_allclose(grad_x_with_harm, grad_x_no_harm + expected_harmonic_x_wf, atol=1e-9, rtol=1e-5)


    def test_11_predict_with_no_harmonics_loaded(self):
        # Load only GIRF spectra, no harmonics
        predictor = TrajectoryPredictor(dt=self.dt, gamma=self.gamma)
        predictor.load_girf({'girf_spectra': self.dummy_girf_spectra_data, 'harmonic_components': {}}) # Empty harmonics

        k_pred_harm_true = predictor.predict_trajectory(self.nominal_k_dict, apply_harmonics=True)
        k_pred_harm_false = predictor.predict_trajectory(self.nominal_k_dict, apply_harmonics=False)

        # Results should be identical as no harmonics were loaded to apply
        np.testing.assert_array_almost_equal(k_pred_harm_true, k_pred_harm_false)


if __name__ == '__main__':
    unittest.main()
