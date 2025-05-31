import unittest
import numpy as np
import json
import tempfile # For creating temporary files for save/load tests
import os

# Adjust path to import from the girf package
# This assumes tests are run from the parent directory of 'girf' or 'girf' is in PYTHONPATH
# For robust testing, a proper test runner setup (e.g. using python -m unittest discover)
# from the project root is better.
# If this script is girf/tests/test_calibrator.py, then parent of girf is two levels up.
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf.calibrator import GIRFCalibrator

class TestGIRFCalibrator(unittest.TestCase):

    def setUp(self):
        self.default_axes = ['x', 'y', 'z']
        self.calibrator = GIRFCalibrator(gradient_axes=self.default_axes)

        # Sample data for measurements
        self.num_points = 128
        self.time_vector = np.linspace(0, self.num_points * 4e-6, self.num_points, endpoint=False)

        # Simple input: a sine wave
        self.input_x_wf = np.sin(2 * np.pi * 1e3 * self.time_vector) # 1 kHz sine
        # Simple response: scaled and phase-shifted version of input
        self.response_x_wf = 0.8 * np.sin(2 * np.pi * 1e3 * self.time_vector - np.pi/4)
        self.response_x_wf += 0.01 * np.random.randn(self.num_points) # Add some noise

        self.input_y_wf = np.cos(2 * np.pi * 2e3 * self.time_vector) # 2 kHz cosine
        self.response_y_wf = 0.9 * np.cos(2 * np.pi * 2e3 * self.time_vector + np.pi/6)
        self.response_y_wf += 0.01 * np.random.randn(self.num_points)


    def test_01_initialization(self):
        self.assertEqual(self.calibrator.gradient_axes, self.default_axes)
        self.assertEqual(self.calibrator.input_waveforms, {})
        self.assertEqual(self.calibrator.measured_responses, {})
        self.assertEqual(self.calibrator.girf_spectra, {})

        custom_axes = ['Gx', 'Gy']
        cal_custom = GIRFCalibrator(gradient_axes=custom_axes)
        self.assertEqual(cal_custom.gradient_axes, custom_axes)

    def test_02_measure_girf(self):
        self.calibrator.measure_girf('x', self.input_x_wf, self.response_x_wf,
                                     waveform_params={'freq_hz': 1000})
        self.assertIn('x', self.calibrator.input_waveforms)
        np.testing.assert_array_equal(self.calibrator.input_waveforms['x'], np.asarray(self.input_x_wf))
        np.testing.assert_array_equal(self.calibrator.measured_responses['x'], np.asarray(self.response_x_wf))
        self.assertEqual(self.calibrator.waveform_params['x']['freq_hz'], 1000)

        with self.assertRaises(ValueError): # Axis not configured
            self.calibrator.measure_girf('invalid_axis', self.input_x_wf, self.response_x_wf)

    def test_03_compute_girf_spectrum_simple_delta(self):
        # Test with a simple delta function input
        # Input: delta function (approximated as one high point, rest zero)
        # Response: scaled and delayed delta function
        delta_input = np.zeros(self.num_points)
        delta_input[0] = 1.0

        # Response: scaled (0.5) and delayed by 1 point (phase shift in freq domain)
        delayed_response = np.zeros(self.num_points)
        delayed_response[1] = 0.5

        self.calibrator.measure_girf('x', delta_input, delayed_response)
        girf_x = self.calibrator.compute_girf_spectrum('x', regularization_factor=1e-9)

        self.assertIsInstance(girf_x, np.ndarray)
        self.assertEqual(len(girf_x), self.num_points)
        self.assertTrue(np.iscomplexobj(girf_x))

        # Expected: FFT(delayed_response) / FFT(delta_input)
        # FFT(delta_input) is approx 1 (constant across frequencies)
        # FFT(delayed_response) is 0.5 * exp(-j*omega*t_delay) where t_delay is 1*dt
        # So GIRF should be approx 0.5 * exp(-j*omega*dt)

        # Check average magnitude (should be around 0.5)
        # Note: FFT of delta_input[0]=1 is array of 1s.
        # FFT of delayed_response[1]=0.5 is 0.5 * exp(-2*pi*j*k*1/N) for k=0..N-1
        # So girf_x[k] = 0.5 * exp(-2*pi*j*k/N)
        expected_girf_manual = 0.5 * np.exp(-2j * np.pi * np.arange(self.num_points) / self.num_points)
        np.testing.assert_array_almost_equal(girf_x, expected_girf_manual, decimal=5)


    def test_04_compute_girf_spectrum_sine_input(self):
        self.calibrator.measure_girf('x', self.input_x_wf, self.response_x_wf)
        girf_x = self.calibrator.compute_girf_spectrum('x')
        self.assertIsInstance(girf_x, np.ndarray)
        self.assertEqual(len(girf_x), self.num_points)

        # For a sine input, the GIRF value at the input frequency should reflect gain and phase shift.
        # This is a more complex check, ensuring the computation runs.
        # Actual value check would require careful setup of expected FFT values.
        # For now, just check type and shape.

    def test_05_compute_girf_spectrum_error_handling(self):
        with self.assertRaises(ValueError): # Data not measured for axis 'y'
            self.calibrator.compute_girf_spectrum('y')

        self.calibrator.measure_girf('z', [], []) # Empty data
        with self.assertRaises(ValueError):
            self.calibrator.compute_girf_spectrum('z')

        self.calibrator.measure_girf('x', [1,2,3], [1,2]) # Mismatched lengths
        with self.assertRaises(ValueError):
            self.calibrator.compute_girf_spectrum('x')


    def test_06_save_and_load_calibration(self):
        # Measure and compute for two axes
        self.calibrator.measure_girf('x', self.input_x_wf, self.response_x_wf, {'p1': 'val_x'})
        girf_x_orig = self.calibrator.compute_girf_spectrum('x')

        self.calibrator.measure_girf('y', self.input_y_wf, self.response_y_wf, {'p2': 'val_y'})
        girf_y_orig = self.calibrator.compute_girf_spectrum('y')

        # Use tempfile for saving and loading
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_file:
            temp_file_path = tmp_file.name

        try:
            self.calibrator.save_calibration(temp_file_path)
            self.assertTrue(os.path.exists(temp_file_path))

            new_calibrator = GIRFCalibrator(gradient_axes=['x', 'y', 'z']) # Can have different initial axes
            new_calibrator.load_calibration(temp_file_path)

            self.assertIn('x', new_calibrator.girf_spectra)
            self.assertIn('y', new_calibrator.girf_spectra)
            self.assertNotIn('z', new_calibrator.girf_spectra) # z was not saved

            np.testing.assert_array_almost_equal(new_calibrator.girf_spectra['x'], girf_x_orig)
            np.testing.assert_array_almost_equal(new_calibrator.girf_spectra['y'], girf_y_orig)

            # Check if waveform_params are loaded
            self.assertEqual(new_calibrator.waveform_params.get('x'), {'p1': 'val_x'})
            self.assertEqual(new_calibrator.waveform_params.get('y'), {'p2': 'val_y'})

            # Check if gradient_axes in the loaded instance are updated based on file content + original.
            # The current load_calibration adds axes from file if not present.
            self.assertIn('x', new_calibrator.gradient_axes)
            self.assertIn('y', new_calibrator.gradient_axes)
            # self.assertIn('z', new_calibrator.gradient_axes) # z was initial, should persist unless file overwrites axes list completely.
                                                               # Current load_calibration merges axes from file.

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_07_save_empty_calibration(self):
        # Test saving when no GIRF spectra are computed
        empty_calibrator = GIRFCalibrator()
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_file:
            temp_file_path = tmp_file.name
        try:
            empty_calibrator.save_calibration(temp_file_path) # Should run without error

            # Load it back and check
            new_empty_calibrator = GIRFCalibrator()
            new_empty_calibrator.load_calibration(temp_file_path)
            self.assertEqual(new_empty_calibrator.girf_spectra, {})
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_08_load_malformed_calibration_file(self):
        malformed_content = "{'this_is_not_json': "

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_file.write(malformed_content)
            temp_file_path = tmp_file.name

        cal = GIRFCalibrator()
        with self.assertRaises(ValueError): # Expecting JSONDecodeError, wrapped in ValueError by load_calibration
            cal.load_calibration(temp_file_path)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # File with missing keys
        missing_keys_content = json.dumps({"gradient_axes": ["x"]}) # Missing "girf_spectra_complex"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_file.write(missing_keys_content)
            temp_file_path = tmp_file.name

        with self.assertRaises(ValueError):
            cal.load_calibration(temp_file_path)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == '__main__':
    unittest.main()
