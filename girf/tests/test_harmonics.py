import unittest
import numpy as np
from scipy.signal import iirnotch # For comparing coefficients if generated

# Adjust path for imports
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf import harmonics # This should work if girf is in path or installed
from girf.utils import DEFAULT_GAMMA_PROTON # For any constants if needed for test data

class TestHarmonics(unittest.TestCase):

    def setUp(self):
        self.dt = 10e-6  # 10 us
        self.num_points = 1024
        self.time_vector = np.arange(self.num_points) * self.dt
        self.sampling_rate_hz = 1.0 / self.dt

        # Sample gradient waveform
        self.grad_wf_clean = 0.01 * np.sin(2 * np.pi * 1e3 * self.time_vector) # 1 kHz sine
        self.grad_wf_with_harmonics = self.grad_wf_clean + \
                                      0.002 * np.sin(2 * np.pi * 10e3 * self.time_vector) + \
                                      0.001 * np.sin(2 * np.pi * 25e3 * self.time_vector)

        self.sensitive_harmonics_info = [
            {'freq_hz': 10000.0, 'amp_sensitivity': 0.0005}, # A/2 value, so actual residual peak was 0.001
            {'freq_hz': 25000.0, 'amp_sensitivity': 0.00025},# Actual residual peak was 0.0005
            {'freq_hz': 50000.0, 'amp_sensitivity': 0.001}  # A frequency not strongly in grad_wf_with_harmonics
        ]

    def test_analyze_nominal_gradient_spectrum_no_excitation(self):
        # Use clean gradient, sensitive freqs are high and not in clean gradient
        analysis = harmonics.analyze_nominal_gradient_spectrum(
            self.grad_wf_clean, self.dt, self.sensitive_harmonics_info,
            analysis_threshold_factor=0.1, fft_scale_factor=2.0 # Use 2/N for one-sided scaling
        )
        self.assertEqual(len(analysis), len(self.sensitive_harmonics_info))
        for item in analysis:
            # For the clean 1kHz sine, its spectrum is mainly at 1kHz.
            # The sensitive frequencies are 10kHz, 25kHz, 50kHz.
            # So, nominal_amp_at_freq for these should be very small (noise/leakage).
            # Thus, is_potentially_excited should be False.
            if item['freq_hz'] in [10000.0, 25000.0, 50000.0]:
                 self.assertFalse(item['is_potentially_excited'], f"Freq {item['freq_hz']} should not be excited by clean 1kHz sine.")
                 self.assertTrue(item['nominal_amp_at_freq'] < item['threshold_value'])

    def test_analyze_nominal_gradient_spectrum_with_excitation(self):
        analysis = harmonics.analyze_nominal_gradient_spectrum(
            self.grad_wf_with_harmonics, self.dt, self.sensitive_harmonics_info,
            analysis_threshold_factor=0.5, fft_scale_factor=2.0 # 2/N scaling
        )
        # Expected amplitudes in grad_wf_with_harmonics: 0.002 at 10kHz, 0.001 at 25kHz
        # Sensitive harmonics amp_sensitivity: 0.0005 at 10kHz, 0.00025 at 25kHz
        # Thresholds (factor=0.5): 0.00025 at 10kHz, 0.000125 at 25kHz

        excited_10k = next(item for item in analysis if item['freq_hz'] == 10000.0)
        # Nominal amp at 10kHz should be approx 0.002 (from input signal)
        self.assertAlmostEqual(excited_10k['nominal_amp_at_freq'], 0.002, delta=0.0005) # Allow some delta due to FFT leakage/scaling
        self.assertTrue(excited_10k['is_potentially_excited'])

        excited_25k = next(item for item in analysis if item['freq_hz'] == 25000.0)
        # Nominal amp at 25kHz should be approx 0.001
        self.assertAlmostEqual(excited_25k['nominal_amp_at_freq'], 0.001, delta=0.0002)
        self.assertTrue(excited_25k['is_potentially_excited'])

        not_excited_50k = next(item for item in analysis if item['freq_hz'] == 50000.0)
        self.assertFalse(not_excited_50k['is_potentially_excited'])


    def test_analyze_nominal_gradient_spectrum_thresholding(self):
        # Use grad_wf_with_harmonics. Sensitive freq at 10kHz has nominal_amp ~0.002.
        # Sensitivity is 0.0005.
        # Test1: threshold_factor makes threshold > 0.002 -> not excited
        analysis_high_thresh = harmonics.analyze_nominal_gradient_spectrum(
            self.grad_wf_with_harmonics, self.dt,
            [{'freq_hz': 10000.0, 'amp_sensitivity': 0.0005}],
            analysis_threshold_factor=5.0, # Threshold = 0.0005 * 5 = 0.0025. Nominal amp ~0.002.
            fft_scale_factor=2.0
        )
        self.assertFalse(analysis_high_thresh[0]['is_potentially_excited'])

        # Test2: threshold_factor makes threshold < 0.002 -> excited
        analysis_low_thresh = harmonics.analyze_nominal_gradient_spectrum(
            self.grad_wf_with_harmonics, self.dt,
            [{'freq_hz': 10000.0, 'amp_sensitivity': 0.0005}],
            analysis_threshold_factor=0.1, # Threshold = 0.0005 * 0.1 = 0.00005. Nominal amp ~0.002.
            fft_scale_factor=2.0
        )
        self.assertTrue(analysis_low_thresh[0]['is_potentially_excited'])


    def test_suggest_notch_filter_params(self):
        freq_to_notch = 10000.0 # Hz
        q_factor = 25.0

        params = harmonics.suggest_notch_filter_params(freq_to_notch, self.sampling_rate_hz, q_factor=q_factor)

        self.assertEqual(params['type'], 'notch')
        self.assertEqual(params['center_freq_hz'], freq_to_notch)
        self.assertEqual(params['q_factor'], q_factor)
        self.assertAlmostEqual(params['bandwidth_hz'], freq_to_notch / q_factor)
        self.assertIn('scipy_coeffs', params)

        if params['scipy_coeffs'] is not None:
            b, a = params['scipy_coeffs']
            self.assertIsInstance(b, list) # Stored as list
            self.assertIsInstance(a, list)
            self.assertEqual(len(b), 3) # Standard IIR notch is 3 coeffs for b and a
            self.assertEqual(len(a), 3)

            # Compare with directly generated coeffs
            b_ref, a_ref = iirnotch(freq_to_notch, q_factor, fs=self.sampling_rate_hz)
            np.testing.assert_array_almost_equal(np.array(b), b_ref)
            np.testing.assert_array_almost_equal(np.array(a), a_ref)

        # Test error conditions
        with self.assertRaises(ValueError): # freq = 0
            harmonics.suggest_notch_filter_params(0, self.sampling_rate_hz)
        with self.assertRaises(ValueError): # freq > Nyquist
            harmonics.suggest_notch_filter_params(self.sampling_rate_hz / 2 + 10, self.sampling_rate_hz)


    def test_evaluate_waveform_smoothing_gaussian(self):
        params = {'sigma_ms': 0.01} # 10 us sigma, which is 1 sample for dt=10us
        eval_results = harmonics.evaluate_waveform_smoothing(
            self.grad_wf_with_harmonics, self.dt, smoothing_type='gaussian', params=params
        )
        self.assertIn('original_max_slew_T_per_m_per_s', eval_results)
        self.assertIn('smoothed_max_slew_T_per_m_per_s', eval_results)
        self.assertIn('smoothed_waveform', eval_results)
        self.assertEqual(len(eval_results['smoothed_waveform']), len(self.grad_wf_with_harmonics))

        # With sigma=1 sample, smoothing effect might be small but slew should typically reduce or stay same
        self.assertTrue(eval_results['smoothed_max_slew_T_per_m_per_s'] <= eval_results['original_max_slew_T_per_m_per_s'] * 1.05) # Allow 5% tolerance due to numerics
        self.assertTrue(eval_results['reduction_metrics']['high_freq_energy_reduction_ratio'] < 1.0) # Expect HF reduction

    def test_evaluate_waveform_smoothing_savgol(self):
        # Window length must be odd and > polyorder. dt=10us. 0.05ms = 5 samples.
        params = {'window_length_ms': 0.05, 'polyorder': 2}
        eval_results = harmonics.evaluate_waveform_smoothing(
            self.grad_wf_with_harmonics, self.dt, smoothing_type='savitzky_golay', params=params
        )
        self.assertIn('smoothed_max_slew_T_per_m_per_s', eval_results)
        self.assertEqual(len(eval_results['smoothed_waveform']), len(self.grad_wf_with_harmonics))
        self.assertTrue(eval_results['smoothed_max_slew_T_per_m_per_s'] <= eval_results['original_max_slew_T_per_m_per_s'] * 1.05)
        self.assertTrue(eval_results['reduction_metrics']['high_freq_energy_reduction_ratio'] < 1.0)


    def test_evaluate_waveform_smoothing_invalid_type(self):
        with self.assertRaises(ValueError):
            harmonics.evaluate_waveform_smoothing(self.grad_wf_with_harmonics, self.dt, smoothing_type='invalid_smoother')

    def test_evaluate_waveform_smoothing_params_conversion(self):
        # Test sigma_samples conversion for Gaussian
        # sigma_ms = 0.01ms, dt = 0.01ms (10us) -> sigma_samples = 1
        gauss_params = {'sigma_ms': 0.01}
        res_gauss = harmonics.evaluate_waveform_smoothing(self.grad_wf_clean, self.dt, 'gaussian', gauss_params)
        # No direct way to check sigma_samples used internally without mocking or more complex setup,
        # but ensure it runs and slew changes as expected.
        self.assertTrue(res_gauss['smoothed_max_slew_T_per_m_per_s'] < res_gauss['original_max_slew_T_per_m_per_s'] * 0.99) # Expect some reduction for sine wave

        # Test window_samples conversion for SavGol
        # window_length_ms = 0.05ms (50us), dt = 0.01ms (10us) -> window_samples = 5 (odd)
        savgol_params = {'window_length_ms': 0.05, 'polyorder': 2}
        res_savgol = harmonics.evaluate_waveform_smoothing(self.grad_wf_clean, self.dt, 'savitzky_golay', savgol_params)
        self.assertTrue(res_savgol['smoothed_max_slew_T_per_m_per_s'] < res_savgol['original_max_slew_T_per_m_per_s'] * 0.99)


if __name__ == '__main__':
    unittest.main()
