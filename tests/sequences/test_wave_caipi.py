import unittest
import numpy as np
from trajgen.sequences.wave_caipi import WaveCAIPISequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestWaveCAIPISequence(unittest.TestCase):

    def setUp(self):
        self.common_params = {
            'name': "TestWaveCAIPI",
            'fov_mm': (220.0, 220.0),
            'resolution_mm': (2.0, 2.0),
            'dt_seconds': 4e-6,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_echoes': 64, # Total lines before undersampling
            'points_per_echo': 128,
            'wave_amplitude_mm': 10.0, # FOV shift amplitude
            'wave_frequency_cycles_per_fov_readout': 2.0, # Two cycles of sine wave
            'wave_phase_offset_rad': 0.0,
            'epi_type': 'flyback',
            'phase_encode_direction': 'y', # kx readout, ky phase
            'undersampling_factor_pe': 1.0,
        }

    def test_creation_and_kspace_shape(self):
        seq = WaveCAIPISequence(**self.common_params)
        self.assertIsNotNone(seq.kspace_points_rad_per_m)
        self.assertEqual(seq.get_num_dimensions(), 2)

        expected_acquired_echoes = self.common_params['num_echoes'] // self.common_params['undersampling_factor_pe']
        expected_points = expected_acquired_echoes * self.common_params['points_per_echo']
        self.assertEqual(seq.kspace_points_rad_per_m.shape, (2, expected_points))

        self.assertEqual(seq.sequence_params.get('wave_amplitude_mm'), self.common_params['wave_amplitude_mm'])
        self.assertEqual(seq.wave_amplitude_mm, self.common_params['wave_amplitude_mm'])

    def test_design_principle_methods(self):
        seq = WaveCAIPISequence(**self.common_params)

        self.assertTrue(len(seq.assess_kspace_coverage()) > 0)
        self.assertTrue(len(seq.estimate_off_resonance_sensitivity()) > 0)
        self.assertTrue(len(seq.assess_motion_robustness()) > 0)
        self.assertTrue(len(seq.suggest_reconstruction_method()) > 0)

        actual_max_grad = seq.get_max_grad_Tm()
        actual_max_slew = seq.get_max_slew_Tm_per_s()

        if actual_max_grad is None or actual_max_slew is None:
            self.fail("Could not retrieve actual max gradient or slew rate for testing.")

        passing_limits = {'max_grad_Tm_per_m': actual_max_grad + 0.01, 'max_slew_Tm_per_s_per_m': actual_max_slew + 10}
        self.assertTrue(seq.check_gradient_limits(passing_limits))
        self.assertTrue(seq.check_gradient_limits({}))


    def test_wave_modulation_applied(self):
        params = self.common_params.copy()
        params['undersampling_factor_pe'] = 1 # Ensure all echoes are generated for simpler indexing
        seq = WaveCAIPISequence(**params)

        k_points = seq.kspace_points_rad_per_m
        points_per_echo = params['points_per_echo']

        # Check the first echo
        ky_echo0 = k_points[1, 0:points_per_echo] # Assuming phase_encode_direction='y'

        # Basic check: are the ky values non-constant along the readout?
        self.assertFalse(np.allclose(ky_echo0, ky_echo0[0]), "Wave modulation not apparent in ky of first echo.")

        # More detailed check for sinusoidal pattern (simplified)
        # Calculate expected base ky (without wave) for the first echo
        fov_m_phase = params['fov_mm'][1] / 1000.0
        res_m_phase = params['resolution_mm'][1] / 1000.0
        k_max_phase_encode = 1.0 / (2.0 * res_m_phase)
        delta_k_phase = 1.0 / fov_m_phase
        expected_ky_base_echo0 = -k_max_phase_encode + 0 * delta_k_phase # For the first echo (i_true_echo=0)

        # The deviation from this base should be sinusoidal
        wave_component_echo0 = ky_echo0 - expected_ky_base_echo0

        # Check if the wave component has a sinusoidal nature (e.g. mean close to zero, specific peaks/troughs)
        # This is an approximation, as the exact shape depends on kx_normalized_for_wave mapping
        self.assertAlmostEqual(np.mean(wave_component_echo0), 0, delta=abs(expected_ky_base_echo0)*0.3 + 1e-3, # Allow some offset
                               msg="Mean of wave component significantly non-zero.")

        # Check amplitude of wave (rough estimate)
        peak_to_peak_wave = np.max(wave_component_echo0) - np.min(wave_component_echo0)

        # Expected k_shift_amplitude_rad_per_m
        k_shift_amplitude_rad_per_m = (params['wave_amplitude_mm'] / (params['fov_mm'][1] / 2.0)) * k_max_phase_encode
        expected_peak_to_peak = 2 * k_shift_amplitude_rad_per_m

        if abs(expected_peak_to_peak) > 1e-9 : # Avoid division by zero if no wave expected
            self.assertAlmostEqual(peak_to_peak_wave, expected_peak_to_peak, delta=expected_peak_to_peak * 0.3, # Allow 30% tolerance
                                   msg=f"Peak-to-peak amplitude of wave component ({peak_to_peak_wave:.3f}) "
                                       f"differs from expected ({expected_peak_to_peak:.3f}).")
        else:
             self.assertTrue(abs(peak_to_peak_wave) < 1e-3) # Should be very small if no wave expected

    def test_invalid_parameters(self):
        invalid_params = self.common_params.copy()
        invalid_params['num_echoes'] = 0
        with self.assertRaises(ValueError):
            WaveCAIPISequence(**invalid_params)

        invalid_params_2 = self.common_params.copy()
        invalid_params_2['points_per_echo'] = 0
        with self.assertRaises(ValueError):
            WaveCAIPISequence(**invalid_params_2)

        invalid_params_3 = self.common_params.copy()
        invalid_params_3['undersampling_factor_pe'] = 0.5
        with self.assertRaises(ValueError):
            WaveCAIPISequence(**invalid_params_3)

if __name__ == '__main__':
    unittest.main()
