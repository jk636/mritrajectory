import unittest
import numpy as np
from trajgen.sequences.tpi import TwistedProjectionImagingSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestTwistedProjectionImagingSequence(unittest.TestCase):

    def setUp(self):
        self.common_params = {
            'name': "TestTPI",
            'fov_mm': 220.0, # Isotropic FOV
            'resolution_mm': 2.5, # Isotropic resolution
            # num_dimensions is implicitly 3 for TPI, set in TPISequence __init__
            'dt_seconds': 4e-6,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_twists': 20, # Reduced for faster tests
            'points_per_segment': 256, # Reduced for faster tests
            'cone_angle_deg': 30.0,
            'spiral_turns_per_twist': 5.0,
            'undersampling_factor': 1.0,
            'dead_time_start_seconds': 0.01,
            'dead_time_end_seconds': 0.01
        }

    def test_creation_and_kspace_shape(self):
        seq = TwistedProjectionImagingSequence(**self.common_params)
        self.assertIsNotNone(seq.kspace_points_rad_per_m)
        self.assertEqual(seq.get_num_dimensions(), 3) # Check if num_dimensions is correctly 3

        expected_points = self.common_params['num_twists'] * self.common_params['points_per_segment']
        self.assertEqual(seq.kspace_points_rad_per_m.shape,
                         (3, expected_points)) # TPI generator returns (3, N)

        self.assertEqual(seq.sequence_params.get('num_twists'), self.common_params['num_twists'])
        self.assertEqual(seq.num_twists, self.common_params['num_twists'])
        self.assertEqual(seq.cone_angle_deg, self.common_params['cone_angle_deg'])


    def test_design_principle_methods(self):
        seq = TwistedProjectionImagingSequence(**self.common_params)

        self.assertTrue(len(seq.assess_kspace_coverage()) > 0)
        self.assertTrue(len(seq.estimate_off_resonance_sensitivity()) > 0)
        self.assertTrue(len(seq.assess_motion_ robustness()) > 0)
        self.assertTrue(len(seq.suggest_reconstruction_method()) > 0)

        actual_max_grad = seq.get_max_grad_Tm()
        actual_max_slew = seq.get_max_slew_Tm_per_s()

        if actual_max_grad is None or actual_max_slew is None:
            self.fail("Could not retrieve actual max gradient or slew rate for testing.")

        passing_limits = {
            'max_grad_Tm_per_m': actual_max_grad + 0.01,
            'max_slew_Tm_per_s_per_m': actual_max_slew + 10
        }
        self.assertTrue(seq.check_gradient_limits(passing_limits))

        failing_grad_limits = {
            'max_grad_Tm_per_m': actual_max_grad - 0.001,
            'max_slew_Tm_per_s_per_m': actual_max_slew + 10
        }
        if actual_max_grad > 1e-9:
            self.assertFalse(seq.check_gradient_limits(failing_grad_limits))

        failing_slew_limits = {
            'max_grad_Tm_per_m': actual_max_grad + 0.01,
            'max_slew_Tm_per_s_per_m': actual_max_slew - 1
        }
        if actual_max_slew > 1e-9:
            self.assertFalse(seq.check_gradient_limits(failing_slew_limits))

        self.assertTrue(seq.check_gradient_limits({}))


    def test_invalid_parameters_generation(self):
        # Test for errors during k-space generation via TPI generator
        invalid_gen_params = self.common_params.copy()
        invalid_gen_params['num_twists'] = 0
        with self.assertRaises(ValueError): # Error from TPI generator
            TwistedProjectionImagingSequence(**invalid_gen_params)

        invalid_gen_params_2 = self.common_params.copy()
        invalid_gen_params_2['points_per_segment'] = 0
        with self.assertRaises(ValueError): # Error from TPI generator
            TwistedProjectionImagingSequence(**invalid_gen_params_2)

        invalid_gen_params_3 = self.common_params.copy()
        invalid_gen_params_3['cone_angle_deg'] = -10 # Invalid angle
        with self.assertRaises(ValueError): # Error from TPI generator
            TwistedProjectionImagingSequence(**invalid_gen_params_3)

        invalid_gen_params_4 = self.common_params.copy()
        invalid_gen_params_4['spiral_turns_per_twist'] = 0 # Invalid turns
        with self.assertRaises(ValueError): # Error from TPI generator
            TwistedProjectionImagingSequence(**invalid_gen_params_4)


if __name__ == '__main__':
    unittest.main()
