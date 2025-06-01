import unittest
import numpy as np
from trajgen.sequences.variable_density_spiral import VariableDensitySpiralSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestVariableDensitySpiralSequence(unittest.TestCase):

    def setUp(self):
        self.common_params = {
            'name': "TestSpiral",
            'fov_mm': 256.0,
            'resolution_mm': 2.0,
            'num_dimensions': 2,
            'dt_seconds': 4e-6,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_interleaves': 16,
            'points_per_interleaf': 1024,
            'spiral_type': 'archimedean',
            'density_transition_radius_factor': 0.5,
            'density_factor_at_center': 2.0,
            'undersampling_factor': 1.0,
            'dead_time_start_seconds': 0.01,
            'dead_time_end_seconds': 0.01
        }

    def test_creation_and_kspace_shape(self):
        seq = VariableDensitySpiralSequence(**self.common_params)
        self.assertIsNotNone(seq.kspace_points_rad_per_m)
        expected_points = self.common_params['num_interleaves'] * self.common_params['points_per_interleaf']
        self.assertEqual(seq.kspace_points_rad_per_m.shape,
                         (self.common_params['num_dimensions'], expected_points))

        # Check if sequence_params (from Trajectory, via MRISequence) got populated
        self.assertEqual(seq.sequence_params.get('num_interleaves'), self.common_params['num_interleaves'])
        self.assertEqual(seq.sequence_params.get('spiral_type'), self.common_params['spiral_type'])
        self.assertEqual(seq.num_interleaves, self.common_params['num_interleaves'])

    def test_design_principle_methods(self):
        seq = VariableDensitySpiralSequence(**self.common_params)

        self.assertTrue(len(seq.assess_kspace_coverage()) > 0)
        self.assertTrue(len(seq.estimate_off_resonance_sensitivity()) > 0)
        self.assertTrue(len(seq.assess_motion_robustness()) > 0)
        self.assertTrue(len(seq.suggest_reconstruction_method()) > 0)

        # Test check_gradient_limits
        # Need actual gradient calculation to proceed, so dt_seconds must be > 0
        # and gamma_Hz_per_T must be > 0.
        # The setUp params ensure gradients can be calculated.

        # Retrieve actual max grad and slew to formulate realistic limits
        # This requires k-space points to be generated, which happens in __init__
        # and gradients/slew to be calculated by the Trajectory class methods.
        actual_max_grad = seq.get_max_grad_Tm()
        actual_max_slew = seq.get_max_slew_Tm_per_s()

        if actual_max_grad is None or actual_max_slew is None:
            self.fail("Could not retrieve actual max gradient or slew rate for testing.")

        # Test passing limits
        passing_limits = {
            'max_grad_Tm_per_m': actual_max_grad + 0.01, # Add a margin
            'max_slew_Tm_per_s_per_m': actual_max_slew + 10
        }
        self.assertTrue(seq.check_gradient_limits(passing_limits))

        # Test failing limits (gradient)
        failing_grad_limits = {
            'max_grad_Tm_per_m': actual_max_grad - 0.001, # Slightly less
            'max_slew_Tm_per_s_per_m': actual_max_slew + 10
        }
        if actual_max_grad > 1e-9 : # Only test if grad is non-zero
             self.assertFalse(seq.check_gradient_limits(failing_grad_limits))

        # Test failing limits (slew)
        failing_slew_limits = {
            'max_grad_Tm_per_m': actual_max_grad + 0.01,
            'max_slew_Tm_per_s_per_m': actual_max_slew - 1 # Slightly less
        }
        if actual_max_slew > 1e-9: # Only test if slew is non-zero
            self.assertFalse(seq.check_gradient_limits(failing_slew_limits))

        # Test with no limits provided (should pass)
        self.assertTrue(seq.check_gradient_limits({}))


    def test_invalid_parameters_generation(self):
        # Test for errors during k-space generation via generator
        # These are typically caught by the generator function itself.
        # VariableDensitySpiralSequence passes parameters to generate_spiral_trajectory

        invalid_gen_params = self.common_params.copy()
        invalid_gen_params['points_per_interleaf'] = 0
        with self.assertRaises(ValueError): # This error comes from generate_spiral_trajectory
            VariableDensitySpiralSequence(**invalid_gen_params)

        invalid_gen_params_2 = self.common_params.copy()
        invalid_gen_params_2['num_interleaves'] = 0
        with self.assertRaises(ValueError):
            VariableDensitySpiralSequence(**invalid_gen_params_2)

        # Test for num_dimensions (though generator might also catch this)
        invalid_dim_params = self.common_params.copy()
        invalid_dim_params['num_dimensions'] = 1 # Spiral generator might not support 1D
        # The specific error might vary based on generator's checks
        with self.assertRaises(ValueError): # Or NotImplementedError
             VariableDensitySpiralSequence(**invalid_dim_params)


if __name__ == '__main__':
    unittest.main()
