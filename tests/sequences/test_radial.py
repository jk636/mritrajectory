import unittest
import numpy as np
from trajgen.sequences.radial import RadialSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestRadialSequence(unittest.TestCase):

    def setUp(self):
        self.common_params = {
            'name': "TestRadial",
            'fov_mm': 200.0,
            'resolution_mm': 2.0,
            'num_dimensions': 2, # For 2D radial
            'dt_seconds': 4e-6,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_spokes': 64,
            'points_per_spoke': 256,
            'projection_angle_increment': 'golden_angle',
            'dead_time_start_seconds': 0.005,
            'dead_time_end_seconds': 0.005
        }

        self.common_params_3d = {
            'name': "TestRadial3D",
            'fov_mm': (200.0, 200.0, 200.0),
            'resolution_mm': (2.5, 2.5, 2.5),
            'num_dimensions': 3, # For 3D radial
            'dt_seconds': 4e-6,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_spokes': 128, # Fewer spokes for faster test
            'points_per_spoke': 128,
            'projection_angle_increment': 'golden_angle', # Uses 3D golden angle
        }


    def test_creation_and_kspace_shape_2d(self):
        seq = RadialSequence(**self.common_params)
        self.assertIsNotNone(seq.kspace_points_rad_per_m)
        expected_points = self.common_params['num_spokes'] * self.common_params['points_per_spoke']
        self.assertEqual(seq.kspace_points_rad_per_m.shape,
                         (self.common_params['num_dimensions'], expected_points))

        self.assertEqual(seq.sequence_params.get('num_spokes'), self.common_params['num_spokes'])
        self.assertEqual(seq.num_spokes, self.common_params['num_spokes'])

    def test_creation_and_kspace_shape_3d(self):
        seq = RadialSequence(**self.common_params_3d)
        self.assertIsNotNone(seq.kspace_points_rad_per_m)
        expected_points = self.common_params_3d['num_spokes'] * self.common_params_3d['points_per_spoke']
        self.assertEqual(seq.kspace_points_rad_per_m.shape,
                         (self.common_params_3d['num_dimensions'], expected_points))
        self.assertEqual(seq.sequence_params.get('num_spokes'), self.common_params_3d['num_spokes'])


    def test_design_principle_methods(self):
        seq = RadialSequence(**self.common_params) # Using 2D for this part

        self.assertTrue(len(seq.assess_kspace_coverage()) > 0)
        self.assertTrue(len(seq.estimate_off_resonance_sensitivity()) > 0)
        self.assertTrue(len(seq.assess_motion_robustness()) > 0)
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
        invalid_params_gen = self.common_params.copy()
        invalid_params_gen['num_spokes'] = 0
        with self.assertRaises(ValueError): # Error from generator
            RadialSequence(**invalid_params_gen)

        invalid_params_gen_2 = self.common_params.copy()
        invalid_params_gen_2['points_per_spoke'] = 0
        with self.assertRaises(ValueError): # Error from generator
            RadialSequence(**invalid_params_gen_2)

        invalid_params_dim = self.common_params.copy()
        invalid_params_dim['num_dimensions'] = 1
        with self.assertRaises(ValueError): # Error from generator for unsupported dimension
            RadialSequence(**invalid_params_dim)

        invalid_angle = self.common_params_3d.copy()
        invalid_angle['projection_angle_increment'] = "not_an_angle_type_for_3d"
        with self.assertRaises(ValueError): # Error from generator for 3D angle spec
             RadialSequence(**invalid_angle)


if __name__ == '__main__':
    unittest.main()
