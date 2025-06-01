import unittest
import numpy as np
from trajgen.sequences.propeller import PropellerBladeSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestPropellerBladeSequence(unittest.TestCase):

    def setUp(self):
        self.common_params = {
            'name': "TestPROPELLER",
            # FOV/res for a blade. Readout FOV usually full, PE FOV is narrow.
            'fov_mm': (256.0, 32.0),
            'resolution_mm': (1.0, 4.0),
            # num_dimensions is implicitly 2
            'dt_seconds': 4e-6,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_blades': 16,
            'lines_per_blade': 8, # Corresponds to fov_mm[1]/resolution_mm[1] = 32/4 = 8
            'points_per_line': 256, # Corresponds to fov_mm[0]/resolution_mm[0] = 256/1 = 256
            'blade_rotation_angle_increment_deg': 180.0 / 16, # Example: 11.25 deg
        }

    def test_creation_and_kspace_shape(self):
        seq = PropellerBladeSequence(**self.common_params)
        self.assertIsNotNone(seq.kspace_points_rad_per_m)
        self.assertEqual(seq.get_num_dimensions(), 2)

        expected_points = self.common_params['num_blades'] * \
                          self.common_params['lines_per_blade'] * \
                          self.common_params['points_per_line']
        self.assertEqual(seq.kspace_points_rad_per_m.shape, (2, expected_points))

        self.assertEqual(seq.sequence_params.get('num_blades'), self.common_params['num_blades'])
        self.assertEqual(seq.num_blades, self.common_params['num_blades'])

    def test_design_principle_methods(self):
        seq = PropellerBladeSequence(**self.common_params)

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
            'max_grad_Tm_per_m': actual_max_grad - 0.001 if actual_max_grad > 0.001 else actual_max_grad / 2,
            'max_slew_Tm_per_s_per_m': actual_max_slew + 10
        }
        if actual_max_grad > 1e-9: # Avoid issues with grad being too close to zero
            self.assertFalse(seq.check_gradient_limits(failing_grad_limits))

        self.assertTrue(seq.check_gradient_limits({}))


    def test_blade_rotation(self):
        params = self.common_params.copy()
        params['num_blades'] = 2
        params['blade_rotation_angle_increment_deg'] = 90.0
        seq = PropellerBladeSequence(**params)

        points_per_blade = params['lines_per_blade'] * params['points_per_line']

        blade0_k_points = seq.kspace_points_rad_per_m[:, 0:points_per_blade]
        blade1_k_points = seq.kspace_points_rad_per_m[:, points_per_blade:2*points_per_blade]

        # Expected points for blade 1 are blade 0 points rotated by 90 degrees
        # Rotation matrix for 90 deg: [[0, -1], [1, 0]]
        rot_mat_90 = np.array([[0, -1], [1, 0]])
        expected_blade1_k_points = rot_mat_90 @ blade0_k_points

        np.testing.assert_array_almost_equal(blade1_k_points, expected_blade1_k_points, decimal=5)

    def test_invalid_parameters(self):
        invalid_params_gen = self.common_params.copy()
        invalid_params_gen['num_blades'] = 0
        with self.assertRaises(ValueError): # Error from generator
            PropellerBladeSequence(**invalid_params_gen)

        invalid_params_gen_2 = self.common_params.copy()
        invalid_params_gen_2['lines_per_blade'] = 0
        with self.assertRaises(ValueError): # Error from generator
            PropellerBladeSequence(**invalid_params_gen_2)

        invalid_params_gen_3 = self.common_params.copy()
        invalid_params_gen_3['points_per_line'] = 0
        with self.assertRaises(ValueError): # Error from generator
            PropellerBladeSequence(**invalid_params_gen_3)


if __name__ == '__main__':
    unittest.main()
