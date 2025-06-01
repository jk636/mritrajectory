import unittest
import numpy as np
from trajgen.sequences.drunken_kooshball import DrunkenKooshballSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestDrunkenKooshballSequence(unittest.TestCase):

    def setUp(self):
        self.common_params = {
            'name': "test_drunken_kooshball",
            'fov_mm': (200.0, 200.0, 100.0), # Example 3D FOV
            'resolution_mm': (2.0, 2.0, 2.0), # Example 3D Res
            'num_points': 1024, # Fewer points for faster tests
            'dt_seconds': 4e-6,
            'base_spherical_spiral_turns': 5,
            'perturbation_amplitude_factor': 0.05,
            'density_sigma_factor': 0.3,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_smoothing_iterations': 1, # Keep low for tests
            'smoothing_kernel_size': 3  # Keep low and odd for tests
        }

    def test_creation_and_kspace_shape(self):
        sequence = DrunkenKooshballSequence(**self.common_params)
        self.assertIsInstance(sequence, DrunkenKooshballSequence)
        self.assertIsNotNone(sequence.kspace_points_rad_per_m)
        self.assertEqual(sequence.kspace_points_rad_per_m.shape, (3, self.common_params['num_points']))
        self.assertEqual(sequence.get_num_dimensions(), 3)

        self.assertEqual(sequence.sequence_params.get('base_spherical_spiral_turns'), self.common_params['base_spherical_spiral_turns'])
        self.assertEqual(sequence.base_spherical_spiral_turns, self.common_params['base_spherical_spiral_turns'])
        self.assertEqual(sequence.num_points, self.common_params['num_points'])

    def test_design_principle_methods(self):
        sequence = DrunkenKooshballSequence(**self.common_params)

        self.assertTrue(len(sequence.assess_kspace_coverage()) > 0)
        self.assertTrue(len(sequence.estimate_off_resonance_sensitivity()) > 0)
        self.assertTrue(len(sequence.assess_motion_robustness()) > 0)
        self.assertTrue(len(sequence.suggest_reconstruction_method()) > 0)

        limits_pass = {'max_grad_Tm_per_m': 0.100, 'max_slew_Tm_per_s_per_m': 500.0} # Generous
        self.assertTrue(sequence.check_gradient_limits(limits_pass))
        self.assertTrue(sequence.check_gradient_limits({})) # No limits

    def test_parameter_influence_perturbation_3d(self):
        params1 = self.common_params.copy()
        params1['perturbation_amplitude_factor'] = 0.01
        # Ensure some randomness by changing seed or just accepting slight chance of collision
        np.random.seed(0) # Set seed for reproducibility of this test case point
        seq1 = DrunkenKooshballSequence(**params1)
        k_points1 = seq1.kspace_points_rad_per_m

        params2 = self.common_params.copy()
        params2['perturbation_amplitude_factor'] = 0.3
        np.random.seed(1) # Different seed for second trajectory
        seq2 = DrunkenKooshballSequence(**params2)
        k_points2 = seq2.kspace_points_rad_per_m

        self.assertFalse(np.allclose(k_points1, k_points2, atol=1e-3),
                         "K-space trajectories with different perturbation factors should not be identical.")

    def test_parameter_influence_density_sigma_3d(self):
        params1 = self.common_params.copy()
        params1['density_sigma_factor'] = 0.1
        np.random.seed(10)
        seq1 = DrunkenKooshballSequence(**params1)
        k_points1 = seq1.kspace_points_rad_per_m

        params2 = self.common_params.copy()
        params2['density_sigma_factor'] = 0.8
        np.random.seed(11)
        seq2 = DrunkenKooshballSequence(**params2)
        k_points2 = seq2.kspace_points_rad_per_m

        self.assertFalse(np.allclose(k_points1, k_points2, atol=1e-3),
                         "K-space trajectories with different density_sigma_factors should not be identical.")

    def test_hardware_constraint_invocation_3d(self):
        params_with_hw = self.common_params.copy()
        params_with_hw['max_grad_Tm_per_m'] = 0.035  # T/m
        params_with_hw['max_slew_Tm_per_s_per_m'] = 120.0 # T/m/s
        params_with_hw['perturbation_amplitude_factor'] = 0.1 # Moderate perturbation
        params_with_hw['num_smoothing_iterations'] = 3

        sequence_hw = DrunkenKooshballSequence(**params_with_hw)
        self.assertIsNotNone(sequence_hw.kspace_points_rad_per_m)

        actual_grad = sequence_hw.get_max_grad_Tm()
        actual_slew = sequence_hw.get_max_slew_Tm_per_s()

        self.assertIsNotNone(actual_grad)
        self.assertIsNotNone(actual_slew)

        if actual_grad is not None:
             print(f"Test HW Constraints (3D): Actual Grad {actual_grad*1000:.2f} mT/m (Target for gen: {params_with_hw['max_grad_Tm_per_m']*1000:.2f} mT/m)")
        if actual_slew is not None:
             print(f"Test HW Constraints (3D): Actual Slew {actual_slew:.2f} T/m/s (Target for gen: {params_with_hw['max_slew_Tm_per_s_per_m']:.2f} T/m/s)")

        # Check if actual values are reasonably close to or below targets (allowing for some margin)
        if actual_grad is not None and params_with_hw['max_grad_Tm_per_m'] is not None:
            self.assertLessEqual(actual_grad, params_with_hw['max_grad_Tm_per_m'] * 1.15, "Actual gradient exceeds target by more than 15%") # Allow some leeway
        if actual_slew is not None and params_with_hw['max_slew_Tm_per_s_per_m'] is not None:
            self.assertLessEqual(actual_slew, params_with_hw['max_slew_Tm_per_s_per_m'] * 1.15, "Actual slew rate exceeds target by more than 15%")


    def test_invalid_parameters_3d(self):
        with self.assertRaisesRegex(ValueError, "num_points must be positive"):
            params_invalid = self.common_params.copy()
            params_invalid['num_points'] = 0
            DrunkenKooshballSequence(**params_invalid)

        with self.assertRaisesRegex(ValueError, "dt_seconds must be positive"):
            params_invalid = self.common_params.copy()
            params_invalid['dt_seconds'] = 0
            DrunkenKooshballSequence(**params_invalid)

        with self.assertRaisesRegex(ValueError, "smoothing_kernel_size must be positive and odd"):
            params_invalid_kernel = self.common_params.copy()
            params_invalid_kernel['smoothing_kernel_size'] = 2 # Even
            DrunkenKooshballSequence(**params_invalid_kernel)

        with self.assertRaisesRegex(ValueError, "smoothing_kernel_size must be positive and odd"):
            params_invalid_kernel_neg = self.common_params.copy()
            params_invalid_kernel_neg['smoothing_kernel_size'] = 0
            DrunkenKooshballSequence(**params_invalid_kernel_neg)

if __name__ == '__main__':
    unittest.main()
