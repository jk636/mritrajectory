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
            'smoothing_kernel_size': 3,  # Keep low and odd for tests
            'smoothness_emphasis_factor': None # Default
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
        self.assertIsNone(sequence.smoothness_emphasis_factor)
        self.assertIsNone(sequence.sequence_params.get('smoothness_emphasis_factor'))


    def test_design_principle_methods(self):
        sequence = DrunkenKooshballSequence(**self.common_params)

        self.assertTrue(len(sequence.assess_kspace_coverage()) > 0)
        self.assertTrue(len(sequence.estimate_off_resonance_sensitivity()) > 0)
        self.assertTrue(len(sequence.assess_motion_robustness()) > 0)
        self.assertTrue(len(sequence.suggest_reconstruction_method()) > 0)

        limits_pass = {'max_grad_Tm_per_m': 0.100, 'max_slew_Tm_per_s_per_m': 500.0} # Generous
        self.assertTrue(sequence.check_gradient_limits(limits_pass))
        self.assertTrue(sequence.check_gradient_limits({})) # No limits


    def test_smoothness_emphasis_factor_effect_3d(self):
        params_default = self.common_params.copy()
        params_default['perturbation_amplitude_factor'] = 0.15
        params_default['num_smoothing_iterations'] = 1
        params_default['max_grad_Tm_per_m'] = None
        params_default['max_slew_Tm_per_s_per_m'] = None

        np.random.seed(42)
        seq_default = DrunkenKooshballSequence(**params_default)
        k_default = seq_default.kspace_points_rad_per_m
        seq_default.dt_seconds = self.common_params['dt_seconds']
        slew_default = seq_default.get_max_slew_Tm_per_s()

        params_smooth = self.common_params.copy()
        params_smooth['perturbation_amplitude_factor'] = 0.15
        params_smooth['num_smoothing_iterations'] = 1
        params_smooth['smoothness_emphasis_factor'] = 1.0 # Max smoothness
        params_smooth['max_grad_Tm_per_m'] = None
        params_smooth['max_slew_Tm_per_s_per_m'] = None

        np.random.seed(42)
        seq_smooth = DrunkenKooshballSequence(**params_smooth)
        k_smooth = seq_smooth.kspace_points_rad_per_m
        seq_smooth.dt_seconds = self.common_params['dt_seconds']
        slew_smooth = seq_smooth.get_max_slew_Tm_per_s()

        self.assertFalse(np.allclose(k_default, k_smooth), "K-space should differ with smoothness factor.")

        # Check that perturbation was reduced (closer to ideal spherical spiral)
        t_base = np.linspace(0, 1, params_default['num_points'], endpoint=True)
        r_ideal = t_base * seq_default.metadata['k_max_rad_per_m']
        theta_ideal = np.arccos(1 - 2 * t_base)
        phi_ideal = np.linspace(0, params_default['base_spherical_spiral_turns'] * 2 * np.pi, params_default['num_points'], endpoint=True)
        kx_ideal = r_ideal * np.sin(theta_ideal) * np.cos(phi_ideal)
        ky_ideal = r_ideal * np.sin(theta_ideal) * np.sin(phi_ideal)
        kz_ideal = r_ideal * np.cos(theta_ideal)
        k_ideal = np.vstack((kx_ideal, ky_ideal, kz_ideal))

        diff_to_ideal_default = np.linalg.norm(k_default - k_ideal)
        diff_to_ideal_smooth = np.linalg.norm(k_smooth - k_ideal)
        self.assertLess(diff_to_ideal_smooth, diff_to_ideal_default * 0.5,
                        "Smoother 3D trajectory should be closer to ideal spherical spiral.")

        if slew_default is not None and slew_smooth is not None:
            self.assertLessEqual(slew_smooth, slew_default * 1.1,
                                 "Max slew for smoother 3D trajectory should ideally be lower or similar.")
            print(f"3D Default Slew: {slew_default:.2f}, Smooth Slew: {slew_smooth:.2f} (factor 1.0)")


    def test_parameter_influence_perturbation_3d(self):
        params1 = self.common_params.copy()
        params1['perturbation_amplitude_factor'] = 0.01
        params1['smoothness_emphasis_factor'] = 0.0
        np.random.seed(0)
        seq1 = DrunkenKooshballSequence(**params1)
        k_points1 = seq1.kspace_points_rad_per_m

        params2 = self.common_params.copy()
        params2['perturbation_amplitude_factor'] = 0.3
        params2['smoothness_emphasis_factor'] = 0.0
        np.random.seed(1)
        seq2 = DrunkenKooshballSequence(**params2)
        k_points2 = seq2.kspace_points_rad_per_m

        self.assertFalse(np.allclose(k_points1, k_points2, atol=1e-3),
                         "K-space trajectories with different perturbation factors should not be identical (smoothness_factor=0).")

    def test_parameter_influence_density_sigma_3d(self):
        params1 = self.common_params.copy()
        params1['density_sigma_factor'] = 0.1
        params1['smoothness_emphasis_factor'] = 0.0
        np.random.seed(10)
        seq1 = DrunkenKooshballSequence(**params1)
        k_points1 = seq1.kspace_points_rad_per_m

        params2 = self.common_params.copy()
        params2['density_sigma_factor'] = 0.8
        params2['smoothness_emphasis_factor'] = 0.0
        np.random.seed(11)
        seq2 = DrunkenKooshballSequence(**params2)
        k_points2 = seq2.kspace_points_rad_per_m

        self.assertFalse(np.allclose(k_points1, k_points2, atol=1e-3),
                         "K-space trajectories with different density_sigma_factors should not be identical (smoothness_factor=0).")

    def test_hardware_constraint_invocation_3d(self):
        params_with_hw = self.common_params.copy()
        params_with_hw['smoothness_emphasis_factor'] = 0.0 # Ensure base behavior for this test
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
