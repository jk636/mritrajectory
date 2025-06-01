import unittest
import numpy as np
from trajgen.sequences.drunken_spiral import DrunkenSpiralSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestDrunkenSpiralSequence(unittest.TestCase):

    def setUp(self):
        self.common_params = {
            'name': "test_drunken_spiral",
            'fov_mm': 200.0,
            'resolution_mm': 1.0,
            'num_points': 1024, # Keep it reasonable for tests
            'dt_seconds': 4e-6,
            'base_spiral_turns': 6,
            'perturbation_amplitude_factor': 0.05,
            'density_sigma_factor': 0.3,
            'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
            'num_smoothing_iterations': 1, # Fewer iterations for faster tests
            'smoothing_kernel_size': 3,   # Smaller kernel for faster tests
            'smoothness_emphasis_factor': None # Default
        }

    def test_creation_and_kspace_shape(self):
        sequence = DrunkenSpiralSequence(**self.common_params)
        self.assertIsInstance(sequence, DrunkenSpiralSequence)
        self.assertIsNotNone(sequence.kspace_points_rad_per_m)
        self.assertEqual(sequence.kspace_points_rad_per_m.shape, (2, self.common_params['num_points']))
        self.assertEqual(sequence.get_num_dimensions(), 2)

        self.assertEqual(sequence.sequence_params.get('base_spiral_turns'), self.common_params['base_spiral_turns'])
        self.assertEqual(sequence.base_spiral_turns, self.common_params['base_spiral_turns'])
        self.assertEqual(sequence.num_points, self.common_params['num_points'])
        self.assertIsNone(sequence.smoothness_emphasis_factor) # Check default
        self.assertIsNone(sequence.sequence_params.get('smoothness_emphasis_factor'))


    def test_design_principle_methods(self):
        sequence = DrunkenSpiralSequence(**self.common_params)

        self.assertTrue(len(sequence.assess_kspace_coverage()) > 0)
        self.assertTrue(len(sequence.estimate_off_resonance_sensitivity()) > 0)
        self.assertTrue(len(sequence.assess_motion_robustness()) > 0)
        self.assertTrue(len(sequence.suggest_reconstruction_method()) > 0)

        # Generous limits that should pass, assuming the trajectory isn't extremely aggressive
        # or if generator-side constraints (if any were passed to __init__) did their job.
        limits_pass = {'max_grad_Tm_per_m': 0.100, 'max_slew_Tm_per_s_per_m': 500.0} # 100 mT/m, 500 T/m/s
        self.assertTrue(sequence.check_gradient_limits(limits_pass))

        # Test with very strict limits that are likely to fail
        params_for_fail = self.common_params.copy()
        params_for_fail['perturbation_amplitude_factor'] = 0.3 # Higher perturbation
        params_for_fail['num_smoothing_iterations'] = 0    # No smoothing from sequence
        params_for_fail['smoothness_emphasis_factor'] = 0.0 # Ensure generator uses base smoothing iters (0 in this case)

        # Need to remove generator-level constraints for this test to purely rely on sequence-level check
        params_for_fail.pop('max_grad_Tm_per_m', None)
        params_for_fail.pop('max_slew_Tm_per_s_per_m', None)

        seq_for_fail = DrunkenSpiralSequence(**params_for_fail)

        actual_max_grad = seq_for_fail.get_max_grad_Tm()
        actual_max_slew = seq_for_fail.get_max_slew_Tm_per_s()

        if actual_max_grad is not None and actual_max_slew is not None:
            limits_fail_grad = {'max_grad_Tm_per_m': actual_max_grad * 0.5, 'max_slew_Tm_per_s_per_m': actual_max_slew * 2.0}
            if actual_max_grad > 1e-9: # Only if grad is significant
                 self.assertFalse(seq_for_fail.check_gradient_limits(limits_fail_grad))

            limits_fail_slew = {'max_grad_Tm_per_m': actual_max_grad * 2.0, 'max_slew_Tm_per_s_per_m': actual_max_slew * 0.5}
            if actual_max_slew > 1e-9: # Only if slew is significant
                 self.assertFalse(seq_for_fail.check_gradient_limits(limits_fail_slew))
        else:
            self.skipTest("Could not retrieve actual gradient/slew for failure condition test.")


    def test_smoothness_emphasis_factor_effect(self):
        params_default = self.common_params.copy()
        params_default['perturbation_amplitude_factor'] = 0.15 # Make it somewhat perturbed
        params_default['num_smoothing_iterations'] = 1      # Low base smoothing
        params_default['max_grad_Tm_per_m'] = None # Disable generator constraint attempt for cleaner test
        params_default['max_slew_Tm_per_s_per_m'] = None

        np.random.seed(42) # Ensure same base noise pattern
        seq_default = DrunkenSpiralSequence(**params_default)
        k_default = seq_default.kspace_points_rad_per_m
        # Ensure dt is set for slew calculation
        seq_default.dt_seconds = self.common_params['dt_seconds']
        slew_default = seq_default.get_max_slew_Tm_per_s()

        params_smooth = self.common_params.copy()
        params_smooth['perturbation_amplitude_factor'] = 0.15 # Same base perturbation
        params_smooth['num_smoothing_iterations'] = 1      # Same base smoothing
        params_smooth['smoothness_emphasis_factor'] = 1.0  # Max smoothness
        params_smooth['max_grad_Tm_per_m'] = None
        params_smooth['max_slew_Tm_per_s_per_m'] = None

        np.random.seed(42) # Ensure same base noise pattern
        seq_smooth = DrunkenSpiralSequence(**params_smooth)
        k_smooth = seq_smooth.kspace_points_rad_per_m
        seq_smooth.dt_seconds = self.common_params['dt_seconds']
        slew_smooth = seq_smooth.get_max_slew_Tm_per_s()

        self.assertFalse(np.allclose(k_default, k_smooth), "K-space should differ with smoothness factor.")

        # Check that perturbation was reduced with smoothness_emphasis_factor = 1.0
        # This means k_smooth should be closer to a non-perturbed spiral
        r_base = np.linspace(0, seq_default.metadata['k_max_rad_per_m'], params_default['num_points'], endpoint=True)
        theta_base = np.linspace(0, params_default['base_spiral_turns'] * 2 * np.pi, params_default['num_points'], endpoint=True)
        kx_ideal = r_base * np.cos(theta_base)
        ky_ideal = r_base * np.sin(theta_base)
        k_ideal = np.vstack((kx_ideal, ky_ideal))

        diff_to_ideal_default = np.linalg.norm(k_default - k_ideal)
        diff_to_ideal_smooth = np.linalg.norm(k_smooth - k_ideal)

        # With smoothness_emphasis_factor=1, perturbation should be minimal (min_perturb_factor)
        # So, k_smooth should be much closer to the ideal spiral.
        self.assertLess(diff_to_ideal_smooth, diff_to_ideal_default * 0.5, # Expect significantly less deviation
                        "Smoother trajectory should be closer to ideal spiral.")

        if slew_default is not None and slew_smooth is not None:
            # Increased smoothing iterations and reduced perturbation should generally lead to lower slew rates
            self.assertLessEqual(slew_smooth, slew_default * 1.1, # Allow 10% margin due to complex interactions
                                 "Max slew for smoother trajectory should ideally be lower or similar.")
            print(f"Default Slew: {slew_default:.2f}, Smooth Slew: {slew_smooth:.2f} (factor 1.0)")

        # Test with smoothness_emphasis_factor = 0.5
        params_mid_smooth = self.common_params.copy()
        params_mid_smooth['perturbation_amplitude_factor'] = 0.15
        params_mid_smooth['num_smoothing_iterations'] = 1
        params_mid_smooth['smoothness_emphasis_factor'] = 0.5
        params_mid_smooth['max_grad_Tm_per_m'] = None
        params_mid_smooth['max_slew_Tm_per_s_per_m'] = None
        np.random.seed(42)
        seq_mid_smooth = DrunkenSpiralSequence(**params_mid_smooth)
        k_mid_smooth = seq_mid_smooth.kspace_points_rad_per_m
        self.assertFalse(np.allclose(k_default, k_mid_smooth), "K-space should differ with smoothness factor 0.5.")
        self.assertFalse(np.allclose(k_smooth, k_mid_smooth), "K-space for factor 0.5 and 1.0 should differ.")


    def test_parameter_influence_perturbation(self):
        params1 = self.common_params.copy()
        params1['perturbation_amplitude_factor'] = 0.01
        params1['smoothness_emphasis_factor'] = 0.0 # Ensure base behavior
        seq1 = DrunkenSpiralSequence(**params1)
        np.random.seed(0)
        seq1 = DrunkenSpiralSequence(**params1)
        k_points1 = seq1.kspace_points_rad_per_m

        params2 = self.common_params.copy()
        params2['perturbation_amplitude_factor'] = 0.3 # Significantly different
        params2['smoothness_emphasis_factor'] = 0.0 # Ensure base behavior
        np.random.seed(1)
        seq2 = DrunkenSpiralSequence(**params2)
        k_points2 = seq2.kspace_points_rad_per_m

        diff = np.linalg.norm(k_points1 - k_points2)
        self.assertGreater(diff, 1e-3, "K-space trajectories with different perturbation factors should not be almost identical (with smoothness_factor=0).")

    def test_parameter_influence_density_sigma(self):
        params1 = self.common_params.copy()
        params1['density_sigma_factor'] = 0.1
        params1['smoothness_emphasis_factor'] = 0.0 # Ensure base behavior
        np.random.seed(10)
        seq1 = DrunkenSpiralSequence(**params1)
        k_points1 = seq1.kspace_points_rad_per_m

        params2 = self.common_params.copy()
        params2['density_sigma_factor'] = 0.8 # Significantly different
        params2['smoothness_emphasis_factor'] = 0.0 # Ensure base behavior
        np.random.seed(11)
        seq2 = DrunkenSpiralSequence(**params2)
        k_points2 = seq2.kspace_points_rad_per_m

        diff = np.linalg.norm(k_points1 - k_points2)
        self.assertGreater(diff, 1e-3, "K-space trajectories with different density_sigma_factors should not be almost identical (with smoothness_factor=0).")

    def test_hardware_constraint_invocation_in_generator(self):
        params_with_hw = self.common_params.copy()
        # These are target limits for the generator
        params_with_hw['smoothness_emphasis_factor'] = 0.0 # Ensure base behavior for this test
        params_with_hw['max_grad_Tm_per_m'] = 0.030  # 30 mT/m
        params_with_hw['max_slew_Tm_per_s_per_m'] = 100.0 # 100 T/m/s
        params_with_hw['perturbation_amplitude_factor'] = 0.15 # Make it a bit wild
        params_with_hw['num_smoothing_iterations'] = 5 # More iterations for generator to try

        sequence_hw = DrunkenSpiralSequence(**params_with_hw)
        self.assertIsNotNone(sequence_hw.kspace_points_rad_per_m) # Check it runs

        # Check that actual performance is somewhat related to the targets
        # The generator prints warnings if not met, this test checks if values are populated.
        actual_grad = sequence_hw.get_max_grad_Tm()
        actual_slew = sequence_hw.get_max_slew_Tm_per_s()

        self.assertIsNotNone(actual_grad)
        self.assertIsNotNone(actual_slew)

        # We expect the actual performance to be AT MOST the target, ideally.
        # But due to discrete nature and smoothing, it might slightly exceed or be well below.
        # A loose check: it shouldn't be drastically higher than a run without constraints.
        if actual_grad is not None:
             print(f"Test HW Constraints: Actual Grad {actual_grad*1000:.2f} mT/m (Target for gen: {params_with_hw['max_grad_Tm_per_m']*1000:.2f} mT/m)")
        if actual_slew is not None:
             print(f"Test HW Constraints: Actual Slew {actual_slew:.2f} T/m/s (Target for gen: {params_with_hw['max_slew_Tm_per_s_per_m']:.2f} T/m/s)")

        # A more robust test might involve comparing to a sequence generated with very loose/no limits.
        # For now, this confirms the parameters are passed and the sequence is generated.

    def test_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "num_points must be positive"):
            params_invalid = self.common_params.copy()
            params_invalid['num_points'] = 0
            DrunkenSpiralSequence(**params_invalid)

        with self.assertRaisesRegex(ValueError, "dt_seconds must be positive"):
            params_invalid = self.common_params.copy()
            params_invalid['dt_seconds'] = 0
            DrunkenSpiralSequence(**params_invalid)

        with self.assertRaisesRegex(ValueError, "smoothing_kernel_size must be positive and odd"):
            params_invalid_kernel = self.common_params.copy()
            params_invalid_kernel['smoothing_kernel_size'] = 2 # Even
            DrunkenSpiralSequence(**params_invalid_kernel)

        with self.assertRaisesRegex(ValueError, "smoothing_kernel_size must be positive and odd"):
            params_invalid_kernel_neg = self.common_params.copy()
            params_invalid_kernel_neg['smoothing_kernel_size'] = -3
            DrunkenSpiralSequence(**params_invalid_kernel_neg)

if __name__ == '__main__':
    unittest.main()
