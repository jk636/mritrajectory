import unittest
import numpy as np
from trajgen.trajectory import Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T
from trajgen.optimizers.cost_components import (
    calculate_signal_decay_penalty,
    calculate_psf_incoherence_penalty,
    calculate_hardware_penalty, # Import for completeness, though not explicitly testing here
    calculate_gradient_roughness_penalty # Import for completeness
)
# from scipy.spatial import Voronoi, ConvexHull, QhullError # Not directly needed for these tests

class TestSignalDecayPenalty(unittest.TestCase):

    def setUp(self):
        self.dt = 4e-6  # 4 us
        self.gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']

        # Short trajectory: 5 points, duration (5-1)*4us = 16 us = 0.016 ms
        self.k_points_short = np.array([[0,1,2,3,4],[0,0,0,0,0]], dtype=float) * 10
        self.traj_short = Trajectory("short", self.k_points_short, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        # Long trajectory: 100 points, duration (100-1)*4us = 396 us = 0.396 ms
        self.k_points_long = np.array([np.arange(100), np.zeros(100)], dtype=float) * 10
        self.traj_long = Trajectory("long", self.k_points_long, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

    def test_basic_t2_decay(self):
        penalty_short_t2_50 = calculate_signal_decay_penalty(self.traj_short, T2_ms=50)
        penalty_long_t2_50 = calculate_signal_decay_penalty(self.traj_long, T2_ms=50)
        # Longer trajectory means more time for decay, so higher penalty (more signal lost)
        self.assertGreater(penalty_long_t2_50, penalty_short_t2_50)

        penalty_long_t2_25 = calculate_signal_decay_penalty(self.traj_long, T2_ms=25)
        # Shorter T2 means faster decay, so higher penalty for the same trajectory
        self.assertGreater(penalty_long_t2_25, penalty_long_t2_50)

        # Test with T2_ms <= 0
        penalty_zero_t2 = calculate_signal_decay_penalty(self.traj_long, T2_ms=0)
        self.assertEqual(penalty_zero_t2, 0.0)
        penalty_neg_t2 = calculate_signal_decay_penalty(self.traj_long, T2_ms=-10)
        self.assertEqual(penalty_neg_t2, 0.0)

    def test_t1_decay_behavior(self):
        # For 13C, T1 is critical
        penalty_13c_no_t1 = calculate_signal_decay_penalty(self.traj_long, T1_ms=None, T2_ms=50, is_13c=True)
        penalty_13c_long_t1 = calculate_signal_decay_penalty(self.traj_long, T1_ms=10000, T2_ms=50, is_13c=True) # T1 very long
        penalty_13c_short_t1 = calculate_signal_decay_penalty(self.traj_long, T1_ms=1, T2_ms=50, is_13c=True) # T1 very short (1ms)

        # Long T1 should have minimal effect beyond T2
        self.assertAlmostEqual(penalty_13c_long_t1, penalty_13c_no_t1, delta=penalty_13c_no_t1*0.1 + 1e-9) # Allow small diff
        # Short T1 should have significant effect
        self.assertGreater(penalty_13c_short_t1, penalty_13c_long_t1)

        # For 1H (is_13c=False), T1 usually less critical for short readouts
        penalty_1h_no_t1 = calculate_signal_decay_penalty(self.traj_long, T1_ms=None, T2_ms=50, is_13c=False)
        penalty_1h_with_t1 = calculate_signal_decay_penalty(self.traj_long, T1_ms=1000, T2_ms=50, is_13c=False) # Assume T1=1s
        # With T1=1000ms and traj_long acq time ~0.4ms, T1 effect should be small
        self.assertAlmostEqual(penalty_1h_with_t1, penalty_1h_no_t1, delta=penalty_1h_no_t1*0.05 + 1e-9)

    def test_b_value_decay(self):
        penalty_no_b = calculate_signal_decay_penalty(self.traj_long, T2_ms=80, b_value_s_per_mm2=0)
        penalty_with_b = calculate_signal_decay_penalty(self.traj_long, T2_ms=80, b_value_s_per_mm2=0.001)
        # With b-value, decay should be stronger, leading to a higher penalty
        self.assertGreater(penalty_with_b, penalty_no_b)

    def test_kspace_weighting_func(self):
        penalty_default_weight = calculate_signal_decay_penalty(self.traj_long, T2_ms=50)

        # Custom weighting: emphasize center of k-space (low radii) less for penalty
        # i.e., signal loss at high-k is penalized more by default (radii^2 or radii^3)
        # If we make weights uniform, penalty might change.
        def uniform_weight_func(radii):
            return np.ones_like(radii)

        penalty_uniform_weighted = calculate_signal_decay_penalty(self.traj_long, T2_ms=50, k_space_weighting_func=uniform_weight_func)
        # Default weights (radii^2) will heavily weight outer points. Uniform weights will average out.
        # Exact relation is complex, just check they are different if trajectory isn't trivial.
        if self.traj_long.kspace_points_rad_per_m.size > 0 and not np.allclose(np.linalg.norm(self.traj_long.kspace_points_rad_per_m, axis=0),0):
             self.assertNotAlmostEqual(penalty_default_weight, penalty_uniform_weighted, delta=1e-9)

    def test_time_threshold_penalty(self):
        # Trajectory duration: (50000-1)*4us = 199996 us = 199.996 ms
        k_very_long = np.array([np.arange(50000), np.zeros(50000)], dtype=float)
        traj_very_long = Trajectory("very_long", k_very_long, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        # T2=80ms -> time_limit_ms = 3*80 = 240ms. max_acq_time = 199.996ms. No time penalty.
        penalty_t2_80 = calculate_signal_decay_penalty(traj_very_long, T2_ms=80)

        # T2=50ms -> time_limit_ms = 3*50 = 150ms. max_acq_time = 199.996ms. Time penalty applies.
        penalty_t2_50 = calculate_signal_decay_penalty(traj_very_long, T2_ms=50)
        self.assertGreater(penalty_t2_50, penalty_t2_80) # Expect higher penalty due to T2 and time penalty

        # Check that the penalty difference is substantial, indicating time penalty triggered
        # Calculate signal_loss_penalty_sum for T2=50ms without time penalty to compare
        decay_t2_50 = np.exp(-(traj_very_long.get_acquisition_times_ms()) / 50.0)
        radii = np.linalg.norm(traj_very_long.kspace_points_rad_per_m, axis=0)
        weights = radii**2
        signal_loss_sum_t2_50 = np.sum((1.0 - decay_t2_50) * weights)

        self.assertGreater(penalty_t2_50, signal_loss_sum_t2_50 + 1.0, "Time penalty component seems missing or too small.")


class TestPSFIncoherencePenalty(unittest.TestCase):

    def setUp(self):
        np.random.seed(0) # for reproducibility
        self.dt = 4e-6
        self.gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
        self.k_max = 200.0 # rad/m

        # Uniform Grid (Cartesian-like)
        num_pts_axis = 10
        x = np.linspace(-self.k_max/2, self.k_max/2, num_pts_axis)
        xx, yy = np.meshgrid(x, x)
        self.k_uniform = np.vstack((xx.ravel(), yy.ravel()))
        self.traj_uniform = Trajectory("uniform", self.k_uniform, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        # Clustered Points
        num_cluster_pts = num_pts_axis * num_pts_axis // 2
        cluster1 = np.random.normal(loc=-self.k_max/4, scale=self.k_max/20, size=(2, num_cluster_pts))
        cluster2 = np.random.normal(loc=self.k_max/4, scale=self.k_max/20, size=(2, num_cluster_pts))
        self.k_clustered = np.hstack((cluster1, cluster2))
        self.traj_clustered = Trajectory("clustered", self.k_clustered, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        # Random Points (more incoherent)
        self.k_random = np.random.uniform(-self.k_max/1.5, self.k_max/1.5, size=(2, num_pts_axis*num_pts_axis)) # Wider spread
        self.traj_random = Trajectory("random", self.k_random, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

    def test_incoherence_penalty_variance(self):
        penalty_uniform = calculate_psf_incoherence_penalty(self.traj_uniform, k_max_rad_per_m=self.k_max)
        penalty_clustered = calculate_psf_incoherence_penalty(self.traj_clustered, k_max_rad_per_m=self.k_max)
        penalty_random = calculate_psf_incoherence_penalty(self.traj_random, k_max_rad_per_m=self.k_max)

        # Uniform should have very low cell variance (ideally zero if perfectly regular)
        # Clustered should have high cell variance
        # Random should be somewhere in between, ideally lower than clustered
        print(f"Uniform Penalty: {penalty_uniform}, Clustered: {penalty_clustered}, Random: {penalty_random}")
        self.assertLess(penalty_uniform, penalty_clustered, "Uniform grid should have lower incoherence penalty than clustered.")
        # Random might sometimes be more uniform than a simple grid if points are well-distributed by chance
        # but clustered should definitely be worse than random if clusters are distinct.
        self.assertLess(penalty_random, penalty_clustered * 0.8, "Random should generally be better than highly clustered, allow some margin.")


    def test_target_density_penalty(self):
        # A trajectory that is actually center heavy
        num_pts_vd = 100
        r_vd = np.random.normal(0, 0.15 * self.k_max, num_pts_vd) # Heavy at center
        phi_vd = np.random.uniform(0, 2*np.pi, num_pts_vd)
        k_vd_center = np.vstack((r_vd * np.cos(phi_vd), r_vd * np.sin(phi_vd)))
        k_vd_center = np.clip(k_vd_center, -self.k_max, self.k_max) # Ensure within bounds
        traj_vd_center = Trajectory("vd_center", k_vd_center, dt_seconds=self.dt)

        def center_heavy_density_target(radii_nd): # Higher density for smaller radii
            return np.exp(-(radii_nd/(0.2*self.k_max + 1e-9))**2)

        def edge_heavy_density_target(radii_nd): # Higher density for larger radii
            return 1.0 - np.exp(-(radii_nd/(0.2*self.k_max + 1e-9))**2)

        penalty_match = calculate_psf_incoherence_penalty(traj_vd_center, k_max_rad_per_m=self.k_max,
                                                          target_density_func=center_heavy_density_target)
        penalty_mismatch = calculate_psf_incoherence_penalty(traj_vd_center, k_max_rad_per_m=self.k_max,
                                                             target_density_func=edge_heavy_density_target)

        print(f"Density Match Penalty: {penalty_match}, Mismatch: {penalty_mismatch}")
        self.assertLess(penalty_match, penalty_mismatch, "Matched density profile should have lower penalty.")

    def test_empty_region_penalty(self):
        # Trajectory with a large hole in the center
        num_pts_hole = 50
        angles = np.linspace(0, 2*np.pi, num_pts_hole, endpoint=False)
        radii_outer = np.linspace(self.k_max*0.6, self.k_max*0.9, num_pts_hole) # Samples only at periphery
        kx_hole = radii_outer * np.cos(angles)
        ky_hole = radii_outer * np.sin(angles)
        k_with_hole = np.vstack((kx_hole, ky_hole))
        traj_with_hole = Trajectory("with_hole", k_with_hole, dt_seconds=self.dt)

        penalty_hole = calculate_psf_incoherence_penalty(traj_with_hole, k_max_rad_per_m=self.k_max)
        # self.traj_random should be more uniformly distributed and have less of an "empty region" penalty component
        penalty_no_hole_random = calculate_psf_incoherence_penalty(self.traj_random, k_max_rad_per_m=self.k_max)

        print(f"Hole Penalty: {penalty_hole}, No Hole (Random): {penalty_no_hole_random}")
        self.assertGreater(penalty_hole, penalty_no_hole_random, "Trajectory with a hole should have higher PSF penalty.")

    def test_edge_cases_few_points(self):
        k_few = self.k_uniform[:,:3] # Only 3 points (dim=2, need dim+2=4 for reliable Voronoi)
        traj_few = Trajectory("few", k_few, dt_seconds=self.dt)
        penalty_few = calculate_psf_incoherence_penalty(traj_few, k_max_rad_per_m=self.k_max)
        # Expect high fixed penalty due to insufficient points
        self.assertGreaterEqual(penalty_few, 100.0)

        k_very_few = self.k_uniform[:,:1] # Only 1 point
        traj_one = Trajectory("one", k_very_few, dt_seconds=self.dt)
        penalty_one = calculate_psf_incoherence_penalty(traj_one, k_max_rad_per_m=self.k_max)
        self.assertGreaterEqual(penalty_one, 100.0)

        # Test empty trajectory
        traj_empty = Trajectory("empty", np.empty((2,0)), dt_seconds=self.dt)
        penalty_empty = calculate_psf_incoherence_penalty(traj_empty, k_max_rad_per_m=self.k_max)
        self.assertGreaterEqual(penalty_empty, 100.0) # High fixed penalty

if __name__ == '__main__':
    unittest.main()
