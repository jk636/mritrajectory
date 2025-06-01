import unittest
import numpy as np

# Adjust path for imports if necessary
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf import kspace_validation

class TestKSpaceValidation(unittest.TestCase):

    def setUp(self):
        # 2D Data (Simple Square)
        self.traj_2d = np.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],  [0, 0],  [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ], dtype=float) # k-space units: m^-1

        # 3D Data (Simple Cube)
        self.traj_3d = np.array([
            [x,y,z] for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]
        ], dtype=float) # m^-1

        self.fov_2d = (0.2, 0.2) # 20cm
        self.matrix_2d = (10, 10) # For coverage/density checks

    def test_get_kspace_extents_2d(self):
        extents = kspace_validation.get_kspace_extents(self.traj_2d)
        np.testing.assert_array_equal(extents['k_min_per_axis'], np.array([-1., -1.]))
        np.testing.assert_array_equal(extents['k_max_per_axis'], np.array([1., 1.]))
        np.testing.assert_array_equal(extents['k_range_per_axis'], np.array([2., 2.]))
        np.testing.assert_array_equal(extents['k_center_per_axis'], np.array([0., 0.]))

    def test_get_kspace_extents_3d(self):
        extents = kspace_validation.get_kspace_extents(self.traj_3d)
        np.testing.assert_array_equal(extents['k_min_per_axis'], np.array([-0.5, -0.5, -0.5]))
        np.testing.assert_array_equal(extents['k_max_per_axis'], np.array([0.5, 0.5, 0.5]))
        np.testing.assert_array_equal(extents['k_range_per_axis'], np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(extents['k_center_per_axis'], np.array([0., 0., 0.]))

    def test_get_kspace_extents_single_point(self):
        traj_single = np.array([[0.1, 0.2, 0.3]])
        extents = kspace_validation.get_kspace_extents(traj_single)
        np.testing.assert_array_equal(extents['k_min_per_axis'], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(extents['k_max_per_axis'], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(extents['k_range_per_axis'], np.array([0., 0., 0.]))
        np.testing.assert_array_equal(extents['k_center_per_axis'], np.array([0.1, 0.2, 0.3]))

    def test_get_kspace_extents_empty(self):
        traj_empty = np.empty((0,3))
        extents = kspace_validation.get_kspace_extents(traj_empty)
        self.assertTrue(np.all(np.isnan(extents['k_min_per_axis'])))
        self.assertTrue(np.all(np.isnan(extents['k_max_per_axis'])))


    def test_check_kspace_coverage_binary_grid_full_hit(self):
        # Design trajectory to hit all cells of a small grid
        # Grid: 3x3. FOV= (0.3, 0.3)m -> dk = 1/0.3 m^-1.
        # k_max_grid = matrix_size * dk / 2 = 3 * (1/0.3) / 2 = 5 m^-1
        # k_coords needed: -dk, 0, dk relative to center of k-space.
        # If k-space center is (0,0), then k values are -1/0.3, 0, 1/0.3
        # Physical k values: -3.33, 0, 3.33
        # k_idx = k_phys * fov + matrix_size / 2
        # k_idx = k_phys * 0.3 + 1.5
        # For k_phys = -1/(2*dk) * dk = -0.5 * (matrix_size/fov), this is wrong
        # k_idx = k_phys * fov + matrix_size/2
        # k_idx = 0 -> k_phys = - (matrix_size/2) / fov = -1.5 / 0.3 = -5
        # k_idx = 1 -> k_phys = (1 - 1.5) / 0.3 = -0.5 / 0.3 = -1.66
        # k_idx = 2 -> k_phys = (2 - 1.5) / 0.3 = 0.5 / 0.3 = 1.66

        fov_small = (3.0, 3.0) # Large FOV means small k-space steps for grid
        matrix_small = (3, 3)
        dk_small = 1.0 / np.asarray(fov_small) # dk = [1/3, 1/3]
        # k_grid_centers = (np.arange(3) - 1) * dk_small[0] # Should be -dk, 0, dk
        # k_grid_centers = [-dk, 0, dk] = [-0.333, 0, 0.333] if center is 0
        # Using the formula: idx = k_phys * fov + matrix_size/2
        # To hit idx=0: k_phys_0 = (0 - 1.5) / 3 = -0.5
        # To hit idx=1: k_phys_1 = (1 - 1.5) / 3 = -1/6
        # To hit idx=2: k_phys_2 = (2 - 1.5) / 3 = 1/6
        # So, k-space points should be at these coordinates.
        k_vals_for_full_hit = np.array([-0.5, -1/6.0, 1/6.0])

        traj_full_hit = np.array([[x,y] for x in k_vals_for_full_hit for y in k_vals_for_full_hit])

        result = kspace_validation.check_kspace_coverage_binary_grid(traj_full_hit, fov_small, matrix_small)
        self.assertAlmostEqual(result['coverage_percentage'], 100.0)
        self.assertEqual(result['num_cells_hit'], 9)
        self.assertEqual(result['total_grid_cells'], 9)
        self.assertEqual(result['k_space_grid_binary'].shape, matrix_small)

    def test_check_kspace_coverage_partial_hit(self):
        # Trajectory that hits only center line of a 3x3 grid
        # To hit y_idx=1 (center row), ky_phys = (1-1.5)/fov_y
        fov = (3.0, 3.0); matrix = (3,3)
        ky_center = (1 - 1.5) / fov[1] # = -1/6
        traj_center_line = np.array([[-0.5, ky_center], [-1/6.0, ky_center], [1/6.0, ky_center]])
        result = kspace_validation.check_kspace_coverage_binary_grid(traj_center_line, fov, matrix)
        self.assertEqual(result['num_cells_hit'], 3)
        self.assertAlmostEqual(result['coverage_percentage'], (3/9.0)*100.0)

    def test_check_kspace_coverage_no_hit(self):
        # Trajectory far outside the FOV/grid mapping
        traj_no_hit = self.traj_2d + 100 # Shift all points far away
        result = kspace_validation.check_kspace_coverage_binary_grid(traj_no_hit, self.fov_2d, self.matrix_2d)
        self.assertEqual(result['num_cells_hit'], 0)
        self.assertAlmostEqual(result['coverage_percentage'], 0.0)

    def test_check_kspace_coverage_with_offset(self):
        # Trajectory is at (0,0), but k-space center is offset such that (0,0) maps outside
        k_center_offset = np.array([self.matrix_2d[0] * (1/self.fov_2d[0]), 0]) # Offset by one FOV in kx

        # A single k-space point at (0,0)
        traj_at_zero = np.array([[0.,0.]])
        # Without offset, it hits center. With offset, it should be shifted out.
        # k_traj_to_grid = traj_at_zero + k_center_offset = [k_fov_x, 0]
        # k_idx_x = k_fov_x * fov_x + matrix_x/2 = (matrix_x/fov_x)*fov_x + matrix_x/2 = matrix_x + matrix_x/2 = 1.5*matrix_x
        # This will be out of bounds.
        result = kspace_validation.check_kspace_coverage_binary_grid(
            traj_at_zero, self.fov_2d, self.matrix_2d, kspace_center_offset_m_inv=k_center_offset
        )
        self.assertEqual(result['num_cells_hit'], 0)


    def test_calculate_kspace_density_map_simple(self):
        # Simple 3-point trajectory on a 3x3 grid
        fov = (3.0,3.0); matrix = (3,3)
        k_val_center = (1-1.5)/fov[0] # -1/6
        traj = np.array([[k_val_center, k_val_center],  # Hits center cell (1,1)
                         [k_val_center, k_val_center],  # Hits center cell again
                         [(-0.5), k_val_center]]) # Hits cell (0,1)

        result = kspace_validation.calculate_kspace_density_map_simple(traj, fov, matrix)
        self.assertEqual(result['density_map'].shape, matrix)
        self.assertEqual(result['density_map'][1,1], 2) # Center cell hit twice
        self.assertEqual(result['density_map'][0,1], 1) # Other cell hit once
        self.assertEqual(result['min_density_overall'], 0) # Unhit cells
        self.assertEqual(result['max_density_overall'], 2)
        self.assertEqual(result['num_hit_cells'], 2)
        self.assertAlmostEqual(result['mean_density_in_hit_cells'], 1.5) # (2+1)/2
        self.assertAlmostEqual(result['std_density_in_hit_cells'], 0.5) # sqrt(((2-1.5)^2 + (1-1.5)^2)/2) = sqrt((0.25+0.25)/2)=sqrt(0.25)=0.5

    def test_analyze_kspace_point_distribution_basic(self):
        # Simple linear trajectory for easy checking
        traj = np.array([[float(i), float(i)*2] for i in range(5)]) # [[0,0],[1,2],[2,4],[3,6],[4,8]]
        result = kspace_validation.analyze_kspace_point_distribution_basic(traj, num_bins=2)

        self.assertIn('radii_stats', result)
        self.assertIn('consecutive_point_distance_stats', result)

        # Radii: sqrt(i^2 + (2i)^2) = sqrt(5i^2) = i*sqrt(5)
        # radii = [0, sqrt(5), 2sqrt(5), 3sqrt(5), 4sqrt(5)]
        # mean_radius = sqrt(5)*(0+1+2+3+4)/5 = sqrt(5)*2 = 4.47
        self.assertAlmostEqual(result['radii_stats']['mean'], np.sqrt(5)*2)
        self.assertEqual(len(result['radii_stats']['histogram_counts']), 2) # num_bins

        # Diffs: diffs are [[1,2],[1,2],[1,2],[1,2]]
        # Norms of diffs: [sqrt(5), sqrt(5), sqrt(5), sqrt(5)]
        self.assertAlmostEqual(result['consecutive_point_distance_stats']['mean'], np.sqrt(5))
        self.assertAlmostEqual(result['consecutive_point_distance_stats']['std'], 0) # All diffs are same
        self.assertEqual(len(result['consecutive_point_distance_stats']['histogram_counts']), 2)

        # Test with trajectory < 2 points
        traj_short = np.array([[1.,1.]])
        res_short = kspace_validation.analyze_kspace_point_distribution_basic(traj_short)
        self.assertTrue(np.isnan(res_short['consecutive_point_distance_stats']['mean']))


if __name__ == '__main__':
    unittest.main()
