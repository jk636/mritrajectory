import unittest
import numpy as np
import os
import sys

# Add the parent directory to the Python path to allow importing trajgen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajgen.generators import generate_spiral_trajectory, generate_radial_trajectory

class TestGenerateSpiralTrajectory(unittest.TestCase):
    def test_basic_2d_spiral(self):
        fov = (200, 200)
        res = (2, 2)
        num_il = 2
        pts_p_il = 256
        k_space = generate_spiral_trajectory(fov, res, num_dimensions=2,
                                             num_interleaves=num_il, points_per_interleaf=pts_p_il)
        self.assertEqual(k_space.shape, (2, num_il * pts_p_il))

        k_max_expected = 1.0 / (2.0 * (np.array(res[0]) / 1000.0))
        k_actual_max_dim0 = np.max(np.abs(k_space[0, :]))
        k_actual_max_dim1 = np.max(np.abs(k_space[1, :]))

        # Check if actual k_max is close to expected (within a tolerance for simplified spiral)
        self.assertAlmostEqual(k_actual_max_dim0, k_max_expected, delta=k_max_expected * 0.1)
        self.assertAlmostEqual(k_actual_max_dim1, k_max_expected, delta=k_max_expected * 0.1)
        self.assertTrue(np.all(np.isfinite(k_space)))

    def test_basic_3d_stack_of_spirals(self):
        fov = (200, 200, 100)
        res = (2, 2, 5)
        num_kz_slices = 8 # num_interleaves is num_kz_slices for 3D stack
        pts_p_slice = 128
        k_space = generate_spiral_trajectory(fov, res, num_dimensions=3,
                                             num_interleaves=num_kz_slices, points_per_interleaf=pts_p_slice)
        self.assertEqual(k_space.shape, (3, num_kz_slices * pts_p_slice))

        k_max_xy_expected = 1.0 / (2.0 * (res[0] / 1000.0))
        k_max_z_expected = 1.0 / (2.0 * (res[2] / 1000.0))

        self.assertAlmostEqual(np.max(np.abs(k_space[0, :])), k_max_xy_expected, delta=k_max_xy_expected * 0.1)
        self.assertAlmostEqual(np.max(np.abs(k_space[1, :])), k_max_xy_expected, delta=k_max_xy_expected * 0.1)
        self.assertAlmostEqual(np.max(np.abs(k_space[2, :])), k_max_z_expected, delta=k_max_z_expected * 0.1)

        # Check kz distribution
        unique_kz = np.unique(k_space[2, :])
        self.assertEqual(len(unique_kz), num_kz_slices)
        self.assertAlmostEqual(np.min(unique_kz), -k_max_z_expected, delta=k_max_z_expected * 0.05)
        self.assertAlmostEqual(np.max(unique_kz), k_max_z_expected, delta=k_max_z_expected * 0.05)
        self.assertTrue(np.all(np.isfinite(k_space)))

    def test_spiral_input_validation(self):
        with self.assertRaisesRegex(ValueError, "num_dimensions must be 2 or 3"):
            generate_spiral_trajectory(200, 2, num_dimensions=1)
        with self.assertRaisesRegex(ValueError, "fov_mm tuple length must match num_dimensions"):
            generate_spiral_trajectory((200,200,200), 2, num_dimensions=2)
        with self.assertRaisesRegex(ValueError, "resolution_mm tuple length must match num_dimensions"):
            generate_spiral_trajectory(200, (2,2,2), num_dimensions=2)
        with self.assertRaisesRegex(ValueError, "points_per_interleaf must be positive"):
            generate_spiral_trajectory(200, 2, points_per_interleaf=0)
        with self.assertRaisesRegex(ValueError, "num_interleaves must be positive"):
            generate_spiral_trajectory(200, 2, num_interleaves=0)
        with self.assertRaisesRegex(ValueError, "undersampling_factor must be positive"):
            generate_spiral_trajectory(200, 2, undersampling_factor=0)
        with self.assertRaisesRegex(NotImplementedError, "Golden angle spiral_type is not yet implemented"):
            generate_spiral_trajectory(200, 2, spiral_type='goldenangle')
        with self.assertRaisesRegex(ValueError, "Unknown spiral_type"):
            generate_spiral_trajectory(200, 2, spiral_type='unknown')

    def test_spiral_edge_cases(self):
        # 1 point per interleaf
        k_space = generate_spiral_trajectory(200, 2, num_interleaves=4, points_per_interleaf=1)
        self.assertEqual(k_space.shape, (2, 4*1))
        # For 1 point, it should be at k_max (or 0 depending on linspace endpoint=True for radius)
        # current_k_radius = np.linspace(0, k_max_xy, points_per_interleaf, endpoint=True)
        # So for 1 point, k_radius will be k_max_xy.
        k_max_expected = 1.0 / (2.0 * (2.0 / 1000.0))
        # Check magnitude of each point (kx,ky)
        for i in range(k_space.shape[1]):
            self.assertAlmostEqual(np.linalg.norm(k_space[:,i]), k_max_expected, delta=k_max_expected*0.01)

        # 1 interleaf
        k_space_single_il = generate_spiral_trajectory(200, 2, num_interleaves=1, points_per_interleaf=100)
        self.assertEqual(k_space_single_il.shape, (2, 100))


class TestGenerateRadialTrajectory(unittest.TestCase):
    def test_basic_2d_radial_golden_angle_normalized(self):
        num_spk = 32
        pts_p_spk = 64
        k_space = generate_radial_trajectory(num_spk, pts_p_spk, num_dimensions=2,
                                             projection_angle_increment='golden_angle')
        self.assertEqual(k_space.shape, (2, num_spk * pts_p_spk))
        k_max_expected_norm = 0.5 * pts_p_spk
        k_mags_actual = np.sqrt(k_space[0,:]**2 + k_space[1,:]**2)
        self.assertAlmostEqual(np.max(k_mags_actual), k_max_expected_norm, delta=k_max_expected_norm*0.01)
        self.assertTrue(np.all(np.isfinite(k_space)))

    def test_basic_2d_radial_fixed_angle_physical_units(self):
        num_spk = 8
        pts_p_spk = 100
        fov = 256
        res = 2.0 # mm
        k_space = generate_radial_trajectory(num_spk, pts_p_spk, num_dimensions=2,
                                             fov_mm=fov, resolution_mm=res,
                                             projection_angle_increment=45.0) # 360/8 = 45 degrees
        self.assertEqual(k_space.shape, (2, num_spk * pts_p_spk))
        k_max_expected_phys = 1.0 / (2.0 * (res / 1000.0))
        k_mags_actual = np.sqrt(k_space[0,:]**2 + k_space[1,:]**2)
        self.assertAlmostEqual(np.max(k_mags_actual), k_max_expected_phys, delta=k_max_expected_phys*0.01)

        # Check angles (crude check)
        # Spoke 0 should be along kx (angle 0)
        # Spoke 1 should be at 45 deg
        spoke0_end_point = k_space[:, pts_p_spk-1] # End point of first spoke
        spoke1_end_point = k_space[:, pts_p_spk + pts_p_spk-1] # End point of second spoke
        self.assertAlmostEqual(np.arctan2(spoke0_end_point[1], spoke0_end_point[0]), np.deg2rad(0), delta=1e-3)
        self.assertAlmostEqual(np.arctan2(spoke1_end_point[1], spoke1_end_point[0]), np.deg2rad(45), delta=1e-3)


    def test_basic_3d_radial_golden_angle(self):
        num_spk = 64
        pts_p_spk = 32
        fov = 200
        res = 4.0 #mm
        k_space = generate_radial_trajectory(num_spk, pts_p_spk, num_dimensions=3,
                                             fov_mm=fov, resolution_mm=res,
                                             projection_angle_increment='golden_angle')
        self.assertEqual(k_space.shape, (3, num_spk * pts_p_spk))
        k_max_expected_phys = 1.0 / (2.0 * (res / 1000.0))
        k_mags_actual = np.linalg.norm(k_space, axis=0)
        self.assertAlmostEqual(np.max(k_mags_actual), k_max_expected_phys, delta=k_max_expected_phys*0.01)
        self.assertTrue(np.all(np.isfinite(k_space)))
        # Check that points are reasonably spread on a sphere (not all on a plane)
        # A simple check: standard deviation of x, y, z for the spoke endpoints should be non-trivial
        # This isn't a perfect test of uniformity but can catch gross errors (e.g. all spokes on xy plane)
        endpoints = k_space[:, (np.arange(num_spk) * pts_p_spk) + pts_p_spk - 1]
        self.assertTrue(np.std(endpoints[0,:]) > k_max_expected_phys * 0.1) # Expect some spread in x
        self.assertTrue(np.std(endpoints[1,:]) > k_max_expected_phys * 0.1) # Expect some spread in y
        self.assertTrue(np.std(endpoints[2,:]) > k_max_expected_phys * 0.1) # Expect some spread in z


    def test_radial_input_validation(self):
        with self.assertRaisesRegex(ValueError, "num_dimensions must be 2 or 3"):
            generate_radial_trajectory(10, 10, num_dimensions=1)
        with self.assertRaisesRegex(ValueError, "num_spokes and points_per_spoke must be positive"):
            generate_radial_trajectory(0, 10)
        with self.assertRaisesRegex(ValueError, "num_spokes and points_per_spoke must be positive"):
            generate_radial_trajectory(10, 0)
        with self.assertRaisesRegex(ValueError, "projection_angle_increment must be 'golden_angle' or a number"):
            generate_radial_trajectory(10, 10, projection_angle_increment='invalid_str')
        with self.assertRaisesRegex(ValueError, "fov_mm and resolution_mm must be numbers or tuples/lists"):
            generate_radial_trajectory(10,10, fov_mm="bad", resolution_mm=2)
        with self.assertRaisesRegex(ValueError, "Both fov_mm and resolution_mm must be provided together, or neither."):
            generate_radial_trajectory(10,10, fov_mm=200, resolution_mm=None)
        with self.assertRaisesRegex(ValueError, "For 3D radial, projection_angle_increment must be 'golden_angle' or a list/tuple"):
            generate_radial_trajectory(10, 10, num_dimensions=3, projection_angle_increment=30.0)
        with self.assertRaisesRegex(ValueError, "Each element in projection_angle_increment list for 3D must be a .* pair"):
             generate_radial_trajectory(2,10, num_dimensions=3, projection_angle_increment=[(1,2,3),(0.1,0.2)])


    def test_radial_edge_cases(self):
        # 1 spoke, 1 point
        k_space = generate_radial_trajectory(1, 1, fov_mm=200, resolution_mm=2)
        self.assertEqual(k_space.shape, (2, 1))
        k_max_expected = 1.0 / (2.0 * (2.0/1000.0))
        # For 1 point, spoke_template_k = np.linspace(0, k_max_val, 1, endpoint=True) -> [k_max_val]
        self.assertAlmostEqual(np.linalg.norm(k_space[:,0]), k_max_expected, delta=1e-9)

        # 1 spoke, many points
        k_space_1sp = generate_radial_trajectory(1, 100, fov_mm=200, resolution_mm=2)
        self.assertEqual(k_space_1sp.shape, (2,100))
        # Should be along kx (angle 0)
        self.assertTrue(np.allclose(k_space_1sp[1,:], 0.0)) # All ky should be zero
        self.assertAlmostEqual(np.max(np.abs(k_space_1sp[0,:])), k_max_expected, delta=1e-9)

        # Many spokes, 1 point
        k_space_1pt = generate_radial_trajectory(100, 1, fov_mm=200, resolution_mm=2, projection_angle_increment=10)
        self.assertEqual(k_space_1pt.shape, (2,100))
        for i in range(100):
            self.assertAlmostEqual(np.linalg.norm(k_space_1pt[:,i]), k_max_expected, delta=1e-9)
            expected_angle_rad = np.deg2rad(i * 10.0)
            actual_angle_rad = np.arctan2(k_space_1pt[1,i], k_space_1pt[0,i])

            # Robust angle comparison:
            # Compare cosines and sines of the angles.
            # Ensure expected_angle_rad is correctly formed without unnecessary wrapping if not needed by function.
            # The current_angle in generator naturally wraps due to cos/sin, arctan2 brings it to (-pi,pi].
            # The expected_angle_rad in test should match this behavior.

            # Let expected_angle_rad be simply np.deg2rad(i * 10.0)
            # No, the generator uses current_angle which increments.
            # The test's expected_angle_rad calculation is likely correct.

            # The issue is that an angle of 0 and 2*pi are the same, but numerically different.
            # We can normalize both to [0, 2*pi) before comparison.
            actual_normalized = actual_angle_rad
            if actual_normalized < 0:
                actual_normalized += 2 * np.pi

            expected_normalized = expected_angle_rad % (2 * np.pi)
            if expected_normalized < 0: # Should not happen with i*10 deg
                expected_normalized += 2 * np.pi

            # If expected is 0, actual might be 2*pi or 0.
            if np.isclose(expected_normalized, 0.0, atol=1e-9) and \
               np.isclose(actual_normalized, 2 * np.pi, atol=1e-9):
                actual_normalized = 0.0
            elif np.isclose(actual_normalized, 0.0, atol=1e-9) and \
                 np.isclose(expected_normalized, 2 * np.pi, atol=1e-9):
                 expected_normalized = 0.0


            self.assertAlmostEqual(actual_normalized, expected_normalized, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
