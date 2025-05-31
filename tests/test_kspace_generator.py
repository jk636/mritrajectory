import unittest
import numpy as np
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajgen.kspace_generator import KSpaceTrajectoryGenerator
from trajgen.trajectory import Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T

class TestKSpaceTrajectoryGeneratorInit(unittest.TestCase):
    def test_basic_instantiation(self):
        gen = KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=1, num_dimensions=2, dt_s=4e-6)
        self.assertIsInstance(gen, KSpaceTrajectoryGenerator)
        self.assertEqual(gen.num_dimensions, 2)
        self.assertEqual(gen.fov_mm, (200.0, 200.0))
        self.assertEqual(gen.resolution_mm, (1.0, 1.0))
        self.assertEqual(gen.dt_s, 4e-6)

    def test_instantiation_3d(self):
        gen = KSpaceTrajectoryGenerator(fov_mm=(200,200,100), resolution_mm=(1,1,2), num_dimensions=3, dt_s=4e-6)
        self.assertEqual(gen.num_dimensions, 3)
        self.assertEqual(gen.fov_mm, (200.0, 200.0, 100.0))
        self.assertEqual(gen.resolution_mm, (1.0, 1.0, 2.0))

    def test_instantiation_with_constraints(self):
        gen = KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=1, num_dimensions=2, dt_s=4e-6,
                                        max_grad_mT_per_m=40.0, max_slew_Tm_per_s_ms=150.0)
        self.assertEqual(gen.max_grad_mT_per_m, 40.0)
        self.assertEqual(gen.max_slew_Tm_per_s_ms, 150.0)

    def test_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "num_dimensions must be 2 or 3"):
            KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=1, num_dimensions=1, dt_s=4e-6)

    def test_fov_resolution_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "fov_mm must be a number or a tuple/list of length 3"):
            KSpaceTrajectoryGenerator(fov_mm=(200,200), resolution_mm=1, num_dimensions=3, dt_s=4e-6)
        with self.assertRaisesRegex(ValueError, "resolution_mm must be a number or a tuple/list of length 3"):
            KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=(1,1), num_dimensions=3, dt_s=4e-6)

    def test_invalid_dt(self):
        with self.assertRaisesRegex(ValueError, "dt_s must be positive"):
            KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=1, num_dimensions=2, dt_s=0)


class TestKSpaceTrajectoryGeneratorCreateSpiral(unittest.TestCase):
    def setUp(self):
        self.gen_2d = KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=2, num_dimensions=2, dt_s=4e-6,
                                                max_grad_mT_per_m=40, max_slew_Tm_per_s_ms=150)
        self.gen_3d = KSpaceTrajectoryGenerator(fov_mm=(200,200,100), resolution_mm=(2,2,4), num_dimensions=3, dt_s=4e-6,
                                                max_grad_mT_per_m=40, max_slew_Tm_per_s_ms=150)

    def test_create_spiral_2d(self):
        num_il, pts_p_il = 4, 256
        traj = self.gen_2d.create_spiral(num_interleaves=num_il, points_per_interleaf=pts_p_il, apply_constraints=False)
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 2)
        self.assertEqual(traj.get_num_points(), num_il * pts_p_il)
        self.assertEqual(traj.dt_seconds, self.gen_2d.dt_s)
        self.assertEqual(traj.metadata['gamma_Hz_per_T'], self.gen_2d.gamma_Hz_per_T)
        self.assertEqual(traj.metadata['num_interleaves'], num_il)
        self.assertFalse(traj.metadata['constraints_actually_applied'])

    def test_create_spiral_3d(self):
        num_il, pts_p_il = 8, 128 # num_il is num_kz_slices for 3D stack-of-spirals
        traj = self.gen_3d.create_spiral(num_interleaves=num_il, points_per_interleaf=pts_p_il, apply_constraints=False)
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 3)
        self.assertEqual(traj.get_num_points(), num_il * pts_p_il)
        self.assertFalse(traj.metadata['constraints_actually_applied'])

    def test_create_spiral_2d_with_constraints(self):
        # This test primarily checks if it runs and if the flag is set.
        # Detailed constraint checking is in TestConstrainTrajectory.
        num_il, pts_p_il = 2, 128
        traj = self.gen_2d.create_spiral(num_interleaves=num_il, points_per_interleaf=pts_p_il, apply_constraints=True)
        self.assertIsInstance(traj, Trajectory)
        self.assertTrue(traj.metadata['constraints_actually_applied'])
        # Check if constraints are somewhat effective (max grad/slew should be limited)
        # These values are rough checks and depend on the constrain_trajectory implementation details.
        if traj.get_max_grad_Tm() is not None: # May be None if only 1 point
             self.assertLessEqual(traj.get_max_grad_Tm(), self.gen_2d.max_grad_mT_per_m / 1000.0 * 1.2) # Allow 20% margin
        if traj.get_max_slew_Tm_per_s() is not None:
             self.assertLessEqual(traj.get_max_slew_Tm_per_s(), self.gen_2d.max_slew_Tm_per_s_ms * 1000.0 * 1.2) # Allow 20% margin

    def test_create_spiral_no_generator_constraints(self):
        gen_no_constr = KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=2, num_dimensions=2, dt_s=4e-6)
        traj = gen_no_constr.create_spiral(num_interleaves=2, points_per_interleaf=128, apply_constraints=True)
        self.assertIsInstance(traj, Trajectory)
        self.assertFalse(traj.metadata['constraints_actually_applied']) # No constraints defined in generator

    def test_create_variable_density_spiral(self):
        num_il, pts_p_il = 1, 1024
        dtrf = 0.2 # density_transition_radius_factor
        dfac = 2.0 # density_factor_at_center

        traj_vd = self.gen_2d.create_spiral(
            num_interleaves=num_il,
            points_per_interleaf=pts_p_il,
            density_transition_radius_factor=dtrf,
            density_factor_at_center=dfac,
            apply_constraints=False
        )
        self.assertIsInstance(traj_vd, Trajectory)
        self.assertEqual(traj_vd.metadata['density_transition_radius_factor'], dtrf)
        self.assertEqual(traj_vd.metadata['density_factor_at_center'], dfac)

        # Qualitative check: more points near the center for VD spiral
        # Compare angular displacement per point at center vs periphery
        k_points_vd = traj_vd.kspace_points_rad_per_m
        angles_vd = np.unwrap(np.arctan2(k_points_vd[1,:], k_points_vd[0,:]))
        radii_vd = np.sqrt(k_points_vd[0,:]**2 + k_points_vd[1,:]**2)

        # Define center region (e.g., first 20% of points, assuming they are mostly central)
        # and peripheral region (e.g., last 20% of points)
        center_slice_vd = slice(0, pts_p_il // 5)
        periphery_slice_vd = slice(-pts_p_il // 5, None)

        avg_d_angle_center_vd = np.mean(np.abs(np.diff(angles_vd[center_slice_vd])))
        avg_d_angle_periphery_vd = np.mean(np.abs(np.diff(angles_vd[periphery_slice_vd])))

        # For denser center, angular displacement per point should be smaller in the center
        self.assertLess(avg_d_angle_center_vd, avg_d_angle_periphery_vd * 0.8, # Expect at least 20% smaller for DFC=2
                        "Angular step at center not significantly smaller for VD spiral.")

        # Check k_max is still reached
        k_nyquist = 1.0 / (2.0 * np.min(np.array(self.gen_2d.resolution_mm) / 1000.0))
        self.assertAlmostEqual(np.max(radii_vd), k_nyquist, delta=k_nyquist*0.05,
                               msg="VD spiral does not reach expected k_max.")

    def test_create_spiral_uniform_density_explicit(self):
        # Test that DFC=1.0 results in uniform-like behavior compared to no VD params
        num_il, pts_p_il = 1, 512
        traj_uniform_ref = self.gen_2d.create_spiral( # Standard uniform
            num_interleaves=num_il, points_per_interleaf=pts_p_il, apply_constraints=False
        )
        traj_vd_uniform = self.gen_2d.create_spiral( # VD params but DFC=1.0
            num_interleaves=num_il, points_per_interleaf=pts_p_il,
            density_transition_radius_factor=0.2,
            density_factor_at_center=1.0,
            apply_constraints=False
        )
        # The point distributions should be very similar (though not necessarily identical due to float precision)
        # A simple check on mean angular displacement diffs
        angles_ref = np.unwrap(np.arctan2(traj_uniform_ref.kspace_points_rad_per_m[1,:], traj_uniform_ref.kspace_points_rad_per_m[0,:]))
        angles_vd_uni = np.unwrap(np.arctan2(traj_vd_uniform.kspace_points_rad_per_m[1,:], traj_vd_uniform.kspace_points_rad_per_m[0,:]))

        avg_d_angle_ref = np.mean(np.abs(np.diff(angles_ref)))
        avg_d_angle_vd_uni = np.mean(np.abs(np.diff(angles_vd_uni)))
        self.assertAlmostEqual(avg_d_angle_ref, avg_d_angle_vd_uni, delta=avg_d_angle_ref*0.01,
                               msg="Spiral with DFC=1.0 differs significantly from standard uniform spiral.")


class TestKSpaceTrajectoryGeneratorCreateRadial(unittest.TestCase):
    def setUp(self):
        self.gen_2d = KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=2, num_dimensions=2, dt_s=4e-6,
                                                max_grad_mT_per_m=40, max_slew_Tm_per_s_ms=150)
        self.gen_3d = KSpaceTrajectoryGenerator(fov_mm=(200,200,100), resolution_mm=(2,2,4), num_dimensions=3, dt_s=4e-6,
                                                max_grad_mT_per_m=40, max_slew_Tm_per_s_ms=150)

    def test_create_radial_2d(self):
        num_spk, pts_p_spk = 32, 128
        traj = self.gen_2d.create_radial(num_spokes=num_spk, points_per_spoke=pts_p_spk, apply_constraints=False)
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 2)
        self.assertEqual(traj.get_num_points(), num_spk * pts_p_spk)
        self.assertEqual(traj.dt_seconds, self.gen_2d.dt_s)
        self.assertEqual(traj.metadata['num_spokes'], num_spk)
        self.assertFalse(traj.metadata['constraints_actually_applied'])

    def test_create_radial_3d(self):
        num_spk, pts_p_spk = 64, 64
        traj = self.gen_3d.create_radial(num_spokes=num_spk, points_per_spoke=pts_p_spk, apply_constraints=False)
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 3)
        self.assertEqual(traj.get_num_points(), num_spk * pts_p_spk)
        self.assertFalse(traj.metadata['constraints_actually_applied'])

    def test_create_radial_2d_with_constraints(self):
        num_spk, pts_p_spk = 16, 64
        traj = self.gen_2d.create_radial(num_spokes=num_spk, points_per_spoke=pts_p_spk, apply_constraints=True)
        self.assertIsInstance(traj, Trajectory)
        self.assertTrue(traj.metadata['constraints_actually_applied'])
        if traj.get_max_grad_Tm() is not None:
             self.assertLessEqual(traj.get_max_grad_Tm(), self.gen_2d.max_grad_mT_per_m / 1000.0 * 1.2)
        if traj.get_max_slew_Tm_per_s() is not None:
             self.assertLessEqual(traj.get_max_slew_Tm_per_s(), self.gen_2d.max_slew_Tm_per_s_ms * 1000.0 * 1.2)

    def test_get_set_params(self):
        gen = KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=1, num_dimensions=2, dt_s=4e-6)
        params = gen.get_params()
        self.assertEqual(params["fov_mm"], (200.0, 200.0))

        with self.assertRaises(AttributeError): # Test setting a non-existent param
            gen.set_params(invalid_param=123)
        with self.assertRaises(ValueError): # Test setting a protected param
            gen.set_params(num_dimensions=3)

        gen.set_params(dt_s=5e-6, max_grad_mT_per_m=50)
        self.assertEqual(gen.dt_s, 5e-6)
        self.assertEqual(gen.max_grad_mT_per_m, 50)


class TestKSpaceTrajectoryGeneratorCreateCones(unittest.TestCase):
    def setUp(self):
        self.gen_3d_rad = KSpaceTrajectoryGenerator(fov_mm=(200,200,150), resolution_mm=(2,2,3), num_dimensions=3, dt_s=4e-6,
                                                max_grad_mT_per_m=40, max_slew_Tm_per_s_ms=150)
        self.gen_2d_rad = KSpaceTrajectoryGenerator(fov_mm=200, resolution_mm=2, num_dimensions=2, dt_s=4e-6)


    def test_create_cones_basic_3d(self):
        num_cones, pts_p_cone, cone_angle = 8, 256, 30.0
        traj = self.gen_3d_rad.create_cones_trajectory(
            num_cones=num_cones,
            points_per_cone=pts_p_cone,
            cone_angle_deg=cone_angle,
            apply_constraints=False
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 3)
        self.assertEqual(traj.get_num_points(), num_cones * pts_p_cone)
        self.assertEqual(traj.dt_seconds, self.gen_3d_rad.dt_s)
        self.assertEqual(traj.metadata['num_cones'], num_cones)
        self.assertEqual(traj.metadata['cone_angle_deg'], cone_angle)
        self.assertFalse(traj.metadata['constraints_actually_applied'])

        # Check k-space extents (approximate)
        k_max_sphere = np.min(1.0 / (2.0 * (np.array(self.gen_3d_rad.resolution_mm) / 1000.0)))

        kz_coords = traj.kspace_points_rad_per_m[2,:]
        # Max kz should be k_max_sphere * cos(cone_angle)
        # Min kz should be 0 if cones point one way, or -k_max_sphere * cos(cone_angle) if they point both ways
        # Current implementation of generate_cones_trajectory makes kz positive (spiraling "out" from origin along cone)
        self.assertLessEqual(np.max(np.abs(kz_coords)), k_max_sphere * np.cos(np.deg2rad(cone_angle)) * 1.05) # allow 5% due to discrete points
        # For points_per_cone > 1, linspace for radius goes from 0 to k_max_on_cone_surface.
        # So min kz should be near 0 for points close to apex.
        self.assertGreaterEqual(np.min(np.abs(kz_coords)), 0.0)


        kx_coords = traj.kspace_points_rad_per_m[0,:]
        ky_coords = traj.kspace_points_rad_per_m[1,:]
        kxy_mag = np.sqrt(kx_coords**2 + ky_coords**2)
        # Max kxy_mag should be k_max_sphere * sin(cone_angle)
        self.assertLessEqual(np.max(kxy_mag), k_max_sphere * np.sin(np.deg2rad(cone_angle)) * 1.05)


    def test_create_cones_with_constraints(self):
        num_cones, pts_p_cone, cone_angle = 4, 128, 45.0
        traj = self.gen_3d_rad.create_cones_trajectory(
            num_cones=num_cones,
            points_per_cone=pts_p_cone,
            cone_angle_deg=cone_angle,
            apply_constraints=True
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertTrue(traj.metadata['constraints_actually_applied'])
        if traj.get_max_grad_Tm() is not None:
            self.assertLessEqual(traj.get_max_grad_Tm(), self.gen_3d_rad.max_grad_mT_per_m / 1000.0 * 1.2)
        if traj.get_max_slew_Tm_per_s() is not None:
            self.assertLessEqual(traj.get_max_slew_Tm_per_s(), self.gen_3d_rad.max_slew_Tm_per_s_ms * 1000.0 * 1.2)

    def test_create_cones_on_2d_generator_fail(self):
        with self.assertRaisesRegex(ValueError, "Cones trajectory is inherently 3D"):
            self.gen_2d_rad.create_cones_trajectory(num_cones=4, points_per_cone=128, cone_angle_deg=30)

    def test_invalid_cone_params_in_generator_call(self):
        # These are tested in generate_cones_trajectory, but ensure KSpaceTrajectoryGenerator passes them through
        with self.assertRaisesRegex(ValueError, "cone_angle_deg must be between 0 and 90 degrees"):
             self.gen_3d_rad.create_cones_trajectory(num_cones=4, points_per_cone=128, cone_angle_deg=0)
        with self.assertRaisesRegex(ValueError, "cone_angle_deg must be between 0 and 90 degrees"):
             self.gen_3d_rad.create_cones_trajectory(num_cones=4, points_per_cone=128, cone_angle_deg=90)
        with self.assertRaisesRegex(ValueError, "num_cones and points_per_cone must be positive"):
             self.gen_3d_rad.create_cones_trajectory(num_cones=0, points_per_cone=128, cone_angle_deg=30)


if __name__ == '__main__':
    unittest.main()


class TestKSpaceTrajectoryGeneratorCreateRosette(unittest.TestCase):
    def setUp(self):
        self.gen_2d = KSpaceTrajectoryGenerator(
            fov_mm=200,
            resolution_mm=2,
            num_dimensions=2,
            dt_s=4e-6,
            max_grad_mT_per_m=40,
            max_slew_Tm_per_s_ms=150
        )
        self.gen_3d = KSpaceTrajectoryGenerator(
            fov_mm=(200,200,100),
            resolution_mm=(2,2,4),
            num_dimensions=3,
            dt_s=4e-6
        )

    def test_create_rosette_basic(self):
        num_petals, total_pts, num_radial_cycles = 5, 1024, 3
        k_max_factor = 0.8
        traj = self.gen_2d.create_rosette_trajectory(
            num_petals=num_petals,
            total_points=total_pts,
            num_radial_cycles=num_radial_cycles,
            k_max_rosette_factor=k_max_factor,
            apply_constraints=False
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 2)
        self.assertEqual(traj.get_num_points(), total_pts)
        self.assertEqual(traj.metadata['num_petals'], num_petals)
        self.assertEqual(traj.metadata['num_radial_cycles'], num_radial_cycles)
        self.assertEqual(traj.metadata['k_max_rosette_factor'], k_max_factor)
        self.assertFalse(traj.metadata['constraints_actually_applied'])

        # Check k-space extent
        k_nyquist = 1.0 / (2.0 * np.min(np.array(self.gen_2d.resolution_mm) / 1000.0))
        expected_k_max = k_nyquist * k_max_factor

        k_magnitudes = np.sqrt(traj.kspace_points_rad_per_m[0,:]**2 + traj.kspace_points_rad_per_m[1,:]**2)
        self.assertLessEqual(np.max(k_magnitudes), expected_k_max * 1.01) # Allow 1% tolerance

    def test_create_rosette_with_constraints(self):
        num_petals, total_pts, num_radial_cycles = 3, 512, 2
        traj = self.gen_2d.create_rosette_trajectory(
            num_petals=num_petals,
            total_points=total_pts,
            num_radial_cycles=num_radial_cycles,
            apply_constraints=True
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertTrue(traj.metadata['constraints_actually_applied'])
        if traj.get_max_grad_Tm() is not None:
            self.assertLessEqual(traj.get_max_grad_Tm(), self.gen_2d.max_grad_mT_per_m / 1000.0 * 1.2)
        if traj.get_max_slew_Tm_per_s() is not None:
            self.assertLessEqual(traj.get_max_slew_Tm_per_s(), self.gen_2d.max_slew_Tm_per_s_ms * 1000.0 * 1.2)

    def test_create_rosette_on_3d_generator_fail(self):
        with self.assertRaisesRegex(ValueError, "Rosette trajectory is 2D"):
            self.gen_3d.create_rosette_trajectory(
                num_petals=5, total_points=1024, num_radial_cycles=3
            )

    def test_invalid_rosette_params_in_generator_call(self):
        # These are tested in generate_rosette_trajectory, but ensure KSpaceTrajectoryGenerator passes them through
        with self.assertRaisesRegex(ValueError, "num_petals must be positive"):
             self.gen_2d.create_rosette_trajectory(num_petals=0, total_points=512, num_radial_cycles=3)
        with self.assertRaisesRegex(ValueError, "total_points must be positive"):
             self.gen_2d.create_rosette_trajectory(num_petals=5, total_points=0, num_radial_cycles=3)
        with self.assertRaisesRegex(ValueError, "num_radial_cycles must be positive"):
             self.gen_2d.create_rosette_trajectory(num_petals=5, total_points=512, num_radial_cycles=0)
        with self.assertRaisesRegex(ValueError, "k_max_rosette_factor must be between 0 .* and 1"):
             self.gen_2d.create_rosette_trajectory(num_petals=5, total_points=512, num_radial_cycles=3, k_max_rosette_factor=0)
        with self.assertRaisesRegex(ValueError, "k_max_rosette_factor must be between 0 .* and 1"):
             self.gen_2d.create_rosette_trajectory(num_petals=5, total_points=512, num_radial_cycles=3, k_max_rosette_factor=1.1)


if __name__ == '__main__':
    unittest.main()


class TestKSpaceTrajectoryGeneratorCreateEPI(unittest.TestCase):
    def setUp(self):
        self.gen_2d = KSpaceTrajectoryGenerator(
            fov_mm=256,
            resolution_mm=2,
            num_dimensions=2,
            dt_s=4e-6,
            max_grad_mT_per_m=40,
            max_slew_Tm_per_s_ms=150
        )
        self.gen_3d = KSpaceTrajectoryGenerator(
            fov_mm=(200,200,100),
            resolution_mm=(2,2,4),
            num_dimensions=3,
            dt_s=4e-6
        )

    def test_create_epi_flyback_y_phase(self):
        num_echoes, pts_p_echo = 64, 128
        traj = self.gen_2d.create_epi_trajectory(
            num_echoes=num_echoes,
            points_per_echo=pts_p_echo,
            epi_type='flyback',
            phase_encode_direction='y',
            apply_constraints=False
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 2)
        self.assertEqual(traj.get_num_points(), num_echoes * pts_p_echo)
        self.assertEqual(traj.metadata['epi_type'], 'flyback')
        self.assertEqual(traj.metadata['phase_encode_direction'], 'y')
        self.assertFalse(traj.metadata['constraints_actually_applied'])

        # Check k-space coverage pattern (kx readout, ky phase)
        # kx should go from -kmax to +kmax for each line
        # ky should be constant for each line and step between lines
        k_max_readout = 1.0 / (2.0 * (self.gen_2d.resolution_mm[0] / 1000.0))
        kx_values = traj.kspace_points_rad_per_m[0, :]
        ky_values = traj.kspace_points_rad_per_m[1, :]

        for i in range(num_echoes):
            echo_kx = kx_values[i*pts_p_echo : (i+1)*pts_p_echo]
            self.assertAlmostEqual(np.min(echo_kx), -k_max_readout, delta=k_max_readout*0.01)
            self.assertAlmostEqual(np.max(echo_kx), k_max_readout, delta=k_max_readout*0.01)

            echo_ky_unique = np.unique(ky_values[i*pts_p_echo : (i+1)*pts_p_echo])
            self.assertEqual(len(echo_ky_unique), 1, f"Ky should be constant for echo {i}")

        # Check phase steps
        unique_phase_steps = np.unique([round(ky_values[i*pts_p_echo],5) for i in range(num_echoes)]) # Round to avoid float precision issues
        self.assertEqual(len(unique_phase_steps), num_echoes)


    def test_create_epi_gradient_recalled_x_phase(self):
        num_echoes, pts_p_echo = 32, 64
        traj = self.gen_2d.create_epi_trajectory(
            num_echoes=num_echoes,
            points_per_echo=pts_p_echo,
            epi_type='gradient_recalled',
            phase_encode_direction='x', # ky readout, kx phase
            apply_constraints=False
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.metadata['epi_type'], 'gradient_recalled')
        self.assertEqual(traj.metadata['phase_encode_direction'], 'x')

        # Check k-space coverage pattern (ky readout, kx phase)
        k_max_readout = 1.0 / (2.0 * (self.gen_2d.resolution_mm[1] / 1000.0)) # resolution_mm[1] for ky readout
        kx_values = traj.kspace_points_rad_per_m[0, :]
        ky_values = traj.kspace_points_rad_per_m[1, :]

        for i in range(num_echoes):
            echo_ky = ky_values[i*pts_p_echo : (i+1)*pts_p_echo]
            # Check if direction reverses for gradient_recalled
            if i % 2 == 0: # Even echoes (0, 2, ...)
                self.assertAlmostEqual(np.min(echo_ky), -k_max_readout, delta=k_max_readout*0.01)
                self.assertAlmostEqual(np.max(echo_ky), k_max_readout, delta=k_max_readout*0.01)
            else: # Odd echoes (1, 3, ...)
                self.assertAlmostEqual(np.min(echo_ky), -k_max_readout, delta=k_max_readout*0.01)
                self.assertAlmostEqual(np.max(echo_ky), k_max_readout, delta=k_max_readout*0.01)
                # More specific check for reversal: first point should be near +k_max, last near -k_max
                self.assertGreater(echo_ky[0], echo_ky[-1]) # k-space traversed from positive to negative

            echo_kx_unique = np.unique(kx_values[i*pts_p_echo : (i+1)*pts_p_echo])
            self.assertEqual(len(echo_kx_unique), 1, f"Kx (phase) should be constant for echo {i}")

        unique_phase_steps = np.unique([round(kx_values[i*pts_p_echo],5) for i in range(num_echoes)])
        self.assertEqual(len(unique_phase_steps), num_echoes)


    def test_create_epi_acquire_every_other_line(self):
        num_echoes_requested, pts_p_echo = 64, 128
        traj = self.gen_2d.create_epi_trajectory(
            num_echoes=num_echoes_requested,
            points_per_echo=pts_p_echo,
            acquire_every_other_line=True,
            apply_constraints=False
        )
        self.assertIsInstance(traj, Trajectory)
        expected_acquired_echoes = num_echoes_requested // 2 + (num_echoes_requested % 2) # Handles odd/even requested
        self.assertEqual(traj.get_num_points(), expected_acquired_echoes * pts_p_echo)
        self.assertEqual(traj.metadata['num_echoes_acquired'], expected_acquired_echoes)

    def test_create_epi_with_constraints(self):
        num_echoes, pts_p_echo = 32, 64
        traj = self.gen_2d.create_epi_trajectory(
            num_echoes=num_echoes,
            points_per_echo=pts_p_echo,
            apply_constraints=True
        )
        self.assertIsInstance(traj, Trajectory)
        self.assertTrue(traj.metadata['constraints_actually_applied'])
        if traj.get_max_grad_Tm() is not None:
            self.assertLessEqual(traj.get_max_grad_Tm(), self.gen_2d.max_grad_mT_per_m / 1000.0 * 1.2)
        if traj.get_max_slew_Tm_per_s() is not None:
            self.assertLessEqual(traj.get_max_slew_Tm_per_s(), self.gen_2d.max_slew_Tm_per_s_ms * 1000.0 * 1.2)

    def test_create_epi_on_3d_generator_fail(self):
        with self.assertRaisesRegex(ValueError, "EPI trajectory is 2D"):
            self.gen_3d.create_epi_trajectory(num_echoes=64, points_per_echo=128)

if __name__ == '__main__':
    unittest.main()
