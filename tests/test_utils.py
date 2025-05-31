import unittest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajgen.trajectory import Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T
from trajgen.generators import generate_radial_trajectory, generate_spiral_trajectory
from trajgen.utils import constrain_trajectory, reconstruct_image, display_trajectory

class TestConstrainTrajectory(unittest.TestCase):
    def setUp(self):
        self.dt = 4e-6
        self.gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
        # Simple linear k-space ramp for testing (2D)
        # kx from 0 to 500, ky = 0
        num_points = 100
        k_max = 500
        k_coords = np.zeros((2, num_points))
        k_coords[0,:] = np.linspace(0, k_max, num_points)
        self.base_traj = Trajectory("test_ramp", k_coords, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        self.max_grad_mT_per_m = 40.0
        self.max_slew_Tm_per_s_ms = 150.0

    def tearDown(self):
        plt.close('all')

    def test_constraints_already_met(self):
        # Create a trajectory that should already meet typical constraints
        k_slow_ramp = np.zeros((2, 50))
        k_slow_ramp[0,:] = np.linspace(0, 10, 50) # Very low k_max -> low grad/slew
        slow_traj = Trajectory("slow_ramp", k_slow_ramp, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        constrained_traj = constrain_trajectory(slow_traj,
                                                self.max_grad_mT_per_m,
                                                self.max_slew_Tm_per_s_ms)

        np.testing.assert_array_almost_equal(
            slow_traj.kspace_points_rad_per_m,
            constrained_traj.kspace_points_rad_per_m, decimal=3)
        self.assertLessEqual(constrained_traj.get_max_grad_Tm() * 1000, self.max_grad_mT_per_m * 1.05) # Allow 5% overshoot from discrete effects
        self.assertLessEqual(constrained_traj.get_max_slew_Tm_per_s() / 1000, self.max_slew_Tm_per_s_ms * 1.05)


    def test_gradient_violation(self):
        # This base_traj likely violates gradient limits if k_max is high over few points & short dt
        # Expected max grad for base_traj: G = dk/dt / gamma = (500 / (100*4e-6)) / 42.576e6
        # G = (500 / 4e-4) / 42.576e6 = 1.25e6 / 42.576e6 approx 0.029 T/m = 29 mT/m
        # This should be fine with max_grad = 40 mT/m.
        # Let's make it violate: k_max = 2000 -> grad = 117 mT/m
        k_viol_grad = np.zeros((2, 100))
        k_viol_grad[0,:] = np.linspace(0, 2000, 100)
        viol_grad_traj = Trajectory("viol_grad", k_viol_grad, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        constrained_traj = constrain_trajectory(viol_grad_traj,
                                                self.max_grad_mT_per_m, # 40 mT/m
                                                200.0) # High slew to isolate grad effect

        self.assertLess(constrained_traj.get_max_grad_Tm(), viol_grad_traj.get_max_grad_Tm())
        self.assertLessEqual(constrained_traj.get_max_grad_Tm() * 1000, self.max_grad_mT_per_m * 1.1) # Allow 10% overshoot

    def test_slew_violation(self):
        # Create a trajectory with sharp turns or fast grad changes for slew violation
        # Rapid square wave gradient -> high slew
        num_points_slew = 50
        grad_slew_test = np.zeros((1, num_points_slew))
        grad_slew_test[0, 10:20] = 0.030 # 30 mT/m
        grad_slew_test[0, 30:40] = -0.030 # Ramp down

        # Convert this gradient to k-space to make a Trajectory object
        k_slew_test = np.cumsum(grad_slew_test * self.gamma * self.dt, axis=1)
        slew_viol_traj = Trajectory("slew_viol", k_slew_test, gradient_waveforms_Tm=grad_slew_test,
                                    dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)

        # Original max slew: (0.030 T/m) / 4e-6 s = 7.5e6 T/m/s = 7500 T/m/s/ms. This will violate.
        constrained_traj = constrain_trajectory(slew_viol_traj,
                                                max_grad_mT_per_m=60.0, # High grad limit
                                                max_slew_Tm_per_s_ms=150.0) # 150 T/m/s/ms

        self.assertLess(constrained_traj.get_max_slew_Tm_per_s(), slew_viol_traj.get_max_slew_Tm_per_s() + 1e-9) # allow for numerical precision
        self.assertLessEqual(constrained_traj.get_max_slew_Tm_per_s() / 1000, self.max_slew_Tm_per_s_ms * 1.15) # Allow 15%

    def test_empty_and_single_point_trajectory(self):
        empty_k = np.empty((2,0))
        empty_traj = Trajectory("empty", empty_k, dt_seconds=self.dt)
        constrained_empty = constrain_trajectory(empty_traj, self.max_grad_mT_per_m, self.max_slew_Tm_per_s_ms)
        self.assertEqual(constrained_empty.get_num_points(), 0)

        single_pt_k = np.array([[10],[10]])
        single_traj = Trajectory("single", single_pt_k, dt_seconds=self.dt)
        constrained_single = constrain_trajectory(single_traj, self.max_grad_mT_per_m, self.max_slew_Tm_per_s_ms)
        self.assertEqual(constrained_single.get_num_points(), 1)
        np.testing.assert_array_almost_equal(constrained_single.kspace_points_rad_per_m, single_pt_k)
        self.assertAlmostEqual(constrained_single.get_max_grad_Tm(), 0.0) # Grad of single point is 0
        self.assertAlmostEqual(constrained_single.get_max_slew_Tm_per_s(), 0.0) # Slew of single point is 0


class TestReconstructImage(unittest.TestCase):
    def setUp(self):
        self.dt = 4e-6
        # Simple 2D radial trajectory for testing
        self.num_spokes = 16
        self.pts_per_spoke = 32
        self.radial_traj_2d = Trajectory(
            "test_radial_2D",
            generate_radial_trajectory(num_spokes=self.num_spokes, points_per_spoke=self.pts_per_spoke, num_dimensions=2),
            dt_seconds=self.dt
        )
        # Calculate Voronoi for density comp
        self.radial_traj_2d.calculate_voronoi_density()


        # Simple K-space data: a bright spot at k=0 (DC component)
        self.k_data_dc = np.zeros(self.radial_traj_2d.get_num_points(), dtype=complex)
        # For radial, k=0 is often sampled multiple times. Find one such point.
        k_norms = np.linalg.norm(self.radial_traj_2d.kspace_points_rad_per_m, axis=0)
        dc_point_indices = np.where(k_norms < 1e-3)[0]
        if len(dc_point_indices) > 0:
            self.k_data_dc[dc_point_indices[0]] = 1.0 * len(dc_point_indices) # Make it bright
        else: # If no exact k=0 point (e.g. half-pixel shift), put at first point
            self.k_data_dc[0] = 1.0


    def tearDown(self):
        plt.close('all')

    def test_reconstruct_dc_signal(self):
        recon_size = (64, 64)
        image = reconstruct_image(self.k_data_dc, self.radial_traj_2d,
                                  recon_matrix_size=recon_size,
                                  oversampling_factor=1.5,
                                  gridding_method='nearest') # Try nearest for DC
        self.assertEqual(image.shape, recon_size)
        # Expect a bright spot at the center of the image for DC signal
        center_x, center_y = recon_size[0]//2, recon_size[1]//2
        # Check a small region around the center
        center_brightness = np.sum(image[center_x-2:center_x+2, center_y-2:center_y+2])
        total_brightness = np.sum(image)
        if total_brightness > 1e-6 : # Avoid division by zero if image is all black
             self.assertGreater(center_brightness / total_brightness, 0.1, "Center not significantly brighter for DC input")
        self.assertTrue(np.all(np.isfinite(image)))


    def test_reconstruct_with_density_comp(self):
        recon_size = (64, 64)
        # Using Voronoi weights computed in setUp
        image_dcw = reconstruct_image(self.k_data_dc, self.radial_traj_2d,
                                      density_comp_weights=self.radial_traj_2d.metadata['density_compensation_weights_voronoi'],
                                      recon_matrix_size=recon_size,
                                      gridding_method='nearest') # Try nearest for DC
        self.assertEqual(image_dcw.shape, recon_size)
        # Basic check, similar to above. Quality assessment is complex.
        center_x, center_y = recon_size[0]//2, recon_size[1]//2
        center_brightness_dcw = np.sum(image_dcw[center_x-2:center_x+2, center_y-2:center_y+2])
        total_brightness_dcw = np.sum(image_dcw)

        if total_brightness_dcw > 1e-6:
            self.assertGreater(center_brightness_dcw / total_brightness_dcw, 0.1)

        # Compare with no DCW - should be different, potentially sharper peak with DCW for radial
        image_no_dcw = reconstruct_image(self.k_data_dc, self.radial_traj_2d,
                                         density_comp_weights=np.ones(self.radial_traj_2d.get_num_points()),
                                         recon_matrix_size=recon_size,
                                         gridding_method='nearest') # Try nearest for DC
        # This is a weak test; difference depends on trajectory and data
        if total_brightness_dcw > 1e-6 and np.sum(np.abs(image_no_dcw)) > 1e-6:
             self.assertNotAlmostEqual(np.max(image_dcw), np.max(image_no_dcw), delta=np.max(image_no_dcw)*0.05,
                                    msg="Image with DCW should generally differ from image without.")


    def test_input_validation_recon(self):
        k_wrong_shape = np.zeros((self.radial_traj_2d.get_num_points(), 2))
        with self.assertRaisesRegex(ValueError, "kspace_data must be a 1D array"):
            reconstruct_image(k_wrong_shape, self.radial_traj_2d)

        k_wrong_size = np.zeros(self.radial_traj_2d.get_num_points() - 1)
        with self.assertRaisesRegex(ValueError, "kspace_data must be a 1D array with length matching"):
            reconstruct_image(k_wrong_size, self.radial_traj_2d)

        with self.assertRaisesRegex(ValueError, "recon_matrix_size tuple length"):
            reconstruct_image(self.k_data_dc, self.radial_traj_2d, recon_matrix_size=(32,32,32)) # 2D traj

        # 1D trajectory
        traj_1d = Trajectory("1d_traj", np.array([[1,2,3]]), dt_seconds=self.dt)
        k_1d = np.zeros(3)
        with self.assertRaisesRegex(NotImplementedError, "Reconstruction for 1D not implemented"):
            reconstruct_image(k_1d, traj_1d)


class TestDisplayTrajectory(unittest.TestCase):
    def setUp(self):
        self.traj_2d = Trajectory("display_2d", generate_spiral_trajectory(200,2,points_per_interleaf=100))
        self.traj_3d = Trajectory("display_3d", generate_radial_trajectory(16,32,num_dimensions=3))
        self.traj_1d = Trajectory("display_1d", np.array([[1.,2.,3.,4.]]), dt_seconds=1e-5)


    def tearDown(self):
        plt.close('all')

    def test_display_2d_kspace_only(self):
        fig = display_trajectory(self.traj_2d)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        self.assertTrue("K-space: display_2d" in fig.axes[0].get_title())

    def test_display_3d_with_waveforms(self):
        # Ensure gradients and slew can be calculated for this trajectory
        self.traj_3d.dt_seconds = 4e-6
        self.traj_3d.get_gradient_waveforms_Tm() # Pre-compute

        fig = display_trajectory(self.traj_3d, show_gradients=True, show_slew=True)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 3)
        self.assertTrue("K-space: display_3d" in fig.axes[0].get_title())
        self.assertTrue("Gradient Waveforms" in fig.axes[1].get_title())
        self.assertTrue("Slew Rate Waveforms" in fig.axes[2].get_title())

    def test_display_1d_trajectory(self):
        fig = display_trajectory(self.traj_1d, show_gradients=True)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 2) # Kspace + Gradients
        self.assertTrue("K-space (1D): display_1d" in fig.axes[0].get_title())


if __name__ == '__main__':
    unittest.main()
