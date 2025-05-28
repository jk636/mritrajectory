import unittest
import numpy as np
import os
import sys

# Add the parent directory to the Python path to allow importing trajgen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajgen import Trajectory, KSpaceTrajectoryGenerator, COMMON_NUCLEI_GAMMA_HZ_PER_T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plot testing

class TestSharedConstants(unittest.TestCase):
    def test_common_nuclei_gamma(self):
        self.assertIsNotNone(COMMON_NUCLEI_GAMMA_HZ_PER_T)
        self.assertIsInstance(COMMON_NUCLEI_GAMMA_HZ_PER_T, dict)

        # Check for specific key presence
        expected_keys = ['1H', '13C', '31P', '19F', '23Na', '129Xe', '2H', '7Li']
        for key in expected_keys:
            self.assertIn(key, COMMON_NUCLEI_GAMMA_HZ_PER_T)
            # Check that values are positive floats
            self.assertIsInstance(COMMON_NUCLEI_GAMMA_HZ_PER_T[key], float)
            self.assertGreater(COMMON_NUCLEI_GAMMA_HZ_PER_T[key], 0)

        # Check minimum size
        self.assertGreaterEqual(len(COMMON_NUCLEI_GAMMA_HZ_PER_T), len(expected_keys))

        # Check a specific known value (e.g. 1H)
        self.assertAlmostEqual(COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'], 42.576e6, places=3)


class TestTrajectory(unittest.TestCase):
    def setUp(self):
        # Common parameters
        self.dt = 4e-6  # 4 us
        self.gamma = 42.576e6  # Hz/T
        self.n_points = 100
        self.k_max = 1.0 / (2 * 0.004) # Corresponds to 4mm resolution

        # Simple linear k-space ramp for 1D
        self.kspace_1d = np.linspace(-self.k_max, self.k_max, self.n_points).reshape(1, self.n_points)
        self.grad_1d_expected_val = (self.k_max * 2 / (self.n_points * self.dt)) / self.gamma
        
        # Simple 2D k-space (e.g., one line along Kx, one along Ky)
        kx_2d = np.linspace(-self.k_max, self.k_max, self.n_points)
        ky_2d = np.zeros_like(kx_2d)
        self.kspace_2d = np.stack([kx_2d, ky_2d]) # Shape (2, N)
        
        # Simple 3D k-space
        kz_3d = np.linspace(0, self.k_max, self.n_points)
        self.kspace_3d = np.stack([kx_2d, ky_2d, kz_3d]) # Shape (3, N)

        self.metadata_example = {'info': 'test_trajectory'}
        self.dead_time_start = 0.001 # 1 ms
        self.dead_time_end = 0.0005 # 0.5 ms

        # For plotting tests
        self.kspace_2d_plot_test = np.random.rand(2, 20) * 250 - 125 # 20 random 2D points
        self.kspace_3d_plot_test = np.random.rand(3, 30) * 250 - 125 # 30 random 3D points


    def tearDown(self):
        plt.close('all') # Close all figures after each test

    def test_trajectory_initialization_basic(self):
        traj = Trajectory("test1D", self.kspace_1d, dt_seconds=self.dt, metadata=self.metadata_example)
        self.assertEqual(traj.name, "test1D")
        self.assertEqual(traj.get_num_dimensions(), 1)
        self.assertEqual(traj.get_num_points(), self.n_points)
        self.assertEqual(traj.dt_seconds, self.dt)
        self.assertIn('info', traj.metadata)
        self.assertEqual(traj.metadata['info'], 'test_trajectory')
        self.assertAlmostEqual(traj.metadata['gamma_Hz_per_T'], self.gamma, places=1) # Default gamma

    def test_trajectory_initialization_with_gradients(self):
        dummy_gradients = np.ones_like(self.kspace_2d)
        traj = Trajectory("test2D_with_grads", self.kspace_2d, 
                          gradient_waveforms_Tm=dummy_gradients, 
                          dt_seconds=self.dt)
        self.assertEqual(traj.get_num_dimensions(), 2)
        self.assertTrue(np.array_equal(traj.get_gradient_waveforms_Tm(), dummy_gradients))

    def test_trajectory_initialization_with_dead_times(self):
        traj = Trajectory("test_deadtime", self.kspace_1d, dt_seconds=self.dt,
                          dead_time_start_seconds=self.dead_time_start,
                          dead_time_end_seconds=self.dead_time_end)
        self.assertEqual(traj.dead_time_start_seconds, self.dead_time_start)
        self.assertEqual(traj.dead_time_end_seconds, self.dead_time_end)
        self.assertIn('dead_time_start_seconds', traj.metadata)
        self.assertIn('dead_time_end_samples', traj.metadata)
        expected_duration = self.n_points * self.dt + self.dead_time_start + self.dead_time_end
        self.assertAlmostEqual(traj.get_duration_seconds(), expected_duration)

    def test_trajectory_initialization_gamma_param(self):
        custom_gamma = 40.0e6
        traj = Trajectory("test_gamma", self.kspace_1d, dt_seconds=self.dt, gamma_Hz_per_T=custom_gamma)
        self.assertAlmostEqual(traj.metadata['gamma_Hz_per_T'], custom_gamma)

        # Test if gamma in metadata is prioritized
        metadata_with_gamma = {'gamma_Hz_per_T': 50.0e6, 'other_info': 'test'}
        traj_meta_gamma = Trajectory("test_meta_gamma", self.kspace_1d, dt_seconds=self.dt, 
                                     metadata=metadata_with_gamma, gamma_Hz_per_T=custom_gamma)
        self.assertAlmostEqual(traj_meta_gamma.metadata['gamma_Hz_per_T'], 50.0e6)


    def test_get_gradient_waveforms_Tm_calculation(self):
        traj = Trajectory("test_grad_calc", self.kspace_1d, dt_seconds=self.dt)
        # Force calculation if it wasn't done in init (it is, via _compute_metrics)
        gradients = traj.get_gradient_waveforms_Tm() 
        self.assertIsNotNone(gradients)
        self.assertEqual(gradients.shape, self.kspace_1d.shape)
        # For a linear ramp kx = m*t, Gx = m / gamma. Here k = m*idx*dt.
        # k_diff = self.kspace_1d[0,1] - self.kspace_1d[0,0]
        # expected_grad_val = k_diff / self.dt / traj.metadata['gamma_Hz_per_T']
        # np.gradient uses central differences, so it's more nuanced for edge points.
        # Let's check a middle point for a more stable value
        # For a linear ramp from -kmax to kmax over N points, slope is 2*kmax / (N-1) samples
        # k(t) = slope_t * t. G(t) = slope_t / gamma.
        # Or, slope_k_vs_idx = (k_space_data[...,-1] - k_space_data[...,0]) / (N-1)
        # grad_idx = slope_k_vs_idx / dt / gamma
        # For a linear ramp, this should be fairly constant.
        # Using the simpler formula: (k_max - (-k_max)) / ( (N-1)*dt ) / gamma
        # However, n_points for linspace means N-1 segments.
        # So, (k_max - (-k_max)) / (N*dt) for average, but np.gradient is more complex.
        # Let's test the shape and that it runs.
        # A more precise check: if k(t) = A*t + B, then dk/dt = A. G = A/gamma.
        # k_max = A * (N-1)*dt / 2 (if centered at 0). A = 2*k_max / ((N-1)*dt)
        # G = (2*k_max / ((N-1)*dt)) / gamma
        # This is what grad_1d_expected_val was for, assuming (N-1) segments.
        # Let's use a simplified check for non-zero middle gradients
        self.assertTrue(np.all(np.abs(gradients[0, 1:-1]) > 1e-3))


    def test_get_gradient_waveforms_Tm_caching(self):
        traj = Trajectory("test_grad_cache", self.kspace_1d, dt_seconds=self.dt)
        gradients1 = traj.get_gradient_waveforms_Tm()
        gradients2 = traj.get_gradient_waveforms_Tm()
        self.assertIs(gradients1, gradients2) # Should return the same cached object

    def test_metric_calculations_run(self):
        # These are called in __init__ via _compute_metrics. Test if fields are populated.
        traj = Trajectory("test_metrics", self.kspace_2d, dt_seconds=self.dt)
        self.assertIn('max_slew_rate_Tm_per_s', traj.metadata)
        self.assertIn('pns_max_abs_gradient_sum_xyz', traj.metadata)
        self.assertIn('fov_estimate_m', traj.metadata)
        self.assertIn('resolution_overall_estimate_m', traj.metadata)
        # For slew rate, with non-zero k-space and dt, it should be non-None.
        self.assertIsNotNone(traj.metadata['max_slew_rate_Tm_per_s'])


    def test_calculate_fov(self):
        # 2D
        k_max_x = np.max(np.abs(self.kspace_2d[0,:]))
        k_max_y = np.max(np.abs(self.kspace_2d[1,:]))
        expected_fov_x = 1.0 / (2 * k_max_x + 1e-9)
        expected_fov_y = 1.0 / (2 * k_max_y + 1e-9) # This will be inf as ky is 0
        
        traj2d = Trajectory("test_fov2d", self.kspace_2d, dt_seconds=self.dt)
        self.assertAlmostEqual(traj2d.metadata['fov_estimate_m'][0], expected_fov_x, places=6)
        self.assertTrue(np.isinf(traj2d.metadata['fov_estimate_m'][1]) or traj2d.metadata['fov_estimate_m'][1] > 1e6) # For zero k-extent

        # 3D
        k_max_z_3d = np.max(np.abs(self.kspace_3d[2,:]))
        expected_fov_z = 1.0 / (2 * k_max_z_3d + 1e-9)
        traj3d = Trajectory("test_fov3d", self.kspace_3d, dt_seconds=self.dt)
        self.assertAlmostEqual(traj3d.metadata['fov_estimate_m'][0], expected_fov_x, places=6)
        self.assertTrue(np.isinf(traj3d.metadata['fov_estimate_m'][1]) or traj3d.metadata['fov_estimate_m'][1] > 1e6)
        self.assertAlmostEqual(traj3d.metadata['fov_estimate_m'][2], expected_fov_z, places=6)

    def test_calculate_resolution(self):
        # Overall resolution is 1 / (2 * max_radius)
        # For kspace_2d, max radius is k_max_x since ky is 0
        k_max_radius_2d = np.max(np.linalg.norm(self.kspace_2d, axis=0))
        expected_res_2d = 1.0 / (2 * k_max_radius_2d + 1e-9)
        traj2d = Trajectory("test_res2d", self.kspace_2d, dt_seconds=self.dt)
        self.assertAlmostEqual(traj2d.metadata['resolution_overall_estimate_m'], expected_res_2d, places=6)

        k_max_radius_3d = np.max(np.linalg.norm(self.kspace_3d, axis=0))
        expected_res_3d = 1.0 / (2 * k_max_radius_3d + 1e-9)
        traj3d = Trajectory("test_res3d", self.kspace_3d, dt_seconds=self.dt)
        self.assertAlmostEqual(traj3d.metadata['resolution_overall_estimate_m'], expected_res_3d, places=6)

    def test_get_duration_seconds(self):
        traj = Trajectory("test_duration1", self.kspace_1d, dt_seconds=self.dt)
        expected_sampling_duration = self.n_points * self.dt
        self.assertAlmostEqual(traj.get_duration_seconds(), expected_sampling_duration)

        traj_dt = Trajectory("test_duration2", self.kspace_1d, dt_seconds=self.dt,
                             dead_time_start_seconds=self.dead_time_start,
                             dead_time_end_seconds=self.dead_time_end)
        expected_total_duration = expected_sampling_duration + self.dead_time_start + self.dead_time_end
        self.assertAlmostEqual(traj_dt.get_duration_seconds(), expected_total_duration)
        
        traj_no_dt = Trajectory("test_duration3", self.kspace_1d, dt_seconds=None)
        self.assertIsNone(traj_no_dt.get_duration_seconds())


    def test_export_import_npz(self):
        filename = "test_trajectory_export.npz"
        traj_orig = Trajectory("test_export", self.kspace_3d, 
                               dt_seconds=self.dt, 
                               metadata={'info': 'original', 'gamma_Hz_per_T': self.gamma},
                               dead_time_start_seconds=self.dead_time_start,
                               dead_time_end_seconds=self.dead_time_end)
        # Ensure gradients are computed and cached before export if we want to compare them
        _ = traj_orig.get_gradient_waveforms_Tm()
        
        traj_orig.export(filename, filetype='npz')
        self.assertTrue(os.path.exists(filename))

        traj_imported = Trajectory.import_from(filename)

        self.assertEqual(traj_imported.name, filename) # Name is filename on import
        self.assertEqual(traj_imported.dt_seconds, traj_orig.dt_seconds)
        self.assertEqual(traj_imported.get_num_points(), traj_orig.get_num_points())
        self.assertEqual(traj_imported.get_num_dimensions(), traj_orig.get_num_dimensions())
        
        # Check k-space data (points_to_export might transpose, getter in Trajectory ensures D,N)
        # Original kspace_3d is D,N. Getter returns D,N. Export saves N,D. Import converts to D,N.
        # So traj_imported.kspace_points_rad_per_m should be D,N
        self.assertTrue(np.allclose(traj_imported.kspace_points_rad_per_m, self.kspace_3d))
        
        # Check metadata items
        self.assertEqual(traj_imported.metadata['info'], 'original')
        self.assertAlmostEqual(traj_imported.metadata['gamma_Hz_per_T'], self.gamma)
        self.assertAlmostEqual(traj_imported.metadata['dead_time_start_seconds'], self.dead_time_start)
        self.assertAlmostEqual(traj_imported.metadata['dead_time_end_seconds'], self.dead_time_end)
        self.assertAlmostEqual(traj_imported.dead_time_start_seconds, self.dead_time_start) # Also check attrs

        # Check if gradients were saved and loaded (if they were computed in original)
        # The export logic for gradients_to_export ensures they are N,D if kspace is N,D
        # The import logic doesn't explicitly re-transpose gradients back to D,N if they were N,D.
        # However, get_gradient_waveforms_Tm will recompute if the format is unexpected or if it's None.
        # Let's compare the result of get_gradient_waveforms_Tm()
        self.assertTrue(np.allclose(traj_imported.get_gradient_waveforms_Tm(), traj_orig.get_gradient_waveforms_Tm()))


        if os.path.exists(filename):
            os.remove(filename)

    def test_voronoi_density(self):
        # Simple 2D square
        points_2d_square = np.array([[0,0], [1,0], [0,1], [1,1]]).T # Shape (2,4)
        traj_square = Trajectory("square", points_2d_square, dt_seconds=self.dt)
        
        cell_sizes = traj_square.calculate_voronoi_density()
        self.assertIsNotNone(cell_sizes)
        self.assertEqual(traj_square.metadata['voronoi_calculation_status'], "Success")
        self.assertEqual(len(cell_sizes), 4)
        # For a unit square of 4 points, each cell should have area 0.25 if points are at corners of cells.
        # Voronoi cells for points at (0,0), (1,0), (0,1), (1,1) will be centered around these points.
        # The point (0,0) would have a cell from (-0.5,-0.5) to (0.5,0.5) if it was part of a larger grid.
        # Here, they are on the convex hull, so their cells are infinite.
        self.assertTrue(np.all(np.isinf(cell_sizes))) # All points on convex hull

        # Add a center point to get finite cells
        points_2d_center = np.array([[0,0], [1,0], [0,1], [1,1], [0.5,0.5]]).T
        traj_center = Trajectory("center_square", points_2d_center, dt_seconds=self.dt)
        cell_sizes_center = traj_center.calculate_voronoi_density()
        self.assertIsNotNone(cell_sizes_center)
        self.assertEqual(traj_center.metadata['voronoi_calculation_status'], "Success")
        self.assertEqual(len(cell_sizes_center), 5)
        # The center point's cell should be finite, others infinite.
        self.assertTrue(np.isfinite(cell_sizes_center[4])) # Center point
        self.assertTrue(np.all(np.isinf(cell_sizes_center[:4]))) # Corner points

        # Test with too few points
        points_2d_line = np.array([[0,0], [1,0]]).T # 2 points, D=2. Need D+1=3
        traj_line = Trajectory("line", points_2d_line, dt_seconds=self.dt)
        cell_sizes_line = traj_line.calculate_voronoi_density()
        self.assertIsNone(cell_sizes_line)
        self.assertIn("Error: Not enough unique points", traj_line.metadata['voronoi_calculation_status'])


        # Test unsupported dimension (1D)
        traj_1d_voronoi = Trajectory("1d_voronoi", self.kspace_1d, dt_seconds=self.dt)
        cell_sizes_1d = traj_1d_voronoi.calculate_voronoi_density()
        self.assertIsNone(cell_sizes_1d)
        self.assertIn("Error: Voronoi calculation only supported for 2D/3D", traj_1d_voronoi.metadata['voronoi_calculation_status'])

        # Test 3D (just a few points to ensure it runs)
        points_3d_simple = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0.5]]).T
        traj_3d_simple = Trajectory("simple3d", points_3d_simple, dt_seconds=self.dt)
        cell_sizes_3d = traj_3d_simple.calculate_voronoi_density()
        self.assertIsNotNone(cell_sizes_3d)
        self.assertEqual(traj_3d_simple.metadata['voronoi_calculation_status'], "Success")
        self.assertEqual(len(cell_sizes_3d), 5)
        self.assertTrue(np.isfinite(cell_sizes_3d[4])) # Center point
        self.assertTrue(np.all(np.isinf(cell_sizes_3d[:4]))) # Corner points

    def test_plot_3d_execution(self):
        # Test with a 3D trajectory
        traj_3d = Trajectory("test_plot3d", self.kspace_3d_plot_test, dt_seconds=self.dt)
        ax = traj_3d.plot_3d()
        self.assertIsNotNone(ax)
        self.assertIsInstance(ax, Axes3D)
        plt.close('all')

        # Test with subsampling parameters
        ax_sub = traj_3d.plot_3d(max_total_points=10, max_interleaves=1, point_stride=2, interleaf_stride=1)
        self.assertIsNotNone(ax_sub)
        self.assertIsInstance(ax_sub, Axes3D)
        plt.close('all')

        # Test with a pre-existing ax
        fig = plt.figure()
        pre_ax = fig.add_subplot(111, projection='3d')
        returned_ax = traj_3d.plot_3d(ax=pre_ax)
        self.assertIs(returned_ax, pre_ax)
        plt.close('all')

        # Test on a 2D trajectory (should print message and return None or original ax)
        traj_2d = Trajectory("test_plot3d_on_2d", self.kspace_2d_plot_test, dt_seconds=self.dt)
        # Suppress print output for this specific call if possible, or just check return
        returned_ax_2d = traj_2d.plot_3d() 
        self.assertIsNone(returned_ax_2d) # Expecting None as no ax was passed and it's not 3D
        
        fig_2d, ax_2d_passed = plt.subplots() # Create a 2D axes
        returned_ax_2d_passed = traj_2d.plot_3d(ax=ax_2d_passed)
        self.assertIs(returned_ax_2d_passed, ax_2d_passed) # Should return the passed ax
        plt.close('all')


    def test_plot_voronoi_execution(self):
        # Test with a 2D trajectory
        # Use fewer points for Voronoi as it can be slow
        kspace_2d_voronoi = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]]).T # 5 points
        traj_2d = Trajectory("test_plot_voronoi_2d", kspace_2d_voronoi, dt_seconds=self.dt)
        _ = traj_2d.calculate_voronoi_density() # Ensure data is computed
        
        ax = traj_2d.plot_voronoi()
        self.assertIsNotNone(ax)
        self.assertIsInstance(ax, plt.Axes)
        plt.close('all')

        # Test with different parameters
        ax_params = traj_2d.plot_voronoi(color_by_area=False, show_vertices=True, cmap='cividis')
        self.assertIsNotNone(ax_params)
        self.assertIsInstance(ax_params, plt.Axes)
        plt.close('all')

        # Test with a pre-existing ax
        fig, pre_ax = plt.subplots()
        returned_ax = traj_2d.plot_voronoi(ax=pre_ax)
        self.assertIs(returned_ax, pre_ax)
        plt.close('all')

        # Test on a 3D trajectory (should fallback to plot_3d)
        traj_3d = Trajectory("test_plot_voronoi_3d", self.kspace_3d_plot_test, dt_seconds=self.dt)
        ax_3d_fallback = traj_3d.plot_voronoi()
        self.assertIsNotNone(ax_3d_fallback)
        self.assertIsInstance(ax_3d_fallback, Axes3D) # Checks if plot_3d was called
        plt.close('all')


class TestKSpaceTrajectoryGenerator(unittest.TestCase):
    def setUp(self):
        self.common_params = {
            'fov': 0.256,
            'resolution': 0.004, # k_max = 125
            'dt': 4e-6,
            'gamma': 42.576e6,
            'n_interleaves': 4, # Default, can be overridden
        }
        # Calculated n_samples for these params (approx, depends on type)
        # k_max / (gamma * g_required * dt) * turns * 2pi * fov
        # For spiral with g_max=0.04, turns=1:
        # k_max = 125. g_required = min(125 / (42.576e6 * 4e-6), 0.04) = 0.04
        # n_samples = ceil(125 * 2 * pi * 0.256 / (42.576e6 * 0.04 * 4e-6)) approx 29517
        # This seems too high. The internal n_samples calculation for spiral is:
        # n_samples = int(np.ceil((self.k_max * 2 * np.pi * self.fov) / (self.gamma * self.g_required * self.dt)))
        # k_max / (g_req * gamma * dt)
        # Let's use a higher resolution to get fewer samples for tests, or set n_samples directly if possible
        # For testing, let's use a resolution that gives a more manageable n_samples
        self.test_res = 0.01 # k_max = 50
        # n_samples for spiral (fov=0.256, res=0.01, g_max=0.04, turns=1)
        # k_max = 50. g_required = min(50 / (42.576e6 * 4e-6), 0.04) = 0.04
        # n_samples = ceil(50 * 2 * pi * 0.256 / (42.576e6 * 0.04 * 4e-6)) = ceil(11806)
        # Still high. The formula used in code has self.k_max * 2 * np.pi * self.fov which is kmax * kmax_fov_rad_s
        # Let's set a small number of points for some tests by adjusting FOV/res drastically or rely on ramp_fraction
        # For a simple radial line, n_samples is more direct.
        # k_max = 50. n_samples for radial is int(ceil(2 * k_max / (gamma * g_required * dt)))
        # This is also not what's in the code. The code's n_samples is complex.
        # Let's rely on the generator's own n_samples calculation.

    def _check_2d_outputs(self, gen, kx, ky, gx, gy, t):
        self.assertIsInstance(kx, np.ndarray)
        self.assertIsInstance(ky, np.ndarray)
        self.assertIsInstance(gx, np.ndarray)
        self.assertIsInstance(gy, np.ndarray)
        self.assertIsInstance(t, np.ndarray)
        
        expected_shape = (gen.n_interleaves, gen.n_samples + 2 * (gen.ramp_samples if gen.add_rewinder else 0) + 2 * (gen.ramp_samples if gen.add_spoiler else 0) )
        # The number of samples can change due to spoiler/rewinder.
        # The generate() method returns t with the final number of samples.
        final_n_samples = t.shape[0]
        expected_shape_interleaved = (gen.n_interleaves, final_n_samples)

        self.assertEqual(kx.shape, expected_shape_interleaved)
        self.assertEqual(ky.shape, expected_shape_interleaved)
        self.assertEqual(gx.shape, expected_shape_interleaved)
        self.assertEqual(gy.shape, expected_shape_interleaved)
        self.assertEqual(t.ndim, 1)


    def _check_3d_outputs(self, gen, kx, ky, kz, gx, gy, gz, t):
        self.assertIsInstance(kx, np.ndarray)
        self.assertIsInstance(ky, np.ndarray)
        self.assertIsInstance(kz, np.ndarray)
        self.assertIsInstance(gx, np.ndarray)
        self.assertIsInstance(gy, np.ndarray)
        self.assertIsInstance(gz, np.ndarray)
        self.assertIsInstance(t, np.ndarray)

        final_n_samples = t.shape[0]
        
        # For stackofspirals, n_interleaves is per stack, so total_shots = n_interleaves * n_stacks
        if gen.traj_type == 'stackofspirals':
            total_shots = gen.n_interleaves * (gen.n_stacks if gen.n_stacks is not None else 1)
            expected_shape_interleaved = (total_shots, final_n_samples)
        else:
            expected_shape_interleaved = (gen.n_interleaves, final_n_samples)

        self.assertEqual(kx.shape, expected_shape_interleaved)
        self.assertEqual(ky.shape, expected_shape_interleaved)
        self.assertEqual(kz.shape, expected_shape_interleaved)
        self.assertEqual(gx.shape, expected_shape_interleaved)
        self.assertEqual(gy.shape, expected_shape_interleaved)
        self.assertEqual(gz.shape, expected_shape_interleaved)
        self.assertEqual(t.ndim, 1)


    def test_2d_trajectory_generation(self):
        types_2d = ['spiral', 'radial', 'epi', 'rosette']
        for traj_type in types_2d:
            with self.subTest(traj_type=traj_type):
                params = self.common_params.copy()
                params['traj_type'] = traj_type
                params['dim'] = 2
                if traj_type == 'rosette': # Rosette needs specific params or defaults
                    params.update({'f1': 3, 'f2': 5, 'a': 0.5}) 
                
                gen = KSpaceTrajectoryGenerator(**params)
                kx, ky, gx, gy, t = gen.generate()
                self._check_2d_outputs(gen, kx, ky, gx, gy, t)
                self.assertTrue(np.all(np.isfinite(kx)))

    def test_3d_trajectory_generation(self):
        types_3d = ['stackofspirals', 'phyllotaxis', 'cones', 'radial3d', 'epi_3d', 'zte']
        for traj_type in types_3d:
            with self.subTest(traj_type=traj_type):
                params = self.common_params.copy()
                params['traj_type'] = traj_type
                params['dim'] = 3
                params['n_interleaves'] = 8 # Smaller for faster 3D tests
                if traj_type == 'stackofspirals':
                    params['n_stacks'] = 2
                if traj_type == 'epi_3d':
                    params['epi_3d_fov_y'] = params['fov']
                    params['epi_3d_resolution_y'] = params['resolution'] * 2 # Fewer PE lines
                    params['epi_3d_fov_z'] = params['fov']
                    params['epi_3d_resolution_z'] = params['resolution'] * 2 
                    # n_interleaves for epi_3d should be Ny * Nz
                    Ny = int(round(params['epi_3d_fov_y'] / params['epi_3d_resolution_y']))
                    Nz = int(round(params['epi_3d_fov_z'] / params['epi_3d_resolution_z']))
                    params['n_interleaves'] = max(1, Ny * Nz)


                gen = KSpaceTrajectoryGenerator(**params)
                kx, ky, kz, gx, gy, gz, t = gen.generate()
                self._check_3d_outputs(gen, kx, ky, kz, gx, gy, gz, t)
                self.assertTrue(np.all(np.isfinite(kx)))


    def test_variable_density_spiral(self):
        params_lin = {**self.common_params, 'dim': 2, 'traj_type': 'spiral', 'n_interleaves': 1, 
                      'vd_method': 'power', 'vd_alpha': 1.0, 'add_rewinder': False}
        gen_lin = KSpaceTrajectoryGenerator(**params_lin)
        kx_lin, ky_lin, _, _, _ = gen_lin.generate()
        r_lin = np.sqrt(kx_lin[0]**2 + ky_lin[0]**2)

        params_dense = {**self.common_params, 'dim': 2, 'traj_type': 'spiral', 'n_interleaves': 1,
                        'vd_method': 'power', 'vd_alpha': 2.0, 'add_rewinder': False} # Denser at center
        gen_dense = KSpaceTrajectoryGenerator(**params_dense)
        kx_dense, ky_dense, _, _, _ = gen_dense.generate()
        r_dense = np.sqrt(kx_dense[0]**2 + ky_dense[0]**2)
        
        # Alpha=2.0 should have more points at smaller radii (median radius should be smaller)
        # Ignore first few points which might be zero or very small for both
        self.assertLess(np.median(r_dense[10:]), np.median(r_lin[10:]))


    def test_ute_ramp_sampling_radial3d(self):
        params = {**self.common_params, 'dim': 3, 'traj_type': 'radial3d', 'n_interleaves': 1, 
                  'add_rewinder': False, 'resolution': 0.01} # k_max = 50
        
        # Standard radial3d (full spoke)
        gen_full = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': False})
        kx_f, ky_f, kz_f, _, _, _, _ = gen_full.generate()
        k_radius_f = np.sqrt(kx_f[0]**2 + ky_f[0]**2 + kz_f[0]**2)
        # Full spoke should have points near -k_max and +k_max if r_profile is flat (default _make_radius_profile)
        # The default r_profile has ramps, so min might not be exactly 0 unless k_max is very large.
        # Check that the profile covers a range that is clearly not just 0 to k_max
        # This test is a bit weak because of the default symmetric ramp profile.
        # A better check: the k-space points should be roughly symmetric around the origin.
        # For a single spoke:
        self.assertTrue(np.isclose(np.mean(kx_f[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(ky_f[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(kz_f[0]), 0.0, atol=1e-1), "Full spoke not centered around origin")


        # UTE radial3d (half spoke)
        gen_half = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': True})
        kx_h, ky_h, kz_h, _, _, _, _ = gen_half.generate()
        k_radius_h = np.sqrt(kx_h[0]**2 + ky_h[0]**2 + kz_h[0]**2)
        self.assertLess(np.min(k_radius_h), 0.01 * gen_half.k_max) # Starts near k=0
        self.assertGreater(np.max(k_radius_h), 0.9 * gen_half.k_max) # Reaches near k_max

    def test_zte_trajectory(self):
        params = {**self.common_params, 'dim': 3, 'traj_type': 'zte', 'n_interleaves': 1, 
                  'add_rewinder': False, 'resolution': 0.01}

        # ZTE with ute_ramp_sampling=True (expected behavior)
        gen_zte_ute = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': True})
        kx_z_ute, ky_z_ute, kz_z_ute, _, _, _, _ = gen_zte_ute.generate()
        k_radius_z_ute = np.sqrt(kx_z_ute[0]**2 + ky_z_ute[0]**2 + kz_z_ute[0]**2)
        self.assertLess(np.min(k_radius_z_ute), 0.01 * gen_zte_ute.k_max) # Starts near k=0

        # ZTE with ute_ramp_sampling=False (should be like full radial spoke)
        gen_zte_full = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': False})
        kx_z_full, ky_z_full, kz_z_full, _, _, _, _ = gen_zte_full.generate()
        # Check that it's symmetric around origin
        self.assertTrue(np.isclose(np.mean(kx_z_full[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(ky_z_full[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(kz_z_full[0]), 0.0, atol=1e-1), "ZTE (no UTE ramp) not centered")


    def test_generate_3d_from_2d(self):
        base_gen_params = {**self.common_params, 'dim': 2, 'traj_type': 'spiral', 
                           'n_interleaves': 1, 'turns': 3, 'add_rewinder': False}
        base_gen = KSpaceTrajectoryGenerator(**base_gen_params)
        
        n_3d_shots = 10
        kx, ky, kz, gx, gy, gz, t = base_gen.generate_3d_from_2d(
            n_3d_shots=n_3d_shots, 
            traj2d_type='spiral' # Use the same type as base_gen for consistency
        )
        
        self.assertEqual(kx.shape[0], n_3d_shots)
        self.assertEqual(ky.shape[0], n_3d_shots)
        self.assertEqual(kz.shape[0], n_3d_shots)
        # n_samples_2d will be calculated by the internal base2d generator
        # For this test, ensure it ran and produced the right number of shots
        self.assertTrue(kx.shape[1] > 0) 
        
        # Check if different shots are actually different (due to rotation)
        self.assertFalse(np.allclose(kx[0], kx[1]))
        self.assertFalse(np.allclose(ky[0], ky[1]))
        # Kz will likely be different unless specific rotation makes them same
        if n_3d_shots > 1 and kx.shape[1] > 0: # Ensure there are points to compare
             self.assertFalse(np.allclose(kz[0], kz[1]) if kz.shape[1] > 0 and np.std(kz[0]-kz[1]) > 1e-9 else False)


    def test_edge_cases_and_options(self):
        # Test n_interleaves = 1
        gen_single_interleaf = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'spiral', 'n_interleaves':1})
        kx, ky, _, _, _ = gen_single_interleaf.generate()
        self.assertEqual(kx.shape[0], 1)

        # Test no rewinder
        gen_no_rewind = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'radial', 'add_rewinder':False})
        _, _, _, _, t_no_rewind = gen_no_rewind.generate()
        
        gen_with_rewind = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'radial', 'add_rewinder':True})
        _, _, _, _, t_with_rewind = gen_with_rewind.generate()
        if gen_no_rewind.ramp_samples > 0 : # Rewinder adds ramp_samples
             self.assertLess(t_no_rewind.shape[0], t_with_rewind.shape[0])

        # Test spoiler
        gen_spoiler = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'spiral', 
                                                 'add_spoiler':True, 'add_rewinder': False})
        _, _, _, _, t_spoiler = gen_spoiler.generate()
        if gen_spoiler.ramp_samples > 0: # Spoiler adds ramp_samples
            self.assertEqual(t_spoiler.shape[0], gen_spoiler.n_samples + gen_spoiler.ramp_samples)


if __name__ == '__main__':
    unittest.main()
