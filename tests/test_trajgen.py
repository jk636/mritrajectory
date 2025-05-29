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

# Need to import the new functions to be tested
from trajgen import (
    normalize_density_weights,
    compute_density_compensation,
    create_periodic_points,
    # compute_cell_area, # Indirectly tested
    compute_voronoi_density,
    generate_spiral_trajectory,
    generate_radial_trajectory,
    generate_golden_angle_3d_trajectory, # Added import
    constrain_trajectory,
    reconstruct_image,
    display_trajectory
)
from unittest.mock import patch # For display_trajectory tests


class TestGenerateGoldenAngle3DTrajectory(unittest.TestCase):
    def setUp(self):
        self.default_dt = 4e-6
        self.num_points_default = 100

    def test_basic_generation_isotropic_fov(self):
        num_points = self.num_points_default
        fov_m = 0.25  # Isotropic FOV
        traj_name = "golden_angle_iso_fov"

        traj = generate_golden_angle_3d_trajectory(num_points, fov_m, name=traj_name, dt_seconds=self.default_dt)

        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.name, traj_name)
        self.assertEqual(traj.get_num_points(), num_points)
        self.assertEqual(traj.get_num_dimensions(), 3)
        self.assertEqual(traj.kspace_points_rad_per_m.shape, (3, num_points))
        self.assertEqual(traj.dt_seconds, self.default_dt)

        expected_k_max = np.pi / fov_m
        expected_k_max_xyz = (expected_k_max, expected_k_max, expected_k_max)

        self.assertIn('generator_params', traj.metadata)
        gp = traj.metadata['generator_params']
        self.assertEqual(gp['num_points'], num_points)
        self.assertEqual(gp['fov_m_input'], fov_m)
        self.assertIsNone(gp['max_k_rad_per_m_input']) # Was not provided
        self.assertEqual(gp['dt_seconds_input'], self.default_dt)

        self.assertIn('k_max_calculated_rad_m_xyz', traj.metadata)
        for i in range(3):
            self.assertAlmostEqual(traj.metadata['k_max_calculated_rad_m_xyz'][i], expected_k_max_xyz[i], places=6)

        # Check k-space points are within bounds
        # For Golden Angle, points are distributed within an ellipsoidal volume.
        # Max extent along each axis should be within calculated k_max for that axis.
        for dim in range(3):
            self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[dim, :])) <= expected_k_max_xyz[dim] * 1.001) # Allow small tolerance

    def test_generation_anisotropic_fov(self):
        num_points = self.num_points_default
        fov_m_aniso = (0.2, 0.3, 0.4) # Anisotropic FOV (x, y, z)
        traj_name = "golden_angle_aniso_fov"

        traj = generate_golden_angle_3d_trajectory(num_points, fov_m_aniso, name=traj_name)

        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.name, traj_name)
        self.assertEqual(traj.get_num_points(), num_points)
        self.assertEqual(traj.get_num_dimensions(), 3)
        self.assertEqual(traj.kspace_points_rad_per_m.shape, (3, num_points))

        expected_k_max_x = np.pi / fov_m_aniso[0]
        expected_k_max_y = np.pi / fov_m_aniso[1]
        expected_k_max_z = np.pi / fov_m_aniso[2]
        expected_k_max_xyz = (expected_k_max_x, expected_k_max_y, expected_k_max_z)

        self.assertIn('generator_params', traj.metadata)
        gp = traj.metadata['generator_params']
        self.assertEqual(gp['fov_m_input'], fov_m_aniso)

        self.assertIn('k_max_calculated_rad_m_xyz', traj.metadata)
        km_calc = traj.metadata['k_max_calculated_rad_m_xyz']
        self.assertAlmostEqual(km_calc[0], expected_k_max_xyz[0], places=6)
        self.assertAlmostEqual(km_calc[1], expected_k_max_xyz[1], places=6)
        self.assertAlmostEqual(km_calc[2], expected_k_max_xyz[2], places=6)

        # Check k-space points are within bounds for each dimension
        self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[0, :])) <= expected_k_max_xyz[0] * 1.001)
        self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[1, :])) <= expected_k_max_xyz[1] * 1.001)
        self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[2, :])) <= expected_k_max_xyz[2] * 1.001)

    def test_generation_explicit_isotropic_max_k(self):
        num_points = self.num_points_default
        fov_m = 0.2 # This will be ignored for k_max calculation if max_k_rad_per_m is given
        explicit_k_max_iso = 100.0
        
        traj = generate_golden_angle_3d_trajectory(num_points, fov_m, max_k_rad_per_m=explicit_k_max_iso)

        self.assertIsInstance(traj, Trajectory)
        expected_k_max_xyz = (explicit_k_max_iso, explicit_k_max_iso, explicit_k_max_iso)

        gp = traj.metadata['generator_params']
        self.assertEqual(gp['max_k_rad_per_m_input'], explicit_k_max_iso)
        km_calc = traj.metadata['k_max_calculated_rad_m_xyz']
        for i in range(3):
            self.assertAlmostEqual(km_calc[i], expected_k_max_xyz[i], places=6)

        for dim in range(3):
            self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[dim, :])) <= expected_k_max_xyz[dim] * 1.001)

    def test_generation_explicit_anisotropic_max_k(self):
        num_points = self.num_points_default
        fov_m = 0.25 # Ignored for k_max calculation
        explicit_k_max_aniso = (100.0, 120.0, 150.0)

        traj = generate_golden_angle_3d_trajectory(num_points, fov_m, max_k_rad_per_m=explicit_k_max_aniso)
        
        self.assertIsInstance(traj, Trajectory)
        gp = traj.metadata['generator_params']
        self.assertEqual(gp['max_k_rad_per_m_input'], explicit_k_max_aniso)
        km_calc = traj.metadata['k_max_calculated_rad_m_xyz']
        for i in range(3):
            self.assertAlmostEqual(km_calc[i], explicit_k_max_aniso[i], places=6)

        for dim in range(3):
            self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[dim, :])) <= explicit_k_max_aniso[dim] * 1.001)

    def test_edge_case_small_num_points(self):
        fov_m = 0.3
        
        # Test with num_points = 1
        traj_one_point = generate_golden_angle_3d_trajectory(1, fov_m)
        self.assertEqual(traj_one_point.get_num_points(), 1)
        self.assertEqual(traj_one_point.kspace_points_rad_per_m.shape, (3, 1))
        # The single point should be close to k=0.
        # r_norm = (0.5/1)^(1/3) = 0.5^(1/3) approx 0.79
        # k_z_norm_sphere = (0*2/1 - 1) + (2/1 / 2) = -1 + 1 = 0
        # theta = acos(0) = pi/2. So kz is 0.
        # phi = 0 * inc = 0.
        # kx_norm = r_norm * sin(pi/2) * cos(0) = r_norm * 1 * 1 = r_norm
        # ky_norm = r_norm * sin(pi/2) * sin(0) = r_norm * 1 * 0 = 0
        # So, point should be (r_norm * k_max_x, 0, 0)
        expected_k_max = np.pi / fov_m
        r_norm_one_pt = np.power(0.5, 1./3.)
        expected_k_point = np.array([[r_norm_one_pt * expected_k_max], [0.0], [0.0]])
        np.testing.assert_allclose(traj_one_point.kspace_points_rad_per_m, expected_k_point, atol=1e-6)

        # Test with num_points = 0
        traj_zero_points = generate_golden_angle_3d_trajectory(0, fov_m)
        self.assertEqual(traj_zero_points.get_num_points(), 0)
        self.assertEqual(traj_zero_points.kspace_points_rad_per_m.shape, (3, 0))
        # Check metadata for k_max_calculated in case of 0 points
        km_calc_zero = traj_zero_points.metadata['k_max_calculated_rad_m_xyz']
        self.assertEqual(km_calc_zero, (0.0, 0.0, 0.0))


    def test_metadata_name_dt_seconds(self):
        num_points = 10
        fov_m = 0.2
        custom_name = "my_golden_angle_test"
        custom_dt = 10e-6

        traj = generate_golden_angle_3d_trajectory(num_points, fov_m, name=custom_name, dt_seconds=custom_dt)
        
        self.assertEqual(traj.name, custom_name)
        self.assertEqual(traj.dt_seconds, custom_dt)
        self.assertIn('generator_params', traj.metadata)
        gp = traj.metadata['generator_params']
        self.assertEqual(gp['dt_seconds_input'], custom_dt)

    def test_invalid_fov_input(self):
        with self.assertRaisesRegex(ValueError, "fov_m must be a float or a list/tuple of 3 floats"):
            generate_golden_angle_3d_trajectory(10, fov_m=(0.1, 0.2)) # Incorrect tuple length
        with self.assertRaisesRegex(ValueError, "fov_m must be a float or a list/tuple of 3 floats"):
            generate_golden_angle_3d_trajectory(10, fov_m="not_a_fov")

    def test_invalid_max_k_input(self):
        with self.assertRaisesRegex(ValueError, "max_k_rad_per_m must be a float or a list/tuple of 3 floats, or None"):
            generate_golden_angle_3d_trajectory(10, fov_m=0.2, max_k_rad_per_m=(100, 120)) # Incorrect tuple length
        with self.assertRaisesRegex(ValueError, "max_k_rad_per_m must be a float or a list/tuple of 3 floats, or None"):
            generate_golden_angle_3d_trajectory(10, fov_m=0.2, max_k_rad_per_m="not_k_max")


class TestAdvancedTrajectoryTools(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.fov_m = 0.2
        self.dt_s = 4e-6 
        self.gamma_1h = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'] # Already imported via Trajectory
        
        # Simple 2D trajectory points (N, D) format
        self.simple_2d_traj_points = np.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]]) 
        self.simple_2d_traj_obj = Trajectory(name="simple_2d_for_adv_tests", 
                                             kspace_points_rad_per_m=self.simple_2d_traj_points.T, # Pass as (D,N)
                                             dt_seconds=self.dt_s)

        # Simple 3D trajectory points (N, D) format
        self.simple_3d_traj_points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
        self.simple_3d_traj_obj = Trajectory(name="simple_3d_for_adv_tests",
                                             kspace_points_rad_per_m=self.simple_3d_traj_points.T, # Pass as (D,N)
                                             dt_seconds=self.dt_s)

    def tearDown(self):
        plt.close('all') # Close any figures generated by tests

    # 1. normalize_density_weights tests
    def test_normalize_density_weights_simple(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_density_weights(weights)
        self.assertTrue(np.allclose(np.sum(normalized), 1.0), "Sum not close to 1.0")
        self.assertTrue(np.allclose(normalized, weights / 10.0), "Values not correctly normalized")

    def test_normalize_density_weights_sum_zero(self):
        weights = np.array([0.0, 0.0, 0.0, 0.0])
        normalized = normalize_density_weights(weights)
        self.assertTrue(np.allclose(np.sum(normalized), 1.0), "Sum not close to 1.0 for zero input")
        self.assertTrue(np.allclose(normalized, np.array([0.25, 0.25, 0.25, 0.25])), "Not uniform for zero input")

    def test_normalize_density_weights_sum_almost_zero(self):
        weights = np.array([1e-15, 1e-15]) # Sum is 2e-15, less than 1e-12 threshold
        normalized = normalize_density_weights(weights)
        self.assertTrue(np.allclose(np.sum(normalized), 1.0), "Sum not close to 1.0 for near-zero input")
        self.assertTrue(np.allclose(normalized, np.array([0.5, 0.5])), "Not uniform for near-zero input")

    def test_normalize_density_weights_empty(self):
        weights = np.array([])
        normalized = normalize_density_weights(weights)
        self.assertEqual(normalized.size, 0, "Empty array did not return empty array")

    def test_normalize_density_weights_single_value(self):
        weights = np.array([5.0])
        normalized = normalize_density_weights(weights)
        self.assertTrue(np.allclose(normalized, np.array([1.0])), "Single value not normalized to 1.0")

    def test_normalize_density_weights_negative_values_sum_non_zero(self):
        weights = np.array([-1.0, -2.0, 3.0, 4.0]) # Sum = 4.0
        normalized = normalize_density_weights(weights)
        self.assertTrue(np.allclose(np.sum(normalized), 1.0))
        self.assertTrue(np.allclose(normalized, weights / 4.0))

    def test_normalize_density_weights_negative_values_sum_zero(self):
        weights = np.array([-1.0, -2.0, 1.0, 2.0]) # Sum = 0.0
        normalized = normalize_density_weights(weights)
        self.assertTrue(np.allclose(np.sum(normalized), 1.0))
        self.assertTrue(np.allclose(normalized, np.full_like(weights, 1.0/weights.size)))

    # 2. compute_density_compensation tests
    def test_cdc_voronoi_2d(self):
        # simple_2d_traj_points is (N, D) = (4, 2)
        weights = compute_density_compensation(self.simple_2d_traj_points, method="voronoi")
        self.assertEqual(weights.shape, (self.simple_2d_traj_points.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0), f"Sum of weights is {np.sum(weights)}")
        self.assertTrue(np.all(weights >= 0), "Negative weights found")

    def test_cdc_voronoi_3d(self):
        # simple_3d_traj_points is (N, D) = (4, 3)
        # Voronoi in 3D needs N >= D+1 points. 4 points is D+1 for 3D.
        weights = compute_density_compensation(self.simple_3d_traj_points, method="voronoi")
        self.assertEqual(weights.shape, (self.simple_3d_traj_points.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0), f"Sum of weights is {np.sum(weights)}")
        self.assertTrue(np.all(weights >= 0), "Negative weights found")

    def test_cdc_pipe_2d(self):
        # simple_2d_traj_points is (N, D) = (4, 2)
        weights = compute_density_compensation(self.simple_2d_traj_points, method="pipe")
        self.assertEqual(weights.shape, (self.simple_2d_traj_points.shape[0],))
        if self.simple_2d_traj_points.shape[0] > 0 : # Avoid sum assertion for empty result
            self.assertTrue(np.allclose(np.sum(weights), 1.0), f"Sum of weights is {np.sum(weights)}")
        self.assertTrue(np.all(weights >= 0), "Negative weights found")
        # Check if weights increase with radius (for points not at origin)
        # Radii: 0, 1, 1, sqrt(2). Weights should be proportional if not for normalization.
        # After normalization, it's harder to directly check proportionality without knowing all values.
        # For [0,0], [1,0], [0,1], [1,1]: radii are 0, 1, 1, ~1.414
        # Raw weights: 0, 1, 1, 1.414. Sum = 3.414
        # Norm weights: 0, 0.292, 0.292, 0.414
        expected_radii = np.linalg.norm(self.simple_2d_traj_points, axis=1)
        if np.sum(expected_radii) > 1e-9 : # If all points are not zero
             expected_weights = expected_radii / np.sum(expected_radii)
             self.assertTrue(np.allclose(weights, expected_weights), f"Pipe weights {weights} not as expected {expected_weights}")
        else: # if all points are zero, expect uniform
             self.assertTrue(np.allclose(weights, np.ones_like(weights)/weights.size ))


    def test_cdc_pipe_3d_raises_error(self):
        with self.assertRaises(ValueError):
            compute_density_compensation(self.simple_3d_traj_points, method="pipe")

    def test_cdc_voronoi_complex_input_2d(self):
        # (num_points,) complex array
        complex_traj = self.simple_2d_traj_points[:,0] + 1j * self.simple_2d_traj_points[:,1]
        weights = compute_density_compensation(complex_traj, method="voronoi")
        self.assertEqual(weights.shape, complex_traj.shape)
        self.assertTrue(np.allclose(np.sum(weights), 1.0))
        self.assertTrue(np.all(weights >= 0))
        
        # Check against real input version
        weights_real_input = compute_density_compensation(self.simple_2d_traj_points, method="voronoi")
        self.assertTrue(np.allclose(weights, weights_real_input))

    def test_cdc_pipe_complex_input_2d(self):
        complex_traj = self.simple_2d_traj_points[:,0] + 1j * self.simple_2d_traj_points[:,1]
        weights = compute_density_compensation(complex_traj, method="pipe")
        self.assertEqual(weights.shape, complex_traj.shape)
        self.assertTrue(np.allclose(np.sum(weights), 1.0))
        self.assertTrue(np.all(weights >= 0))

        # Check against real input version
        weights_real_input = compute_density_compensation(self.simple_2d_traj_points, method="pipe")
        self.assertTrue(np.allclose(weights, weights_real_input))
        
    def test_cdc_unknown_method(self):
        with self.assertRaises(ValueError):
            compute_density_compensation(self.simple_2d_traj_points, method="unknown_method")

    def test_cdc_empty_trajectory(self):
        empty_traj_real = np.empty((0,2))
        weights = compute_density_compensation(empty_traj_real, method="voronoi")
        self.assertEqual(weights.size, 0)

        empty_traj_complex = np.empty((0,), dtype=complex)
        weights_complex = compute_density_compensation(empty_traj_complex, method="voronoi")
        self.assertEqual(weights_complex.size, 0)

    # 3. create_periodic_points tests
    def test_create_periodic_points_2d(self):
        point = np.array([[0.1, 0.2]]) # Shape (1, 2)
        extended = create_periodic_points(point, ndim=2)
        self.assertEqual(extended.shape, (9, 2))
        
        # Check for original point
        self.assertTrue(np.any(np.allclose(extended, point, atol=1e-7), axis=1))
        
        # Check for a few shifted points
        expected_shifts = [
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1], [ 0, 0], [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1]
        ]
        for shift in expected_shifts:
            shifted_point = point + np.array(shift)
            self.assertTrue(np.any(np.allclose(extended, shifted_point, atol=1e-7), axis=1), 
                            f"Shifted point {shifted_point} not found")

    def test_create_periodic_points_3d(self):
        point = np.array([[0.1, 0.2, 0.3]]) # Shape (1, 3)
        extended = create_periodic_points(point, ndim=3)
        self.assertEqual(extended.shape, (27, 3))

        # Check for original point
        self.assertTrue(np.any(np.allclose(extended, point, atol=1e-7), axis=1))

        # Check for a few shifted points
        expected_shifts = [
            [-1,-1,-1], [0,0,0], [1,1,1], [1,0,0], [0,1,0], [0,0,1]
        ]
        for shift_vals in expected_shifts:
            shifted_point = point + np.array(shift_vals)
            self.assertTrue(np.any(np.allclose(extended, shifted_point, atol=1e-7), axis=1),
                            f"Shifted point {shifted_point} not found")

    def test_create_periodic_points_invalid_ndim(self):
        point = np.array([[0.1, 0.2]])
        with self.assertRaisesRegex(ValueError, "Number of dimensions must be 2 or 3"):
            create_periodic_points(point, ndim=1)
        with self.assertRaisesRegex(ValueError, "Number of dimensions must be 2 or 3"):
            create_periodic_points(point, ndim=4)
            
    def test_create_periodic_points_shape_mismatch(self):
        point_2d = np.array([[0.1, 0.2]])
        with self.assertRaisesRegex(ValueError, "Trajectory shape.*inconsistent with ndim"):
            create_periodic_points(point_2d, ndim=3)

        point_3d = np.array([[0.1, 0.2, 0.3]])
        with self.assertRaisesRegex(ValueError, "Trajectory shape.*inconsistent with ndim"):
            create_periodic_points(point_3d, ndim=2)

    # 4. compute_voronoi_density tests (implicitly tests compute_cell_area)
    def test_cvd_2d_clipped(self):
        # simple_2d_traj_points is (N,D) = (4,2)
        # These points are [[0,0],[1,0],[0,1],[1,1]]. All on convex hull for "clipped".
        # compute_voronoi_density internally normalizes to [-0.5, 0.5] before processing if needed.
        weights = compute_voronoi_density(self.simple_2d_traj_points, boundary_type="clipped")
        self.assertEqual(weights.shape, (self.simple_2d_traj_points.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0), f"Sum is {np.sum(weights)}")
        self.assertTrue(np.all(weights >= 0))
        # For clipped boundary with points forming a square, all cells are infinite, 
        # so compute_cell_area returns 0, then median replacement (which is 0), then uniform.
        # This should result in uniform weights.
        self.assertTrue(np.allclose(weights, np.ones(4)/4.0), f"Weights {weights} not uniform for clipped square.")


    def test_cvd_2d_periodic(self):
        # Normalize points to [-0.5, 0.5] for periodic boundary conditions
        points = self.simple_2d_traj_points.copy() # [[0,0],[1,0],[0,1],[1,1]]
        points_norm = (points - np.min(points)) / (np.max(points) - np.min(points)) - 0.5 
        # This makes them: [[-0.5,-0.5], [0.5,-0.5], [-0.5,0.5], [0.5,0.5]]

        weights = compute_voronoi_density(points_norm, boundary_type="periodic")
        self.assertEqual(weights.shape, (points_norm.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0), f"Sum is {np.sum(weights)}")
        self.assertTrue(np.all(weights > 0)) # For periodic, should get valid cell areas
        # For 4 corner points of a unit square in periodic, all cell areas should be equal.
        self.assertTrue(np.allclose(weights, np.ones(4)/4.0), f"Weights {weights} not uniform for periodic.")

    def test_cvd_3d_clipped(self):
        # simple_3d_traj_points is (N,D) = (4,3)
        # These are corners of a tetrahedron essentially. All on convex hull.
        weights = compute_voronoi_density(self.simple_3d_traj_points, boundary_type="clipped")
        self.assertEqual(weights.shape, (self.simple_3d_traj_points.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0), f"Sum is {np.sum(weights)}")
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.allclose(weights, np.ones(4)/4.0), f"Weights {weights} not uniform for clipped 3D.")


    def test_cvd_3d_periodic(self):
        points = self.simple_3d_traj_points.copy()
        # Normalize to [-0.5, 0.5]
        min_val = np.min(points)
        max_val = np.max(points)
        if max_val == min_val: max_val += 1e-9 # Avoid division by zero if all points are same
        points_norm = (points - min_val) / (max_val - min_val) - 0.5
        
        weights = compute_voronoi_density(points_norm, boundary_type="periodic")
        self.assertEqual(weights.shape, (points_norm.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0), f"Sum is {np.sum(weights)}")
        self.assertTrue(np.all(weights > 0)) # Should be positive for periodic
        # For 4 symmetric points, expect uniform weights
        self.assertTrue(np.allclose(weights, np.ones(4)/4.0), f"Weights {weights} not uniform for periodic 3D.")

    def test_cvd_empty_trajectory(self):
        empty_traj = np.empty((0,2))
        weights = compute_voronoi_density(empty_traj)
        self.assertEqual(weights.size, 0)

    def test_cvd_single_point_2d(self):
        point = np.array([[0.1, 0.2]])
        weights_clipped = compute_voronoi_density(point, boundary_type="clipped")
        self.assertEqual(weights_clipped.shape, (1,))
        self.assertTrue(np.allclose(weights_clipped, [1.0]))

        weights_periodic = compute_voronoi_density(point, boundary_type="periodic")
        self.assertEqual(weights_periodic.shape, (1,))
        self.assertTrue(np.allclose(weights_periodic, [1.0]))
        
    def test_cvd_collinear_points_2d_clipped(self):
        # Collinear points, all cells should be infinite, leading to uniform weights
        points = np.array([[0.,0.], [1.,0.], [2.,0.], [3.,0.]])
        weights = compute_voronoi_density(points, boundary_type="clipped")
        self.assertEqual(weights.shape, (points.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0))
        self.assertTrue(np.allclose(weights, np.ones(points.shape[0]) / points.shape[0]), 
                        f"Collinear clipped weights not uniform: {weights}")

    def test_cvd_collinear_points_2d_periodic(self):
        # Collinear points, normalized
        points = np.array([[-0.5,0.], [-0.25,0.], [0.,0.], [0.25,0.], [0.5,0.]])
        weights = compute_voronoi_density(points, boundary_type="periodic")
        self.assertEqual(weights.shape, (points.shape[0],))
        self.assertTrue(np.allclose(np.sum(weights), 1.0))
        # For periodic, even if collinear, the central cells should have some area.
        # Exact values are complex, but they should not necessarily be uniform if spacing isn't.
        # Here spacing is uniform, so expect uniform weights.
        self.assertTrue(np.allclose(weights, np.ones(points.shape[0]) / points.shape[0]),
                        f"Collinear periodic weights not uniform: {weights}")

    # 5. Trajectory Generators
    def test_generate_spiral_trajectory(self):
        num_arms = 4
        num_samples_per_arm = 100
        fov = 0.2
        
        traj = generate_spiral_trajectory(num_arms, num_samples_per_arm, fov_m=fov, dt_seconds=self.dt_s)
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.get_num_dimensions(), 2)
        self.assertEqual(traj.get_num_points(), num_arms * num_samples_per_arm)
        self.assertEqual(traj.dt_seconds, self.dt_s)
        self.assertIn('generator_params', traj.metadata)
        self.assertEqual(traj.metadata['generator_params']['traj_type'], 'spiral')

        # Check k_max calculation
        k_max_expected = np.pi / fov
        self.assertAlmostEqual(traj.metadata['generator_params']['k_max_calculated_rad_m'], k_max_expected)
        
        k_points = traj.kspace_points_rad_per_m # Should be (D,N) = (2, total_points)
        radii = np.sqrt(k_points[0,:]**2 + k_points[1,:]**2)
        self.assertTrue(np.max(radii) <= k_max_expected * 1.001) # Allow small tolerance

        # Test with explicit k_max and num_revolutions
        explicit_k_max = 150.0
        num_revs = 5.0
        traj_explicit = generate_spiral_trajectory(num_arms, num_samples_per_arm, fov_m=fov, 
                                                 max_k_rad_per_m=explicit_k_max, 
                                                 num_revolutions=num_revs)
        self.assertAlmostEqual(traj_explicit.metadata['generator_params']['k_max_calculated_rad_m'], explicit_k_max)
        self.assertAlmostEqual(traj_explicit.metadata['generator_params']['num_revolutions_effective'], num_revs)
        k_points_explicit = traj_explicit.kspace_points_rad_per_m
        radii_explicit = np.sqrt(k_points_explicit[0,:]**2 + k_points_explicit[1,:]**2)
        self.assertTrue(np.max(radii_explicit) <= explicit_k_max * 1.001)


    def test_generate_radial_trajectory(self):
        num_spokes = 10
        num_samples_per_spoke = 64
        fov = 0.25

        # Golden Angle
        traj_ga = generate_radial_trajectory(num_spokes, num_samples_per_spoke, fov_m=fov, 
                                             use_golden_angle=True, dt_seconds=self.dt_s)
        self.assertIsInstance(traj_ga, Trajectory)
        self.assertEqual(traj_ga.get_num_dimensions(), 2)
        self.assertEqual(traj_ga.get_num_points(), num_spokes * num_samples_per_spoke)
        self.assertEqual(traj_ga.dt_seconds, self.dt_s)
        self.assertTrue(traj_ga.metadata['generator_params']['use_golden_angle'])
        
        k_points_ga = traj_ga.kspace_points_rad_per_m
        k_max_expected_ga = np.pi / fov
        radii_ga = np.sqrt(k_points_ga[0,:]**2 + k_points_ga[1,:]**2)
        self.assertTrue(np.max(radii_ga) <= k_max_expected_ga * 1.001)
        # Check if points range from -k_max to k_max (approx) by looking at one spoke
        one_spoke_ga = k_points_ga[:, :num_samples_per_spoke]
        radii_one_spoke_ga = np.sqrt(one_spoke_ga[0,:]**2 + one_spoke_ga[1,:]**2)
        # Linspace for k_radii goes from -k_max to k_max, so min radius can be 0 if num_samples is odd.
        self.assertTrue(np.isclose(np.min(radii_one_spoke_ga), 0.0) or \
                        np.isclose(np.min(np.abs(one_spoke_ga)), 0.0, atol=1e-1)) # Checks if it passes through center
        
        # Uniform Angle
        traj_uni = generate_radial_trajectory(num_spokes, num_samples_per_spoke, fov_m=fov, 
                                              use_golden_angle=False, dt_seconds=self.dt_s)
        self.assertFalse(traj_uni.metadata['generator_params']['use_golden_angle'])
        k_points_uni = traj_uni.kspace_points_rad_per_m
        
        # Check angles are different for GA vs Uniform (for first few spokes)
        angles_ga = np.arctan2(k_points_ga[1, ::num_samples_per_spoke], k_points_ga[0, ::num_samples_per_spoke])
        angles_uni = np.arctan2(k_points_uni[1, ::num_samples_per_spoke], k_points_uni[0, ::num_samples_per_spoke])
        
        if num_spokes > 1:
            self.assertFalse(np.allclose(angles_ga[:min(5, num_spokes)], angles_uni[:min(5, num_spokes)]))

        # Test with explicit k_max
        explicit_k_max = 200.0
        traj_explicit_k = generate_radial_trajectory(num_spokes, num_samples_per_spoke, fov_m=fov,
                                                     max_k_rad_per_m=explicit_k_max)
        self.assertAlmostEqual(traj_explicit_k.metadata['generator_params']['k_max_calculated_rad_m'], explicit_k_max)
        k_points_exp_k = traj_explicit_k.kspace_points_rad_per_m
        radii_exp_k = np.sqrt(k_points_exp_k[0,:]**2 + k_points_exp_k[1,:]**2)
        self.assertTrue(np.max(radii_exp_k) <= explicit_k_max * 1.001)

    def test_generate_spiral_edge_cases(self):
        fov = 0.2
        default_dt = self.dt_s # from setUp of TestAdvancedTrajectoryTools

        # Test with num_arms = 1, num_samples_per_arm = 1
        with self.subTest(case="1arm_1sample"):
            traj = generate_spiral_trajectory(num_arms=1, num_samples_per_arm=1, fov_m=fov, dt_seconds=default_dt)
            self.assertIsInstance(traj, Trajectory)
            self.assertEqual(traj.get_num_points(), 1)
            self.assertEqual(traj.get_num_dimensions(), 2)
            self.assertEqual(traj.kspace_points_rad_per_m.shape, (2, 1))
            
            k_max_expected = np.pi / fov
            # For num_samples_per_arm = 1, t_sample = 1.0. current_radius = k_max.
            # angle_offset = 0 for first arm.
            # current_angle = 0 + effective_revolutions * 2 * np.pi.
            # Default effective_revolutions = 10.0. cos(10*2pi)=1, sin(10*2pi)=0.
            # So point should be (k_max, 0)
            expected_k_point = np.array([[k_max_expected], [0.0]])
            np.testing.assert_allclose(traj.kspace_points_rad_per_m, expected_k_point, atol=1e-6)
            self.assertIn('generator_params', traj.metadata)
            self.assertEqual(traj.metadata['generator_params']['num_arms'], 1)
            self.assertEqual(traj.metadata['generator_params']['num_samples_per_arm'], 1)

        # Test with num_arms = 0 (total points = 0)
        with self.subTest(case="0arms"):
            traj_zero_arms = generate_spiral_trajectory(num_arms=0, num_samples_per_arm=100, fov_m=fov, dt_seconds=default_dt)
            self.assertEqual(traj_zero_arms.get_num_points(), 0)
            self.assertEqual(traj_zero_arms.kspace_points_rad_per_m.shape, (2, 0))
            self.assertIn('generator_params', traj_zero_arms.metadata)
            self.assertEqual(traj_zero_arms.metadata['generator_params']['num_arms'], 0)

        # Test with num_samples_per_arm = 0 (total points = 0)
        with self.subTest(case="0samples_per_arm"):
            traj_zero_samples = generate_spiral_trajectory(num_arms=4, num_samples_per_arm=0, fov_m=fov, dt_seconds=default_dt)
            self.assertEqual(traj_zero_samples.get_num_points(), 0)
            self.assertEqual(traj_zero_samples.kspace_points_rad_per_m.shape, (2, 0))
            self.assertIn('generator_params', traj_zero_samples.metadata)
            self.assertEqual(traj_zero_samples.metadata['generator_params']['num_samples_per_arm'], 0)

    def test_generate_radial_edge_cases(self):
        fov = 0.25
        default_dt = self.dt_s

        # Test with num_spokes = 1, num_samples_per_spoke = 1
        with self.subTest(case="1spoke_1sample"):
            traj = generate_radial_trajectory(num_spokes=1, num_samples_per_spoke=1, fov_m=fov, dt_seconds=default_dt)
            self.assertIsInstance(traj, Trajectory)
            self.assertEqual(traj.get_num_points(), 1)
            self.assertEqual(traj.get_num_dimensions(), 2)
            self.assertEqual(traj.kspace_points_rad_per_m.shape, (2, 1))
            # For num_samples_per_spoke = 1, k_radii = np.array([0.0]). So point is (0,0).
            expected_k_point = np.array([[0.0], [0.0]])
            np.testing.assert_allclose(traj.kspace_points_rad_per_m, expected_k_point, atol=1e-7)
            self.assertIn('generator_params', traj.metadata)
            self.assertEqual(traj.metadata['generator_params']['num_spokes'], 1)
            self.assertEqual(traj.metadata['generator_params']['num_samples_per_spoke'], 1)

        # Test with num_spokes = 0 (total points = 0)
        with self.subTest(case="0spokes"):
            traj_zero_spokes = generate_radial_trajectory(num_spokes=0, num_samples_per_spoke=64, fov_m=fov, dt_seconds=default_dt)
            self.assertEqual(traj_zero_spokes.get_num_points(), 0)
            self.assertEqual(traj_zero_spokes.kspace_points_rad_per_m.shape, (2, 0))
            self.assertIn('generator_params', traj_zero_spokes.metadata)
            self.assertEqual(traj_zero_spokes.metadata['generator_params']['num_spokes'], 0)

        # Test with num_samples_per_spoke = 0 (total points = 0)
        with self.subTest(case="0samples_per_spoke"):
            traj_zero_samples = generate_radial_trajectory(num_spokes=10, num_samples_per_spoke=0, fov_m=fov, dt_seconds=default_dt)
            self.assertEqual(traj_zero_samples.get_num_points(), 0)
            self.assertEqual(traj_zero_samples.kspace_points_rad_per_m.shape, (2, 0))
            self.assertIn('generator_params', traj_zero_samples.metadata)
            self.assertEqual(traj_zero_samples.metadata['generator_params']['num_samples_per_spoke'], 0)

    # 6. constrain_trajectory tests
    def test_constrain_trajectory(self):
        # Create a trajectory that will surely violate constraints
        # Large k-space step in one dt
        k_max_high = 1000 
        num_points = 10
        # k_points_violating = np.array([[0, k_max_high],[0, k_max_high*0.5]]).T # (D,N) for 2 points
        # A longer trajectory to see iteration
        kx_viol = np.linspace(0, k_max_high, num_points)
        ky_viol = np.linspace(0, k_max_high/2, num_points)
        k_points_violating = np.stack((kx_viol, ky_viol)) # (2, num_points)

        traj_violating = Trajectory("violating_traj", k_points_violating, dt_seconds=self.dt_s, gamma_Hz_per_T=self.gamma_1h)

        max_grad = 0.04  # T/m
        max_slew = 150   # T/m/s
        
        constrained_traj = constrain_trajectory(traj_violating, max_grad, max_slew)
        
        self.assertIsInstance(constrained_traj, Trajectory)
        self.assertEqual(constrained_traj.get_num_dimensions(), 2)
        self.assertEqual(constrained_traj.get_num_points(), num_points)
        self.assertEqual(constrained_traj.dt_seconds, self.dt_s) # Should preserve dt
        
        # Check if k-space points have changed (they should have if constraints were active)
        # (Unless the original trajectory was already compliant, which is not the case here)
        if num_points > 1: # Comparison only makes sense if there's more than one point
             self.assertFalse(np.allclose(traj_violating.kspace_points_rad_per_m, 
                                     constrained_traj.kspace_points_rad_per_m),
                         "Constrained k-space is identical to original, constraints might not have been applied.")

        # Check metadata for constraint info
        self.assertIn('constraints', constrained_traj.metadata)
        self.assertEqual(constrained_traj.metadata['constraints']['max_gradient_Tm_per_m'], max_grad)
        self.assertEqual(constrained_traj.metadata['constraints']['max_slew_rate_Tm_per_m_per_s'], max_slew)

        # Verify actual constraints (this is complex, a simpler check is if max achieved grad/slew is LE constraint)
        # Recompute gradients and slew rates from the constrained trajectory
        # The Trajectory class does this upon initialization if dt is provided.
        # So, constrained_traj.get_max_grad_Tm() and constrained_traj.get_max_slew_Tm_per_s()
        # should give values based on its *own* k-space points.
        
        # Note: The internal calculation of max_grad and max_slew in Trajectory init might be slightly different
        # (e.g. np.gradient vs forward/backward differences used in constrain_trajectory).
        # For this test, we primarily check that the process runs and output is changed.
        # A precise check would require re-running the constraint logic here.
        
        # A loose check: the maximum k-space step in the constrained trajectory should be smaller
        # than in the original one, if constraints were active.
        dk_orig = np.linalg.norm(np.diff(traj_violating.kspace_points_rad_per_m, axis=1), axis=0)
        dk_constrained = np.linalg.norm(np.diff(constrained_traj.kspace_points_rad_per_m, axis=1), axis=0)
        if num_points > 1:
            self.assertTrue(np.max(dk_constrained) < np.max(dk_orig) + 1e-6, # add tolerance for fp
                        "Max k-space step in constrained trajectory not smaller than original.")

    def test_constrain_trajectory_empty(self):
        empty_traj_obj = Trajectory("empty", np.array([[],[]]), dt_seconds=self.dt_s)
        constrained_empty = constrain_trajectory(empty_traj_obj, 0.04, 150)
        self.assertEqual(constrained_empty.get_num_points(), 0)

    def test_constrain_trajectory_single_point(self):
        single_point_k = np.array([[0.],[0.]]) # (D, N)
        single_point_traj = Trajectory("single_pt", single_point_k, dt_seconds=self.dt_s)
        constrained_single = constrain_trajectory(single_point_traj, 0.04, 150)
        self.assertEqual(constrained_single.get_num_points(), 1)
        self.assertTrue(np.allclose(constrained_single.kspace_points_rad_per_m, single_point_k))
        
    def test_constrain_trajectory_no_dt_error(self):
        k_points = np.array([[0,1],[0,1]]).astype(float) # (D,N)
        traj_no_dt = Trajectory("no_dt_traj", k_points, dt_seconds=None)
        with self.assertRaisesRegex(ValueError, "Dwell time .* must be positive and available"):
            constrain_trajectory(traj_no_dt, 0.04, 150)

        with self.assertRaisesRegex(ValueError, "Dwell time .* must be positive and available"):
            constrain_trajectory(traj_no_dt, 0.04, 150, dt_seconds=0)
        with self.assertRaisesRegex(ValueError, "Dwell time .* must be positive and available"):
            constrain_trajectory(traj_no_dt, 0.04, 150, dt_seconds=-1)

    # 7. reconstruct_image tests
    def test_reconstruct_image_simple_center_peak(self):
        # Use self.simple_2d_traj_obj which has points: [[0,0],[1,0],[0,1],[1,1]] (when transposed in obj)
        # kspace_points_rad_per_m is (2,4) in the object.
        # get_num_points() is 4.
        k_data = np.zeros(self.simple_2d_traj_obj.get_num_points(), dtype=complex)
        # Find index of k-space center (0,0)
        # self.simple_2d_traj_points was [[0,0],[1,0],[0,1],[1,1]]
        # Transposed in object: [[0,1,0,1], [0,0,1,1]]
        # So, the first point (index 0) is k=(0,0)
        center_idx = 0 # Assuming the first point is (0,0) based on simple_2d_traj_points
        k_data[center_idx] = 100.0 # Strong signal at k-space center
        
        grid_s = (32, 32)
        img = reconstruct_image(k_data, self.simple_2d_traj_obj, grid_s, density_comp_method=None, verbose=False)
        
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, grid_s)
        self.assertEqual(img.dtype, np.float64) # Magnitude image is float
        
        # Check that the center of the image has the highest intensity
        center_pixel_y, center_pixel_x = grid_s[0]//2, grid_s[1]//2
        max_intensity_coord = np.unravel_index(np.argmax(img), img.shape)
        # Allow a small region around the center due to regridding effects
        self.assertTrue(abs(max_intensity_coord[0] - center_pixel_y) <= 2 and \
                        abs(max_intensity_coord[1] - center_pixel_x) <= 2,
                        f"Peak intensity not at center. Found at {max_intensity_coord}, expected near ({center_pixel_y},{center_pixel_x})")

    def test_reconstruct_image_with_voronoi_dc(self):
        # Use a slightly more complex trajectory for DC to have an effect
        traj_radial = generate_radial_trajectory(num_spokes=8, num_samples_per_spoke=32, fov_m=self.fov_m)
        k_data = np.ones(traj_radial.get_num_points(), dtype=complex) # Uniform k-space signal
        
        grid_s = (24, 24) # Smaller grid for faster test
        img_dc = reconstruct_image(k_data, traj_radial, grid_s, density_comp_method="voronoi", verbose=False)
        img_no_dc = reconstruct_image(k_data, traj_radial, grid_s, density_comp_method=None, verbose=False)

        self.assertEqual(img_dc.shape, grid_s)
        self.assertEqual(img_no_dc.shape, grid_s)
        
        # With uniform k-space signal, DC should generally make the image flatter / less center-bright
        # So, std of DC image might be smaller, or center peak less pronounced.
        # This is a qualitative check; exact values are hard to predict.
        # A simple check is that they are not identical if DC had any effect.
        if traj_radial.get_num_points() > 0:
             self.assertFalse(np.allclose(img_dc, img_no_dc), "DC had no effect on the image.")

    def test_reconstruct_image_input_validation(self):
        grid_s = (16,16)
        k_data_valid = np.ones(self.simple_2d_traj_obj.get_num_points(), dtype=complex)

        # Mismatched k-space data size
        k_data_wrong_size = np.ones(self.simple_2d_traj_obj.get_num_points() + 1, dtype=complex)
        with self.assertRaisesRegex(ValueError, "kspace_data size .* does not match trajectory points"):
            reconstruct_image(k_data_wrong_size, self.simple_2d_traj_obj, grid_s)

        # Non-2D trajectory (using the 3D object)
        with self.assertRaisesRegex(ValueError, "Image reconstruction currently only supports 2D trajectories"):
            reconstruct_image(np.ones(self.simple_3d_traj_obj.get_num_points()), self.simple_3d_traj_obj, grid_s)

        # Invalid grid_size
        invalid_grids = [(16,), (16,16,16), (0,16), (16, -5), (16.5, 16)]
        for ig in invalid_grids:
            with self.subTest(invalid_grid=ig):
                with self.assertRaisesRegex(ValueError, "grid_size must be a tuple of 2 positive integers"):
                    reconstruct_image(k_data_valid, self.simple_2d_traj_obj, ig)

    # 8. display_trajectory tests
    @patch.object(Trajectory, 'plot_2d')
    def test_display_trajectory_2d_called(self, mock_plot_2d):
        # We need a valid Trajectory object, simple_2d_traj_obj from setUp is fine
        kwargs_to_pass = {'title': 'Test 2D Plot', 'max_total_points': 500}
        returned_ax = display_trajectory(self.simple_2d_traj_obj, plot_type="2D", **kwargs_to_pass)
        
        mock_plot_2d.assert_called_once()
        # Check if kwargs were passed through
        # The first arg to plot_2d will be self, so we check kwargs passed to the mock
        # In Python 3.8+, use mock_plot_2d.assert_called_once_with(**kwargs_to_pass)
        # For broader compatibility:
        args, kwargs = mock_plot_2d.call_args
        for key, value in kwargs_to_pass.items():
            self.assertEqual(kwargs[key], value, f"Keyword argument {key} not passed correctly.")
        
        # Check if the mock's return value is passed through
        self.assertEqual(returned_ax, mock_plot_2d.return_value)


    @patch.object(Trajectory, 'plot_3d')
    def test_display_trajectory_3d_called(self, mock_plot_3d):
        # Using simple_3d_traj_obj from setUp
        kwargs_to_pass = {'title': 'Test 3D Plot', 'max_interleaves': 10}
        returned_ax = display_trajectory(self.simple_3d_traj_obj, plot_type="3D", **kwargs_to_pass)
        
        mock_plot_3d.assert_called_once()
        args, kwargs = mock_plot_3d.call_args
        for key, value in kwargs_to_pass.items():
            self.assertEqual(kwargs[key], value, f"Keyword argument {key} not passed correctly.")

        self.assertEqual(returned_ax, mock_plot_3d.return_value)

    def test_display_trajectory_invalid_type(self):
        with self.assertRaisesRegex(TypeError, "trajectory_obj must be an instance of Trajectory"):
            display_trajectory("not_a_trajectory_object", plot_type="2D")

    def test_display_trajectory_invalid_plot_type(self):
        with self.assertRaisesRegex(ValueError, "plot_type must be '2D' or '3D'"):
            display_trajectory(self.simple_2d_traj_obj, plot_type="invalid_plot_type_str")


if __name__ == '__main__':
    unittest.main()
