import unittest
import numpy as np
import os
import sys
import io
import contextlib

# Add the parent directory to the Python path to allow importing trajgen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajgen.trajectory import Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T
from trajgen.trajectory import normalize_density_weights, create_periodic_points, compute_density_compensation, compute_voronoi_density #, compute_cell_area # For new test class
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TestSharedConstants(unittest.TestCase):
    def test_common_nuclei_gamma(self):
        self.assertIsNotNone(COMMON_NUCLEI_GAMMA_HZ_PER_T)
        self.assertIsInstance(COMMON_NUCLEI_GAMMA_HZ_PER_T, dict)
        expected_keys = ['1H', '13C', '31P', '19F', '23Na', '129Xe', '2H', '7Li']
        for key in expected_keys:
            self.assertIn(key, COMMON_NUCLEI_GAMMA_HZ_PER_T)
            self.assertIsInstance(COMMON_NUCLEI_GAMMA_HZ_PER_T[key], float)
            self.assertGreater(COMMON_NUCLEI_GAMMA_HZ_PER_T[key], 0)
        self.assertGreaterEqual(len(COMMON_NUCLEI_GAMMA_HZ_PER_T), len(expected_keys))
        self.assertAlmostEqual(COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'], 42.576e6, places=3)

class TestTrajectory(unittest.TestCase):
    def setUp(self):
        self.dt = 4e-6
        self.gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
        self.n_points = 100
        self.k_max = 1.0 / (2 * 0.004)
        self.kspace_1d_flat = np.linspace(-self.k_max, self.k_max, self.n_points) # Shape (100,)
        self.kspace_1d = self.kspace_1d_flat.reshape(1, self.n_points) # Shape (1,100) for Trajectory class
        
        kx_2d = np.linspace(-self.k_max, self.k_max, self.n_points)
        ky_2d = np.zeros_like(kx_2d)
        self.kspace_2d = np.stack([kx_2d, ky_2d]) # D=2, N=100
        
        kz_3d = np.linspace(0, self.k_max, self.n_points)
        self.kspace_3d = np.stack([kx_2d, ky_2d, kz_3d]) # D=3, N=100

        self.metadata_example = {'info': 'test_trajectory'}
        self.dead_time_start = 0.001
        self.dead_time_end = 0.0005

        self.kspace_2d_plot_test = np.random.rand(2, 20) * 250 - 125 # D=2, N=20
        self.kspace_3d_plot_test = np.random.rand(3, 30) * 250 - 125 # D=3, N=30
        self.kspace_empty_2d = np.empty((2,0))
        self.kspace_empty_3d = np.empty((3,0))

        self.kspace_voronoi_good = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]]).T # D=2, N=5
        self.kspace_voronoi_bad = np.array([[0,0],[1,0]]).T # D=2, N=2 (too few for Voronoi)
        self.kspace_voronoi_1d_for_test = np.array([[-1, 0, 1, 2, 3]]).T # N=5, D=1 (when transposed in class) -> (1,5) in class
        self.kspace_voronoi_3d_for_test = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0.5]]).T # D=3, N=5

    def tearDown(self):
        plt.close('all')

    def test_trajectory_initialization_basic(self):
        traj = Trajectory("test1D", self.kspace_1d, dt_seconds=self.dt, metadata=self.metadata_example)
        self.assertEqual(traj.name, "test1D")
        self.assertEqual(traj.get_num_dimensions(), 1)
        self.assertEqual(traj.get_num_points(), self.n_points)

    def test_trajectory_initialization_with_gradients(self):
        dummy_gradients_dn = np.ones_like(self.kspace_2d)
        traj = Trajectory("test2D_with_grads_dn", self.kspace_2d,
                          gradient_waveforms_Tm=dummy_gradients_dn, dt_seconds=self.dt)
        self.assertEqual(traj.get_num_dimensions(), 2); self.assertEqual(traj.get_num_points(), 100)
        np.testing.assert_array_equal(traj.get_gradient_waveforms_Tm(), dummy_gradients_dn)

        dummy_gradients_nd = dummy_gradients_dn.T
        traj_nd_grad = Trajectory("test2D_ND_grads", self.kspace_2d,
                                  gradient_waveforms_Tm=dummy_gradients_nd, dt_seconds=self.dt)
        self.assertEqual(traj_nd_grad.get_num_dimensions(), 2); self.assertEqual(traj_nd_grad.get_num_points(), 100)
        np.testing.assert_array_equal(traj_nd_grad.get_gradient_waveforms_Tm(), dummy_gradients_dn)

    def test_get_gradient_waveforms_Tm_detailed(self):
        traj_empty_dn = Trajectory("empty_dn", np.empty((2,0)), dt_seconds=self.dt)
        self.assertEqual(traj_empty_dn.get_num_dimensions(), 2); self.assertEqual(traj_empty_dn.get_num_points(), 0)
        g = traj_empty_dn.get_gradient_waveforms_Tm(); self.assertIsNotNone(g); self.assertEqual(g.shape, (2,0))

        traj_empty_nd = Trajectory("empty_nd", np.empty((0,2)), dt_seconds=self.dt)
        self.assertEqual(traj_empty_nd.get_num_dimensions(), 2); self.assertEqual(traj_empty_nd.get_num_points(), 0)
        g_nd = traj_empty_nd.get_gradient_waveforms_Tm(); self.assertIsNotNone(g_nd); self.assertEqual(g_nd.shape, (2,0))

        traj_empty_flat = Trajectory("empty_flat", np.array([]), dt_seconds=self.dt)
        self.assertEqual(traj_empty_flat.get_num_dimensions(), 1); self.assertEqual(traj_empty_flat.get_num_points(), 0)
        g_flat = traj_empty_flat.get_gradient_waveforms_Tm(); self.assertIsNotNone(g_flat); self.assertEqual(g_flat.shape, (1,0))

        k_single_d1 = np.array([[1.],[2.]])
        traj_single_d1 = Trajectory("single_d1", k_single_d1, dt_seconds=self.dt)
        self.assertEqual(traj_single_d1.get_num_dimensions(), 2); self.assertEqual(traj_single_d1.get_num_points(), 1)
        np.testing.assert_array_equal(traj_single_d1.get_gradient_waveforms_Tm(), np.zeros((2,1)))

        k_single_1d = np.array([[1., 2.]])
        traj_single_1d = Trajectory("single_1d", k_single_1d, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)
        self.assertEqual(traj_single_1d.get_num_dimensions(), 1); self.assertEqual(traj_single_1d.get_num_points(), 2)
        g_single_1d_expected = np.gradient(k_single_1d, self.dt, axis=1) / self.gamma
        np.testing.assert_array_almost_equal(traj_single_1d.get_gradient_waveforms_Tm(), g_single_1d_expected)

        k_single_flat_d = np.array([1.,2.,3.])
        traj_single_flat_d = Trajectory("single_flat_d", k_single_flat_d, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)
        self.assertEqual(traj_single_flat_d.get_num_dimensions(), 1); self.assertEqual(traj_single_flat_d.get_num_points(), 3)
        g_single_flat_d_expected = np.gradient(k_single_flat_d.reshape(1,3), self.dt, axis=1) / self.gamma
        np.testing.assert_array_almost_equal(traj_single_flat_d.get_gradient_waveforms_Tm(), g_single_flat_d_expected)

        k_dn = np.array([[1,2,3,4,5],[6,7,8,9,10]], dtype=float)
        traj_dn = Trajectory("dn", k_dn, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)
        g_dn_expected = np.gradient(k_dn, self.dt, axis=1) / self.gamma
        np.testing.assert_array_almost_equal(traj_dn.get_gradient_waveforms_Tm(), g_dn_expected)

        k_nd_ambiguous = np.array([[1,4],[2,5],[3,6]], dtype=float)
        traj_nd_ambiguous = Trajectory("nd_ambiguous", k_nd_ambiguous, dt_seconds=self.dt, gamma_Hz_per_T=self.gamma)
        self.assertEqual(traj_nd_ambiguous.get_num_dimensions(), 2)
        self.assertEqual(traj_nd_ambiguous.get_num_points(), 3)
        np.testing.assert_array_almost_equal(traj_nd_ambiguous.kspace_points_rad_per_m, k_nd_ambiguous.T)
        grads_nd_ambiguous = traj_nd_ambiguous.get_gradient_waveforms_Tm()
        self.assertEqual(grads_nd_ambiguous.shape, (2,3))
        expected_g_nd_ambiguous = np.gradient(k_nd_ambiguous.T, self.dt, axis=1) / self.gamma
        np.testing.assert_array_almost_equal(grads_nd_ambiguous, expected_g_nd_ambiguous)

        traj_dt_none = Trajectory("dt_none", self.kspace_1d, dt_seconds=None)
        self.assertIsNone(traj_dt_none.get_gradient_waveforms_Tm())
        traj_dt_zero = Trajectory("dt_zero", self.kspace_1d, dt_seconds=0)
        self.assertIsNone(traj_dt_zero.get_gradient_waveforms_Tm(), "Grads should be None if dt is 0")

    def test_compute_metrics_edge_cases(self):
        traj_empty = Trajectory("empty_metrics", np.empty((2,0)), dt_seconds=self.dt)
        self.assertIsNone(traj_empty.metadata.get('max_slew_rate_Tm_per_s'))
        self.assertIsNone(traj_empty.metadata.get('pns_max_abs_gradient_sum_xyz'))
        self.assertIsNone(traj_empty.metadata.get('pns_max_abs_slew_sum_xyz'))
        self.assertIsNone(traj_empty.metadata.get('fov_estimate_m'))
        self.assertIsNone(traj_empty.metadata.get('resolution_overall_estimate_m'))

        traj_single_k0 = Trajectory("single_k0", np.zeros((2,1)), dt_seconds=self.dt)
        self.assertEqual(traj_single_k0.get_num_dimensions(),2)
        self.assertEqual(traj_single_k0.get_num_points(),1)
        np.testing.assert_array_almost_equal(traj_single_k0.get_gradient_waveforms_Tm(), np.zeros((2,1)))
        self.assertEqual(traj_single_k0.metadata.get('max_slew_rate_Tm_per_s'), 0.0)
        self.assertEqual(traj_single_k0.metadata.get('pns_max_abs_gradient_sum_xyz'), 0.0)
        self.assertEqual(traj_single_k0.metadata.get('pns_max_abs_slew_sum_xyz'), 0.0)
        self.assertIsNone(traj_single_k0.metadata.get('fov_estimate_m'))
        self.assertIsNone(traj_single_k0.metadata.get('resolution_overall_estimate_m'))

        k_single_non0 = np.array([[1.],[0.]])
        traj_single_non0 = Trajectory("single_non0", k_single_non0, dt_seconds=self.dt)
        self.assertEqual(traj_single_non0.get_num_dimensions(), 2)
        self.assertEqual(traj_single_non0.get_num_points(), 1)
        expected_grads_single_non0 = np.zeros((2,1))
        np.testing.assert_array_almost_equal(traj_single_non0.get_gradient_waveforms_Tm(), expected_grads_single_non0,
                                             err_msg="Gradients for single non-zero point should be zero.")
        self.assertEqual(traj_single_non0.metadata.get('max_slew_rate_Tm_per_s'), 0.0)
        self.assertAlmostEqual(traj_single_non0.metadata.get('pns_max_abs_gradient_sum_xyz'), 0.0, places=7)
        self.assertEqual(traj_single_non0.metadata.get('pns_max_abs_slew_sum_xyz'), 0.0)
        self.assertIsNotNone(traj_single_non0.metadata['fov_estimate_m'])
        self.assertAlmostEqual(traj_single_non0.metadata['fov_estimate_m'][0], 0.5)
        self.assertTrue(np.isinf(traj_single_non0.metadata['fov_estimate_m'][1]))
        self.assertAlmostEqual(traj_single_non0.metadata['resolution_overall_estimate_m'], 0.5)

        traj_all_zeros = Trajectory("all_zeros", np.zeros((2,10)), dt_seconds=self.dt)
        self.assertEqual(traj_all_zeros.metadata.get('max_slew_rate_Tm_per_s'), 0.0)
        self.assertEqual(traj_all_zeros.metadata.get('pns_max_abs_gradient_sum_xyz'), 0.0)
        self.assertEqual(traj_all_zeros.metadata.get('pns_max_abs_slew_sum_xyz'), 0.0)
        self.assertIsNone(traj_all_zeros.metadata.get('fov_estimate_m'), "FOV for all-zero k-space should be None")
        self.assertIsNone(traj_all_zeros.metadata.get('resolution_overall_estimate_m'), "Resolution for all-zero k-space should be None")

        traj_dt_none = Trajectory("dt_none_metrics", self.kspace_1d, dt_seconds=None)
        self.assertIsNone(traj_dt_none.metadata.get('max_slew_rate_Tm_per_s'))
        self.assertIsNone(traj_dt_none.metadata.get('pns_max_abs_gradient_sum_xyz'))
        self.assertIsNone(traj_dt_none.metadata.get('pns_max_abs_slew_sum_xyz'))
        self.assertIsNotNone(traj_dt_none.metadata.get('fov_estimate_m'))
        self.assertIsNotNone(traj_dt_none.metadata.get('resolution_overall_estimate_m'))

    def test_export_import_npz_focused(self):
        filename = "test_traj_npz_focused.npz"
        k_orig = np.array([[1.,2.,3.],[4.,5.,6.]])
        g_orig = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
        traj_orig = Trajectory("orig_npz", k_orig, gradient_waveforms_Tm=g_orig, dt_seconds=self.dt)
        
        traj_orig.export(filename, filetype='npz')
        self.assertTrue(os.path.exists(filename))
        traj_imported = Trajectory.import_from(filename)
        
        np.testing.assert_array_almost_equal(traj_imported.kspace_points_rad_per_m, k_orig)
        self.assertIsNotNone(traj_imported.get_gradient_waveforms_Tm())
        np.testing.assert_array_almost_equal(traj_imported.get_gradient_waveforms_Tm(), g_orig)
        self.assertEqual(traj_imported.dt_seconds, self.dt)
        if os.path.exists(filename): os.remove(filename)

    def test_export_import_csv_npy_txt(self):
        base_filename = "test_export_simple"
        file_extensions = ['csv', 'npy', 'txt']

        k_space_data_orig = np.random.rand(3, 50)
        dt_orig = 4e-6
        metadata_orig = {'test_info': 'export_test_simple', 'gamma_Hz_per_T': self.gamma}

        for file_ext in file_extensions:
            filename = f"{base_filename}.{file_ext}"
            with self.subTest(file_ext=file_ext):
                traj_orig = Trajectory(f"orig_{file_ext}", k_space_data_orig,
                                       dt_seconds=dt_orig, metadata=metadata_orig)

                traj_orig.export(filename, filetype=file_ext)
                self.assertTrue(os.path.exists(filename))

                traj_imported = Trajectory.import_from(filename)

                self.assertEqual(traj_imported.name, filename)
                np.testing.assert_array_almost_equal(
                    traj_imported.kspace_points_rad_per_m,
                    k_space_data_orig,
                    err_msg=f"K-space data mismatch for {file_ext}"
                )
                self.assertIsNone(traj_imported.dt_seconds, f"dt_seconds for {file_ext}")
                self.assertIsNone(traj_imported.get_gradient_waveforms_Tm(), f"Gradients for {file_ext}")
                self.assertIn('gamma_Hz_per_T', traj_imported.metadata)
                self.assertNotIn('test_info', traj_imported.metadata)

                if os.path.exists(filename):
                    os.remove(filename)

        with self.assertRaises((FileNotFoundError, ValueError)):
             Trajectory.import_from("non_existent_file.csv")
        dummy_unsupported_file = "test_dummy.unsupported_ext"
        with open(dummy_unsupported_file, "w") as f: f.write("dummy data")
        with self.assertRaises(ValueError): Trajectory.import_from(dummy_unsupported_file)
        if os.path.exists(dummy_unsupported_file): os.remove(dummy_unsupported_file)
        traj_for_bad_export = Trajectory("bad_export", np.random.rand(2,10))
        with self.assertRaises(ValueError):
            traj_for_bad_export.export("test_dummy.badtype_export", filetype="badtype")

    def test_summary_method_runs(self):
        traj = Trajectory("summary_test_traj", self.kspace_1d, dt_seconds=self.dt, metadata={'info':'test summary'})
        captured_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_output):
                traj.summary()
        except Exception as e:
            self.fail(f"summary() method raised an exception: {e}")
        output_str = captured_output.getvalue()
        self.assertIn(traj.name, output_str)
        self.assertIn(str(traj.get_num_points()), output_str)
        self.assertIn(str(traj.get_num_dimensions()), output_str)
        if 'info' in traj.metadata:
             self.assertIn(str(traj.metadata['info']), output_str)

    def test_plot_2d_execution_detailed(self):
        traj_empty = Trajectory("plot_empty_2d", self.kspace_empty_2d)
        ax = traj_empty.plot_2d(); self.assertIsNone(ax)
        fig, ax_passed = plt.subplots(); returned_ax = traj_empty.plot_2d(ax=ax_passed); self.assertIs(returned_ax, ax_passed)
        plt.close('all')

        traj_1d = Trajectory("plot_1d_on_2d", self.kspace_1d)
        ax = traj_1d.plot_2d(); self.assertIsNone(ax)
        fig, ax_passed = plt.subplots(); returned_ax = traj_1d.plot_2d(ax=ax_passed); self.assertIs(returned_ax, ax_passed)
        plt.close('all')

        traj_2d = Trajectory("plot_2d", self.kspace_2d_plot_test)
        ax = traj_2d.plot_2d(); self.assertIsInstance(ax, plt.Axes)
        fig, ax_passed = plt.subplots(); returned_ax = traj_2d.plot_2d(ax=ax_passed); self.assertIs(returned_ax, ax_passed)
        
        metadata_interleaf = {'interleaf_structure': (4, self.kspace_2d_plot_test.shape[1]//4)}
        traj_2d_interleaved = Trajectory("plot_2d_il", self.kspace_2d_plot_test, metadata=metadata_interleaf)
        ax = traj_2d_interleaved.plot_2d(max_total_points=10, max_interleaves=2, interleaf_stride=1, point_stride=1, legend_on=True)
        self.assertIsNotNone(ax)
        if ax.lines and hasattr(ax, 'get_legend') and callable(ax.get_legend) and ax.get_legend() is not None:
             self.assertTrue(len(ax.get_legend().get_texts()) > 0)
        plt.close('all')

        traj_3d = Trajectory("plot_3d_on_2d", self.kspace_3d_plot_test)
        ax = traj_3d.plot_2d(); self.assertIsInstance(ax, plt.Axes)
        plt.close('all')

    def test_plot_3d_execution_detailed(self):
        traj_empty = Trajectory("plot_empty_3d", self.kspace_empty_3d)
        ax = traj_empty.plot_3d(); self.assertIsNone(ax)
        plt.close('all')

        traj_1d = Trajectory("plot_1d_on_3d", self.kspace_1d)
        ax = traj_1d.plot_3d(); self.assertIsNone(ax)
        plt.close('all')
        
        traj_2d = Trajectory("plot_2d_on_3d", self.kspace_2d_plot_test)
        ax = traj_2d.plot_3d(); self.assertIsNone(ax)
        plt.close('all')

        traj_3d = Trajectory("plot_3d", self.kspace_3d_plot_test)
        ax = traj_3d.plot_3d(); self.assertIsInstance(ax, Axes3D)
        fig, ax_passed_3d = plt.subplots(subplot_kw={'projection': '3d'});
        returned_ax = traj_3d.plot_3d(ax=ax_passed_3d); self.assertIs(returned_ax, ax_passed_3d)
        
        metadata_interleaf_3d = {'interleaf_structure': (6, self.kspace_3d_plot_test.shape[1]//6)}
        traj_3d_interleaved = Trajectory("plot_3d_il", self.kspace_3d_plot_test, metadata=metadata_interleaf_3d)
        ax = traj_3d_interleaved.plot_3d(max_total_points=10, max_interleaves=3, interleaf_stride=2, point_stride=1)
        self.assertIsInstance(ax, Axes3D)
        plt.close('all')

        fig, ax_2d_passed = plt.subplots()
        returned_ax_recreated = traj_3d.plot_3d(ax=ax_2d_passed)
        self.assertIsInstance(returned_ax_recreated, Axes3D)
        self.assertIsNot(returned_ax_recreated, ax_2d_passed)
        plt.close('all')

# Removed old test_plot_voronoi_detailed as it's replaced by TestTrajectoryPlotVoronoi

class TestTrajectoryCalculateVoronoiDensity(unittest.TestCase):
    def setUp(self):
        self.dt = 4e-6
        self.kspace_2d_simple = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]]).T # 2x5
        self.kspace_1d_simple = np.array([[-1, 0, 1, 2, 3]]) # 1x5
        self.kspace_3d_simple = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0.5]]).T # 3x5
        self.kspace_empty = np.empty((2,0))
        self.kspace_2d_insufficient = np.array([[0,0],[1,0]]).T # 2x2

    def tearDown(self):
        plt.close('all')

    def test_calculate_voronoi_2d(self):
        traj = Trajectory("vor_2d", self.kspace_2d_simple, dt_seconds=self.dt)
        weights = traj.calculate_voronoi_density()
        self.assertEqual(traj.metadata['voronoi_calculation_status'], "Success")
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (self.kspace_2d_simple.shape[1],))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertIn('density_compensation_weights_voronoi', traj.metadata)
        self.assertIn('voronoi_cell_sizes', traj.metadata)
        np.testing.assert_array_equal(traj.metadata['density_compensation_weights_voronoi'], weights)
        np.testing.assert_array_equal(traj.metadata['voronoi_cell_sizes'], weights)

        # Test force_recompute
        traj.metadata['voronoi_calculation_status'] = "Old Status"
        weights_recomputed = traj.calculate_voronoi_density(force_recompute=True)
        self.assertEqual(traj.metadata['voronoi_calculation_status'], "Success")
        np.testing.assert_array_almost_equal(weights, weights_recomputed)


    def test_calculate_voronoi_1d(self):
        # compute_voronoi_density internally handles 1D by returning uniform weights if it can't process
        # The Trajectory class method should reflect this behavior.
        traj = Trajectory("vor_1d", self.kspace_1d_simple, dt_seconds=self.dt)
        weights = traj.calculate_voronoi_density()
        self.assertEqual(traj.get_num_dimensions(), 1)
        # The compute_voronoi_density function has a check:
        # if ndim not in [2, 3]: raise ValueError(...)
        # So this should result in an error status if compute_voronoi_density is called directly with 1D
        # Let's check the behavior of the class method given this.
        # The helper `compute_density_compensation` (not directly used by Trajectory's method) has a fallback for 1D.
        # However, `Trajectory.calculate_voronoi_density` calls `compute_voronoi_density` directly.
        # `compute_voronoi_density` raises ValueError for ndim not in [2,3].
        self.assertTrue(traj.metadata['voronoi_calculation_status'].startswith("Error: Number of dimensions (ndim=1) must be 2 or 3"))
        self.assertIsNone(weights)
        self.assertIsNone(traj.metadata.get('density_compensation_weights_voronoi'))


    def test_calculate_voronoi_3d(self):
        traj = Trajectory("vor_3d", self.kspace_3d_simple, dt_seconds=self.dt)
        weights = traj.calculate_voronoi_density()
        self.assertEqual(traj.metadata['voronoi_calculation_status'], "Success")
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (self.kspace_3d_simple.shape[1],))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertIn('density_compensation_weights_voronoi', traj.metadata)

    def test_calculate_voronoi_empty(self):
        traj = Trajectory("vor_empty", self.kspace_empty, dt_seconds=self.dt)
        weights = traj.calculate_voronoi_density()
        self.assertEqual(traj.metadata['voronoi_calculation_status'], "Skipped: No k-space points")
        self.assertIsNotNone(weights) # Should be an empty array
        self.assertEqual(weights.size, 0)
        self.assertIsNone(traj.metadata.get('density_compensation_weights_voronoi'))


    def test_calculate_voronoi_insufficient_points(self):
        # compute_voronoi_density has: if unique_pts.shape[0] < ndim + 1: return normalize_density_weights(np.ones(num_points))
        traj = Trajectory("vor_insufficient", self.kspace_2d_insufficient, dt_seconds=self.dt)
        self.assertEqual(traj.get_num_points(), 2)
        self.assertEqual(traj.get_num_dimensions(), 2)
        weights = traj.calculate_voronoi_density()
        # This should now return uniform weights as per compute_voronoi_density's fallback
        self.assertEqual(traj.metadata['voronoi_calculation_status'], "Success")
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape, (self.kspace_2d_insufficient.shape[1],))
        np.testing.assert_array_almost_equal(weights, np.full(self.kspace_2d_insufficient.shape[1], 1.0/self.kspace_2d_insufficient.shape[1]))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)

class TestTrajectoryPlotVoronoi(unittest.TestCase):
    def setUp(self):
        self.kspace_2d = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]]).T # 2x5
        self.kspace_3d = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0.5]]).T # 3x5
        self.kspace_1d = np.array([[-1, 0, 1, 2, 3]]) # 1x5
        self.kspace_empty = np.empty((2,0))
        self.kspace_2d_few_points = np.array([[0,0],[1,0]]).T # 2x2

    def tearDown(self):
        plt.close('all') # Ensure all figures are closed after each test

    def test_plot_voronoi_2d(self):
        traj = Trajectory("plot_vor_2d", self.kspace_2d)
        ax = traj.plot_voronoi(title="Test 2D Voronoi")
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue(ax.get_title() == "Test 2D Voronoi")
        self.assertTrue(len(ax.collections) > 0 or len(ax.lines) > 0) # Check if something was plotted

    def test_plot_voronoi_3d(self):
        traj = Trajectory("plot_vor_3d", self.kspace_3d)
        # Should fall back to plot_3d
        ax = traj.plot_voronoi(title="Test 3D Voronoi Fallback")
        self.assertIsInstance(ax, Axes3D) # plot_3d returns Axes3D
        self.assertTrue("3D Scatter Fallback" in ax.get_title())

    def test_plot_voronoi_1d(self):
        traj = Trajectory("plot_vor_1d", self.kspace_1d)
        ax = traj.plot_voronoi(title="Test 1D Voronoi")
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue("(1D points)" in ax.get_title())
        self.assertTrue(len(ax.lines) > 0) # Should plot points

    def test_plot_voronoi_empty(self):
        traj = Trajectory("plot_vor_empty", self.kspace_empty)
        ax = traj.plot_voronoi(title="Test Empty Voronoi")
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue("(No points)" in ax.get_title())

    def test_plot_voronoi_with_ax(self):
        traj = Trajectory("plot_vor_2d_ax", self.kspace_2d)
        fig, existing_ax = plt.subplots()
        returned_ax = traj.plot_voronoi(ax=existing_ax, title="Test Existing Ax")
        self.assertIs(returned_ax, existing_ax)
        self.assertTrue(returned_ax.get_title() == "Test Existing Ax")

    def test_plot_voronoi_2d_few_points(self):
        # Not enough points for Voronoi, should plot points
        traj = Trajectory("plot_vor_2d_few", self.kspace_2d_few_points)
        ax = traj.plot_voronoi(title="Test Few Points")
        self.assertIsInstance(ax, plt.Axes)
        self.assertTrue("(Too few points for Voronoi)" in ax.get_title())
        self.assertTrue(len(ax.lines) > 0) # Check points are plotted

    def test_plot_voronoi_2d_clip_boundary(self):
        traj = Trajectory("plot_vor_2d_clip", self.kspace_2d)
        ax = traj.plot_voronoi(title="Test Clip Boundary", clip_boundary_m=1.0)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlim(), (-1.0, 1.0))
        self.assertEqual(ax.get_ylim(), (-1.0, 1.0))
        # Check if patches were added (even if empty, collection might be there)
        # or points if Voronoi failed but still within clip.
        # This test mainly ensures the API for clipping is hit and plot limits are set.
        self.assertTrue(len(ax.collections) > 0 or len(ax.lines) > 0)


class TestTrajectoryHelperFunctions(unittest.TestCase):
    def setUp(self):
        self.simple_2d_points = np.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]])
        self.simple_3d_points = np.array([[-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [-0.5,0.5,-0.5], [-0.5,-0.5,0.5], [0.25,0.25,0.25]])


    def test_normalize_density_weights_detailed(self):
        weights1 = np.array([1.0, 2.0, 3.0, 4.0])
        expected1 = weights1 / 10.0
        np.testing.assert_array_almost_equal(normalize_density_weights(weights1), expected1)
        self.assertAlmostEqual(np.sum(normalize_density_weights(weights1)), 1.0)

        weights2 = np.array([0.0, 0.0, 0.0])
        expected2 = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(normalize_density_weights(weights2), expected2)
        self.assertAlmostEqual(np.sum(normalize_density_weights(weights2)), 1.0)

        weights3 = np.array([1e-15, -1e-15, 2e-15])
        expected3 = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(normalize_density_weights(weights3), expected3)
        self.assertAlmostEqual(np.sum(normalize_density_weights(weights3)), 1.0)

        weights4 = np.array([])
        expected4 = np.array([])
        np.testing.assert_array_almost_equal(normalize_density_weights(weights4), expected4)

        weights5 = np.array([5.0])
        expected5 = np.array([1.0])
        np.testing.assert_array_almost_equal(normalize_density_weights(weights5), expected5)
        self.assertAlmostEqual(np.sum(normalize_density_weights(weights5)), 1.0)

        weights6 = np.array([-1.0, 3.0])
        expected6 = np.array([-0.5, 1.5])
        np.testing.assert_array_almost_equal(normalize_density_weights(weights6), expected6)
        self.assertAlmostEqual(np.sum(normalize_density_weights(weights6)), 1.0)

        weights7 = np.array([-2.0, 2.0])
        expected7 = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(normalize_density_weights(weights7), expected7)
        self.assertAlmostEqual(np.sum(normalize_density_weights(weights7)), 1.0)

    def test_create_periodic_points_detailed(self):
        point_2d = np.array([[0.1, 0.2]])
        point_2d_flat = point_2d[0]
        extended_2d = create_periodic_points(point_2d, ndim=2)
        self.assertEqual(extended_2d.shape, (9, 2))
        self.assertTrue(np.any([np.allclose(row, point_2d_flat) for row in extended_2d]))
        self.assertTrue(np.any([np.allclose(row, point_2d_flat + np.array([-1,-1])) for row in extended_2d]))
        self.assertTrue(np.any([np.allclose(row, point_2d_flat + np.array([1,1])) for row in extended_2d]))

        point_3d = np.array([[0.1, 0.2, 0.3]])
        point_3d_flat = point_3d[0]
        extended_3d = create_periodic_points(point_3d, ndim=3)
        self.assertEqual(extended_3d.shape, (27, 3))
        self.assertTrue(np.any([np.allclose(row, point_3d_flat) for row in extended_3d]))
        self.assertTrue(np.any([np.allclose(row, point_3d_flat + np.array([-1,-1,-1])) for row in extended_3d]))
        self.assertTrue(np.any([np.allclose(row, point_3d_flat + np.array([1,1,1])) for row in extended_3d]))

        with self.assertRaisesRegex(ValueError, "Number of dimensions must be 2 or 3"):
            create_periodic_points(point_2d, ndim=1)
        with self.assertRaisesRegex(ValueError, "Number of dimensions must be 2 or 3"):
            create_periodic_points(point_3d, ndim=4)

        with self.assertRaisesRegex(ValueError, "Trajectory shape.*inconsistent with ndim"):
            create_periodic_points(np.array([[0,0]]), ndim=3)
        with self.assertRaisesRegex(ValueError, "Trajectory shape.*inconsistent with ndim"):
            create_periodic_points(np.array([[0,0,0]]), ndim=2)

    def test_compute_density_compensation_detailed(self):
        # Voronoi 2D
        weights_vor_real = compute_density_compensation(self.simple_2d_points, method="voronoi")
        self.assertEqual(weights_vor_real.shape, (self.simple_2d_points.shape[0],))
        self.assertAlmostEqual(np.sum(weights_vor_real), 1.0)
        self.assertTrue(np.all(weights_vor_real >= 0))

        complex_pts_2d = self.simple_2d_points[:,0] + 1j*self.simple_2d_points[:,1]
        weights_vor_complex = compute_density_compensation(complex_pts_2d, method="voronoi")
        self.assertEqual(weights_vor_complex.shape, (complex_pts_2d.shape[0],))
        np.testing.assert_array_almost_equal(weights_vor_complex, weights_vor_real)

        # Voronoi edge cases
        two_points = self.simple_2d_points[:2,:] # (2,2)
        weights_two_pts = compute_density_compensation(two_points, method="voronoi") # Fallback to uniform
        np.testing.assert_array_almost_equal(weights_two_pts, np.full(two_points.shape[0], 1.0/two_points.shape[0]))

        weights_empty_real = compute_density_compensation(np.empty((0,2)), method="voronoi")
        self.assertEqual(weights_empty_real.shape, (0,))
        weights_empty_complex = compute_density_compensation(np.empty((0,),dtype=complex), method="voronoi")
        self.assertEqual(weights_empty_complex.shape, (0,))

        # Pipe 2D
        pipe_pts = np.array([[0.,0.], [1.,0.], [2.,0.]])
        weights_pipe_real = compute_density_compensation(pipe_pts, method="pipe")
        expected_pipe_weights = np.array([0., 1/3., 2/3.]) # Raw [0,1,2], sum=3
        np.testing.assert_array_almost_equal(weights_pipe_real, expected_pipe_weights)
        self.assertAlmostEqual(np.sum(weights_pipe_real), 1.0)

        complex_pipe_pts = pipe_pts[:,0] + 1j*pipe_pts[:,1]
        weights_pipe_complex = compute_density_compensation(complex_pipe_pts, method="pipe")
        np.testing.assert_array_almost_equal(weights_pipe_complex, expected_pipe_weights)

        # Pipe edge cases
        weights_pipe_empty = compute_density_compensation(np.empty((0,2)), method="pipe")
        self.assertEqual(weights_pipe_empty.shape, (0,))

        zeros_pts = np.zeros((5,2))
        weights_pipe_zeros = compute_density_compensation(zeros_pts, method="pipe") # All radii 0 -> uniform
        np.testing.assert_array_almost_equal(weights_pipe_zeros, np.full(5, 0.2))


        # Invalid cases
        with self.assertRaisesRegex(ValueError, "Pipe method is only supported for 2D"):
            compute_density_compensation(self.simple_3d_points, method="pipe")
        with self.assertRaisesRegex(ValueError, "Unknown density compensation method"):
            compute_density_compensation(self.simple_2d_points, method="unknown")

    def test_compute_voronoi_density_detailed(self):
        points_2d_norm = np.array([[-0.5,-0.5], [0.5,-0.5], [-0.5,0.5], [0.5,0.5]])
        points_3d_norm = np.array([[-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [-0.5,0.5,-0.5], [-0.5,-0.5,0.5], [0.25,0.25,0.25]])

        # 2D Clipped
        weights_2d_clip = compute_voronoi_density(points_2d_norm, boundary_type="clipped")
        self.assertEqual(weights_2d_clip.shape, (4,)); self.assertAlmostEqual(np.sum(weights_2d_clip),1.0)
        np.testing.assert_array_almost_equal(weights_2d_clip, np.full(4,0.25)) # Expect uniform

        # 2D Periodic
        weights_2d_period = compute_voronoi_density(points_2d_norm, boundary_type="periodic")
        self.assertEqual(weights_2d_period.shape, (4,)); self.assertAlmostEqual(np.sum(weights_2d_period),1.0)
        np.testing.assert_array_almost_equal(weights_2d_period, np.full(4,0.25)) # Expect uniform

        # 3D (focus on execution and basic properties)
        weights_3d_clip = compute_voronoi_density(points_3d_norm, boundary_type="clipped")
        self.assertEqual(weights_3d_clip.shape, (5,)); self.assertAlmostEqual(np.sum(weights_3d_clip),1.0)
        self.assertTrue(np.all(weights_3d_clip >= 0))

        weights_3d_period = compute_voronoi_density(points_3d_norm, boundary_type="periodic")
        self.assertEqual(weights_3d_period.shape, (5,)); self.assertAlmostEqual(np.sum(weights_3d_period),1.0)
        self.assertTrue(np.all(weights_3d_period > 0)) # Periodic should have finite areas

        # Edge cases
        self.assertEqual(compute_voronoi_density(np.empty((0,2))).size, 0)
        np.testing.assert_array_almost_equal(compute_voronoi_density(np.array([[0.,0.]])), np.array([1.0]))

        collinear_pts = np.array([[-0.5,0], [0,0], [0.5,0]])
        weights_coll_clip = compute_voronoi_density(collinear_pts, boundary_type="clipped")
        np.testing.assert_array_almost_equal(weights_coll_clip, np.full(3,1/3.))
        weights_coll_period = compute_voronoi_density(collinear_pts, boundary_type="periodic")
        np.testing.assert_array_almost_equal(weights_coll_period, np.full(3,1/3.))

        # Invalid inputs
        with self.assertRaisesRegex(ValueError, "Number of dimensions.*must be 2 or 3"):
            compute_voronoi_density(np.random.rand(5,1))
        with self.assertRaisesRegex(ValueError, "Trajectory must be 2D"):
            compute_voronoi_density(np.random.rand(5))
        with self.assertRaisesRegex(ValueError, "Unknown boundary_type"):
            compute_voronoi_density(points_2d_norm, boundary_type="invalid")

if __name__ == '__main__':
    unittest.main()
