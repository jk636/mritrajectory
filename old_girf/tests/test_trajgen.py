import unittest
import numpy as np
import os
import sys
from typing import Union # Added for type hinting

# Add the parent directory to the Python path to allow importing trajgen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajgen import Trajectory, KSpaceTrajectoryGenerator, COMMON_NUCLEI_GAMMA_HZ_PER_T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plot testing

# Need to import the new functions to be tested
from trajgen import (
    normalize_density_weights,
    compute_density_compensation,
    create_periodic_points,
    # compute_cell_area, # Indirectly tested
    compute_voronoi_density,
    generate_spiral_trajectory,
    generate_radial_trajectory,
    generate_golden_angle_3d_trajectory, 
    constrain_trajectory,
    reconstruct_image,
    display_trajectory,
    predict_actual_gradients, 
    correct_kspace_with_girf, 
    sGIRF, 
    GIRF, 
    predict_actual_gradients_from_sgirf, 
    generate_tw_ssi_pulse, 
    apply_girf_convolution 
)
from unittest.mock import patch 
import tempfile 
import shutil   


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
        self.gamma = 42.576e6
        self.n_points = 100
        self.k_max = 1.0 / (2 * 0.004)
        self.kspace_1d = np.linspace(-self.k_max, self.k_max, self.n_points).reshape(1, self.n_points)
        kx_2d = np.linspace(-self.k_max, self.k_max, self.n_points)
        ky_2d = np.zeros_like(kx_2d)
        self.kspace_2d = np.stack([kx_2d, ky_2d])
        kz_3d = np.linspace(0, self.k_max, self.n_points)
        self.kspace_3d = np.stack([kx_2d, ky_2d, kz_3d])
        self.metadata_example = {'info': 'test_trajectory'}
        self.dead_time_start = 0.001
        self.dead_time_end = 0.0005
        self.kspace_2d_plot_test = np.random.rand(2, 20) * 250 - 125
        self.kspace_3d_plot_test = np.random.rand(3, 30) * 250 - 125

    def tearDown(self):
        plt.close('all')

    def test_trajectory_initialization_basic(self):
        traj = Trajectory("test1D", self.kspace_1d, dt_seconds=self.dt, metadata=self.metadata_example)
        self.assertEqual(traj.name, "test1D")
        self.assertEqual(traj.get_num_dimensions(), 1)
        self.assertEqual(traj.get_num_points(), self.n_points)
        self.assertEqual(traj.dt_seconds, self.dt)
        self.assertIn('info', traj.metadata)
        self.assertEqual(traj.metadata['info'], 'test_trajectory')
        self.assertAlmostEqual(traj.metadata['gamma_Hz_per_T'], self.gamma, places=1)

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
        metadata_with_gamma = {'gamma_Hz_per_T': 50.0e6, 'other_info': 'test'}
        traj_meta_gamma = Trajectory("test_meta_gamma", self.kspace_1d, dt_seconds=self.dt, 
                                     metadata=metadata_with_gamma, gamma_Hz_per_T=custom_gamma)
        self.assertAlmostEqual(traj_meta_gamma.metadata['gamma_Hz_per_T'], 50.0e6)

    def test_get_gradient_waveforms_Tm_calculation(self):
        traj = Trajectory("test_grad_calc", self.kspace_1d, dt_seconds=self.dt)
        gradients = traj.get_gradient_waveforms_Tm() 
        self.assertIsNotNone(gradients)
        self.assertEqual(gradients.shape, self.kspace_1d.shape)
        self.assertTrue(np.all(np.abs(gradients[0, 1:-1]) > 1e-3))

    def test_get_gradient_waveforms_Tm_caching(self):
        traj = Trajectory("test_grad_cache", self.kspace_1d, dt_seconds=self.dt)
        gradients1 = traj.get_gradient_waveforms_Tm()
        gradients2 = traj.get_gradient_waveforms_Tm()
        self.assertIs(gradients1, gradients2)

    def test_metric_calculations_run(self):
        traj = Trajectory("test_metrics", self.kspace_2d, dt_seconds=self.dt)
        self.assertIn('max_slew_rate_Tm_per_s', traj.metadata)
        self.assertIn('pns_max_abs_gradient_sum_xyz', traj.metadata)
        self.assertIn('fov_estimate_m', traj.metadata)
        self.assertIn('resolution_overall_estimate_m', traj.metadata)
        self.assertIsNotNone(traj.metadata['max_slew_rate_Tm_per_s'])

    def test_calculate_fov(self):
        k_max_x = np.max(np.abs(self.kspace_2d[0,:]))
        k_max_y = np.max(np.abs(self.kspace_2d[1,:]))
        expected_fov_x = 1.0 / (2 * k_max_x + 1e-9)
        traj2d = Trajectory("test_fov2d", self.kspace_2d, dt_seconds=self.dt)
        self.assertAlmostEqual(traj2d.metadata['fov_estimate_m'][0], expected_fov_x, places=6)
        self.assertTrue(np.isinf(traj2d.metadata['fov_estimate_m'][1]) or traj2d.metadata['fov_estimate_m'][1] > 1e6)
        k_max_z_3d = np.max(np.abs(self.kspace_3d[2,:]))
        expected_fov_z = 1.0 / (2 * k_max_z_3d + 1e-9)
        traj3d = Trajectory("test_fov3d", self.kspace_3d, dt_seconds=self.dt)
        self.assertAlmostEqual(traj3d.metadata['fov_estimate_m'][0], expected_fov_x, places=6)
        self.assertTrue(np.isinf(traj3d.metadata['fov_estimate_m'][1]) or traj3d.metadata['fov_estimate_m'][1] > 1e6)
        self.assertAlmostEqual(traj3d.metadata['fov_estimate_m'][2], expected_fov_z, places=6)

    def test_calculate_resolution(self):
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
        _ = traj_orig.get_gradient_waveforms_Tm()
        traj_orig.export(filename, filetype='npz')
        self.assertTrue(os.path.exists(filename))
        traj_imported = Trajectory.import_from(filename)
        self.assertEqual(traj_imported.name, filename)
        self.assertEqual(traj_imported.dt_seconds, traj_orig.dt_seconds)
        self.assertEqual(traj_imported.get_num_points(), traj_orig.get_num_points())
        self.assertEqual(traj_imported.get_num_dimensions(), traj_orig.get_num_dimensions())
        self.assertTrue(np.allclose(traj_imported.kspace_points_rad_per_m, self.kspace_3d))
        self.assertEqual(traj_imported.metadata['info'], 'original')
        self.assertAlmostEqual(traj_imported.metadata['gamma_Hz_per_T'], self.gamma)
        self.assertAlmostEqual(traj_imported.metadata['dead_time_start_seconds'], self.dead_time_start)
        self.assertAlmostEqual(traj_imported.metadata['dead_time_end_seconds'], self.dead_time_end)
        self.assertTrue(np.allclose(traj_imported.get_gradient_waveforms_Tm(), traj_orig.get_gradient_waveforms_Tm()))
        if os.path.exists(filename): os.remove(filename)

    def test_voronoi_density(self):
        points_2d_square = np.array([[0,0], [1,0], [0,1], [1,1]]).T
        traj_square = Trajectory("square", points_2d_square, dt_seconds=self.dt)
        cell_sizes = traj_square.calculate_voronoi_density()
        self.assertIsNotNone(cell_sizes)
        self.assertEqual(traj_square.metadata['voronoi_calculation_status'], "Success")
        self.assertTrue(np.all(np.isinf(cell_sizes)))
        points_2d_center = np.array([[0,0], [1,0], [0,1], [1,1], [0.5,0.5]]).T
        traj_center = Trajectory("center_square", points_2d_center, dt_seconds=self.dt)
        cell_sizes_center = traj_center.calculate_voronoi_density()
        self.assertTrue(np.isfinite(cell_sizes_center[4]))
        self.assertTrue(np.all(np.isinf(cell_sizes_center[:4])))
        points_2d_line = np.array([[0,0], [1,0]]).T
        traj_line = Trajectory("line", points_2d_line, dt_seconds=self.dt)
        self.assertIsNone(traj_line.calculate_voronoi_density())
        self.assertIn("Error: Not enough unique points", traj_line.metadata['voronoi_calculation_status'])
        traj_1d_voronoi = Trajectory("1d_voronoi", self.kspace_1d, dt_seconds=self.dt)
        self.assertIsNone(traj_1d_voronoi.calculate_voronoi_density())
        self.assertIn("Error: Voronoi calculation only supported for 2D/3D", traj_1d_voronoi.metadata['voronoi_calculation_status'])
        points_3d_simple = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0.5]]).T
        traj_3d_simple = Trajectory("simple3d", points_3d_simple, dt_seconds=self.dt)
        cell_sizes_3d = traj_3d_simple.calculate_voronoi_density()
        self.assertIsNotNone(cell_sizes_3d)
        self.assertEqual(traj_3d_simple.metadata['voronoi_calculation_status'], "Success")
        self.assertTrue(np.isfinite(cell_sizes_3d[4]))
        self.assertTrue(np.all(np.isinf(cell_sizes_3d[:4])))

    def test_plot_3d_execution(self):
        traj_3d = Trajectory("test_plot3d", self.kspace_3d_plot_test, dt_seconds=self.dt)
        ax = traj_3d.plot_3d()
        self.assertIsInstance(ax, Axes3D); plt.close('all')
        ax_sub = traj_3d.plot_3d(max_total_points=10, max_interleaves=1, point_stride=2, interleaf_stride=1)
        self.assertIsInstance(ax_sub, Axes3D); plt.close('all')
        fig = plt.figure(); pre_ax = fig.add_subplot(111, projection='3d')
        returned_ax = traj_3d.plot_3d(ax=pre_ax)
        self.assertIs(returned_ax, pre_ax); plt.close('all')
        traj_2d = Trajectory("test_plot3d_on_2d", self.kspace_2d_plot_test, dt_seconds=self.dt)
        self.assertIsNone(traj_2d.plot_3d()); plt.close('all')
        fig_2d, ax_2d_passed = plt.subplots()
        returned_ax_2d_passed = traj_2d.plot_3d(ax=ax_2d_passed)
        self.assertIs(returned_ax_2d_passed, ax_2d_passed); plt.close('all')

    def test_plot_voronoi_execution(self):
        kspace_2d_voronoi = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]]).T
        traj_2d = Trajectory("test_plot_voronoi_2d", kspace_2d_voronoi, dt_seconds=self.dt)
        _ = traj_2d.calculate_voronoi_density()
        ax = traj_2d.plot_voronoi()
        self.assertIsInstance(ax, plt.Axes); plt.close('all')
        ax_params = traj_2d.plot_voronoi(color_by_area=False, show_vertices=True, cmap='cividis')
        self.assertIsInstance(ax_params, plt.Axes); plt.close('all')
        fig, pre_ax = plt.subplots()
        returned_ax = traj_2d.plot_voronoi(ax=pre_ax)
        self.assertIs(returned_ax, pre_ax); plt.close('all')
        traj_3d = Trajectory("test_plot_voronoi_3d", self.kspace_3d_plot_test, dt_seconds=self.dt)
        ax_3d_fallback = traj_3d.plot_voronoi()
        self.assertIsInstance(ax_3d_fallback, Axes3D); plt.close('all')


class TestKSpaceTrajectoryGenerator(unittest.TestCase):
    def setUp(self):
        self.common_params = {
            'fov': 0.256,
            'resolution': 0.004, # k_max = 125
            'dt': 4e-6,
            'gamma': 42.576e6,
            'n_interleaves': 4, # Default, can be overridden
        }
        self.test_res = 0.01 # k_max = 50

    def _check_2d_outputs(self, gen, kx, ky, gx, gy, t):
        self.assertIsInstance(kx, np.ndarray)
        self.assertIsInstance(ky, np.ndarray)
        self.assertIsInstance(gx, np.ndarray)
        self.assertIsInstance(gy, np.ndarray)
        self.assertIsInstance(t, np.ndarray)
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
                if traj_type == 'rosette': params.update({'f1': 3, 'f2': 5, 'a': 0.5}) 
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
                params['n_interleaves'] = 8 
                if traj_type == 'stackofspirals': params['n_stacks'] = 2
                if traj_type == 'epi_3d':
                    params['epi_3d_fov_y'] = params['fov']
                    params['epi_3d_resolution_y'] = params['resolution'] * 2 
                    params['epi_3d_fov_z'] = params['fov']
                    params['epi_3d_resolution_z'] = params['resolution'] * 2 
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
                        'vd_method': 'power', 'vd_alpha': 2.0, 'add_rewinder': False}
        gen_dense = KSpaceTrajectoryGenerator(**params_dense)
        kx_dense, ky_dense, _, _, _ = gen_dense.generate()
        r_dense = np.sqrt(kx_dense[0]**2 + ky_dense[0]**2)
        self.assertLess(np.median(r_dense[10:]), np.median(r_lin[10:]))

    def test_ute_ramp_sampling_radial3d(self):
        params = {**self.common_params, 'dim': 3, 'traj_type': 'radial3d', 'n_interleaves': 1, 
                  'add_rewinder': False, 'resolution': 0.01}
        gen_full = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': False})
        kx_f, ky_f, kz_f, _, _, _, _ = gen_full.generate()
        self.assertTrue(np.isclose(np.mean(kx_f[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(ky_f[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(kz_f[0]), 0.0, atol=1e-1))
        gen_half = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': True})
        kx_h, ky_h, kz_h, _, _, _, _ = gen_half.generate()
        k_radius_h = np.sqrt(kx_h[0]**2 + ky_h[0]**2 + kz_h[0]**2)
        self.assertLess(np.min(k_radius_h), 0.01 * gen_half.k_max)
        self.assertGreater(np.max(k_radius_h), 0.9 * gen_half.k_max)

    def test_zte_trajectory(self):
        params = {**self.common_params, 'dim': 3, 'traj_type': 'zte', 'n_interleaves': 1, 
                  'add_rewinder': False, 'resolution': 0.01}
        gen_zte_ute = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': True})
        kx_z_ute, ky_z_ute, kz_z_ute, _, _, _, _ = gen_zte_ute.generate()
        k_radius_z_ute = np.sqrt(kx_z_ute[0]**2 + ky_z_ute[0]**2 + kz_z_ute[0]**2)
        self.assertLess(np.min(k_radius_z_ute), 0.01 * gen_zte_ute.k_max)
        gen_zte_full = KSpaceTrajectoryGenerator(**{**params, 'ute_ramp_sampling': False})
        kx_z_full, ky_z_full, kz_z_full, _, _, _, _ = gen_zte_full.generate()
        self.assertTrue(np.isclose(np.mean(kx_z_full[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(ky_z_full[0]), 0.0, atol=1e-1) and \
                        np.isclose(np.mean(kz_z_full[0]), 0.0, atol=1e-1))

    def test_generate_3d_from_2d(self):
        base_gen_params = {**self.common_params, 'dim': 2, 'traj_type': 'spiral', 
                           'n_interleaves': 1, 'turns': 3, 'add_rewinder': False}
        base_gen = KSpaceTrajectoryGenerator(**base_gen_params)
        n_3d_shots = 10
        kx, ky, kz, gx, gy, gz, t = base_gen.generate_3d_from_2d(
            n_3d_shots=n_3d_shots, traj2d_type='spiral')
        self.assertEqual(kx.shape[0], n_3d_shots)
        self.assertEqual(ky.shape[0], n_3d_shots)
        self.assertEqual(kz.shape[0], n_3d_shots)
        self.assertTrue(kx.shape[1] > 0) 
        self.assertFalse(np.allclose(kx[0], kx[1]))
        self.assertFalse(np.allclose(ky[0], ky[1]))
        if n_3d_shots > 1 and kx.shape[1] > 0:
             self.assertFalse(np.allclose(kz[0], kz[1]) if kz.shape[1] > 0 and np.std(kz[0]-kz[1]) > 1e-9 else False)

    def test_edge_cases_and_options(self):
        gen_single_interleaf = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'spiral', 'n_interleaves':1})
        kx, ky, _, _, _ = gen_single_interleaf.generate()
        self.assertEqual(kx.shape[0], 1)
        gen_no_rewind = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'radial', 'add_rewinder':False})
        _, _, _, _, t_no_rewind = gen_no_rewind.generate()
        gen_with_rewind = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'radial', 'add_rewinder':True})
        _, _, _, _, t_with_rewind = gen_with_rewind.generate()
        if gen_no_rewind.ramp_samples > 0 : self.assertLess(t_no_rewind.shape[0], t_with_rewind.shape[0])
        gen_spoiler = KSpaceTrajectoryGenerator(**{**self.common_params, 'dim':2, 'traj_type':'spiral', 
                                                 'add_spoiler':True, 'add_rewinder': False})
        _, _, _, _, t_spoiler = gen_spoiler.generate()
        if gen_spoiler.ramp_samples > 0: self.assertEqual(t_spoiler.shape[0], gen_spoiler.n_samples + gen_spoiler.ramp_samples)


class TestGenerateGoldenAngle3DTrajectory(unittest.TestCase):
    def setUp(self):
        self.default_dt = 4e-6
        self.num_points_default = 100

    def test_basic_generation_isotropic_fov(self):
        num_points = self.num_points_default; fov_m = 0.25; traj_name = "golden_angle_iso_fov"
        traj = generate_golden_angle_3d_trajectory(num_points, fov_m, name=traj_name, dt_seconds=self.default_dt)
        self.assertIsInstance(traj, Trajectory); self.assertEqual(traj.name, traj_name)
        self.assertEqual(traj.get_num_points(), num_points); self.assertEqual(traj.get_num_dimensions(), 3)
        self.assertEqual(traj.kspace_points_rad_per_m.shape, (3, num_points)); self.assertEqual(traj.dt_seconds, self.default_dt)
        expected_k_max = np.pi / fov_m; expected_k_max_xyz = (expected_k_max, expected_k_max, expected_k_max)
        gp = traj.metadata['generator_params']
        self.assertEqual(gp['num_points'], num_points); self.assertEqual(gp['fov_m_input'], fov_m)
        self.assertIsNone(gp['max_k_rad_per_m_input']); self.assertEqual(gp['dt_seconds_input'], self.default_dt)
        for i in range(3): self.assertAlmostEqual(traj.metadata['k_max_calculated_rad_m_xyz'][i], expected_k_max_xyz[i], places=6)
        for dim in range(3): self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[dim, :])) <= expected_k_max_xyz[dim] * 1.001)

    def test_generation_anisotropic_fov(self):
        num_points = self.num_points_default; fov_m_aniso = (0.2, 0.3, 0.4); traj_name = "golden_angle_aniso_fov"
        traj = generate_golden_angle_3d_trajectory(num_points, fov_m_aniso, name=traj_name)
        self.assertIsInstance(traj, Trajectory); self.assertEqual(traj.name, traj_name)
        expected_k_max_xyz = (np.pi / fov_m_aniso[0], np.pi / fov_m_aniso[1], np.pi / fov_m_aniso[2])
        km_calc = traj.metadata['k_max_calculated_rad_m_xyz']
        for i in range(3): self.assertAlmostEqual(km_calc[i], expected_k_max_xyz[i], places=6)
        for dim in range(3): self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[dim, :])) <= expected_k_max_xyz[dim] * 1.001)

    def test_generation_explicit_isotropic_max_k(self):
        num_points = self.num_points_default; fov_m = 0.2; explicit_k_max_iso = 100.0
        traj = generate_golden_angle_3d_trajectory(num_points, fov_m, max_k_rad_per_m=explicit_k_max_iso)
        expected_k_max_xyz = (explicit_k_max_iso, explicit_k_max_iso, explicit_k_max_iso)
        gp = traj.metadata['generator_params']; km_calc = traj.metadata['k_max_calculated_rad_m_xyz']
        self.assertEqual(gp['max_k_rad_per_m_input'], explicit_k_max_iso)
        for i in range(3): self.assertAlmostEqual(km_calc[i], expected_k_max_xyz[i], places=6)
        for dim in range(3): self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[dim, :])) <= expected_k_max_xyz[dim] * 1.001)

    def test_generation_explicit_anisotropic_max_k(self):
        num_points = self.num_points_default; fov_m = 0.25; explicit_k_max_aniso = (100.0, 120.0, 150.0)
        traj = generate_golden_angle_3d_trajectory(num_points, fov_m, max_k_rad_per_m=explicit_k_max_aniso)
        gp = traj.metadata['generator_params']; km_calc = traj.metadata['k_max_calculated_rad_m_xyz']
        self.assertEqual(gp['max_k_rad_per_m_input'], explicit_k_max_aniso)
        for i in range(3): self.assertAlmostEqual(km_calc[i], explicit_k_max_aniso[i], places=6)
        for dim in range(3): self.assertTrue(np.max(np.abs(traj.kspace_points_rad_per_m[dim, :])) <= explicit_k_max_aniso[dim] * 1.001)

    def test_edge_case_small_num_points(self):
        fov_m = 0.3
        traj_one_point = generate_golden_angle_3d_trajectory(1, fov_m)
        self.assertEqual(traj_one_point.get_num_points(), 1)
        expected_k_max = np.pi / fov_m; r_norm_one_pt = np.power(0.5, 1./3.)
        expected_k_point = np.array([[r_norm_one_pt * expected_k_max], [0.0], [0.0]])
        np.testing.assert_allclose(traj_one_point.kspace_points_rad_per_m, expected_k_point, atol=1e-6)
        traj_zero_points = generate_golden_angle_3d_trajectory(0, fov_m)
        self.assertEqual(traj_zero_points.get_num_points(), 0)
        self.assertEqual(traj_zero_points.metadata['k_max_calculated_rad_m_xyz'], (0.0, 0.0, 0.0))

    def test_metadata_name_dt_seconds(self):
        traj = generate_golden_angle_3d_trajectory(10, 0.2, name="my_golden_angle_test", dt_seconds=10e-6)
        self.assertEqual(traj.name, "my_golden_angle_test"); self.assertEqual(traj.dt_seconds, 10e-6)
        self.assertEqual(traj.metadata['generator_params']['dt_seconds_input'], 10e-6)

    def test_invalid_fov_input(self):
        with self.assertRaisesRegex(ValueError, "fov_m must be a float or a list/tuple of 3 floats"):
            generate_golden_angle_3d_trajectory(10, fov_m=(0.1, 0.2))
        with self.assertRaisesRegex(ValueError, "fov_m must be a float or a list/tuple of 3 floats"):
            generate_golden_angle_3d_trajectory(10, fov_m="not_a_fov")

    def test_invalid_max_k_input(self):
        with self.assertRaisesRegex(ValueError, "max_k_rad_per_m must be a float or a list/tuple of 3 floats, or None"):
            generate_golden_angle_3d_trajectory(10, 0.2, max_k_rad_per_m=(100, 120))
        with self.assertRaisesRegex(ValueError, "max_k_rad_per_m must be a float or a list/tuple of 3 floats, or None"):
            generate_golden_angle_3d_trajectory(10, 0.2, max_k_rad_per_m="not_k_max")


class TestAdvancedTrajectoryTools(unittest.TestCase):
    def setUp(self):
        self.fov_m = 0.2; self.dt_s = 4e-6; self.gamma_1h = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
        self.simple_2d_traj_points = np.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]]) 
        self.simple_2d_traj_obj = Trajectory("simple_2d_for_adv_tests", self.simple_2d_traj_points.T, dt_seconds=self.dt_s)
        self.simple_3d_traj_points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
        self.simple_3d_traj_obj = Trajectory("simple_3d_for_adv_tests",self.simple_3d_traj_points.T, dt_seconds=self.dt_s)
    def tearDown(self): plt.close('all')
    def test_normalize_density_weights_simple(self):
        w = np.array([1.,2.,3.,4.]); n = normalize_density_weights(w)
        self.assertTrue(np.allclose(np.sum(n),1.)); self.assertTrue(np.allclose(n,w/10.))
    def test_normalize_density_weights_sum_zero(self):
        w=np.zeros(4); n=normalize_density_weights(w)
        self.assertTrue(np.allclose(np.sum(n),1.)); self.assertTrue(np.allclose(n,np.ones(4)/4.))
    def test_normalize_density_weights_sum_almost_zero(self):
        w=np.array([1e-15,1e-15]); n=normalize_density_weights(w)
        self.assertTrue(np.allclose(np.sum(n),1.)); self.assertTrue(np.allclose(n,np.ones(2)/2.))
    def test_normalize_density_weights_empty(self): self.assertEqual(normalize_density_weights(np.array([])).size,0)
    def test_normalize_density_weights_single_value(self): self.assertTrue(np.allclose(normalize_density_weights(np.array([5.])),[1.]))
    def test_normalize_density_weights_negative_values_sum_non_zero(self):
        w=np.array([-1.,-2.,3.,4.]); n=normalize_density_weights(w)
        self.assertTrue(np.allclose(np.sum(n),1.)); self.assertTrue(np.allclose(n,w/4.))
    def test_normalize_density_weights_negative_values_sum_zero(self):
        w=np.array([-1.,-2.,1.,2.]); n=normalize_density_weights(w)
        self.assertTrue(np.allclose(np.sum(n),1.)); self.assertTrue(np.allclose(n,np.full_like(w,1./w.size)))
    def test_cdc_voronoi_2d(self):
        w=compute_density_compensation(self.simple_2d_traj_points); self.assertEqual(w.shape,(4,))
        self.assertTrue(np.allclose(np.sum(w),1.)); self.assertTrue(np.all(w>=0))
    def test_cdc_voronoi_3d(self):
        w=compute_density_compensation(self.simple_3d_traj_points); self.assertEqual(w.shape,(4,))
        self.assertTrue(np.allclose(np.sum(w),1.)); self.assertTrue(np.all(w>=0))
    def test_cdc_pipe_2d(self):
        w=compute_density_compensation(self.simple_2d_traj_points,method="pipe"); self.assertEqual(w.shape,(4,))
        if self.simple_2d_traj_points.shape[0]>0: self.assertTrue(np.allclose(np.sum(w),1.))
        self.assertTrue(np.all(w>=0))
        r=np.linalg.norm(self.simple_2d_traj_points,axis=1)
        if np.sum(r)>1e-9: self.assertTrue(np.allclose(w,r/np.sum(r)))
        else: self.assertTrue(np.allclose(w,np.ones_like(w)/w.size))
    def test_cdc_pipe_3d_raises_error(self):
        with self.assertRaises(ValueError): compute_density_compensation(self.simple_3d_traj_points,method="pipe")
    def test_cdc_voronoi_complex_input_2d(self):
        ct=self.simple_2d_traj_points[:,0]+1j*self.simple_2d_traj_points[:,1]
        w=compute_density_compensation(ct); wr=compute_density_compensation(self.simple_2d_traj_points)
        self.assertEqual(w.shape,ct.shape); self.assertTrue(np.allclose(np.sum(w),1.)); self.assertTrue(np.allclose(w,wr))
    def test_cdc_pipe_complex_input_2d(self):
        ct=self.simple_2d_traj_points[:,0]+1j*self.simple_2d_traj_points[:,1]
        w=compute_density_compensation(ct,method="pipe"); wr=compute_density_compensation(self.simple_2d_traj_points,method="pipe")
        self.assertEqual(w.shape,ct.shape); self.assertTrue(np.allclose(np.sum(w),1.)); self.assertTrue(np.allclose(w,wr))
    def test_cdc_unknown_method(self):
        with self.assertRaises(ValueError): compute_density_compensation(self.simple_2d_traj_points,method="unknown")
    def test_cdc_empty_trajectory(self):
        self.assertEqual(compute_density_compensation(np.empty((0,2))).size,0)
        self.assertEqual(compute_density_compensation(np.empty((0,),dtype=complex)).size,0)
    def test_create_periodic_points_2d(self):
        p=np.array([[.1,.2]]); e=create_periodic_points(p,ndim=2); self.assertEqual(e.shape,(9,2))
        self.assertTrue(np.any(np.allclose(e,p,atol=1e-7),axis=1))
        s=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
        for sh in s: self.assertTrue(np.any(np.allclose(e,p+np.array(sh),atol=1e-7),axis=1))
    def test_create_periodic_points_3d(self):
        p=np.array([[.1,.2,.3]]); e=create_periodic_points(p,ndim=3); self.assertEqual(e.shape,(27,3))
        self.assertTrue(np.any(np.allclose(e,p,atol=1e-7),axis=1))
        s=[[-1,-1,-1],[0,0,0],[1,1,1],[1,0,0],[0,1,0],[0,0,1]]
        for sh in s: self.assertTrue(np.any(np.allclose(e,p+np.array(sh),atol=1e-7),axis=1))
    def test_create_periodic_points_invalid_ndim(self):
        p=np.array([[.1,.2]])
        with self.assertRaisesRegex(ValueError,"Number of dimensions must be 2 or 3"): create_periodic_points(p,ndim=1)
        with self.assertRaisesRegex(ValueError,"Number of dimensions must be 2 or 3"): create_periodic_points(p,ndim=4)
    def test_create_periodic_points_shape_mismatch(self):
        p2d=np.array([[.1,.2]]); p3d=np.array([[.1,.2,.3]])
        with self.assertRaisesRegex(ValueError,"Trajectory shape.*inconsistent with ndim"): create_periodic_points(p2d,ndim=3)
        with self.assertRaisesRegex(ValueError,"Trajectory shape.*inconsistent with ndim"): create_periodic_points(p3d,ndim=2)
    def test_cvd_2d_clipped(self):
        w=compute_voronoi_density(self.simple_2d_traj_points,boundary_type="clipped")
        self.assertEqual(w.shape,(4,)); self.assertTrue(np.allclose(np.sum(w),1.))
        self.assertTrue(np.allclose(w,np.ones(4)/4.))
    def test_cvd_2d_periodic(self):
        p=(self.simple_2d_traj_points-np.min(self.simple_2d_traj_points))/(np.max(self.simple_2d_traj_points)-np.min(self.simple_2d_traj_points))-.5
        w=compute_voronoi_density(p,boundary_type="periodic"); self.assertEqual(w.shape,(4,))
        self.assertTrue(np.allclose(np.sum(w),1.)); self.assertTrue(np.allclose(w,np.ones(4)/4.))
    def test_cvd_3d_clipped(self):
        w=compute_voronoi_density(self.simple_3d_traj_points,boundary_type="clipped")
        self.assertEqual(w.shape,(4,)); self.assertTrue(np.allclose(np.sum(w),1.))
        self.assertTrue(np.allclose(w,np.ones(4)/4.))
    def test_cvd_3d_periodic(self):
        p=self.simple_3d_traj_points.copy(); mn,mx=np.min(p),np.max(p)
        if mx==mn:mx+=1e-9
        pn=(p-mn)/(mx-mn)-.5; w=compute_voronoi_density(pn,boundary_type="periodic")
        self.assertEqual(w.shape,(4,)); self.assertTrue(np.allclose(np.sum(w),1.))
        self.assertTrue(np.allclose(w,np.ones(4)/4.))
    def test_cvd_empty_trajectory(self): self.assertEqual(compute_voronoi_density(np.empty((0,2))).size,0)
    def test_cvd_single_point_2d(self):
        p=np.array([[.1,.2]]); wc=compute_voronoi_density(p,boundary_type="clipped")
        self.assertEqual(wc.shape,(1,)); self.assertTrue(np.allclose(wc,[1.]))
        wp=compute_voronoi_density(p,boundary_type="periodic")
        self.assertEqual(wp.shape,(1,)); self.assertTrue(np.allclose(wp,[1.]))
    def test_cvd_collinear_points_2d_clipped(self):
        p=np.array([[0,0],[1,0],[2,0],[3,0]]); w=compute_voronoi_density(p,boundary_type="clipped")
        self.assertTrue(np.allclose(w,np.ones(p.shape[0])/p.shape[0]))
    def test_cvd_collinear_points_2d_periodic(self):
        p=np.array([[-.5,0],[-.25,0],[0,0],[.25,0],[.5,0]]); w=compute_voronoi_density(p,boundary_type="periodic")
        self.assertTrue(np.allclose(w,np.ones(p.shape[0])/p.shape[0]))
    def test_generate_spiral_trajectory(self):
        na,ns,fov=4,100,.2; tj=generate_spiral_trajectory(na,ns,fov_m=fov,dt_seconds=self.dt_s)
        self.assertIsInstance(tj,Trajectory); self.assertEqual(tj.get_num_dimensions(),2)
        self.assertEqual(tj.get_num_points(),na*ns); gp=tj.metadata['generator_params']
        km_exp=np.pi/fov; self.assertAlmostEqual(gp['k_max_calculated_rad_m'],km_exp)
        r=np.sqrt(tj.kspace_points_rad_per_m[0,:]**2+tj.kspace_points_rad_per_m[1,:]**2)
        self.assertTrue(np.max(r)<=km_exp*1.001)
        km_ex=150.; nr=5.; tj_ex=generate_spiral_trajectory(na,ns,fov_m=fov,max_k_rad_per_m=km_ex,num_revolutions=nr)
        self.assertAlmostEqual(tj_ex.metadata['generator_params']['k_max_calculated_rad_m'],km_ex)
        r_ex=np.sqrt(tj_ex.kspace_points_rad_per_m[0,:]**2+tj_ex.kspace_points_rad_per_m[1,:]**2)
        self.assertTrue(np.max(r_ex)<=km_ex*1.001)
    def test_generate_radial_trajectory(self):
        nsp,nss,fov=10,64,.25
        tj_ga=generate_radial_trajectory(nsp,nss,fov_m=fov,use_golden_angle=True,dt_seconds=self.dt_s)
        self.assertTrue(tj_ga.metadata['generator_params']['use_golden_angle'])
        km_exp=np.pi/fov; r_ga=np.sqrt(tj_ga.kspace_points_rad_per_m[0,:]**2+tj_ga.kspace_points_rad_per_m[1,:]**2)
        self.assertTrue(np.max(r_ga)<=km_exp*1.001)
        tj_uni=generate_radial_trajectory(nsp,nss,fov_m=fov,use_golden_angle=False,dt_seconds=self.dt_s)
        self.assertFalse(tj_uni.metadata['generator_params']['use_golden_angle'])
        if nsp>1:
            a_ga=np.arctan2(tj_ga.kspace_points_rad_per_m[1,::nss],tj_ga.kspace_points_rad_per_m[0,::nss])
            a_uni=np.arctan2(tj_uni.kspace_points_rad_per_m[1,::nss],tj_uni.kspace_points_rad_per_m[0,::nss])
            self.assertFalse(np.allclose(a_ga[:min(5,nsp)],a_uni[:min(5,nsp)]))
    def test_generate_spiral_edge_cases(self):
        fov=.2; dt=self.dt_s
        with self.subTest(c="1a1s"):
            tj=generate_spiral_trajectory(1,1,fov,dt); km=np.pi/fov
            np.testing.assert_allclose(tj.kspace_points_rad_per_m,np.array([[km],[0.]]),atol=1e-6)
        with self.subTest(c="0a"): self.assertEqual(generate_spiral_trajectory(0,100,fov,dt).get_num_points(),0)
        with self.subTest(c="0s"): self.assertEqual(generate_spiral_trajectory(4,0,fov,dt).get_num_points(),0)
    def test_generate_radial_edge_cases(self):
        fov=.25; dt=self.dt_s
        with self.subTest(c="1s1s"):
            tj=generate_radial_trajectory(1,1,fov,dt)
            np.testing.assert_allclose(tj.kspace_points_rad_per_m,np.array([[0.],[0.]]),atol=1e-7)
        with self.subTest(c="0s"): self.assertEqual(generate_radial_trajectory(0,64,fov,dt).get_num_points(),0)
        with self.subTest(c="0ss"): self.assertEqual(generate_radial_trajectory(10,0,fov,dt).get_num_points(),0)
    def test_constrain_trajectory(self):
        k_viol=np.stack((np.linspace(0,1000,10),np.linspace(0,500,10)))
        tj_v=Trajectory("v_traj",k_viol,dt_seconds=self.dt_s,gamma_Hz_per_T=self.gamma_1h)
        mg,ms=.04,150; c_tj=constrain_trajectory(tj_v,mg,ms)
        self.assertIsInstance(c_tj,Trajectory)
        if k_viol.shape[1]>1: self.assertFalse(np.allclose(tj_v.kspace_points_rad_per_m,c_tj.kspace_points_rad_per_m))
        self.assertEqual(c_tj.metadata['constraints']['post_processed_max_gradient_Tm_per_m'],mg) 
        self.assertEqual(c_tj.metadata['constraints']['post_processed_max_slew_rate_Tm_per_s'],ms) 
    def test_constrain_trajectory_empty(self):
        et=Trajectory("empty",np.array([[],[]]),dt_seconds=self.dt_s)
        self.assertEqual(constrain_trajectory(et,.04,150).get_num_points(),0)
    def test_constrain_trajectory_single_point(self):
        spk=np.array([[0.],[0.]]);spt=Trajectory("spt",spk,dt_seconds=self.dt_s)
        cs=constrain_trajectory(spt,.04,150)
        self.assertEqual(cs.get_num_points(),1); np.testing.assert_allclose(cs.kspace_points_rad_per_m,spk)
    def test_constrain_trajectory_no_dt_error(self):
        kps=np.array([[0,1],[0,1]]).astype(float); tj_ndt=Trajectory("ndt",kps,dt_seconds=None)
        with self.assertRaisesRegex(ValueError,"Dwell time .* must be positive"): constrain_trajectory(tj_ndt,.04,150)
    def test_constrain_trajectory_on_already_constrained_spiral(self):
        fov,dt,gma=.2,4e-6,COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
        na,nsps=2,64; gg,gs=.015,75.
        tj_gen_constr=generate_spiral_trajectory(na,nsps,fov,dt,gma,max_gradient_Tm_per_m=gg,max_slew_rate_Tm_per_s=gs)
        self.assertTrue(tj_gen_constr.metadata['generator_params']['constraints_applied'])
        ppg,pps=gg*.999,gs*.999
        tj_post_constr=constrain_trajectory(tj_gen_constr,ppg,pps,dt_seconds=dt)
        np.testing.assert_allclose(tj_gen_constr.kspace_points_rad_per_m,tj_post_constr.kspace_points_rad_per_m,atol=1e-5,rtol=1e-3)
        self.assertTrue(tj_post_constr.metadata['constraints'].get('post_processing_applied'))
    def test_reconstruct_image_simple_center_peak(self):
        kd=np.zeros(self.simple_2d_traj_obj.get_num_points(),dtype=complex); kd[0]=100.
        gs=(32,32); img=reconstruct_image(kd,self.simple_2d_traj_obj,gs,density_comp_method=None,verbose=False)
        self.assertEqual(img.shape,gs); cpy,cpx=gs[0]//2,gs[1]//2
        max_ic=np.unravel_index(np.argmax(img),img.shape)
        self.assertTrue(abs(max_ic[0]-cpy)<=2 and abs(max_ic[1]-cpx)<=2)
    def test_reconstruct_image_with_voronoi_dc(self):
        tj_r=generate_radial_trajectory(8,32,self.fov_m); kd=np.ones(tj_r.get_num_points(),dtype=complex)
        gs=(24,24); img_dc=reconstruct_image(kd,tj_r,gs,density_comp_method="voronoi",verbose=False)
        img_ndc=reconstruct_image(kd,tj_r,gs,density_comp_method=None,verbose=False)
        if tj_r.get_num_points()>0: self.assertFalse(np.allclose(img_dc,img_ndc))
    def test_reconstruct_image_input_validation(self):
        gs=(16,16); kd_v=np.ones(self.simple_2d_traj_obj.get_num_points(),dtype=complex)
        kd_ws=np.ones(self.simple_2d_traj_obj.get_num_points()+1,dtype=complex)
        with self.assertRaisesRegex(ValueError,"kspace_data size .* does not match"): reconstruct_image(kd_ws,self.simple_2d_traj_obj,gs)
        with self.assertRaisesRegex(ValueError,"Image reconstruction currently only supports 2D"): reconstruct_image(np.ones(self.simple_3d_traj_obj.get_num_points()),self.simple_3d_traj_obj,gs)
        igs=[(16,),(16,16,16),(0,16),(16,-5),(16.5,16)]
        for i in igs: 
            with self.subTest(ig=i): 
                with self.assertRaisesRegex(ValueError,"grid_size must be a tuple of 2 positive integers"): reconstruct_image(kd_v,self.simple_2d_traj_obj,i)
    @patch.object(Trajectory,'plot_2d')
    def test_display_trajectory_2d_called(self,mp2d):
        kw={'title':'Test 2D Plot','max_total_points':500}; ra=display_trajectory(self.simple_2d_traj_obj,plot_type="2D",**kw)
        mp2d.assert_called_once(); _,mk=mp2d.call_args
        for k,v in kw.items():self.assertEqual(mk[k],v)
        self.assertEqual(ra,mp2d.return_value)
    @patch.object(Trajectory,'plot_3d')
    def test_display_trajectory_3d_called(self,mp3d):
        kw={'title':'Test 3D Plot','max_interleaves':10}; ra=display_trajectory(self.simple_3d_traj_obj,plot_type="3D",**kw)
        mp3d.assert_called_once(); _,mk=mp3d.call_args
        for k,v in kw.items():self.assertEqual(mk[k],v)
        self.assertEqual(ra,mp3d.return_value)
    def test_display_trajectory_invalid_type(self):
        with self.assertRaisesRegex(TypeError,"trajectory_obj must be an instance of Trajectory"):display_trajectory("not_a_traj",plot_type="2D")
    def test_display_trajectory_invalid_plot_type(self):
        with self.assertRaisesRegex(ValueError,"plot_type must be '2D' or '3D'"):display_trajectory(self.simple_2d_traj_obj,plot_type="invalid")


class TestGIRF(unittest.TestCase):
    def setUp(self):
        self.sample_ht_x = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        self.sample_ht_y = np.array([0.2, 0.4, 0.2])
        self.sample_ht_z = np.array([0.5, 0.5, 0.5, 0.5])
        self.valid_dt_girf = 4e-6
        self.test_girf_name = "TestGIRFProfile"
        self.temp_dir = tempfile.mkdtemp()
        self.filepath_x = os.path.join(self.temp_dir, 'girf_x.npy')
        self.filepath_y = os.path.join(self.temp_dir, 'girf_y.npy')
        self.filepath_z = os.path.join(self.temp_dir, 'girf_z.npy')
        np.save(self.filepath_x, self.sample_ht_x); np.save(self.filepath_y, self.sample_ht_y); np.save(self.filepath_z, self.sample_ht_z)
        self.dt_traj = 1e-5; self.num_points_traj = 100; k_max_val = np.pi / 0.01
        kx_3d = np.linspace(0,k_max_val,self.num_points_traj); ky_3d = np.linspace(0,k_max_val/2,self.num_points_traj); kz_3d = np.zeros(self.num_points_traj)
        self.kspace_3d_for_girf_test = np.stack([kx_3d,ky_3d,kz_3d],axis=0)
        self.traj_3d_for_girf = Trajectory("TestTraj3D_forGIRF",self.kspace_3d_for_girf_test,dt_seconds=self.dt_traj)
        kx_2d = np.linspace(0,k_max_val,self.num_points_traj); ky_2d = np.linspace(0,k_max_val/3,self.num_points_traj)
        self.kspace_2d_for_girf_test = np.stack([kx_2d,ky_2d],axis=0)
        self.traj_2d_for_girf = Trajectory("TestTraj2D_forGIRF",self.kspace_2d_for_girf_test,dt_seconds=self.dt_traj)
        self.identity_girf = GIRF(np.array([1.0]),np.array([1.0]),np.array([1.0]),dt_girf=self.dt_traj,name="IdentityGIRF")
        self.scaling_girf = GIRF(np.array([0.5]),np.array([2.0]),np.array([1.0]),dt_girf=self.dt_traj,name="ScalingGIRF")
        blur_kernel_raw = np.array([0.25,0.5,0.25]); blur_kernel_normalized = blur_kernel_raw/np.sum(blur_kernel_raw)
        self.blurring_girf = GIRF(blur_kernel_normalized.copy(),blur_kernel_normalized.copy(),np.array([1.0]),dt_girf=self.dt_traj,name="BlurringGIRF")
        self.zero_h_t = np.array([0.0]); self.delta_h_t = np.array([1.0])
        self.identity_sgirf = sGIRF(self.delta_h_t,self.zero_h_t,self.zero_h_t,self.zero_h_t,self.delta_h_t,self.zero_h_t,self.zero_h_t,self.zero_h_t,self.delta_h_t,dt_sgirf=self.dt_traj,name="Identity_sGIRF")
        self.scaling_sgirf = sGIRF(np.array([0.5]),self.zero_h_t,self.zero_h_t,self.zero_h_t,np.array([1.5]),self.zero_h_t,self.zero_h_t,self.zero_h_t,np.array([1.0]),dt_sgirf=self.dt_traj,name="Scaling_sGIRF")
        self.crossterm_sgirf = sGIRF(self.delta_h_t,np.array([0.1]),self.zero_h_t,np.array([0.2]),self.delta_h_t,np.array([-0.1]),self.zero_h_t,self.zero_h_t,self.delta_h_t,dt_sgirf=self.dt_traj,name="Crossterm_sGIRF")

    def tearDown(self): shutil.rmtree(self.temp_dir)
    def test_girf_initialization_valid(self):
        g=GIRF(self.sample_ht_x,self.sample_ht_y,self.sample_ht_z,self.valid_dt_girf,name=self.test_girf_name)
        self.assertTrue(np.array_equal(g.h_t_x,self.sample_ht_x)); self.assertTrue(np.array_equal(g.h_t_y,self.sample_ht_y))
        self.assertTrue(np.array_equal(g.h_t_z,self.sample_ht_z)); self.assertEqual(g.dt_girf,self.valid_dt_girf); self.assertEqual(g.name,self.test_girf_name)
    def test_girf_initialization_default_name(self):
        g=GIRF(self.sample_ht_x,self.sample_ht_y,self.sample_ht_z,self.valid_dt_girf); self.assertEqual(g.name,"CustomGIRF")
    def test_girf_initialization_invalid_dt(self):
        with self.assertRaisesRegex(ValueError,"dt_girf must be positive"): GIRF(self.sample_ht_x,self.sample_ht_y,self.sample_ht_z,dt_girf=0)
    def test_girf_initialization_invalid_ht_dims(self):
        h2d=np.array([[.1,.2],[.3,.4]])
        with self.assertRaisesRegex(ValueError,"h_t_x must be a 1D array"): GIRF(h2d,self.sample_ht_y,self.sample_ht_z,self.valid_dt_girf)
    def test_girf_repr_method(self):
        g=GIRF(self.sample_ht_x,self.sample_ht_y,self.sample_ht_z,self.valid_dt_girf,name=self.test_girf_name)
        er=(f"GIRF(name='{self.test_girf_name}', dt_girf={self.valid_dt_girf:.2e}, x_len={len(self.sample_ht_x)}, y_len={len(self.sample_ht_y)}, z_len={len(self.sample_ht_z)})")
        self.assertEqual(repr(g),er)
    def test_from_files_successful_load_with_name(self):
        g=GIRF.from_files(self.filepath_x,self.filepath_y,self.filepath_z,self.valid_dt_girf,name="LoadedGIRF")
        self.assertTrue(np.array_equal(g.h_t_x,self.sample_ht_x)); self.assertEqual(g.name,"LoadedGIRF")
    def test_from_files_successful_load_auto_name(self):
        g=GIRF.from_files(self.filepath_x,self.filepath_y,self.filepath_z,self.valid_dt_girf); self.assertEqual(g.name,"girf")
    def test_from_files_file_not_found(self):
        nf=os.path.join(self.temp_dir,"non_existent.npy")
        with self.assertRaisesRegex(FileNotFoundError,f"Could not load GIRF data: {nf} not found."): GIRF.from_files(nf,self.filepath_y,self.filepath_z,self.valid_dt_girf)
    def test_from_files_invalid_npy_file(self):
        invf=os.path.join(self.temp_dir,"invalid.npy"); open(invf,'w').write("bad")
        with self.assertRaisesRegex(ValueError,"Error loading GIRF data"): GIRF.from_files(invf,self.filepath_y,self.filepath_z,self.valid_dt_girf)
    def test_from_files_invalid_ht_dims_in_file(self):
        f2d=os.path.join(self.temp_dir,"ht_2d.npy"); np.save(f2d,np.array([[.1,.2],[.3,.4]]))
        with self.assertRaisesRegex(ValueError,f"h_t_x from {f2d} must be a 1D array"):GIRF.from_files(f2d,self.filepath_y,self.filepath_z,self.valid_dt_girf)
    def test_from_files_invalid_dt(self):
        with self.assertRaisesRegex(ValueError,"dt_girf must be positive"):GIRF.from_files(self.filepath_x,self.filepath_y,self.filepath_z,dt_girf=0)
    def test_precompensate_identity_girf(self):
        pt=precompensate_gradients_with_girf(self.traj_3d_for_girf,self.identity_girf,num_iterations=5,alpha=0.5)
        pm=pt.metadata.get('girf_precompensation',{}); self.assertTrue(pm.get('applied'))
        for err in pm.get('final_relative_errors_per_axis',[1.]): self.assertLess(err,1e-3)
        og=self.traj_3d_for_girf.get_gradient_waveforms_Tm()
        np.testing.assert_allclose(pt.gradient_waveforms_Tm,og,atol=1e-4,rtol=1e-3)
        np.testing.assert_allclose(pt.kspace_points_rad_per_m,self.traj_3d_for_girf.kspace_points_rad_per_m,atol=1e-4,rtol=1e-3)
    def test_precompensate_with_distorting_girf(self):
        ni,av=30,.5; tt=self.traj_3d_for_girf
        pt=precompensate_gradients_with_girf(tt,self.blurring_girf,num_iterations=ni,alpha=av,tolerance=1e-3)
        pm=pt.metadata.get('girf_precompensation',{}); self.assertTrue(pm.get('applied'))
        pcg=pt.gradient_waveforms_Tm
        sim_traj=Trajectory("sim",pt.kspace_points_rad_per_m,gradient_waveforms_Tm=pcg,dt_seconds=pt.dt_seconds,metadata={'gamma_Hz_per_T':pt.metadata['gamma_Hz_per_T']})
        sag=predict_actual_gradients(sim_traj,self.blurring_girf); tg=tt.get_gradient_waveforms_Tm()
        np.testing.assert_allclose(sag,tg,rtol=.15,atol=.005)
    def test_precompensate_convergence_tolerance(self):
        tt=self.traj_3d_for_girf
        pt_li=precompensate_gradients_with_girf(tt,self.blurring_girf,num_iterations=2,alpha=.5,tolerance=1e-7)
        er_li=pt_li.metadata['girf_precompensation']['final_relative_errors_per_axis']
        pt_hi=precompensate_gradients_with_girf(tt,self.blurring_girf,num_iterations=50,alpha=.5,tolerance=.05)
        er_hi=pt_hi.metadata['girf_precompensation']['final_relative_errors_per_axis']
        for i in range(tt.get_num_dimensions()):
            if np.linalg.norm(tt.get_gradient_waveforms_Tm()[i,:])>1e-9:
                self.assertTrue(er_li[i]>1e-7+1e-9); self.assertLessEqual(er_hi[i],.05+1e-5); self.assertLess(er_hi[i],er_li[i])
    def test_precompensate_input_validation_and_failure_modes(self):
        tnd=Trajectory("NoDt",self.kspace_3d_for_girf_test,dt_seconds=None)
        ftnd=precompensate_gradients_with_girf(tnd,self.identity_girf); self.assertFalse(ftnd.metadata['girf_precompensation']['applied'])
        ek=np.empty((self.traj_3d_for_girf.get_num_dimensions(),0)); tek=Trajectory("EmptyK",ek,dt_seconds=self.dt_traj)
        fteg=precompensate_gradients_with_girf(tek,self.identity_girf); self.assertFalse(fteg.metadata['girf_precompensation']['applied'])
        with self.assertRaisesRegex(ValueError,"num_iterations must be positive"):precompensate_gradients_with_girf(self.traj_3d_for_girf,self.identity_girf,num_iterations=0)
    def test_precompensate_kspace_recalculation_accuracy(self):
        pt=precompensate_gradients_with_girf(self.traj_3d_for_girf,self.blurring_girf,num_iterations=5)
        pg=pt.gradient_waveforms_Tm; dt=pt.dt_seconds; ga=pt.metadata['girf_precompensation']['gamma_used_for_kspace_recalc']
        ekr=np.zeros_like(pg); iko=self.traj_3d_for_girf.kspace_points_rad_per_m[:,0].reshape(-1,1)
        if pg.shape[1]>0:ekr=iko+np.concatenate((np.zeros((pg.shape[0],1)),np.cumsum(pg[:,:-1]*ga*dt,axis=1)),axis=1)
        np.testing.assert_allclose(pt.kspace_points_rad_per_m,ekr,atol=1e-6,rtol=1e-5)

    # Tests for correct_kspace_with_girf using GIRF
    def test_correct_kspace_identity_girf_input(self):
        corrected_traj = correct_kspace_with_girf(self.traj_3d_for_girf, girf_system=self.identity_girf)
        meta = corrected_traj.metadata.get('girf_correction', {})
        self.assertTrue(meta.get('applied'))
        self.assertEqual(meta.get('type'), 'GIRF')
        self.assertEqual(meta.get('girf_name'), self.identity_girf.name)
        np.testing.assert_allclose(corrected_traj.kspace_points_rad_per_m, self.traj_3d_for_girf.kspace_points_rad_per_m, atol=1e-6)
        np.testing.assert_allclose(corrected_traj.gradient_waveforms_Tm, self.traj_3d_for_girf.get_gradient_waveforms_Tm(), atol=1e-7)

    def test_correct_kspace_scaling_girf_input(self):
        corrected_traj = correct_kspace_with_girf(self.traj_3d_for_girf, girf_system=self.scaling_girf)
        meta = corrected_traj.metadata.get('girf_correction', {})
        self.assertTrue(meta.get('applied'))
        self.assertEqual(meta.get('type'), 'GIRF')
        cmd_grads = self.traj_3d_for_girf.get_gradient_waveforms_Tm()
        actual_grads = corrected_traj.gradient_waveforms_Tm
        np.testing.assert_allclose(actual_grads[0,:], cmd_grads[0,:] * 0.5, atol=1e-7)
        np.testing.assert_allclose(actual_grads[1,:], cmd_grads[1,:] * 2.0, atol=1e-7)
        np.testing.assert_allclose(actual_grads[2,:], cmd_grads[2,:] * 1.0, atol=1e-7)
        gamma = meta['gamma_used_for_correction']; dt = corrected_traj.dt_seconds
        k0 = self.traj_3d_for_girf.kspace_points_rad_per_m[:,0].reshape(-1,1)
        exp_k = k0 + np.concatenate((np.zeros((actual_grads.shape[0],1)), np.cumsum(actual_grads[:,:-1]*gamma*dt,axis=1)),axis=1)
        np.testing.assert_allclose(corrected_traj.kspace_points_rad_per_m, exp_k, atol=1e-6)

    def test_correct_kspace_gamma_override_girf_input(self): 
        custom_gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'] * 0.75
        corrected_traj = correct_kspace_with_girf(self.traj_3d_for_girf, girf_system=self.identity_girf, gamma_Hz_per_T=custom_gamma)
        meta = corrected_traj.metadata['girf_correction']
        self.assertTrue(meta['applied']); self.assertEqual(meta['type'], 'GIRF')
        self.assertEqual(meta['gamma_used_for_correction'], custom_gamma)
        self.assertEqual(meta['gamma_override_Hz_per_T'], custom_gamma)
        self.assertEqual(corrected_traj.metadata['gamma_Hz_per_T'], custom_gamma)
        self.assertFalse(np.allclose(corrected_traj.kspace_points_rad_per_m, self.traj_3d_for_girf.kspace_points_rad_per_m))

    def test_correct_kspace_trajectory_no_dt_girf_input(self): 
        traj_no_dt = Trajectory("NoDt", self.kspace_3d_for_girf_test, dt_seconds=None)
        corrected_traj = correct_kspace_with_girf(traj_no_dt, girf_system=self.identity_girf)
        meta = corrected_traj.metadata['girf_correction']
        self.assertFalse(meta['applied']); self.assertIn("dt_seconds is missing", meta['status'])
        self.assertEqual(meta['type'], 'GIRF')

    @patch('trajgen.predict_actual_gradients')
    def test_correct_kspace_empty_predicted_gradients_girf_input(self, mock_predict_gradients_girf): 
        mock_predict_gradients_girf.return_value = np.empty((self.traj_3d_for_girf.get_num_dimensions(), 0))
        corrected_traj = correct_kspace_with_girf(self.traj_3d_for_girf, girf_system=self.identity_girf)
        meta = corrected_traj.metadata['girf_correction']
        self.assertFalse(meta['applied']); self.assertIn("No actual gradients processed", meta['status'])
        self.assertEqual(meta['type'], 'GIRF')
        mock_predict_gradients_girf.assert_called_once_with(self.traj_3d_for_girf, self.identity_girf)

    def test_correct_kspace_start_point_preservation_girf_input(self): 
        k_start_offset = np.array([10., -5., 100.]).reshape(-1,1)
        k_data = self.traj_3d_for_girf.kspace_points_rad_per_m - self.traj_3d_for_girf.kspace_points_rad_per_m[:,0].reshape(-1,1) + k_start_offset
        traj_offset = Trajectory("Offset", k_data, dt_seconds=self.dt_traj, metadata={'gamma_Hz_per_T': self.traj_3d_for_girf.metadata['gamma_Hz_per_T']})
        corrected_traj = correct_kspace_with_girf(traj_offset, girf_system=self.identity_girf)
        self.assertTrue(corrected_traj.metadata['girf_correction']['applied'])
        self.assertEqual(corrected_traj.metadata.get('girf_correction',{}).get('type'), 'GIRF')
        np.testing.assert_allclose(corrected_traj.kspace_points_rad_per_m[:,0], k_start_offset.flatten(), atol=1e-7)

    # Tests for correct_kspace_with_girf using sGIRF
    def test_correct_kspace_with_identity_sgirf_3d_traj(self):
        corrected_traj = correct_kspace_with_girf(self.traj_3d_for_girf, girf_system=self.identity_sgirf)
        meta = corrected_traj.metadata['girf_correction']
        self.assertTrue(meta['applied']); self.assertEqual(meta['type'], 'sGIRF')
        self.assertEqual(meta['girf_name'], self.identity_sgirf.name)
        np.testing.assert_allclose(corrected_traj.kspace_points_rad_per_m, self.traj_3d_for_girf.kspace_points_rad_per_m, atol=1e-6)
        np.testing.assert_allclose(corrected_traj.gradient_waveforms_Tm, self.traj_3d_for_girf.get_gradient_waveforms_Tm(), atol=1e-7)
        self.assertEqual(corrected_traj.get_num_dimensions(), 3)

    def test_correct_kspace_with_identity_sgirf_2d_traj(self):
        corrected_traj = correct_kspace_with_girf(self.traj_2d_for_girf, girf_system=self.identity_sgirf)
        meta = corrected_traj.metadata['girf_correction']
        self.assertTrue(meta['applied']); self.assertEqual(meta['type'], 'sGIRF')
        cmd_2d_grads = self.traj_2d_for_girf.get_gradient_waveforms_Tm()
        actual_3d_grads = corrected_traj.gradient_waveforms_Tm
        self.assertEqual(actual_3d_grads.shape[0], 3)
        np.testing.assert_allclose(actual_3d_grads[0,:], cmd_2d_grads[0,:], atol=1e-7)
        np.testing.assert_allclose(actual_3d_grads[1,:], cmd_2d_grads[1,:], atol=1e-7)
        np.testing.assert_allclose(actual_3d_grads[2,:], np.zeros_like(cmd_2d_grads[0,:]), atol=1e-7)
        self.assertEqual(corrected_traj.get_num_dimensions(), 3) 
        k0_2d = self.traj_2d_for_girf.kspace_points_rad_per_m[:,0]
        exp_k0_3d = np.array([k0_2d[0], k0_2d[1], 0.0])
        gamma = meta['gamma_used_for_correction']; dt = corrected_traj.dt_seconds
        exp_k_corr = np.zeros((3, self.num_points_traj)); exp_k_corr[:,0] = exp_k0_3d
        if self.num_points_traj > 1:
            deltas = actual_3d_grads * gamma * dt
            exp_k_corr[:,1:] = exp_k0_3d.reshape(-1,1) + np.cumsum(deltas[:,:-1], axis=1)
        np.testing.assert_allclose(corrected_traj.kspace_points_rad_per_m, exp_k_corr, atol=1e-6)

    def test_correct_kspace_with_crossterm_sgirf_3d_traj(self):
        corrected_traj = correct_kspace_with_girf(self.traj_3d_for_girf, girf_system=self.crossterm_sgirf)
        meta = corrected_traj.metadata['girf_correction']
        self.assertTrue(meta['applied']); self.assertEqual(meta['type'], 'sGIRF')
        self.assertFalse(np.allclose(corrected_traj.gradient_waveforms_Tm, self.traj_3d_for_girf.get_gradient_waveforms_Tm()))
        self.assertFalse(np.allclose(corrected_traj.kspace_points_rad_per_m, self.traj_3d_for_girf.kspace_points_rad_per_m))
        self.assertEqual(corrected_traj.get_num_dimensions(), 3)

    @patch('trajgen.predict_actual_gradients_from_sgirf')
    def test_correct_kspace_with_sgirf_empty_predicted_gradients(self, mock_predict_sgirf_grads):
        mock_predict_sgirf_grads.return_value = np.empty((3, 0)) 
        corrected_traj = correct_kspace_with_girf(self.traj_3d_for_girf, girf_system=self.identity_sgirf)
        meta = corrected_traj.metadata['girf_correction']
        self.assertFalse(meta['applied']); self.assertIn("No actual gradients processed", meta['status'])
        self.assertEqual(meta['type'], 'sGIRF')
        mock_predict_sgirf_grads.assert_called_once_with(self.traj_3d_for_girf, self.identity_sgirf)

    def test_correct_kspace_with_invalid_girf_system_type(self):
        with self.assertRaisesRegex(TypeError, "girf_system must be an instance of GIRF or sGIRF"):
            correct_kspace_with_girf(self.traj_3d_for_girf, "not_a_girf_object")

    # Tests for predict_actual_gradients_from_sgirf
    def test_predict_actual_gradients_from_identity_sgirf_3d_traj(self):
        commanded_gradients_3d = self.traj_3d_for_girf.get_gradient_waveforms_Tm()
        predicted_gradients = predict_actual_gradients_from_sgirf(self.traj_3d_for_girf, self.identity_sgirf)
        self.assertEqual(predicted_gradients.shape, (3, self.num_points_traj))
        np.testing.assert_allclose(predicted_gradients, commanded_gradients_3d, atol=1e-7)

    def test_predict_actual_gradients_from_identity_sgirf_2d_traj(self):
        commanded_gradients_2d = self.traj_2d_for_girf.get_gradient_waveforms_Tm()
        predicted_gradients = predict_actual_gradients_from_sgirf(self.traj_2d_for_girf, self.identity_sgirf)
        self.assertEqual(predicted_gradients.shape, (3, self.num_points_traj))
        np.testing.assert_allclose(predicted_gradients[0,:], commanded_gradients_2d[0,:], atol=1e-7)
        np.testing.assert_allclose(predicted_gradients[1,:], commanded_gradients_2d[1,:], atol=1e-7)
        np.testing.assert_allclose(predicted_gradients[2,:], np.zeros(self.num_points_traj), atol=1e-7)

    def test_predict_actual_gradients_from_scaling_sgirf(self):
        commanded_gradients_3d = self.traj_3d_for_girf.get_gradient_waveforms_Tm()
        predicted_gradients = predict_actual_gradients_from_sgirf(self.traj_3d_for_girf, self.scaling_sgirf)
        self.assertEqual(predicted_gradients.shape, (3, self.num_points_traj))
        expected_gx = commanded_gradients_3d[0,:] * 0.5
        expected_gy = commanded_gradients_3d[1,:] * 1.5
        expected_gz = commanded_gradients_3d[2,:] * 1.0 
        np.testing.assert_allclose(predicted_gradients[0,:], expected_gx, atol=1e-7)
        np.testing.assert_allclose(predicted_gradients[1,:], expected_gy, atol=1e-7)
        np.testing.assert_allclose(predicted_gradients[2,:], expected_gz, atol=1e-7)

    def test_predict_actual_gradients_from_crossterm_sgirf(self):
        commanded_gradients_3d = self.traj_3d_for_girf.get_gradient_waveforms_Tm()
        Gx_cmd, Gy_cmd, Gz_cmd = commanded_gradients_3d[0,:], commanded_gradients_3d[1,:], commanded_gradients_3d[2,:]
        predicted_gradients = predict_actual_gradients_from_sgirf(self.traj_3d_for_girf, self.crossterm_sgirf)
        self.assertEqual(predicted_gradients.shape, (3, self.num_points_traj))
        expected_gx_act = Gx_cmd * 1.0 + Gy_cmd * 0.1 
        expected_gy_act = Gx_cmd * 0.2 + Gy_cmd * 1.0 + Gz_cmd * (-0.1)
        expected_gz_act = Gz_cmd * 1.0
        np.testing.assert_allclose(predicted_gradients[0,:], expected_gx_act, atol=1e-7)
        np.testing.assert_allclose(predicted_gradients[1,:], expected_gy_act, atol=1e-7)
        np.testing.assert_allclose(predicted_gradients[2,:], expected_gz_act, atol=1e-7)

    def test_predict_actual_gradients_from_sgirf_empty_trajectory(self):
        empty_kspace = np.empty((3,0))
        traj_empty = Trajectory("EmptyTraj", empty_kspace, dt_seconds=self.dt_traj)
        predicted_gradients = predict_actual_gradients_from_sgirf(traj_empty, self.identity_sgirf)
        self.assertEqual(predicted_gradients.shape, (3,0))
        empty_kspace_0d = np.empty((0,0))
        traj_empty_0d = Trajectory("Empty0DTraj", empty_kspace_0d, dt_seconds=self.dt_traj)
        predicted_gradients_0d = predict_actual_gradients_from_sgirf(traj_empty_0d, self.identity_sgirf)
        self.assertEqual(predicted_gradients_0d.shape, (3,0))

    def test_predict_actual_gradients_from_sgirf_no_dt(self):
        traj_no_dt = Trajectory("NoDtTraj", self.kspace_3d_for_girf_test, dt_seconds=None)
        with self.assertRaisesRegex(ValueError, "trajectory.dt_seconds must be positive and available"):
            predict_actual_gradients_from_sgirf(traj_no_dt, self.identity_sgirf)
        traj_zero_dt = Trajectory("ZeroDtTraj", self.kspace_3d_for_girf_test, dt_seconds=0)
        with self.assertRaisesRegex(ValueError, "trajectory.dt_seconds must be positive and available"):
            predict_actual_gradients_from_sgirf(traj_zero_dt, self.identity_sgirf)


class TestApplyGirfConvolution(unittest.TestCase):
    def setUp(self):
        self.dt_std = 1e-5 
        self.grad_wave_simple = np.array([0., 0., 1., 0., 0.])
        self.girf_delta_1pt = np.array([1.0])
        self.girf_delta_3pt = np.array([0., 1., 0.])
        self.girf_boxcar_3pt = np.array([1/3., 1/3., 1/3.])

    def test_convolution_input_validation(self):
        grad_2d = np.array([[1,2,3],[4,5,6]]); girf_2d = np.array([[0.1],[0.1]])
        with self.assertRaisesRegex(ValueError, "gradient_waveform_1d must be a 1D NumPy array"):
            apply_girf_convolution(grad_2d, self.girf_delta_1pt, self.dt_std, self.dt_std)
        with self.assertRaisesRegex(ValueError, "girf_h_t_1d must be a 1D NumPy array"):
            apply_girf_convolution(self.grad_wave_simple, girf_2d, self.dt_std, self.dt_std)
        with self.assertRaisesRegex(ValueError, "dt_gradient must be positive"):
            apply_girf_convolution(self.grad_wave_simple, self.girf_delta_1pt, 0, self.dt_std)
        with self.assertRaisesRegex(ValueError, "dt_girf must be positive"):
            apply_girf_convolution(self.grad_wave_simple, self.girf_delta_1pt, self.dt_std, 0)

    def test_convolution_empty_inputs(self):
        empty_arr = np.array([])
        self.assertTrue(np.array_equal(apply_girf_convolution(empty_arr, self.girf_delta_1pt, self.dt_std, self.dt_std), empty_arr))
        self.assertTrue(np.array_equal(apply_girf_convolution(self.grad_wave_simple, empty_arr, self.dt_std, self.dt_std), empty_arr))

    def test_convolution_same_dt(self):
        out_delta1 = apply_girf_convolution(self.grad_wave_simple, self.girf_delta_1pt, self.dt_std, self.dt_std)
        np.testing.assert_allclose(out_delta1, self.grad_wave_simple, atol=1e-7)
        out_delta3 = apply_girf_convolution(self.grad_wave_simple, self.girf_delta_3pt, self.dt_std, self.dt_std)
        np.testing.assert_allclose(out_delta3, self.grad_wave_simple, atol=1e-7)
        expected_boxcar_out = np.array([0., 1/3., 1/3., 1/3., 0.])
        out_boxcar = apply_girf_convolution(self.grad_wave_simple, self.girf_boxcar_3pt, self.dt_std, self.dt_std)
        np.testing.assert_allclose(out_boxcar, expected_boxcar_out, atol=1e-7)

    def test_convolution_different_dt_resample_girf(self):
        gradient_long = np.zeros(100); gradient_long[50] = 1.0
        girf_short_boxcar = np.array([0.25, 0.25, 0.25, 0.25])
        out_upsample = apply_girf_convolution(gradient_long, girf_short_boxcar, 1e-5, 4e-5)
        self.assertAlmostEqual(np.sum(out_upsample), np.sum(gradient_long), places=6)
        out_downsample = apply_girf_convolution(gradient_long, girf_short_boxcar, 5e-5, 1e-5)
        self.assertAlmostEqual(np.sum(out_downsample), np.sum(gradient_long), places=6)

    def test_convolution_normalization_effect(self):
        gradient_impulse = np.array([0.,0.,1.,0.,0.]); girf_unnorm = np.array([.5,.5,.5])
        output = apply_girf_convolution(gradient_impulse,girf_unnorm,1e-5,2e-5)
        self.assertAlmostEqual(np.sum(output),np.sum(gradient_impulse)*np.sum(girf_unnorm),places=6)
        girf_sum_zero = np.array([-.5,1.,-.5])
        output_sum_zero = apply_girf_convolution(gradient_impulse,girf_sum_zero,1e-5,2e-5)
        self.assertAlmostEqual(np.sum(output_sum_zero),0.,places=6)

    def test_convolution_short_girf_resampling(self):
        grad_wave=np.array([0,0,1,0,0],dtype=float); girf_single=np.array([2.]); dt_g,dt_h=1e-3,1e-6
        np.testing.assert_allclose(apply_girf_convolution(grad_wave,girf_single,dt_g,dt_h),grad_wave*2.,atol=1e-7)
        girf_two=np.array([1.,1.]); dt_g_long=1e-2
        np.testing.assert_allclose(apply_girf_convolution(grad_wave,girf_two,dt_g_long,dt_h),grad_wave*2.,atol=1e-7)
        girf_bipolar_short=np.array([1.,-1.])
        np.testing.assert_allclose(apply_girf_convolution(grad_wave,girf_bipolar_short,dt_g_long,dt_h),np.zeros_like(grad_wave),atol=1e-7)


class TestGenerateSpiralTrajectoryWithLimits(unittest.TestCase):
    def setUp(self):
        self.fov_m = 0.256; self.dt_seconds = 4e-6; self.num_arms = 2; self.num_samples_per_arm = 128
        self.gamma_Hz_per_T = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']; self.k_max_ideal = np.pi / self.fov_m
        self.num_revolutions = 10.0
        self.ideal_k_points = self._generate_ideal_spiral_points(self.num_arms,self.num_samples_per_arm,self.fov_m,self.k_max_ideal,self.num_revolutions)
    def _generate_ideal_spiral_points(self,na,nsa,fov,km,nrev):
        kmtu=km if km is not None else np.pi/fov; apts=[]
        for j in range(na):
            ao=j*(2*np.pi/na)
            for s in range(nsa):
                if s==0: kx,ky=0.,0.
                else:
                    tsid=s/(nsa-1) if nsa>1 else 1.
                    ir=tsid*kmtu; ia=ao+nrev*2*np.pi*tsid
                    kx=ir*np.cos(ia); ky=ir*np.sin(ia)
                apts.append(np.array([kx,ky]))
        return np.array(apts).T if apts else np.empty((2,0))
    def test_spiral_no_limits_applied(self):
        tj=generate_spiral_trajectory(self.num_arms,self.num_samples_per_arm,self.fov_m,self.dt_seconds,self.gamma_Hz_per_T,None,None,self.num_revolutions,self.k_max_ideal)
        self.assertFalse(tj.metadata['generator_params']['constraints_applied'])
        np.testing.assert_allclose(tj.kspace_points_rad_per_m,self.ideal_k_points,atol=1e-6)
    def test_spiral_with_very_loose_limits(self):
        tj=generate_spiral_trajectory(self.num_arms,self.num_samples_per_arm,self.fov_m,self.dt_seconds,self.gamma_Hz_per_T,10.,20000.,self.num_revolutions,self.k_max_ideal)
        self.assertTrue(tj.metadata['generator_params']['constraints_applied'])
        np.testing.assert_allclose(tj.kspace_points_rad_per_m,self.ideal_k_points,atol=1e-3)
    def test_spiral_gradient_limiting_active(self):
        tj=generate_spiral_trajectory(self.num_arms,self.num_samples_per_arm,self.fov_m,self.dt_seconds,self.gamma_Hz_per_T,.005,10000.,self.num_revolutions,self.k_max_ideal)
        self.assertTrue(tj.metadata['generator_params']['constraints_applied'])
        self.assertLessEqual(tj.get_max_grad_Tm(),.005*1.01)
        self.assertTrue(np.max(np.linalg.norm(tj.kspace_points_rad_per_m,axis=0))<self.k_max_ideal*.99)
    def test_spiral_slew_limiting_active(self):
        tj=generate_spiral_trajectory(self.num_arms,self.num_samples_per_arm,self.fov_m,self.dt_seconds,self.gamma_Hz_per_T,10.,20.,self.num_revolutions,self.k_max_ideal)
        self.assertTrue(tj.metadata['generator_params']['constraints_applied'])
        self.assertLessEqual(tj.get_max_slew_Tm_per_s(),20.*1.01)
        self.assertTrue(np.max(np.linalg.norm(tj.kspace_points_rad_per_m,axis=0))<self.k_max_ideal*.99)
    def test_spiral_both_limits_active(self):
        tj=generate_spiral_trajectory(self.num_arms,self.num_samples_per_arm,self.fov_m,self.dt_seconds,self.gamma_Hz_per_T,.01,50.,self.num_revolutions,self.k_max_ideal)
        self.assertTrue(tj.metadata['generator_params']['constraints_applied'])
        self.assertLessEqual(tj.get_max_grad_Tm(),.01*1.01); self.assertLessEqual(tj.get_max_slew_Tm_per_s(),50.*1.01)
        self.assertTrue(np.max(np.linalg.norm(tj.kspace_points_rad_per_m,axis=0))<self.k_max_ideal*.99)
    def test_spiral_metadata_constraints(self):
        gl,sl=.02,100.
        tj=generate_spiral_trajectory(self.num_arms,self.num_samples_per_arm,self.fov_m,self.dt_seconds,self.gamma_Hz_per_T,gl,sl,self.num_revolutions,self.k_max_ideal)
        gp=tj.metadata['generator_params']; self.assertTrue(gp['constraints_applied'])
        self.assertEqual(gp['max_gradient_Tm_per_m'],gl); self.assertEqual(gp['max_slew_rate_Tm_per_s'],sl)
    def test_spiral_zero_or_negative_limits_behavior(self):
        tj0g=generate_spiral_trajectory(self.num_arms,self.num_samples_per_arm,self.fov_m,self.dt_seconds,max_gradient_Tm_per_m=0,max_slew_rate_Tm_per_s=100.)
        self.assertFalse(tj0g.metadata['generator_params']['constraints_applied'])
    def test_spiral_invalid_dt_or_gamma_with_limits(self):
        with self.assertRaisesRegex(ValueError,"dt_seconds must be positive"):generate_spiral_trajectory(1,10,self.fov_m,0,max_gradient_Tm_per_m=.01,max_slew_rate_Tm_per_s=50.)
        with self.assertRaisesRegex(ValueError,"gamma_Hz_per_T must be positive"):generate_spiral_trajectory(1,10,self.fov_m,self.dt_seconds,gamma_Hz_per_T=0,max_gradient_Tm_per_m=.01,max_slew_rate_Tm_per_s=50.)


class TestSGIRF(unittest.TestCase):
    def setUp(self):
        self.dt_sgirf = 4e-6; self.response_len = 64; self.test_sgirf_name = "TestSGIRFProfile"
        self.h_data_arrays = {}; self.base_data = np.arange(self.response_len,dtype=float)/self.response_len
        comps=['xx','xy','xz','yx','yy','yz','zx','zy','zz']
        for i,c in enumerate(comps): self.h_data_arrays[c]=self.base_data+(i*.1)
        self.temp_dir=tempfile.mkdtemp(); self.filepaths={}
        for c,d in self.h_data_arrays.items():
            fp=os.path.join(self.temp_dir,f"sgirf_{c}.npy"); np.save(fp,d); self.filepaths[c]=fp
    def tearDown(self): shutil.rmtree(self.temp_dir)
    def test_sgirf_initialization_valid(self):
        sg=sGIRF(**self.h_data_arrays,dt_sgirf=self.dt_sgirf,name=self.test_sgirf_name)
        self.assertEqual(sg.dt_sgirf,self.dt_sgirf); self.assertEqual(sg.name,self.test_sgirf_name)
        self.assertEqual(sg.h_t_matrix.shape,(3,3)); self.assertEqual(sg._response_len,self.response_len)
        for i,r in enumerate(['x','y','z']):
            for j,c_ax in enumerate(['x','y','z']): np.testing.assert_array_equal(sg.h_t_matrix[i,j],self.h_data_arrays[r+c_ax])
    def test_sgirf_initialization_invalid_dt(self):
        a=[self.h_data_arrays[c] for c in ['xx','xy','xz','yx','yy','yz','zx','zy','zz']]
        with self.assertRaisesRegex(ValueError,"dt_sgirf must be positive"): sGIRF(*a,dt_sgirf=0)
    def test_sgirf_initialization_invalid_h_dim(self):
        v={k:a.copy() for k,a in self.h_data_arrays.items()}; inv=v.copy(); inv['xx']=np.random.rand(self.response_len,2)
        with self.assertRaisesRegex(ValueError,"h_xx_t must be a 1D array"): sGIRF(**inv,dt_sgirf=self.dt_sgirf)
    def test_sgirf_initialization_mismatched_lengths(self):
        v={k:a.copy() for k,a in self.h_data_arrays.items()}; inv=v.copy(); inv['xy']=np.random.rand(self.response_len+1)
        with self.assertRaisesRegex(ValueError,"All h_ij_t arrays must have the same length"): sGIRF(**inv,dt_sgirf=self.dt_sgirf)
    def test_sgirf_initialization_zero_length_responses(self):
        zl={k:np.array([]) for k in self.h_data_arrays.keys()}
        with self.assertRaisesRegex(ValueError,"must not be an empty array"): sGIRF(**zl,dt_sgirf=self.dt_sgirf)
    def test_sgirf_repr_method(self):
        sg=sGIRF(**self.h_data_arrays,dt_sgirf=self.dt_sgirf,name=self.test_sgirf_name)
        er=(f"sGIRF(name='{self.test_sgirf_name}', dt_sgirf={self.dt_sgirf:.2e}, response_len={self.response_len}, shape=(3,3))")
        self.assertEqual(repr(sg),er)
    def test_from_numpy_files_successful_load(self):
        sg=sGIRF.from_numpy_files(self.filepaths,self.dt_sgirf,name="LoadedSGIRF"); self.assertEqual(sg.name,"LoadedSGIRF")
        sgan=sGIRF.from_numpy_files(self.filepaths,self.dt_sgirf); self.assertTrue(sgan.name.startswith("sGIRF_sgirf"))
    def test_from_numpy_files_missing_key_in_dict(self):
        inc_fp=self.filepaths.copy(); del inc_fp['xy']
        with self.assertRaisesRegex(ValueError,"Missing required filepaths"): sGIRF.from_numpy_files(inc_fp,self.dt_sgirf)
    def test_from_numpy_files_file_not_found(self):
        bad_fp=self.filepaths.copy(); bad_fp['xx']=os.path.join(self.temp_dir,"non_existent.npy")
        with self.assertRaisesRegex(FileNotFoundError,"Could not load sGIRF data"): sGIRF.from_numpy_files(bad_fp,self.dt_sgirf)
    def test_from_numpy_files_invalid_npy_content(self):
        fp2d=os.path.join(self.temp_dir,"sgirf_xx_2d.npy"); np.save(fp2d,np.random.rand(self.response_len,2))
        bad_fp=self.filepaths.copy(); bad_fp['xx']=fp2d
        with self.assertRaisesRegex(ValueError,"Data in .*sgirf_xx_2d.npy.*is not 1D"): sGIRF.from_numpy_files(bad_fp,self.dt_sgirf)


class TestGenerateTwSsiPulse(unittest.TestCase):
    def setUp(self):
        self.default_duration_s = 1e-3; self.default_bandwidth_hz = 4000
        self.default_dt_s = 4e-6; self.default_tukey_alpha = 0.3
    def test_generate_pulse_defaults(self):
        p=generate_tw_ssi_pulse(self.default_duration_s,self.default_bandwidth_hz,self.default_dt_s,self.default_tukey_alpha)
        self.assertIsInstance(p,np.ndarray); self.assertEqual(p.ndim,1)
        enp=int(round(self.default_duration_s/self.default_dt_s)); self.assertEqual(p.shape[0],enp)
        if enp>0:self.assertAlmostEqual(np.max(np.abs(p)),1.,places=6)
    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError,"duration_s must be positive"):generate_tw_ssi_pulse(0,self.default_bandwidth_hz)
        with self.assertRaisesRegex(ValueError,"bandwidth_hz must be positive"):generate_tw_ssi_pulse(self.default_duration_s,0)
        with self.assertRaisesRegex(ValueError,"dt_s must be positive"):generate_tw_ssi_pulse(self.default_duration_s,self.default_bandwidth_hz,dt_s=0)
        with self.assertRaisesRegex(ValueError,"Tukey window alpha must be between 0 and 1"):generate_tw_ssi_pulse(self.default_duration_s,self.default_bandwidth_hz,tukey_alpha=1.1)
        with self.assertRaisesRegex(ValueError,"too short for dt_s"):generate_tw_ssi_pulse(1e-6,1000,1e-6)
    def test_tukey_alpha_effects(self):
        pa0=generate_tw_ssi_pulse(self.default_duration_s,self.default_bandwidth_hz,tukey_alpha=0.)
        if pa0.size>0: self.assertTrue(np.abs(pa0[0])>1e-3); self.assertTrue(np.abs(pa0[-1])>1e-3)
        pa1=generate_tw_ssi_pulse(self.default_duration_s,self.default_bandwidth_hz,tukey_alpha=1.)
        if pa1.size>0: self.assertAlmostEqual(pa1[0],0.,places=6); self.assertAlmostEqual(pa1[-1],0.,places=6)
    def test_pulse_shape_properties(self):
        p=generate_tw_ssi_pulse(self.default_duration_s,self.default_bandwidth_hz,self.default_dt_s,self.default_tukey_alpha)
        if p.size>1: np.testing.assert_allclose(p,p[::-1],atol=1e-6)

if __name__ == '__main__':
    unittest.main()
