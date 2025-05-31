import unittest
import numpy as np
import tempfile
import os

# Adjust path for imports
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf.corrector import TrajectoryCorrector
# Attempt to import nibabel for NIFTI, but make it optional for tests not strictly needing it
try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class TestTrajectoryCorrector(unittest.TestCase):

    def setUp(self):
        self.default_config = {'matrix_size': (64, 64), 'fov': (0.2, 0.2)} # 20cm FOV
        self.corrector = TrajectoryCorrector(reconstruction_config=self.default_config)

        # Dummy data for reconstruction
        self.num_samples = 100
        # Simple trajectory: line along kx, from -kmax/2 to kmax/2
        # k_max for grid = matrix_size / (2 * FOV_dim)
        # For matrix_size=64, fov=0.2, k_max_grid_dim = 64 / (2*0.2) = 160 m^-1
        # Let our trajectory span part of this.
        k_max_traj = 100 # m^-1
        self.dummy_kx = np.linspace(-k_max_traj/2, k_max_traj/2, self.num_samples)
        self.dummy_ky = np.zeros(self.num_samples)
        self.dummy_predicted_trajectory = np.stack([self.dummy_kx, self.dummy_ky], axis=-1)

        # Raw k-space data: constant signal (simulates point at image center)
        self.dummy_raw_kspace_samples = np.ones(self.num_samples, dtype=np.complex128) * 10.0
        self.dummy_raw_kspace_samples += (np.random.randn(self.num_samples) + 1j*np.random.randn(self.num_samples)) * 0.1 # Noise


    def test_01_initialization(self):
        self.assertEqual(self.corrector.reconstruction_config['matrix_size'], (64,64))
        custom_config = {'matrix_size': (32,32), 'fov': (0.1, 0.1)}
        custom_corrector = TrajectoryCorrector(reconstruction_config=custom_config)
        self.assertEqual(custom_corrector.reconstruction_config['matrix_size'], (32,32))
        self.assertEqual(custom_corrector.reconstruction_config['fov'], (0.1,0.1))
        self.assertIn('kernel_name', custom_corrector.reconstruction_config) # Check default is merged


    def test_02_prepare_trajectory_for_gridding(self):
        # Test dict input
        traj_dict = {'x': self.dummy_kx, 'y': self.dummy_ky}
        prepared_from_dict = self.corrector._prepare_trajectory_for_gridding(traj_dict)
        np.testing.assert_array_equal(prepared_from_dict, self.dummy_predicted_trajectory)

        # Test list input
        traj_list = [[self.dummy_kx[i], self.dummy_ky[i]] for i in range(self.num_samples)]
        prepared_from_list = self.corrector._prepare_trajectory_for_gridding(traj_list)
        np.testing.assert_array_equal(prepared_from_list, self.dummy_predicted_trajectory)

        # Test ndarray input (already correct format)
        prepared_from_array = self.corrector._prepare_trajectory_for_gridding(self.dummy_predicted_trajectory)
        np.testing.assert_array_equal(prepared_from_array, self.dummy_predicted_trajectory)

        # Single axis trajectory (list of numbers) -> (N,1) array
        single_axis_list = [1.0, 2.0, 3.0]
        prepared_single = self.corrector._prepare_trajectory_for_gridding(single_axis_list)
        self.assertEqual(prepared_single.shape, (3,1))

        with self.assertRaises(ValueError): # Inconsistent lengths in dict
            self.corrector._prepare_trajectory_for_gridding({'x': np.array([1,2]), 'y': np.array([1,2,3])})
        with self.assertRaises(TypeError): # Unsupported type
            self.corrector._prepare_trajectory_for_gridding(None)


    def test_03_reconstruct_image_simple_case(self):
        # Test with a single k-space point at (0,0) which should be center of k-space grid
        # This point should receive the sum of all k-space sample values if all map to it.
        matrix_size = (4, 4) # Very small grid
        fov = (0.1, 0.1) # meters
        recon_config_simple = {'matrix_size': matrix_size, 'fov': fov}
        corrector_simple = TrajectoryCorrector(reconstruction_config=recon_config_simple)

        # Trajectory: all points map to kx=0, ky=0
        # Physical kx=0, ky=0 maps to grid center: (matrix_size/2, matrix_size/2)
        # Grid indices: k_val * fov_val + matrix_size_val / 2.0
        # kx=0 * 0.1 + 4/2 = 2. ky=0 * 0.1 + 4/2 = 2. So center is (2,2)
        simple_traj_k_coords = np.zeros((5, 2)) # 5 samples, all at k=(0,0)
        simple_raw_samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.complex128) # Sum = 15

        reconstructed_img = corrector_simple.reconstruct_image(simple_traj_k_coords, simple_raw_samples)

        self.assertEqual(reconstructed_img.shape, matrix_size)

        # Check the gridded k-space (before IFFT)
        # Center of k-space grid (DC component)
        center_idx_x = matrix_size[0] // 2
        center_idx_y = matrix_size[1] // 2

        # With nearest neighbor and all points at (0,0), the center grid cell should get sum of samples
        # (assuming density compensation of 1/N_hits is applied)
        # The current density comp is value/N_hits. So 15.0/5 = 3.0
        self.assertAlmostEqual(corrector_simple.gridded_kspace[center_idx_x, center_idx_y], 3.0)

        # After IFFT of a DC component, image should be flat (constant value)
        # The value is DC_component / N_pixels (for np.fft.ifft2)
        expected_img_val = corrector_simple.gridded_kspace[center_idx_x, center_idx_y] / (matrix_size[0]*matrix_size[1])

        # Since image_data is complex, check magnitude if needed, or check all points
        # For a pure DC k-space, the image should be constant.
        # np.testing.assert_array_almost_equal(np.abs(reconstructed_img), np.full(matrix_size, np.abs(expected_img_val)))
        # More simply, check that the sum of image pixels relates to DC component
        self.assertAlmostEqual(np.sum(reconstructed_img), corrector_simple.gridded_kspace[center_idx_x, center_idx_y])


    def test_04_reconstruct_image_line_trajectory(self):
        # Uses data from setUp: line along kx
        # This is more of an integration test that it runs
        img = self.corrector.reconstruct_image(self.dummy_predicted_trajectory, self.dummy_raw_kspace_samples)
        self.assertEqual(img.shape, self.default_config['matrix_size'])

        # For a line of constant k-space samples, expect a sinc-like response perpendicular to the line in image domain
        # Peak intensity should be near the center of the image.
        center_pixel_value = np.abs(img[img.shape[0]//2, img.shape[1]//2])
        self.assertTrue(center_pixel_value > 0) # Basic check

        # Test with dict trajectory input
        img_dict_traj = self.corrector.reconstruct_image(
            {'x': self.dummy_kx, 'y': self.dummy_ky}, self.dummy_raw_kspace_samples
        )
        self.assertEqual(img_dict_traj.shape, self.default_config['matrix_size'])


    def test_05_reconstruct_image_error_handling(self):
        # Mismatch between trajectory points and k-space samples
        with self.assertRaises(ValueError):
            self.corrector.reconstruct_image(self.dummy_predicted_trajectory, self.dummy_raw_kspace_samples[:-1])

        # Unsupported trajectory dimensions (e.g., 1D or 4D)
        with self.assertRaises(ValueError):
            self.corrector.reconstruct_image(np.zeros((10,1)), self.dummy_raw_kspace_samples[:10]) # 1D traj
        with self.assertRaises(ValueError):
             self.corrector.reconstruct_image(np.zeros((10,4)), self.dummy_raw_kspace_samples[:10]) # 4D traj

        # Config mismatch
        bad_config = {'matrix_size': (64,64,64), 'fov': (0.2,0.2)} # 3D matrix, 2D FOV for 2D traj
        corr_bad_cfg = TrajectoryCorrector(reconstruction_config=bad_config)
        with self.assertRaises(ValueError): # matrix_size (3D) and fov (2D) inconsistent for 2D trajectory
            corr_bad_cfg.reconstruct_image(self.dummy_predicted_trajectory, self.dummy_raw_kspace_samples)


    def test_06_evaluate_image_quality_placeholder(self):
        # Test that it returns expected structure when image is not reconstructed
        metrics_no_img = self.corrector.evaluate_image_quality()
        self.assertIn('error', metrics_no_img)

        # Reconstruct a dummy image first
        self.corrector.reconstruct_image(self.dummy_predicted_trajectory, self.dummy_raw_kspace_samples)
        metrics_with_img = self.corrector.evaluate_image_quality()
        self.assertNotIn('error', metrics_with_img)
        self.assertIn('mean_intensity', metrics_with_img)
        self.assertIn('snr_placeholder', metrics_with_img)
        self.assertTrue(metrics_with_img['mean_intensity'] >= 0)


    def test_07_save_image(self):
        self.corrector.reconstruct_image(self.dummy_predicted_trajectory, self.dummy_raw_kspace_samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test NPY saving
            npy_file_path = os.path.join(tmpdir, "test_image.npy")
            self.corrector.save_image(npy_file_path, file_format='NPY')
            self.assertTrue(os.path.exists(npy_file_path))
            loaded_npy_img = np.load(npy_file_path)
            np.testing.assert_array_almost_equal(np.abs(self.corrector.image_data), loaded_npy_img)

            # Test NIFTI saving (if nibabel is installed)
            if nib:
                nifti_file_path = os.path.join(tmpdir, "test_image.nii.gz")
                self.corrector.save_image(nifti_file_path, file_format='NIFTI')
                self.assertTrue(os.path.exists(nifti_file_path))
                loaded_nifti_img = nib.load(nifti_file_path).get_fdata()
                # NIFTI save might alter shape (e.g., add z-dim for 2D) or data type
                np.testing.assert_array_almost_equal(np.abs(self.corrector.image_data), loaded_nifti_img.squeeze(), decimal=5)
            else:
                print("Skipping NIFTI save test as nibabel is not installed.")

            # Test PNG saving (if matplotlib is installed)
            if plt:
                png_file_path = os.path.join(tmpdir, "test_image.png")
                self.corrector.save_image(png_file_path, file_format='PNG')
                self.assertTrue(os.path.exists(png_file_path))
                # Further checks could involve reading the PNG and comparing pixel data,
                # but that's more complex and depends on image reading libraries.
            else:
                print("Skipping PNG save test as matplotlib is not installed.")

            # Test unsupported format
            unsupported_file_path = os.path.join(tmpdir, "test_image.xyz")
            # This should print an error but not raise an exception based on current implementation
            self.corrector.save_image(unsupported_file_path, file_format='XYZ')


    def test_08_save_image_no_data(self):
        # Test saving when no image data is available
        empty_corrector = TrajectoryCorrector()
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_file_path = os.path.join(tmpdir, "no_image.npy")
            # This should print an error but not raise an exception
            empty_corrector.save_image(npy_file_path, file_format='NPY')
            self.assertFalse(os.path.exists(npy_file_path)) # File should not be created


if __name__ == '__main__':
    unittest.main()
