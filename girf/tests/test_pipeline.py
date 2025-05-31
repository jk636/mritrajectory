import unittest
import numpy as np
import json
import tempfile
import os

# Adjust path for imports
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf.pipeline import girf_trajectory_pipeline, DEFAULT_GAMMA_PROTON
from girf.calibrator import GIRFCalibrator # To create dummy GIRF file
from girf import utils # For trajectory standardization if needed for dummy data

# Helper to generate dummy spiral for pipeline input
def _generate_dummy_spiral_for_pipeline(num_points, k_max, num_revolutions, num_dims=2):
    if num_dims not in [2,3]: raise ValueError("Only 2D or 3D spiral for this helper.")
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_points)
    radius_norm = theta / np.max(theta) if np.max(theta) > 0 else np.zeros_like(theta)
    radius = radius_norm * k_max

    kx = radius * np.cos(theta)
    ky = radius * np.sin(theta)
    if num_dims == 2:
        return {'x': kx, 'y': ky} # Pipeline expects dict or array
    else:
        kz = np.linspace(-k_max/5, k_max/5, num_points)
        return {'x': kx, 'y': ky, 'z': kz}


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.num_k_points = 256 # Reduced from 1024 for faster tests
        self.dt = 4e-6

        self.base_config = {
            'dt': self.dt,
            'gamma': DEFAULT_GAMMA_PROTON,
            'Gmax_T_per_m': 0.080, # Relaxed Gmax for tests
            'Smax_T_per_m_per_s': 250.0, # Relaxed Smax
            'pns_thresholds': {
                'rheobase_T_per_s': 30.0,
                'chronaxie_ms': 0.50,
                'max_total_pns_normalized': 1.5 # Relaxed PNS for basic tests
            },
            'axis_weights_pns': {'x': 0.75, 'y': 0.75, 'z': 1.0},
            'reconstruction_config': {
                'matrix_size': (32, 32), # Small matrix for faster test recon
                'fov': (0.256, 0.256)
            },
            'girf_interpolation_kind': 'linear',
            'verbose': False # Keep tests quiet unless debugging
        }

        # Nominal trajectory (2D Spiral)
        self.nominal_traj_dict = _generate_dummy_spiral_for_pipeline(
            self.num_k_points, k_max=150, num_revolutions=8, num_dims=2
        )

        # Dummy raw k-space data
        self.raw_kspace_samples = np.random.randn(self.num_k_points) + 1j * np.random.randn(self.num_k_points)

        # Create a dummy GIRF file
        self.temp_girf_file_handle, self.temp_girf_file_path = tempfile.mkstemp(suffix='.json')
        fft_freqs = np.fft.fftfreq(self.num_k_points, d=self.dt)
        girf_x_ideal = np.ones(self.num_k_points, dtype=np.complex128)
        girf_y_atten = 0.8 * np.exp(-1j * np.pi/4 * np.sign(fft_freqs)) # Attenuation + phase shift
        girf_y_atten[0] = 0.8

        dummy_girf_content = {
            "gradient_axes": ["x", "y"],
            "girf_spectra_complex": {
                "x": [[val.real, val.imag] for val in girf_x_ideal],
                "y": [[val.real, val.imag] for val in girf_y_atten]
            }, "waveform_params": {}
        }
        with open(self.temp_girf_file_path, 'w') as f:
            json.dump(dummy_girf_content, f)

    def tearDown(self):
        os.close(self.temp_girf_file_handle)
        if os.path.exists(self.temp_girf_file_path): # Ensure it exists before removing
            os.remove(self.temp_girf_file_path)

    def test_01_pipeline_runs_ideal_girf_no_recon(self):
        # Test with ideal GIRF (no girf_data_path), no image reconstruction
        result = girf_trajectory_pipeline(
            nominal_trajectory_kspace=self.nominal_traj_dict,
            config=self.base_config,
            image_raw_data=None, # No recon
            girf_data_path=None  # Ideal GIRF
        )
        self.assertIn('final_status', result)
        # Expect success or success with warnings (due to relaxed constraints it might pass all)
        self.assertTrue(result['final_status'].startswith('SUCCESS'))
        self.assertIn('nominal_data', result)
        self.assertIn('predicted_data', result)
        # With ideal GIRF, nominal and predicted k-space should be identical
        np.testing.assert_array_almost_equal(
            result['nominal_data']['kspace_array'],
            result['predicted_data']['kspace_array']
        )
        self.assertEqual(result['reconstruction']['status'], 'Skipped - No raw image data')
        self.assertTrue(result['pns_analysis']['status'] == 'Performed' or result['pns_analysis']['status'] == 'Skipped - PNSModel not available')


    def test_02_pipeline_runs_with_girf_and_recon(self):
        result = girf_trajectory_pipeline(
            nominal_trajectory_kspace=self.nominal_traj_dict,
            config=self.base_config,
            image_raw_data=self.raw_kspace_samples,
            girf_data_path=self.temp_girf_file_path
        )
        self.assertTrue(result['final_status'].startswith('SUCCESS')) # or SUCCESS_WITH_..._WARNINGS
        self.assertIn('kspace_array', result['predicted_data'])
        # Predicted should differ from nominal for y-axis due to GIRF
        diff_sum_y = np.sum(np.abs(result['nominal_data']['kspace_array'][:,1] - result['predicted_data']['kspace_array'][:,1]))
        self.assertTrue(diff_sum_y > 1e-3) # Check they are meaningfully different for y

        self.assertEqual(result['reconstruction']['status'], 'Performed')
        self.assertIn('reconstructed_image_shape', result['reconstruction'])
        self.assertEqual(tuple(result['reconstruction']['reconstructed_image_shape']),
                         tuple(self.base_config['reconstruction_config']['matrix_size']))
        self.assertIn('image_quality_metrics', result['reconstruction'])


    def test_03_pipeline_nominal_hw_violation(self):
        config_tight_hw = self.base_config.copy()
        config_tight_hw['Gmax_T_per_m'] = 0.001 # Very tight Gmax, likely to fail

        result = girf_trajectory_pipeline(
            nominal_trajectory_kspace=self.nominal_traj_dict,
            config=config_tight_hw,
            girf_data_path=None
        )
        self.assertFalse(result['nominal_data']['hw_constraints']['G_ok'])
        # Status might be SUCCESS_WITH_NOMINAL_HW_WARNINGS or specific fail if configured
        self.assertIn("NOMINAL_HW_WARNINGS", result['final_status']) # Or similar based on logic


    def test_04_pipeline_pns_violation(self):
        config_tight_pns = self.base_config.copy()
        config_tight_pns['pns_thresholds'] = {
            'rheobase_T_per_s': 5.0, # Very sensitive PNS model
            'chronaxie_ms': 0.20,
            'max_total_pns_normalized': 0.1 # Very low limit
        }
        result = girf_trajectory_pipeline(
            nominal_trajectory_kspace=self.nominal_traj_dict, # A reasonably dynamic trajectory
            config=config_tight_pns,
            girf_data_path=None # Ideal GIRF for simplicity here
        )
        if result['pns_analysis']['status'] == 'Performed': # Check if PNS ran
            self.assertFalse(result['pns_analysis']['compliant'])
            self.assertIn("PNS_WARNINGS", result['final_status'])
        else:
            self.skipTest("PNSModel might not be available or test setup needs review if PNS not performed.")


    def test_05_pipeline_girf_load_failure(self):
        # Create an invalid GIRF file path
        invalid_girf_path = "non_existent_girf.json"
        result = girf_trajectory_pipeline(
            nominal_trajectory_kspace=self.nominal_traj_dict,
            config=self.base_config,
            girf_data_path=invalid_girf_path
        )
        # Should proceed with ideal GIRF, log error for loading
        self.assertTrue(any("Failed to load GIRF" in msg for msg in result['status_messages'] if "[ERROR]" in msg))
        np.testing.assert_array_almost_equal(
            result['nominal_data']['kspace_array'],
            result['predicted_data']['kspace_array'] # Predicted = Nominal
        )
        self.assertTrue(result['final_status'].startswith('SUCCESS')) # Still success, but with error logged


    def test_06_pipeline_missing_critical_config(self):
        config_no_dt = self.base_config.copy()
        del config_no_dt['dt']
        result = girf_trajectory_pipeline(self.nominal_traj_dict, config_no_dt)
        self.assertTrue(result['final_status'].startswith('FAILED_CONFIGURATION_OR_DATA_ERROR'))
        self.assertTrue(any("'dt' (time step) must be provided" in msg for msg in result['status_messages'] if "[ERROR]" in msg))

        config_no_gmax = self.base_config.copy()
        del config_no_gmax['Gmax_T_per_m']
        result = girf_trajectory_pipeline(self.nominal_traj_dict, config_no_gmax)
        self.assertTrue(result['final_status'].startswith('FAILED_CONFIGURATION_OR_DATA_ERROR'))
        self.assertTrue(any("'Gmax_T_per_m' must be provided" in msg for msg in result['status_messages'] if "[ERROR]" in msg))

    def test_07_pipeline_input_trajectory_array(self):
        # Test with nominal trajectory as array instead of dict
        nominal_traj_arr = utils.standardize_trajectory_format(self.nominal_traj_dict, target_format='array')
        result = girf_trajectory_pipeline(
            nominal_trajectory_kspace=nominal_traj_arr,
            config=self.base_config,
            girf_data_path=None
        )
        self.assertTrue(result['final_status'].startswith('SUCCESS'))
        self.assertEqual(result['nominal_data']['kspace_array'].shape, nominal_traj_arr.shape)


if __name__ == '__main__':
    unittest.main()
