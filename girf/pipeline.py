import numpy as np
import json
import os # For __main__ example file handling

# Import GIRF package components
from . import utils
from .calibrator import GIRFCalibrator # Needed for __main__ to create dummy GIRF file
from .predictor import TrajectoryPredictor
from .pns import PNSModel
from .corrector import TrajectoryCorrector

# Default Gyromagnetic ratio for protons in Hz/T
DEFAULT_GAMMA_PROTON = 42.576e6

def girf_trajectory_pipeline(nominal_trajectory_kspace,
                             config,
                             image_raw_data=None,
                             girf_data_path=None):
    """
    Main GIRF trajectory processing and image reconstruction pipeline.

    Args:
        nominal_trajectory_kspace (dict or np.ndarray): The initial, ideal k-space trajectory (units: m^-1).
        config (dict): Configuration parameters for the pipeline. Expected keys:
            - 'dt' (float): Time step (seconds).
            - 'gamma' (float, optional): Gyromagnetic ratio (Hz/T). Defaults to proton gamma.
            - 'Gmax_T_per_m' (float): Maximum gradient strength (T/m).
            - 'Smax_T_per_m_per_s' (float): Maximum slew rate (T/m/s).
            - 'pns_thresholds' (dict, optional): For PNSModel (e.g., rheobase, chronaxie, max_total_pns_normalized).
            - 'axis_weights_pns' (dict, optional): For PNSModel axis weighting.
            - 'reconstruction_config' (dict, optional): For TrajectoryCorrector (e.g., matrix_size, fov).
            - 'girf_interpolation_kind' (str, optional): Interpolation kind for GIRF resampling (e.g., 'linear').
            - 'verbose' (bool, optional): If True, print more messages. Defaults to False.
        image_raw_data (np.ndarray, optional): Raw k-space data for image reconstruction (complex). (num_samples,).
        girf_data_path (str, optional): Path to a file containing GIRF spectra (e.g., JSON).
                                        If None, ideal GIRF (no effect) is assumed.

    Returns:
        dict: Results including trajectories, gradients, slew rates, status messages, PNS info,
              and image data/quality if reconstructed.
    """

    result = {
        'status_messages': [],
        'final_status': 'FAILED_PIPELINE_INITIALIZATION', # Will be updated
        'nominal_data': {},
        'predicted_data': {},
        'pns_analysis': {'compliant': False, 'status': 'Not Performed'},
        'reconstruction': {'status': 'Not Performed'},
        'config_used': config, # Store config for reference
        'inputs_summary': {
            'nominal_trajectory_provided': nominal_trajectory_kspace is not None,
            'image_raw_data_provided': image_raw_data is not None,
            'girf_data_path_provided': girf_data_path is not None
        }
    }

    verbose = config.get('verbose', False)
    def log_message(msg, level="INFO"):
        if verbose or level == "ERROR" or level == "WARNING": print(f"Pipeline [{level}]: {msg}")
        result['status_messages'].append(f"[{level}] {msg}")

    log_message("Pipeline started.")

    try:
        # --- 0. Configuration and Parameter Extraction ---
        dt = config.get('dt')
        if dt is None: raise ValueError("'dt' (time step) must be provided in config.")
        gamma = config.get('gamma', DEFAULT_GAMMA_PROTON)
        gmax = config.get('Gmax_T_per_m')
        if gmax is None: raise ValueError("'Gmax_T_per_m' must be provided in config.")
        smax = config.get('Smax_T_per_m_per_s')
        if smax is None: raise ValueError("'Smax_T_per_m_per_s' must be provided in config.")

        pns_threshold_config = config.get('pns_thresholds') # PNSModel has defaults if None
        pns_axis_weights = config.get('axis_weights_pns')   # PNSModel uses 1.0 if None
        recon_config = config.get('reconstruction_config')  # Corrector has defaults if None
        girf_interp_kind = config.get('girf_interpolation_kind', 'linear')

        log_message(f"Extracted parameters: dt={dt*1e6:.1f}us, Gmax={gmax*1e3:.1f}mT/m, Smax={smax:.0f}T/m/s.")

        # --- 1. Standardize Nominal Trajectory ---
        if nominal_trajectory_kspace is None:
            raise ValueError("Nominal k-space trajectory must be provided.")

        # Determine number of spatial dimensions from nominal trajectory before standardizing fully to array
        # This helps standardize_trajectory_format pick better default axis names if needed for dict output later
        num_dims_nominal = 0
        if isinstance(nominal_trajectory_kspace, dict):
            num_dims_nominal = len(nominal_trajectory_kspace)
            # Store original axis names if input is dict, for consistent output later
            original_nom_traj_axis_names = sorted(nominal_trajectory_kspace.keys())
        elif isinstance(nominal_trajectory_kspace, (np.ndarray, list)):
            temp_arr = np.asarray(nominal_trajectory_kspace)
            num_dims_nominal = temp_arr.shape[1] if temp_arr.ndim == 2 else (1 if temp_arr.ndim == 1 else 0)
            original_nom_traj_axis_names = [f'axis_{i}' for i in range(num_dims_nominal)]


        nominal_k_array = utils.standardize_trajectory_format(nominal_trajectory_kspace,
                                                              num_spatial_dims=num_dims_nominal,
                                                              target_format='array')
        result['nominal_data']['kspace_array'] = nominal_k_array
        log_message(f"Standardized nominal k-space trajectory to array shape: {nominal_k_array.shape}")

        # --- 2. Nominal Gradient & Slew Rate Calculation ---
        nominal_g_array = utils.compute_gradient_waveforms(nominal_k_array, gamma, dt, output_format='array')
        nominal_sr_array = utils.compute_slew_rates(nominal_g_array, dt, output_format='array')
        result['nominal_data']['gradients_array_T_per_m'] = nominal_g_array
        result['nominal_data']['slew_rates_array_T_per_m_per_s'] = nominal_sr_array
        log_message("Computed nominal gradients and slew rates.")

        # --- 3. Hardware Constraint Check (Nominal) ---
        nom_g_ok, nom_max_g, nom_g_details = utils.check_gradient_strength(nominal_g_array, gmax)
        nom_sr_ok, nom_max_sr, nom_sr_details = utils.check_slew_rate(nominal_sr_array, smax)
        result['nominal_data']['hw_constraints'] = {
            'G_ok': nom_g_ok, 'max_G_found_T_per_m': nom_max_g, 'G_details': nom_g_details,
            'SR_ok': nom_sr_ok, 'max_SR_found_T_per_m_per_s': nom_max_sr, 'SR_details': nom_sr_details
        }
        if not nom_g_ok: log_message(f"Nominal gradient strength exceeds Gmax. Details: {nom_g_details}", "WARNING")
        if not nom_sr_ok: log_message(f"Nominal slew rate exceeds Smax. Details: {nom_sr_details}", "WARNING")
        # Option to fail early: if config.get('fail_on_nominal_hw_violation', False) and not (nom_g_ok and nom_sr_ok):
        #    result['final_status'] = 'FAILED_NOMINAL_HW_VIOLATION'; return result

        # --- 4. GIRF Loading & Trajectory Prediction ---
        predictor = TrajectoryPredictor(dt=dt, gamma=gamma)
        girf_loaded_successfully = False
        if girf_data_path:
            try:
                predictor.load_girf(girf_data_path)
                log_message(f"GIRF loaded successfully from {girf_data_path}.")
                girf_loaded_successfully = True
                result['inputs_summary']['girf_spectra_keys'] = list(predictor.girf_spectra.keys())
            except Exception as e:
                log_message(f"Failed to load GIRF from {girf_data_path}: {e}. Proceeding with ideal GIRF (no effect).", "ERROR")
                result['inputs_summary']['girf_load_error'] = str(e)
        else:
            log_message("No GIRF data path provided. Assuming ideal GIRF (predicted trajectory = nominal).", "INFO")

        if girf_loaded_successfully and predictor.girf_spectra:
            # Pass nominal_k_array (standardized) to predictor
            # The predictor needs to know the axis names for its internal GIRF application.
            # We use the original axis names if the input was a dict, or generic ones.
            # Predictor's predict_trajectory can now take dict or array.
            # To ensure correct axis mapping, it's safer to pass dict if original was dict.
            if isinstance(nominal_trajectory_kspace, dict):
                # Use original dict with standardized array values
                nominal_k_for_predictor = utils.standardize_trajectory_format(nominal_k_array,
                                                                              target_format='dict',
                                                                              default_axis_names=original_nom_traj_axis_names)
            else: # input was array-like
                nominal_k_for_predictor = nominal_k_array

            predicted_k_array = predictor.predict_trajectory(nominal_k_for_predictor,
                                                             girf_resample_kind=girf_interp_kind)
            log_message("Trajectory predicted using loaded GIRF.")
        else:
            predicted_k_array = nominal_k_array.copy() # Or deepcopy if modification is done later
            log_message("Using nominal trajectory as predicted trajectory (ideal or no GIRF).")

        result['predicted_data']['kspace_array'] = predicted_k_array

        predicted_g_array = utils.compute_gradient_waveforms(predicted_k_array, gamma, dt, output_format='array')
        predicted_sr_array = utils.compute_slew_rates(predicted_g_array, dt, output_format='array')
        result['predicted_data']['gradients_array_T_per_m'] = predicted_g_array
        result['predicted_data']['slew_rates_array_T_per_m_per_s'] = predicted_sr_array
        log_message("Computed predicted gradients and slew rates.")

        # --- 5. Hardware Constraint Check (Predicted) ---
        pred_g_ok, pred_max_g, pred_g_details = utils.check_gradient_strength(predicted_g_array, gmax)
        pred_sr_ok, pred_max_sr, pred_sr_details = utils.check_slew_rate(predicted_sr_array, smax)
        result['predicted_data']['hw_constraints'] = {
            'G_ok': pred_g_ok, 'max_G_found_T_per_m': pred_max_g, 'G_details': pred_g_details,
            'SR_ok': pred_sr_ok, 'max_SR_found_T_per_m_per_s': pred_max_sr, 'SR_details': pred_sr_details
        }
        if not pred_g_ok: log_message(f"Predicted gradient strength exceeds Gmax. Details: {pred_g_details}", "WARNING")
        if not pred_sr_ok: log_message(f"Predicted slew rate exceeds Smax. Details: {pred_sr_details}", "WARNING")

        # --- 6. PNS Check (using Predicted Slew Rates) ---
        if PNSModel is not None : # Check if PNSModel class was successfully imported
            pns_model = PNSModel(pns_thresholds=pns_threshold_config, dt=dt)

            # PNSModel expects slew rates as dict {'axis': waveform} or array (T, N_axes)
            # Convert predicted_sr_array to dict if pns_axis_weights are named
            if pns_axis_weights and isinstance(pns_axis_weights, dict):
                 # Ensure predicted_sr_array is mapped to dict with correct axis names for PNSModel
                 sr_for_pns = utils.standardize_trajectory_format(predicted_sr_array,
                                                                  target_format='dict',
                                                                  default_axis_names=original_nom_traj_axis_names)
            else: # PNSModel can take array and use generic axis names or match with weights if weights keys are integers/generic
                 sr_for_pns = predicted_sr_array

            pns_total_ts = pns_model.compute_pns(sr_for_pns, axis_weights=pns_axis_weights)
            pns_ok, peak_pns = pns_model.check_limits(pns_total_ts)

            result['pns_analysis'] = {
                'compliant': pns_ok,
                'peak_total_normalized_pns': peak_pns,
                'pns_total_normalized_timeseries': pns_total_ts, # Can be large, consider summarizing
                'status': 'Performed',
                'pns_model_thresholds_used': pns_model.pns_thresholds # Store what model used
            }
            log_message(f"PNS check performed. Compliant: {pns_ok}, Peak PNS: {peak_pns:.3f}")
            if not pns_ok: log_message("PNS limits exceeded for predicted trajectory.", "WARNING")
        else:
            log_message("PNSModel not available. Skipping PNS check.", "WARNING")
            result['pns_analysis']['status'] = 'Skipped - PNSModel not available'


        # --- 7. Image Reconstruction ---
        if image_raw_data is not None:
            if TrajectoryCorrector is not None:
                log_message("Starting image reconstruction.")
                corrector = TrajectoryCorrector(reconstruction_config=recon_config)
                try:
                    # TrajectoryCorrector expects k-space trajectory (coords) and raw samples
                    # Pass predicted_k_array (coords)
                    reconstructed_image = corrector.reconstruct_image(predicted_k_array, image_raw_data)
                    image_quality = corrector.evaluate_image_quality() # Placeholder metrics for now
                    result['reconstruction'] = {
                        'status': 'Performed',
                        'reconstructed_image_shape': reconstructed_image.shape, # Store shape, not full image in result dict for now
                        # 'reconstructed_image_array': reconstructed_image, # Optional: if image is small or path is not preferred
                        'image_quality_metrics': image_quality
                    }
                    log_message(f"Image reconstruction complete. Image shape: {reconstructed_image.shape}")

                    # Optional: save image from pipeline, or let user do it with returned image
                    # save_path = config.get('output_image_path', None)
                    # if save_path: corrector.save_image(save_path, format=config.get('output_image_format','NPY'))

                except Exception as e:
                    log_message(f"Image reconstruction failed: {e}", "ERROR")
                    result['reconstruction']['status'] = f'Failed: {e}'
            else:
                log_message("TrajectoryCorrector not available. Skipping image reconstruction.", "WARNING")
                result['reconstruction']['status'] = 'Skipped - TrajectoryCorrector not available'
        else:
            log_message("No raw image data provided. Skipping image reconstruction.")
            result['reconstruction']['status'] = 'Skipped - No raw image data'

        # --- 8. Finalize Status ---
        # Define success based on critical steps. For now, if it runs to end without Python exceptions.
        # Specific failure modes (like HW violation or PNS non-compliance) are warnings here,
        # but could be made pipeline failures based on config.
        result['final_status'] = 'SUCCESS'
        if not nom_g_ok or not nom_sr_ok : result['final_status'] = 'SUCCESS_WITH_NOMINAL_HW_WARNINGS'
        if not pred_g_ok or not pred_sr_ok : result['final_status'] = 'SUCCESS_WITH_PREDICTED_HW_WARNINGS'
        if not result['pns_analysis'].get('compliant', True) : result['final_status'] = 'SUCCESS_WITH_PNS_WARNINGS'
        if result['reconstruction']['status'].startswith('Failed') : result['final_status'] = 'FAILED_RECONSTRUCTION'

        log_message(f"Pipeline finished with status: {result['final_status']}", "INFO")

    except ValueError as ve: # Config errors or critical data errors
        log_message(f"Pipeline halted due to ValueError: {ve}", "ERROR")
        result['final_status'] = f"FAILED_CONFIGURATION_OR_DATA_ERROR: {ve}"
    except Exception as e:
        log_message(f"Pipeline halted due to unexpected critical error: {e}", "ERROR")
        result['final_status'] = f"FAILED_UNEXPECTED_ERROR: {e}"
        import traceback
        result['error_traceback'] = traceback.format_exc() # For debugging

    return result


# --- Example Usage ---
def _generate_dummy_spiral_kspace(num_points, k_max, num_revolutions, num_dims=2):
    """Generates a simple 2D or 3D Archimedean spiral k-space trajectory (m^-1)."""
    if num_dims not in [2,3]: raise ValueError("Only 2D or 3D spiral supported for dummy data.")
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_points)
    radius_norm = theta / np.max(theta) if np.max(theta) > 0 else np.zeros_like(theta)
    radius = radius_norm * k_max

    kx = radius * np.cos(theta)
    ky = radius * np.sin(theta)
    if num_dims == 2:
        return {'x': kx, 'y': ky}
    else: # num_dims == 3
        # Simple z modulation for 3D spiral (e.g., conical or variable pitch)
        kz = np.linspace(-k_max/5, k_max/5, num_points) # Example: slow variation on z
        return {'x': kx, 'y': ky, 'z': kz}

if __name__ == '__main__':
    print("--- Running girf.pipeline.py Main Pipeline Example ---")

    # --- 1. Setup: Create dummy data and config ---

    # Pipeline Configuration
    config_pipeline = {
        'dt': 4e-6,  # seconds
        'gamma': DEFAULT_GAMMA_PROTON,
        'Gmax_T_per_m': 0.040,
        'Smax_T_per_m_per_s': 180.0,
        'pns_thresholds': {
            'rheobase_T_per_s': 20.0,
            'chronaxie_ms': 0.36,
            'max_total_pns_normalized': 0.8
        },
        'axis_weights_pns': {'x': 0.75, 'y': 0.75, 'z': 1.0}, # Example weights
        'reconstruction_config': {
            'matrix_size': (64, 64),
            'fov': (0.256, 0.256) # meters (256mm FOV)
        },
        'girf_interpolation_kind': 'linear',
        'verbose': True
    }

    # Generate Nominal Trajectory (2D Spiral for this example)
    num_k_points = 1024
    nominal_traj_dict = _generate_dummy_spiral_kspace(num_k_points, k_max=250, num_revolutions=16, num_dims=2) # k_max in m^-1
    # nominal_traj_array = utils.standardize_trajectory_format(nominal_traj_dict, target_format='array')


    # Generate Dummy GIRF data and save to file
    dummy_girf_file = "temp_dummy_girf_pipeline.json"
    try:
        if GIRFCalibrator is not None: # Check if class is available
            cal = GIRFCalibrator(gradient_axes=['x', 'y']) # Match axes in nominal_traj_dict
            # Create some simple GIRF spectra (e.g., low-pass filter effect)
            # FFT frequencies for a trajectory of num_k_points with dt
            fft_freqs = np.fft.fftfreq(num_k_points, d=config_pipeline['dt'])
            girf_x_example = 1.0 / (1 + 1j * fft_freqs / 15e3) # 15kHz LPF
            girf_y_example = 1.0 / (1 + 1j * fft_freqs / 12e3) # 12kHz LPF

            # Calibrator expects input_waveforms and measured_responses to compute.
            # For dummy file, we can manually set girf_spectra if needed, or just save empty.
            # Or, more simply, create a JSON that mimics the save_calibration output.
            dummy_girf_content = {
                "gradient_axes": ["x", "y"],
                "girf_spectra_complex": {
                    "x": [[val.real, val.imag] for val in girf_x_example],
                    "y": [[val.real, val.imag] for val in girf_y_example]
                },
                "waveform_params": {}
            }
            with open(dummy_girf_file, 'w') as f:
                json.dump(dummy_girf_content, f, indent=4)
            print(f"Saved dummy GIRF data to {dummy_girf_file}")
            girf_path_for_pipeline = dummy_girf_file
        else:
            print("GIRFCalibrator not available, cannot create dummy GIRF file. Pipeline will run with ideal GIRF.")
            girf_path_for_pipeline = None
    except Exception as e_girf_setup:
        print(f"Error creating dummy GIRF file: {e_girf_setup}. Pipeline may run with ideal GIRF.")
        girf_path_for_pipeline = None


    # Generate Dummy Raw K-space Data for Image Reconstruction
    # For a 2D trajectory, raw data is typically 1D array of complex numbers
    # Simulate a simple signal (e.g., from a point source -> constant k-space signal + noise)
    dummy_raw_kspace_samples = np.ones(num_k_points, dtype=np.complex128) * 10
    dummy_raw_kspace_samples += (np.random.randn(num_k_points) + 1j * np.random.randn(num_k_points)) * 1.0


    # --- 2. Run Pipeline ---
    pipeline_output = girf_trajectory_pipeline(nominal_trajectory_kspace=nominal_traj_dict,
                                               config=config_pipeline,
                                               image_raw_data=dummy_raw_kspace_samples,
                                               girf_data_path=girf_path_for_pipeline)

    # --- 3. Print Summary of Results ---
    print("\n--- Pipeline Execution Finished. Results Summary: ---")
    print(f"Final Pipeline Status: {pipeline_output['final_status']}")

    print("\nKey Outputs:")
    print(f"  Nominal K-space Array Shape: {pipeline_output['nominal_data'].get('kspace_array', {}).shape if pipeline_output['nominal_data'].get('kspace_array') is not None else 'N/A'}")
    print(f"  Predicted K-space Array Shape: {pipeline_output['predicted_data'].get('kspace_array', {}).shape if pipeline_output['predicted_data'].get('kspace_array') is not None else 'N/A'}")

    print("\nNominal HW Constraints:")
    if 'hw_constraints' in pipeline_output['nominal_data']:
        print(f"  Gradients OK: {pipeline_output['nominal_data']['hw_constraints']['G_ok']} (Max: {pipeline_output['nominal_data']['hw_constraints']['max_G_found_T_per_m']:.3f} T/m)")
        print(f"  Slew Rates OK: {pipeline_output['nominal_data']['hw_constraints']['SR_ok']} (Max: {pipeline_output['nominal_data']['hw_constraints']['max_SR_found_T_per_m_per_s']:.1f} T/m/s)")

    print("\nPredicted HW Constraints:")
    if 'hw_constraints' in pipeline_output['predicted_data']:
        print(f"  Gradients OK: {pipeline_output['predicted_data']['hw_constraints']['G_ok']} (Max: {pipeline_output['predicted_data']['hw_constraints']['max_G_found_T_per_m']:.3f} T/m)")
        print(f"  Slew Rates OK: {pipeline_output['predicted_data']['hw_constraints']['SR_ok']} (Max: {pipeline_output['predicted_data']['hw_constraints']['max_SR_found_T_per_m_per_s']:.1f} T/m/s)")

    print("\nPNS Analysis:")
    pns_info = pipeline_output['pns_analysis']
    print(f"  Status: {pns_info['status']}")
    if pns_info['status'] == 'Performed':
        print(f"  Compliant: {pns_info['compliant']}")
        print(f"  Peak Total Normalized PNS: {pns_info.get('peak_total_normalized_pns', 'N/A'):.3f}")

    print("\nImage Reconstruction:")
    recon_info = pipeline_output['reconstruction']
    print(f"  Status: {recon_info['status']}")
    if recon_info['status'] == 'Performed':
        print(f"  Reconstructed Image Shape: {recon_info.get('reconstructed_image_shape', 'N/A')}")
        print(f"  Image Quality Metrics: {recon_info.get('image_quality_metrics', {})}")

    # print("\nAll Status Messages:")
    # for msg in pipeline_output['status_messages']:
    #    print(f"  {msg}")

    # Clean up dummy GIRF file
    if girf_path_for_pipeline and os.path.exists(girf_path_for_pipeline):
        try:
            os.remove(girf_path_for_pipeline)
            print(f"\nCleaned up dummy GIRF file: {girf_path_for_pipeline}")
        except Exception as e_cleanup:
            print(f"Error cleaning up dummy GIRF file {girf_path_for_pipeline}: {e_cleanup}")

    print("\n--- Pipeline Example Script Finished ---")
