import numpy as np
import json
import logging # Added for logging

# Attempt to import GIRFCalibrator for loading, handle if it's not directly available
try:
    from .calibrator import GIRFCalibrator
except ImportError:
    GIRFCalibrator = None
# Import utils for resampling if needed, though GIRFCalibrator's output should be primary source
from . import utils


class TrajectoryPredictor:
    def __init__(self, girf_spectra=None, harmonic_components=None, dt=None, gamma=42.576e6): # Hz/T, common for 1H
        """
        Initializes the TrajectoryPredictor.

        Args:
            girf_spectra (dict, optional): Pre-loaded GIRF spectra.
                                           Maps axis (str) to GIRF spectrum (np.array).
            harmonic_components (dict, optional): Pre-loaded harmonic components.
                                                  Maps axis (str) to list of harmonic dicts.
            dt (float, optional): Time step (sampling interval) in seconds. Required for predictions.
            gamma (float, optional): Gyromagnetic ratio in Hz/T. Defaults to 42.576 MHz/T for protons.
        """
        self.girf_spectra = girf_spectra if girf_spectra is not None else {}
        self.harmonic_components = harmonic_components if harmonic_components is not None else {}
        self.nominal_trajectory_kspace = None
        self.predicted_trajectory_kspace = None
        self.nominal_gradients_time = None
        self.predicted_gradients_time = None
        self.axes_names = [] # Store axis names corresponding to columns if input is array

        if dt is None:
            # Using print here as logger might not be set up if dt is critical before logger init
            print("CRITICAL WARNING: 'dt' (time step) not provided during TrajectoryPredictor initialization. Essential for most operations.")
        self.dt = dt
        self.gamma = gamma # Hz/T
        self.gamma_rad_per_T_per_s = self.gamma * 2 * np.pi # For convenience if needed in rad/s/T

        self.logger = logging.getLogger(self.__class__.__name__)
        # Configure basic logging if no handlers are set up for this logger or root,
        # to ensure messages are visible if this class is used standalone.
        if not self.logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')

        self.logger.info(f"TrajectoryPredictor initialized. dt={self.dt} s, gamma={self.gamma} Hz/T")

    def load_girf(self, girf_data_or_path):
        """
        Loads GIRF spectra and harmonic components.

        Args:
            girf_data_or_path (dict or str):
                - If dict: Assumed to be {'girf_spectra': {...}, 'harmonic_components': {...}, ...}.
                - If str: Path to a JSON file compatible with GIRFCalibrator output.
        """
        if isinstance(girf_data_or_path, dict):
            # Direct dictionary load
            raw_spectra = girf_data_or_path.get('girf_spectra', girf_data_or_path.get('girf_spectra_data', {})) # Handle older/direct format
            if isinstance(raw_spectra, dict) and raw_spectra.get('spectrum_complex_list') and isinstance(raw_spectra.get('spectrum_complex_list'),list): # check if it's a single axis entry from Calibrator save format
                 # This means girf_data_or_path was likely a single entry from calibrator.girf_spectra_data
                 # This load_girf expects a dict of {axis_name: spectrum_array} for girf_spectra
                 # Or a full calibrator save structure
                 # For simplicity, assume if 'girf_spectra' key exists, it's the {axis: spectrum_array} dict
                 # If 'girf_spectra_data' key exists, it's the full calibrator save structure
                 if 'girf_spectra_data' in girf_data_or_path: # Full structure from save_calibration
                     self.girf_spectra = {}
                     for axis, spec_data_dict in raw_spectra.items():
                         if 'spectrum_complex_list' in spec_data_dict:
                             s_list = spec_data_dict['spectrum_complex_list']
                             self.girf_spectra[axis] = np.array([complex(r, i) for r, i in s_list])
                         elif 'spectrum' in spec_data_dict : # if it directly contains the spectrum array
                             self.girf_spectra[axis] = np.asarray(spec_data_dict['spectrum'])

                 else: # Assume it's already in {'axis': spectrum_array} format
                     self.girf_spectra = {k: np.asarray(v) for k, v in raw_spectra.items()}

            else: # Assume it's already in {'axis': spectrum_array} format
                self.girf_spectra = {k: np.asarray(v) for k, v in raw_spectra.items()}


            raw_harmonics = girf_data_or_path.get('harmonic_components', girf_data_or_path.get('harmonic_components_data', {}))
            if 'complex_value_real_imag' in str(raw_harmonics): # Check if it's from calibrator save format
                self.harmonic_components = {}
                for axis, h_list_data in raw_harmonics.items():
                    self.harmonic_components[axis] = [
                        {'freq_hz': h['freq_hz'],
                         'complex_value': complex(h['complex_value_real_imag'][0], h['complex_value_real_imag'][1])}
                        for h in h_list_data
                    ]
            else: # Assume it's already in {'axis': [{'freq_hz': f, 'complex_value': cv}, ...]} format
                self.harmonic_components = raw_harmonics

            print("GIRF data (spectra and harmonics) loaded from dictionary.")

        elif isinstance(girf_data_or_path, str):
            # Load from file path
            if GIRFCalibrator is not None:
                try:
                    calibrator = GIRFCalibrator(dt=self.dt) # Pass dt for consistency
                    calibrator.load_calibration(girf_data_or_path)
                    self.girf_spectra = {axis: data.get('spectrum') for axis, data in calibrator.girf_spectra.items() if data.get('spectrum') is not None}
                    self.harmonic_components = calibrator.harmonic_components
                    print(f"GIRF data loaded from file: {girf_data_or_path} using GIRFCalibrator.")
                except Exception as e:
                    raise IOError(f"Failed to load GIRF data from path {girf_data_or_path} using GIRFCalibrator: {e}")
            else: # Fallback direct JSON load if GIRFCalibrator not available
                try:
                    with open(girf_data_or_path, 'r') as f: loaded_data = json.load(f)

                    # Load GIRF spectra
                    raw_spectra_data = loaded_data.get("girf_spectra_data", {})
                    self.girf_spectra = {}
                    for axis, spec_data_dict in raw_spectra_data.items():
                        if 'spectrum_complex_list' in spec_data_dict:
                            s_list = spec_data_dict['spectrum_complex_list']
                            self.girf_spectra[axis] = np.array([complex(r, i) for r, i in s_list])

                    # Load harmonic components
                    raw_harmonics_data = loaded_data.get("harmonic_components_data", {})
                    self.harmonic_components = {}
                    for axis, h_list_data in raw_harmonics_data.items():
                        self.harmonic_components[axis] = [
                            {'freq_hz': h['freq_hz'],
                             'complex_value': complex(h['complex_value_real_imag'][0], h['complex_value_real_imag'][1])}
                            for h in h_list_data ]
                    print(f"GIRF data loaded directly from JSON file (GIRFCalibrator not found): {girf_data_or_path}")
                except Exception as e_json:
                    raise IOError(f"Failed to load GIRF data from path {girf_data_or_path} (direct JSON): {e_json}")
        else:
            raise TypeError("girf_data_or_path must be a dictionary or a file path string.")


    def _trajectory_to_gradients(self, trajectory_kspace):
        # Using the version from utils for consistency now
        return utils.compute_gradient_waveforms(trajectory_kspace, self.gamma, self.dt, output_format='array', input_format_hint='array')

    def _gradients_to_trajectory(self, gradients_time, initial_kspace_point=None):
        # Using the version from utils for consistency now
        return utils.integrate_trajectory(gradients_time, self.gamma, self.dt, initial_k0=initial_kspace_point, output_format='array', input_format_hint='array')


    def predict_trajectory(self, nominal_trajectory_kspace_data, girf_resample_kind='linear', apply_harmonics=True):
        """
        Predicts the actual k-space trajectory. Optionally adds harmonic distortions.
        """
        if self.dt is None:
            raise ValueError("Time step 'dt' is not set.")

        self.nominal_trajectory_kspace = utils.standardize_trajectory_format(
            nominal_trajectory_kspace_data, target_format='array'
        )
        # Determine axis names based on input type or defaults
        if isinstance(nominal_trajectory_kspace_data, dict):
            self.axes_names = sorted(nominal_trajectory_kspace_data.keys())
        else: # Array input, generate default names based on number of columns
            num_axes = self.nominal_trajectory_kspace.shape[1]
            self.axes_names = [f'axis_{i}' for i in range(num_axes)] \
                              if num_axes > 3 else ['x', 'y', 'z'][:num_axes]


        initial_k_point = self.nominal_trajectory_kspace[0, :].copy()
        self.nominal_gradients_time = self._trajectory_to_gradients(self.nominal_trajectory_kspace)

        num_points, num_axes = self.nominal_gradients_time.shape
        predicted_gradients_time_val = np.zeros_like(self.nominal_gradients_time, dtype=float) # Store real part

        for i in range(num_axes):
            axis_name = self.axes_names[i]
            nominal_grad_axis = self.nominal_gradients_time[:, i]

            if axis_name not in self.girf_spectra or not self.girf_spectra[axis_name].any():
                print(f"Warning: GIRF for axis '{axis_name}' not found or empty. Using nominal gradient.")
                predicted_gradients_time_val[:, i] = nominal_grad_axis
                continue

            girf_spectrum_axis = self.girf_spectra[axis_name]
            fft_nominal_grad = np.fft.fft(nominal_grad_axis)
            n_fft = len(fft_nominal_grad)

            if len(girf_spectrum_axis) != n_fft:
                girf_spectrum_axis_resized = utils.resample_waveform(girf_spectrum_axis, n_fft, kind=girf_resample_kind)
                print(f"Resampled GIRF for axis '{axis_name}' from {len(girf_spectrum_axis)} to {n_fft} points.")
            else:
                girf_spectrum_axis_resized = girf_spectrum_axis

            fft_predicted_grad = fft_nominal_grad * girf_spectrum_axis_resized
            predicted_grad_axis_time = np.fft.ifft(fft_predicted_grad).real

            # Add harmonic components if requested
            if apply_harmonics and axis_name in self.harmonic_components and self.harmonic_components[axis_name]:
                print(f"Applying {len(self.harmonic_components[axis_name])} harmonics for axis '{axis_name}'.")
                total_harmonic_waveform_axis = np.zeros(n_fft, dtype=float)
                axis_freq_bins = np.fft.fftfreq(n_fft, d=self.dt)

                for harmonic_info in self.harmonic_components[axis_name]:
                    h_freq = harmonic_info['freq_hz']
                    # cv_calib is Xk_residual_peak / N_calibration_points
                    cv_calib = harmonic_info['complex_value']

                    # Construct spectrum for this single harmonic component
                    # The value C_stored = (A/2)exp(j*phi)
                    # To reconstruct A*cos(2*pi*f*t+phi), the ifft input spectrum should be:
                    # S[idx_pos] = C_stored, S[idx_neg] = conj(C_stored)
                    # Then, time_waveform = ifft(S) * N_target_points

                    h_spectrum_single_comp = np.zeros(n_fft, dtype=np.complex128)

                    idx_pos = np.argmin(np.abs(axis_freq_bins - h_freq))
                    # Check if the found bin is close enough to the actual harmonic frequency
                    if np.isclose(axis_freq_bins[idx_pos], h_freq):
                        h_spectrum_single_comp[idx_pos] = cv_calib

                    if h_freq != 0: # Avoid double counting DC or Nyquist issues if h_freq is exactly that
                        idx_neg = np.argmin(np.abs(axis_freq_bins - (-h_freq)))
                        if np.isclose(axis_freq_bins[idx_neg], -h_freq):
                             h_spectrum_single_comp[idx_neg] = np.conj(cv_calib)

                    # The ifft includes 1/N. The cv_calib was Xk/N_calib.
                    # If N (current n_fft) == N_calib, then ifft(cv_calib_component_spectrum * N) is right.
                    # This assumes that n_fft (current trajectory length) is same as n_points during calibration for that harmonic.
                    # This is a simplification. A more robust way would be to store harmonic physical amplitude & phase.
                    # For now, assume n_fft used for calibration of harmonics is same as current n_fft.
                    single_harmonic_time = np.fft.ifft(h_spectrum_single_comp * n_fft).real
                    total_harmonic_waveform_axis += single_harmonic_time

                predicted_grad_axis_time += total_harmonic_waveform_axis
                print(f"  Max abs value of added harmonics for axis {axis_name}: {np.max(np.abs(total_harmonic_waveform_axis)):.3e}")

            predicted_gradients_time_val[:, i] = predicted_grad_axis_time

        self.predicted_gradients_time = predicted_gradients_time_val
        self.logger.info("Predicted trajectory with optional harmonics computed.")
        return self.predicted_trajectory_kspace

    def apply_b0_inhomogeneity_effects(self, trajectory_kspace, b0_map_hz, time_points_s):
        """
        Conceptual placeholder for applying B0 inhomogeneity effects.
        Currently simulates a simple k-space shift on axis 1 if b0_map_hz is scalar.

        Args:
            trajectory_kspace (np.ndarray): (Npoints, Ndims) array of k-space points (m^-1).
            b0_map_hz (float or np.ndarray): Average B0 offset (Hz) or spatial B0 map (Hz).
            time_points_s (np.ndarray): (Npoints) array of time for each k-space point (s).

        Returns:
            np.ndarray: Modified trajectory_kspace.
        """
        modified_trajectory_kspace = trajectory_kspace.copy()

        if not isinstance(b0_map_hz, np.ndarray) or b0_map_hz.ndim == 0: # Scalar average effect
            avg_b0_offset_hz = float(b0_map_hz)

            # Simplified k-space shift model: delta_k_axis = B0_offset_Hz * time_points_s (units 1/m)
            # This is a direct addition to k-space coordinates in 1/m.
            delta_k_shift_values = avg_b0_offset_hz * time_points_s

            if modified_trajectory_kspace.shape[1] > 1: # Apply to 'y' axis (axis 1) if it exists
                modified_trajectory_kspace[:, 1] += delta_k_shift_values
                self.logger.info(f"Conceptual: Applied B0 inhomogeneity (k-space shift on axis 1) using avg offset {avg_b0_offset_hz:.1f} Hz. Max shift: {np.max(np.abs(delta_k_shift_values)):.2f} m^-1.")
            elif modified_trajectory_kspace.shape[1] == 1: # Apply to the only axis (axis 0)
                modified_trajectory_kspace[:, 0] += delta_k_shift_values
                self.logger.info(f"Conceptual: Applied B0 inhomogeneity (k-space shift on axis 0 for 1D traj) using avg offset {avg_b0_offset_hz:.1f} Hz. Max shift: {np.max(np.abs(delta_k_shift_values)):.2f} m^-1.")
            else:
                 self.logger.warning("Conceptual B0 effect: Trajectory has no spatial dimensions to apply shift to.")
            return modified_trajectory_kspace
        else: # Spatially varying B0 map
            self.logger.info("Conceptual: B0 inhomogeneity effect with spatial map is complex and not yet implemented. Returning original trajectory.")
            return trajectory_kspace # Return copy to maintain consistency of returning a copy

    def apply_gradient_nonlinearity_effects(self, trajectory_kspace, gradient_nonlinearity_model=None):
        """
        Conceptual placeholder for applying gradient non-linearity effects.

        Args:
            trajectory_kspace (np.ndarray): (Npoints, Ndims) array of k-space points.
            gradient_nonlinearity_model (dict or object, optional): Model describing non-linearities
                                         (e.g., SH coeffs).

        Returns:
            np.ndarray: Modified trajectory_kspace (currently returns original).
        """
        self.logger.info("Conceptual: Gradient non-linearity effects are complex. This function is a placeholder.")
        if gradient_nonlinearity_model:
            self.logger.info(f"  Received gradient_nonlinearity_model of type: {type(gradient_nonlinearity_model)}")
        return trajectory_kspace.copy()


    def validate_trajectory(self, validation_threshold=0.1):
        """ Compares the predicted trajectory with the nominal trajectory. """
        if self.predicted_trajectory_kspace is None or self.nominal_trajectory_kspace is None:
            self.logger.error("Nominal or predicted trajectory not available for validation.")
            return False
        if self.nominal_trajectory_kspace.shape != self.predicted_trajectory_kspace.shape:
            self.logger.error("Shape mismatch between nominal and predicted trajectories.")
            return False

        diff = np.abs(self.predicted_trajectory_kspace - self.nominal_trajectory_kspace)
        abs_nominal = np.abs(self.nominal_trajectory_kspace)

        relative_error = np.zeros_like(diff)
        non_zero_mask = abs_nominal > 1e-9
        zero_mask = ~non_zero_mask

        relative_error[non_zero_mask] = diff[non_zero_mask] / abs_nominal[non_zero_mask]
        relative_error[zero_mask] = diff[zero_mask]

        mean_deviation = np.mean(relative_error)
        max_abs_deviation = np.max(diff)

        self.logger.info(f"Validation: Mean Rel/Abs Deviation = {mean_deviation:.4f}, Max Abs Deviation = {max_abs_deviation:.4f}")
        if mean_deviation <= validation_threshold:
            self.logger.info(f"Validation PASSED (Mean Dev {mean_deviation:.4f} <= Threshold {validation_threshold:.4f})")
            return True
        else:
            self.logger.info(f"Validation FAILED (Mean Dev {mean_deviation:.4f} > Threshold {validation_threshold:.4f})")
            return False

if __name__ == '__main__':
    # Setup basic logging for the __main__ example if no other logging is configured
    if not logging.getLogger().hasHandlers(): # Check root logger
        # BasicConfig should ideally be called only once at application start.
        # In a library, it's often left to the application to configure logging.
        # However, for this example to run standalone and show logs, we add it.
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')
    logger_main = logging.getLogger("TrajectoryPredictorExample") # Use a specific name for module example

    logger_main.info("--- Running TrajectoryPredictor Example with Harmonics & Conceptual Effects ---")
    dt_val = 4e-6
    num_pts = 256
    gamma_val = DEFAULT_GAMMA_PROTON
    time_pts_s_example = np.arange(num_pts) * dt_val


    # 1. Create Dummy GIRF and Harmonics Data
    fft_freqs = np.fft.fftfreq(num_pts, d=dt_val)
    girf_x_main = 0.9 * np.exp(1j * np.pi/8 * np.sign(fft_freqs))
    girf_x_main[0] = 0.9

    harmonic_x1_freq = 10000
    harmonic_x1_amp_phys = 0.001 # Amplitude of this gradient harmonic component (T/m)
    harmonic_x1_phase = np.pi/3
    # complex_value = (Amp/2) * exp(j*phase) as stored by Calibrator
    harmonic_x1_cv = (harmonic_x1_amp_phys / 2) * np.exp(1j * harmonic_x1_phase)

    harmonics_data_x = [{'freq_hz': harmonic_x1_freq, 'complex_value': harmonic_x1_cv}]

    dummy_full_girf_data = {
        'girf_spectra': {'x': girf_x_main, 'y': np.ones(num_pts, dtype=np.complex128)},
        'harmonic_components': {'x': harmonics_data_x, 'y': []}
    }

    # 2. Initialize Predictor
    predictor = TrajectoryPredictor(dt=dt_val, gamma=gamma_val)
    predictor.load_girf(dummy_full_girf_data)

    # 3. Create Nominal Trajectory
    nominal_kx = np.linspace(0, 150, num_pts)
    nominal_ky = 80 * np.sin(2 * np.pi * 2e3 * time_pts_s_example)
    nominal_k_data_dict = {'x': nominal_kx, 'y': nominal_ky}

    # 4. Predict Trajectory - WITHOUT Harmonics
    logger_main.info("\n--- Predicting WITHOUT Harmonics ---")
    predicted_k_no_harm = predictor.predict_trajectory(nominal_k_data_dict, apply_harmonics=False)
    grad_pred_no_harm_x = predictor.predicted_gradients_time[:, predictor.axes_names.index('x')].copy()

    # 5. Predict Trajectory - WITH Harmonics
    logger_main.info("\n--- Predicting WITH Harmonics ---")
    predicted_k_with_harm = predictor.predict_trajectory(nominal_k_data_dict, apply_harmonics=True)
    grad_pred_with_harm_x = predictor.predicted_gradients_time[:, predictor.axes_names.index('x')].copy()

    # 6. Compare
    logger_main.info("\n--- Comparison (Harmonics Effect) ---")
    max_abs_grad_diff = np.max(np.abs(grad_pred_with_harm_x - grad_pred_no_harm_x))
    logger_main.info(f"Max absolute difference in X-gradients (with/without harmonics): {max_abs_grad_diff:.3e} T/m")
    if harmonics_data_x and max_abs_grad_diff > 1e-9 * harmonic_x1_amp_phys :
        logger_main.info("Harmonic application had a noticeable effect on gradients.")
    elif not harmonics_data_x and max_abs_grad_diff < 1e-9:
         logger_main.info("No harmonics defined, and difference is negligible, as expected.")
    else:
        logger_main.warning("Harmonic effect on gradients is smaller than expected or unexpected difference found.")

    # 7. Apply Conceptual B0 Inhomogeneity Effect
    logger_main.info("\n--- Applying Conceptual B0 Inhomogeneity Effect ---")
    avg_b0_offset_example = 5.0 # Hz average offset
    k_traj_after_b0 = predictor.apply_b0_inhomogeneity_effects(
        predicted_k_with_harm,
        avg_b0_offset_example,
        time_pts_s_example
    )
    b0_k_diff_norm = np.linalg.norm(k_traj_after_b0 - predicted_k_with_harm)
    logger_main.info(f"Norm of k-space difference after applying conceptual B0 effect: {b0_k_diff_norm:.3e}")
    if b0_k_diff_norm > 1e-9: logger_main.info("Conceptual B0 effect modified the trajectory.")


    # 8. Apply Conceptual Gradient Non-linearity Effect
    logger_main.info("\n--- Applying Conceptual Gradient Non-linearity Effect ---")
    dummy_gnl_model = {'sh_coeffs_x': [0.001, 0.0005], 'sh_coeffs_y': [0.0008]}
    k_traj_after_gnl = predictor.apply_gradient_nonlinearity_effects(
        k_traj_after_b0,
        gradient_nonlinearity_model=dummy_gnl_model
    )
    gnl_k_diff_norm = np.linalg.norm(k_traj_after_gnl - k_traj_after_b0)
    if gnl_k_diff_norm < 1e-9:
        logger_main.info("Conceptual GNL effect is a placeholder and did not modify the trajectory, as expected.")
    else:
        logger_main.warning("Conceptual GNL effect (placeholder) unexpectedly modified the trajectory.")


    # Validate the final trajectory (with all effects)
    logger_main.info("\nValidating final conceptual trajectory (vs. original nominal):")
    predictor.predicted_trajectory_kspace = k_traj_after_gnl
    predictor.validate_trajectory(validation_threshold=0.5)

    logger_main.info("\n--- TrajectoryPredictor Conceptual Effects Example Finished ---")
