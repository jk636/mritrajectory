import numpy as np
import json
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
            print("Warning: 'dt' (time step) not provided during initialization. It must be set before prediction.")
        self.dt = dt
        self.gamma = gamma

        print(f"TrajectoryPredictor initialized. dt={self.dt} s, gamma={self.gamma} Hz/T")

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
        self.predicted_trajectory_kspace = self._gradients_to_trajectory(self.predicted_gradients_time,
                                                                         initial_kspace_point=initial_k_point)
        print("Predicted trajectory with optional harmonics computed.")
        return self.predicted_trajectory_kspace


    def validate_trajectory(self, validation_threshold=0.1):
        """ Compares the predicted trajectory with the nominal trajectory. """
        # (Implementation from previous version, ensure it uses self.nominal_trajectory_kspace etc.)
        if self.predicted_trajectory_kspace is None or self.nominal_trajectory_kspace is None:
            print("Error: Nominal or predicted trajectory not available for validation.")
            return False
        if self.nominal_trajectory_kspace.shape != self.predicted_trajectory_kspace.shape:
            print("Error: Shape mismatch between nominal and predicted trajectories.")
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

        print(f"Validation: Mean Rel/Abs Deviation = {mean_deviation:.4f}, Max Abs Deviation = {max_abs_deviation:.4f}")
        if mean_deviation <= validation_threshold:
            print(f"Validation PASSED (Mean Dev {mean_deviation:.4f} <= Threshold {validation_threshold:.4f})")
            return True
        else:
            print(f"Validation FAILED (Mean Dev {mean_deviation:.4f} > Threshold {validation_threshold:.4f})")
            return False

if __name__ == '__main__':
    print("--- Running TrajectoryPredictor Example with Harmonics ---")
    dt_val = 4e-6
    num_pts = 256
    gamma_val = DEFAULT_GAMMA_PROTON

    # 1. Create Dummy GIRF and Harmonics Data
    fft_freqs = np.fft.fftfreq(num_pts, d=dt_val)
    girf_x_main = 0.9 * np.exp(1j * np.pi/8 * np.sign(fft_freqs)) # Attenuation and phase shift
    girf_x_main[0] = 0.9

    # Harmonics for x-axis: e.g., one at 10kHz
    # The complex_value is Xk_residual_peak / N_calibration_points
    # So, if residual time signal has A*cos(2*pi*f0*t+phi), then Xk_peak/N = (A/2)*e^(j*phi)
    harmonic_x1_freq = 10000 # 10 kHz
    harmonic_x1_amp_phys = 0.05 # Relative to what? Let's say this is the A in A*cos(...)
    harmonic_x1_phase = np.pi/3
    harmonic_x1_cv = (harmonic_x1_amp_phys / 2) * np.exp(1j * harmonic_x1_phase)

    harmonics_data_x = [{'freq_hz': harmonic_x1_freq, 'complex_value': harmonic_x1_cv}]

    dummy_full_girf_data = {
        'girf_spectra': {'x': girf_x_main, 'y': np.ones(num_pts, dtype=np.complex128)}, # y is ideal
        'harmonic_components': {'x': harmonics_data_x, 'y': []} # No harmonics for y
    }

    # 2. Initialize Predictor
    predictor = TrajectoryPredictor(dt=dt_val, gamma=gamma_val)
    predictor.load_girf(dummy_full_girf_data) # Load from dict

    # 3. Create Nominal Trajectory (ramp for x, sine for y)
    time_vec = np.arange(num_pts) * dt_val
    nominal_kx = np.linspace(0, 150, num_pts) # m^-1
    nominal_ky = 80 * np.sin(2 * np.pi * 2e3 * time_vec) # 2kHz sine, amp 80 m^-1
    nominal_k_data_dict = {'x': nominal_kx, 'y': nominal_ky}

    # 4. Predict Trajectory - WITHOUT Harmonics
    print("\n--- Predicting WITHOUT Harmonics ---")
    predicted_k_no_harm = predictor.predict_trajectory(nominal_k_data_dict, apply_harmonics=False)
    grad_pred_no_harm_x = predictor.predicted_gradients_time[:, predictor.axes_names.index('x')]

    # 5. Predict Trajectory - WITH Harmonics
    print("\n--- Predicting WITH Harmonics ---")
    predicted_k_with_harm = predictor.predict_trajectory(nominal_k_data_dict, apply_harmonics=True)
    grad_pred_with_harm_x = predictor.predicted_gradients_time[:, predictor.axes_names.index('x')]

    # 6. Compare
    print("\n--- Comparison ---")
    k_diff_norm = np.linalg.norm(predicted_k_with_harm - predicted_k_no_harm)
    grad_diff_norm = np.linalg.norm(grad_pred_with_harm_x - grad_pred_no_harm_x)

    print(f"Norm of difference between k-space trajectories (with/without harmonics): {k_diff_norm:.3e}")
    print(f"Norm of difference between X-gradients (with/without harmonics): {grad_diff_norm:.3e}")

    # Expect grad_diff_norm to be non-zero if harmonics were applied
    # The harmonic amplitude was 0.05. Its effect on gradients should be noticeable.
    # A simple check: max abs diff of gradients
    max_abs_grad_diff = np.max(np.abs(grad_pred_with_harm_x - grad_pred_no_harm_x))
    print(f"Max absolute difference in X-gradients: {max_abs_grad_diff:.3e}")

    # The time domain signal from harmonic: A*cos(2*pi*f*t + phi)
    # Its derivative (related to gradient) will be A*2*pi*f * -sin(...)
    # Max of that is A*2*pi*f. Here A=0.05, f=10kHz. Max grad component = 0.05 * 2*pi*10000 = 3141 T/m (if A was gradient units)
    # However, harmonic['complex_value'] is for the *gradient residual*. So it's already in gradient-like units.
    # The harmonic was A_grad_residual * cos(...). So max_abs_grad_diff should be around A_grad_residual (0.05).
    # This depends on how well the single freq bin matches the harmonic_x1_freq.
    if harmonics_data_x: # if harmonics were actually defined
        if max_abs_grad_diff > 1e-9: # Check if it's meaningfully non-zero
            print("Harmonic application seems to have an effect on gradients, as expected.")
        else:
            print("Warning: Harmonics applied, but difference in gradients is very small or zero.")
    else:
        if max_abs_grad_diff < 1e-9:
            print("No harmonics were defined, and difference is zero, as expected.")
        else:
            print("Warning: No harmonics defined, but difference in gradients is non-zero.")


    # Validate the final trajectory (with harmonics)
    print("\nValidating trajectory predicted WITH harmonics:")
    predictor.validate_trajectory(validation_threshold=0.5) # Higher threshold as harmonics add deviation

    print("\n--- TrajectoryPredictor Harmonics Example Finished ---")
