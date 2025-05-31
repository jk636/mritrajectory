import numpy as np
import json
# Attempt to import GIRFCalibrator for loading, handle if it's not directly available
try:
    from .calibrator import GIRFCalibrator
except ImportError:
    GIRFCalibrator = None # Placeholder if calibrator is not in the same package structure

class TrajectoryPredictor:
    def __init__(self, girf_spectra=None, dt=None, gamma=42.576e6): # Hz/T, common for 1H
        """
        Initializes the TrajectoryPredictor.

        Args:
            girf_spectra (dict, optional): Pre-loaded GIRF spectra.
                                           Maps axis (str) to GIRF spectrum (np.array).
            dt (float, optional): Time step (sampling interval) in seconds. Required for predictions.
            gamma (float, optional): Gyromagnetic ratio in Hz/T. Defaults to 42.576 MHz/T for protons.
        """
        self.girf_spectra = girf_spectra if girf_spectra is not None else {}
        self.nominal_trajectory_kspace = None # Stores k-space trajectory (time_points, num_axes)
        self.predicted_trajectory_kspace = None
        self.nominal_gradients_time = None # Stores time-domain gradients
        self.predicted_gradients_time = None

        if dt is None:
            # dt is crucial, raise an error or warning if not provided,
            # or set a default and warn. Forcing it for predictability.
            print("Warning: 'dt' (time step) not provided during initialization. It must be set before prediction.")
        self.dt = dt
        self.gamma = gamma # Hz/T

        print(f"TrajectoryPredictor initialized. dt={self.dt} s, gamma={self.gamma} Hz/T")

    def load_girf(self, girf_data_or_path):
        """
        Loads GIRF spectra.

        Args:
            girf_data_or_path (dict or str):
                - If dict: Assumed to be GIRF spectra {'axis': np.array, ...}.
                - If str: Path to a JSON file compatible with GIRFCalibrator output.
        """
        if isinstance(girf_data_or_path, dict):
            self.girf_spectra = girf_data_or_path
            # Ensure values are numpy arrays
            for axis, spectrum in self.girf_spectra.items():
                if not isinstance(spectrum, np.ndarray):
                    self.girf_spectra[axis] = np.asarray(spectrum)
            print("GIRF spectra loaded from dictionary.")
        elif isinstance(girf_data_or_path, str):
            if GIRFCalibrator is not None:
                try:
                    calibrator = GIRFCalibrator() # Temporary instance for loading
                    calibrator.load_calibration(girf_data_or_path)
                    self.girf_spectra = calibrator.girf_spectra # These are already np.arrays
                    print(f"GIRF spectra loaded from file: {girf_data_or_path} using GIRFCalibrator.")
                except Exception as e:
                    print(f"Error loading GIRF from file using GIRFCalibrator: {e}. Trying direct JSON load.")
                    # Fallback to direct JSON load if GIRFCalibrator fails or is not available
                    try:
                        with open(girf_data_or_path, 'r') as f:
                            loaded_data = json.load(f)
                        # Assuming the structure matches what GIRFCalibrator saves
                        raw_spectra = loaded_data.get("girf_spectra_complex", {})
                        self.girf_spectra = {}
                        for axis, spec_list in raw_spectra.items():
                            if spec_list and isinstance(spec_list[0], list) and len(spec_list[0]) == 2: # Complex [real, imag]
                                self.girf_spectra[axis] = np.array([complex(item[0], item[1]) for item in spec_list])
                            else: # Assuming real or already complex numbers (less likely from json dump)
                                self.girf_spectra[axis] = np.asarray(spec_list)
                        print(f"GIRF spectra loaded directly from JSON file: {girf_data_or_path}")
                    except Exception as e_json:
                        raise IOError(f"Failed to load GIRF data from path {girf_data_or_path}. Errors: Calibrator way '{e}', JSON way '{e_json}'")

            else: # GIRFCalibrator not available, try direct JSON load
                 try:
                    with open(girf_data_or_path, 'r') as f:
                        loaded_data = json.load(f)
                    raw_spectra = loaded_data.get("girf_spectra_complex", {})
                    self.girf_spectra = {}
                    for axis, spec_list in raw_spectra.items():
                        if spec_list and isinstance(spec_list[0], list) and len(spec_list[0]) == 2:
                            self.girf_spectra[axis] = np.array([complex(item[0], item[1]) for item in spec_list])
                        else:
                            self.girf_spectra[axis] = np.asarray(spec_list)
                    print(f"GIRF spectra loaded directly from JSON file (GIRFCalibrator not found): {girf_data_or_path}")
                 except Exception as e_json:
                    raise IOError(f"Failed to load GIRF data from path {girf_data_or_path} (GIRFCalibrator not found). JSON error: {e_json}")
        else:
            raise TypeError("girf_data_or_path must be a dictionary or a file path string.")

    def _trajectory_to_gradients(self, trajectory_kspace):
        """
        Converts a k-space trajectory to gradient waveforms.
        Assumes trajectory_kspace is (num_points, num_axes).
        gradients = diff(trajectory_kspace, axis=0) / (gamma * dt)
        """
        if not isinstance(trajectory_kspace, np.ndarray):
            trajectory_kspace = np.asarray(trajectory_kspace)
        if self.dt is None:
            raise ValueError("Time step 'dt' must be set before converting trajectory to gradients.")
        if trajectory_kspace.ndim == 1: # Single axis trajectory
            trajectory_kspace = trajectory_kspace[:, np.newaxis] # Ensure 2D for consistent diff

        # Calculate difference along the time axis (axis 0)
        # np.diff prepends a point to match length, or we handle it:
        # Gradients correspond to the interval, so one less point than trajectory.
        # Or, assume gradient is constant during dt, applied at start of interval.
        # g[n] = (k[n+1] - k[n]) / (gamma * dt)
        gradients = np.diff(trajectory_kspace, axis=0) / (self.gamma * self.dt)

        # To maintain length, often the first gradient point is assumed zero or extrapolated.
        # Here, we'll pad with zeros at the beginning to match trajectory length.
        padding = np.zeros((1, trajectory_kspace.shape[1]), dtype=gradients.dtype)
        gradients_padded = np.concatenate([padding, gradients], axis=0)

        return gradients_padded

    def _gradients_to_trajectory(self, gradients_time, initial_kspace_point=None):
        """
        Converts gradient waveforms to a k-space trajectory.
        Assumes gradients_time is (num_points, num_axes).
        trajectory = cumsum(gradients_time * gamma * dt, axis=0)
        """
        if not isinstance(gradients_time, np.ndarray):
            gradients_time = np.asarray(gradients_time)
        if self.dt is None:
            raise ValueError("Time step 'dt' must be set before converting gradients to trajectory.")
        if gradients_time.ndim == 1: # Single axis
            gradients_time = gradients_time[:, np.newaxis]

        # k_delta[n] = g[n] * gamma * dt
        k_deltas = gradients_time * self.gamma * self.dt

        # Integrate: k[t] = sum(g[tau]*gamma*dt) from 0 to t
        trajectory_kspace = np.cumsum(k_deltas, axis=0)

        if initial_kspace_point is not None:
            if not isinstance(initial_kspace_point, np.ndarray):
                initial_kspace_point = np.asarray(initial_kspace_point)
            if initial_kspace_point.shape != (trajectory_kspace.shape[1],):
                 raise ValueError(f"Shape of initial_kspace_point {initial_kspace_point.shape} "
                                  f"must match number of axes {trajectory_kspace.shape[1]}")
            trajectory_kspace += initial_kspace_point - trajectory_kspace[0,:] # Adjust to start from initial_k_point

        return trajectory_kspace

    def predict_trajectory(self, nominal_trajectory_kspace_data):
        """
        Predicts the actual k-space trajectory based on nominal trajectory and GIRF.

        Args:
            nominal_trajectory_kspace_data (np.array or list of lists):
                Shape (num_time_points, num_axes) e.g., for x, y, z.
                Or a dictionary {'axis_name': np.array_1D, ...}

        Returns:
            np.array: The predicted k-space trajectory (num_time_points, num_axes).
        """
        if self.dt is None:
            raise ValueError("Time step 'dt' is not set. Call set_dt() or provide at initialization.")

        if isinstance(nominal_trajectory_kspace_data, dict):
            # Convert dict to (T, N) array, assuming consistent lengths and defined axes order
            # For simplicity, assume standard axes order like ['x', 'y', 'z'] or from GIRF keys
            axes_order = [ax for ax in ['x', 'y', 'z'] if ax in nominal_trajectory_kspace_data] # crude sort
            if not axes_order: raise ValueError("Nominal trajectory dictionary is empty or has no standard axes.")

            # Check lengths
            traj_len = -1
            for axis in axes_order:
                current_len = len(nominal_trajectory_kspace_data[axis])
                if traj_len == -1: traj_len = current_len
                elif traj_len != current_len: raise ValueError("Inconsistent lengths in nominal_trajectory_kspace_data dict.")

            self.nominal_trajectory_kspace = np.zeros((traj_len, len(axes_order)))
            for idx, axis in enumerate(axes_order):
                self.nominal_trajectory_kspace[:, idx] = nominal_trajectory_kspace_data[axis]
            self.axes_names = axes_order # Store the order for mapping back if needed
            print(f"Converted nominal trajectory from dict to array with shape {self.nominal_trajectory_kspace.shape} for axes {self.axes_names}")

        elif isinstance(nominal_trajectory_kspace_data, (list, np.ndarray)):
            self.nominal_trajectory_kspace = np.asarray(nominal_trajectory_kspace_data)
            if self.nominal_trajectory_kspace.ndim == 1: # If single axis trajectory of length T
                self.nominal_trajectory_kspace = self.nominal_trajectory_kspace[:, np.newaxis] # Shape (T,1)
            if self.nominal_trajectory_kspace.ndim != 2:
                raise ValueError("nominal_trajectory_kspace_data must be 2D (num_points, num_axes).")
            # Assume axes names from GIRF keys if possible, or generic like 'axis_0', 'axis_1'
            self.axes_names = list(self.girf_spectra.keys()) # Hope this matches columns
            if not self.axes_names or len(self.axes_names) != self.nominal_trajectory_kspace.shape[1]:
                self.axes_names = [f'axis_{i}' for i in range(self.nominal_trajectory_kspace.shape[1])]
            print(f"Using nominal trajectory array of shape {self.nominal_trajectory_kspace.shape} for axes {self.axes_names}")
        else:
            raise TypeError("nominal_trajectory_kspace_data must be a dict, list, or numpy array.")

        # 1. Convert nominal k-space trajectory to nominal gradient waveforms
        initial_k_point = self.nominal_trajectory_kspace[0, :].copy() # Save initial k-space point
        self.nominal_gradients_time = self._trajectory_to_gradients(self.nominal_trajectory_kspace)

        num_points, num_axes = self.nominal_gradients_time.shape
        self.predicted_gradients_time = np.zeros_like(self.nominal_gradients_time, dtype=np.complex128)

        # 2. For each axis, apply GIRF
        for i in range(num_axes):
            axis_name = self.axes_names[i]
            if axis_name not in self.girf_spectra:
                print(f"Warning: GIRF spectrum for axis '{axis_name}' not found. Using nominal gradient for this axis.")
                self.predicted_gradients_time[:, i] = self.nominal_gradients_time[:, i]
                continue

            nominal_grad_axis = self.nominal_gradients_time[:, i]
            girf_spectrum_axis = self.girf_spectra[axis_name]

            # FFT of nominal gradient
            fft_nominal_grad = np.fft.fft(nominal_grad_axis)

            # Ensure GIRF spectrum matches length of FFT nominal gradient
            # This is a critical step: GIRF spectrum length from calibration might differ
            # from the length of the current trajectory's FFT.
            # Simplest: truncate or pad GIRF. More advanced: interpolate GIRF.
            n_fft = len(fft_nominal_grad)
            if len(girf_spectrum_axis) < n_fft:
                # Pad GIRF spectrum with its last value or zeros
                padding_val = girf_spectrum_axis[-1] if len(girf_spectrum_axis) > 0 else 1.0 # Default to 1 if empty
                girf_spectrum_axis_resized = np.pad(girf_spectrum_axis.astype(np.complex128),
                                                    (0, n_fft - len(girf_spectrum_axis)),
                                                    'constant', constant_values=(padding_val,))
                print(f"Padded GIRF for axis '{axis_name}' from {len(girf_spectrum_axis)} to {n_fft}")
            elif len(girf_spectrum_axis) > n_fft:
                # Truncate GIRF spectrum
                girf_spectrum_axis_resized = girf_spectrum_axis[:n_fft]
                print(f"Truncated GIRF for axis '{axis_name}' from {len(girf_spectrum_axis)} to {n_fft}")
            else:
                girf_spectrum_axis_resized = girf_spectrum_axis

            # Multiply by GIRF spectrum in frequency domain
            fft_predicted_grad = fft_nominal_grad * girf_spectrum_axis_resized

            # IFFT to get predicted gradient in time domain
            predicted_grad_axis_complex = np.fft.ifft(fft_predicted_grad)
            self.predicted_gradients_time[:, i] = predicted_grad_axis_complex.real # Assume physical gradients are real

        # 3. Convert predicted gradients back to k-space trajectory
        self.predicted_trajectory_kspace = self._gradients_to_trajectory(self.predicted_gradients_time, initial_kspace_point=initial_k_point)

        print("Predicted trajectory computed.")
        return self.predicted_trajectory_kspace

    def validate_trajectory(self, validation_threshold=0.1):
        """
        Compares the predicted trajectory with the nominal trajectory.

        Args:
            validation_threshold (float): Threshold for deviation.
                                          Interpretation depends on the metric (e.g., max relative error).

        Returns:
            bool: True if deviation is within threshold, False otherwise.
        """
        if self.predicted_trajectory_kspace is None or self.nominal_trajectory_kspace is None:
            print("Error: Nominal or predicted trajectory not available for validation.")
            return False

        if self.nominal_trajectory_kspace.shape != self.predicted_trajectory_kspace.shape:
            print("Error: Shape mismatch between nominal and predicted trajectories.")
            return False # Or handle resample/comparison of subsegment

        # Metric: Mean Absolute Relative Error (MARE) over all points and axes
        # Avoid division by zero if nominal_trajectory has zeros.
        diff = np.abs(self.predicted_trajectory_kspace - self.nominal_trajectory_kspace)
        abs_nominal = np.abs(self.nominal_trajectory_kspace)

        # Calculate relative error where nominal is not zero
        # For points where nominal is zero, use absolute difference
        relative_error = np.zeros_like(diff)
        non_zero_mask = abs_nominal > 1e-9 # Small epsilon to avoid division by tiny numbers
        zero_mask = ~non_zero_mask

        relative_error[non_zero_mask] = diff[non_zero_mask] / abs_nominal[non_zero_mask]
        relative_error[zero_mask] = diff[zero_mask] # Use absolute difference where nominal is zero

        mean_deviation = np.mean(relative_error)
        max_abs_deviation = np.max(diff)

        print(f"Validation: Mean Relative/Absolute Deviation = {mean_deviation:.4f}, Max Absolute Deviation = {max_abs_deviation:.4f}")

        # For this example, let's use mean_deviation against threshold
        if mean_deviation <= validation_threshold:
            print(f"Validation PASSED (Mean Deviation {mean_deviation:.4f} <= Threshold {validation_threshold:.4f})")
            return True
        else:
            print(f"Validation FAILED (Mean Deviation {mean_deviation:.4f} > Threshold {validation_threshold:.4f})")
            return False

if __name__ == '__main__':
    print("--- Running TrajectoryPredictor Example with Actual Logic ---")

    # Time parameters
    dt_val = 4e-6  # 4 us dwell time
    num_points = 256 # Number of points in trajectory
    duration = num_points * dt_val

    # 1. Create Dummy GIRF Data (e.g., a simple low-pass filter effect)
    # GIRF should be complex, frequency domain. Length can be arbitrary for this test,
    # as the predictor will resize it. Let's make it same length as trajectory FFT.
    girf_len = num_points
    freqs = np.fft.fftfreq(girf_len, d=dt_val)

    # Simple low-pass filter GIRF: H(f) = 1 / (1 + j*f/f_cutoff)
    f_cutoff_x = 20e3 # 20 kHz
    f_cutoff_y = 15e3 # 15 kHz

    girf_x = 1.0 / (1 + 1j * freqs / f_cutoff_x)
    girf_y = 1.0 / (1 + 1j * freqs / f_cutoff_y)
    dummy_girf_spectra = {'x': girf_x, 'y': girf_y}

    # Save dummy GIRF to a JSON to test loading (optional, can pass dict directly)
    # temp_girf_file = "temp_dummy_girf_for_predictor.json"
    # if GIRFCalibrator: # Quick save if calibrator is available
    #     temp_calibrator = GIRFCalibrator(gradient_axes=['x', 'y'])
    #     temp_calibrator.girf_spectra = {'x': girf_x, 'y': girf_y} # Manually set
    #     temp_calibrator.save_calibration(temp_girf_file)


    # 2. Initialize Predictor
    predictor = TrajectoryPredictor(dt=dt_val, gamma=42.576e6) # Standard proton gamma in Hz/T
    # predictor.load_girf(temp_girf_file) # Test loading from file
    predictor.load_girf(dummy_girf_spectra) # Test loading from dict

    # 3. Create a Nominal K-Space Trajectory (e.g., a ramp for x, sine for y)
    time_vector = np.linspace(0, duration - dt_val, num_points, endpoint=True)

    # k-space (m^-1)
    nominal_kx = 100 * time_vector / duration # Ramp up to 100 m^-1
    nominal_ky = 50 * np.sin(2 * np.pi * 5e3 * time_vector) # 5 kHz sine wave, amplitude 50 m^-1

    # Combine into (num_points, num_axes) array
    nominal_trajectory_data = np.stack([nominal_kx, nominal_ky], axis=-1)
    # Or provide as dict:
    # nominal_trajectory_data_dict = {'x': nominal_kx, 'y': nominal_ky}


    # 4. Predict Trajectory
    try:
        predicted_k_traj = predictor.predict_trajectory(nominal_trajectory_data)
        # predicted_k_traj_from_dict = predictor.predict_trajectory(nominal_trajectory_data_dict)

        print(f"Nominal k-space trajectory shape: {predictor.nominal_trajectory_kspace.shape}")
        print(f"Predicted k-space trajectory shape: {predicted_k_traj.shape}")

        # Check first few points
        print(f"\nNominal kx (first 5): {predictor.nominal_trajectory_kspace[:5, 0]}")
        print(f"Predicted kx (first 5): {predicted_k_traj[:5, 0]}")
        print(f"Nominal ky (first 5): {predictor.nominal_trajectory_kspace[:5, 1]}")
        print(f"Predicted ky (first 5): {predicted_k_traj[:5, 1]}")

        # 5. Validate Trajectory
        is_valid = predictor.validate_trajectory(validation_threshold=0.2) # Allow 20% mean deviation
        print(f"Trajectory validation result: {is_valid}")

    except ValueError as e:
        print(f"Error during prediction or validation: {e}")
    except Exception as e_gen:
        print(f"An unexpected error occurred: {e_gen}")
        import traceback
        traceback.print_exc()


    # Clean up dummy file (if created)
    # import os
    # if os.path.exists(temp_girf_file):
    #     os.remove(temp_girf_file)

    print("\n--- TrajectoryPredictor Example Finished ---")
