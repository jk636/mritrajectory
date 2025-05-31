import numpy as np
import json
from scipy.signal import find_peaks # For harmonic analysis

class GIRFCalibrator:
    def __init__(self, gradient_axes=None, dt=None):
        """
        Initializes the GIRFCalibrator.

        Args:
            gradient_axes (list of str, optional): The gradient axes to consider (e.g., ['x', 'y', 'z']).
                                                   Defaults to ['x', 'y', 'z'] if None.
            dt (float, optional): Default time step (sampling interval) in seconds for waveforms.
                                  This is crucial for FFT frequency calculations.
                                  Can be overridden if 'dt' is in waveform_params during measure_girf.
        """
        if gradient_axes is None:
            self.gradient_axes = ['x', 'y', 'z']
        else:
            self.gradient_axes = gradient_axes

        self.input_waveforms = {}  # Stores input waveforms per axis: {'x': {'data': np.array, 'params': {}}, ...}
        self.measured_responses = {} # Stores measured responses per axis: {'x': {'data': np.array, 'params': {}}, ...}
        self.girf_spectra = {}  # Stores computed GIRF info per axis: {'x': {'spectrum': np.array, 'residual_info': {...}}, ...}
        self.harmonic_components = {} # Stores identified harmonics: {'x': [{'freq': f, 'complex_value': val}, ...]}

        if dt is None:
            print("Warning: 'dt' not provided to GIRFCalibrator. It must be available in waveform_params or computations might fail.")
        self.dt = dt # Default dt

        print(f"GIRFCalibrator initialized for axes: {self.gradient_axes}, default dt: {self.dt}s")

    def measure_girf(self, axis, input_waveform, measured_response, waveform_params=None):
        """
        Stores the input waveform and measured response for a given gradient axis.

        Args:
            axis (str): The gradient axis (e.g., 'x', 'y', 'z').
            input_waveform (list or np.array): The input gradient waveform data.
            measured_response (list or np.array): The measured system response waveform data.
            waveform_params (dict, optional): Additional parameters (e.g., {'dt': specific_dt}).
        """
        if axis not in self.gradient_axes:
            raise ValueError(f"Axis '{axis}' is not configured. Configured axes: {self.gradient_axes}")

        current_params = waveform_params if waveform_params is not None else {}

        self.input_waveforms[axis] = {
            'data': np.asarray(input_waveform),
            'params': current_params
        }
        self.measured_responses[axis] = {
            'data': np.asarray(measured_response),
            'params': current_params # Assuming same params for input/response pair
        }

        input_len = len(self.input_waveforms[axis]['data'])
        response_len = len(self.measured_responses[axis]['data'])
        dt_used = current_params.get('dt', self.dt)
        log_dt = f"{dt_used*1e6:.2f}us" if dt_used else "Not Set"

        print(f"Measurement data stored for axis '{axis}'. Input len: {input_len}, Resp len: {response_len}, dt: {log_dt}")


    def compute_girf_spectrum(self, axis, regularization_factor=1e-6, harmonic_height_factor=0.1, harmonic_N_std=3, harmonic_distance_factor=0.01):
        """
        Computes the GIRF spectrum and analyzes residuals for harmonics for the specified axis.
        GIRF(omega) = FFT(Measured_Response(t)) / FFT(Input_Waveform(t))

        Args:
            axis (str): The gradient axis for which to compute the GIRF.
            regularization_factor (float): Small value for regularizing GIRF division.
            harmonic_height_factor (float): For find_peaks, fraction of max residual FFT magnitude.
                                            Used if N_std method yields lower height.
            harmonic_N_std (float): For find_peaks, number of std devs above mean for height.
            harmonic_distance_factor (float): For find_peaks, fraction of total positive frequencies
                                              for minimum distance between peaks.

        Returns:
            np.array: The computed GIRF spectrum (complex-valued).
        """
        if axis not in self.input_waveforms or axis not in self.measured_responses:
            raise ValueError(f"Data not found for axis '{axis}'. Call measure_girf() first.")

        input_wf_entry = self.input_waveforms[axis]
        measured_resp_entry = self.measured_responses[axis]

        input_data = input_wf_entry['data']
        measured_data = measured_resp_entry['data']

        current_dt = input_wf_entry['params'].get('dt', self.dt)
        if current_dt is None:
            raise ValueError(f"Time step 'dt' not available for axis '{axis}' (not in waveform_params or instance default).")

        if len(input_data) == 0 or len(measured_data) == 0:
            raise ValueError(f"Waveform data for axis '{axis}' is empty.")
        if len(input_data) != len(measured_data):
            raise ValueError(f"Input and measured waveforms for axis '{axis}' must have same length.")

        print(f"Computing GIRF spectrum for axis '{axis}' (dt: {current_dt*1e6:.2f}us)...")
        n_points = len(input_data)

        fft_input = np.fft.fft(input_data)
        fft_measured = np.fft.fft(measured_data)

        denominator = fft_input.copy()
        # Add regularization to magnitude before division, or use a fixed small number for very small components
        # A common way: den_reg = fft_input + eps * max(abs(fft_input)) (if fft_input is small)
        # Or simply add to abs: den_reg = fft_input * (abs(fft_input) + reg) / abs(fft_input)
        # Simplified: add small constant to very small values in denominator
        den_magnitude = np.abs(denominator)
        min_den_magnitude_for_direct_division = regularization_factor # Arbitrary threshold
        small_indices = den_magnitude < min_den_magnitude_for_direct_division
        # For small inputs, GIRF is ill-defined. Result can be set to 0 or regularized.
        # Option 1: Set GIRF to 0 where input FFT is too small.
        # fft_input_reg = fft_input.copy()
        # fft_input_reg[small_indices] = np.inf # Leads to GIRF = 0
        # girf_spectrum_fft = fft_measured / fft_input_reg
        # Option 2: Regularize denominator (common in Wiener-like deconvolution)
        denominator[small_indices] = min_den_magnitude_for_direct_division # Avoid division by zero
        if np.any(small_indices):
             print(f"  Regularized {np.sum(small_indices)} points in GIRF denominator for axis '{axis}'.")
        girf_spectrum_fft = fft_measured / denominator

        if axis not in self.girf_spectra: self.girf_spectra[axis] = {}
        self.girf_spectra[axis]['spectrum'] = girf_spectrum_fft
        print(f"  GIRF spectrum computed. Length: {len(girf_spectrum_fft)}")

        # --- Harmonic Analysis from Residuals ---
        print(f"  Performing harmonic analysis for axis '{axis}'...")
        predicted_response_fft = fft_input * girf_spectrum_fft
        predicted_response_time = np.fft.ifft(predicted_response_fft)

        residual_time = measured_data - predicted_response_time.real # Assume physical response is real
        residual_fft_complex = np.fft.fft(residual_time)
        residual_fft_magnitude = np.abs(residual_fft_complex)
        freqs = np.fft.fftfreq(n_points, d=current_dt)

        self.girf_spectra[axis]['residual_info'] = {
            'fft_complex': residual_fft_complex,
            'frequencies_hz': freqs
        }

        positive_freq_indices = np.where(freqs > 0)[0] # Ensure it's 1D array of indices
        if len(positive_freq_indices) == 0:
            print("  No positive frequencies available for harmonic peak detection.")
            self.harmonic_components[axis] = []
            return girf_spectrum_fft

        positive_freqs = freqs[positive_freq_indices]
        positive_residual_fft_mag = residual_fft_magnitude[positive_freq_indices]
        positive_residual_fft_cplx = residual_fft_complex[positive_freq_indices]

        # Determine peak detection parameters
        height_thresh_std = np.mean(positive_residual_fft_mag) + harmonic_N_std * np.std(positive_residual_fft_mag)
        height_thresh_factor = harmonic_height_factor * np.max(positive_residual_fft_mag) if np.max(positive_residual_fft_mag) > 0 else 0
        peak_height_threshold = max(height_thresh_std, height_thresh_factor)

        min_peak_dist_samples = int(len(positive_freqs) * harmonic_distance_factor)
        if min_peak_dist_samples < 1: min_peak_dist_samples = 1

        peaks_indices, _ = find_peaks(positive_residual_fft_mag,
                                      height=peak_height_threshold,
                                      distance=min_peak_dist_samples)

        axis_harmonics = []
        for peak_idx in peaks_indices:
            h_freq = positive_freqs[peak_idx]
            # Scale FFT component by 1/n_points to get amplitude of sine component
            # Or by 2/n_points for positive frequencies if representing one side of spectrum
            # For now, store complex value as is (scaled by 1/n_points for generalizability)
            h_complex_val = positive_residual_fft_cplx[peak_idx] / n_points
            axis_harmonics.append({'freq_hz': h_freq, 'complex_value': h_complex_val})

        self.harmonic_components[axis] = sorted(axis_harmonics, key=lambda x: np.abs(x['complex_value']), reverse=True) # Sort by magnitude
        print(f"  Found {len(axis_harmonics)} harmonic components for axis '{axis}'.")
        if axis_harmonics:
            top_n = min(3, len(axis_harmonics))
            for i in range(top_n):
                h = axis_harmonics[i]
                print(f"    - Freq: {h['freq_hz']:.1f} Hz, Amp: {np.abs(h['complex_value']):.3e}, Phase: {np.angle(h['complex_value']):.2f} rad")

        return girf_spectrum_fft


    def save_calibration(self, file_path):
        """ Saves computed GIRF spectra and harmonic components to a JSON file. """
        if not self.girf_spectra and not self.harmonic_components:
            print("Warning: No GIRF spectra or harmonics computed. Saving an empty calibration.")

        # Prepare girf_spectra for saving (convert numpy arrays to lists)
        girf_spectra_to_save = {}
        for axis, data_dict in self.girf_spectra.items():
            girf_spectra_to_save[axis] = {}
            if 'spectrum' in data_dict and isinstance(data_dict['spectrum'], np.ndarray):
                girf_spectra_to_save[axis]['spectrum_complex_list'] = [[val.real, val.imag] for val in data_dict['spectrum']]
            if 'residual_info' in data_dict:
                girf_spectra_to_save[axis]['residual_info'] = {
                    'fft_complex_list': [[val.real, val.imag] for val in data_dict['residual_info']['fft_complex']],
                    'frequencies_hz_list': data_dict['residual_info']['frequencies_hz'].tolist()
                }

        # Prepare harmonic_components for saving
        harmonics_to_save = {}
        for axis, harmonics_list in self.harmonic_components.items():
            harmonics_to_save[axis] = [
                {'freq_hz': h['freq_hz'], 'complex_value_real_imag': [h['complex_value'].real, h['complex_value'].imag]}
                for h in harmonics_list
            ]

        calibration_data = {
            "gradient_axes": self.gradient_axes,
            "default_dt_s": self.dt,
            "girf_spectra_data": girf_spectra_to_save,
            "harmonic_components_data": harmonics_to_save,
            # Save input_waveforms params for reference if needed
            "input_waveform_params": {axis: entry['params'] for axis, entry in self.input_waveforms.items() if entry['params']}
        }

        print(f"Saving GIRF calibration data to {file_path}...")
        try:
            with open(file_path, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            print("GIRF calibration saved successfully.")
        except IOError as e:
            print(f"Error saving calibration data to {file_path}: {e}")
            raise


    def load_calibration(self, file_path):
        """ Loads GIRF spectra and harmonic components from a JSON file. """
        print(f"Loading GIRF calibration data from {file_path}...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except IOError as e:
            print(f"Error loading calibration data: {e}"); raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}") from e

        self.gradient_axes = data.get("gradient_axes", ['x', 'y', 'z'])
        self.dt = data.get("default_dt_s", self.dt) # Load default dt, or keep existing if not in file

        loaded_girf_spectra_data = data.get("girf_spectra_data", {})
        self.girf_spectra = {}
        for axis, spec_data_dict in loaded_girf_spectra_data.items():
            self.girf_spectra[axis] = {}
            if 'spectrum_complex_list' in spec_data_dict:
                s_list = spec_data_dict['spectrum_complex_list']
                self.girf_spectra[axis]['spectrum'] = np.array([complex(r, i) for r, i in s_list])
            if 'residual_info' in spec_data_dict:
                r_info = spec_data_dict['residual_info']
                self.girf_spectra[axis]['residual_info'] = {
                    'fft_complex': np.array([complex(r,i) for r,i in r_info['fft_complex_list']]),
                    'frequencies_hz': np.array(r_info['frequencies_hz_list'])
                }

        loaded_harmonics_data = data.get("harmonic_components_data", {})
        self.harmonic_components = {}
        for axis, harmonics_list_data in loaded_harmonics_data.items():
            self.harmonic_components[axis] = [
                {'freq_hz': h['freq_hz'], 'complex_value': complex(h['complex_value_real_imag'][0], h['complex_value_real_imag'][1])}
                for h in harmonics_list_data
            ]

        # Load input waveform params if saved (might be useful for context)
        loaded_input_params = data.get("input_waveform_params", {})
        for axis, params in loaded_input_params.items():
            if axis in self.input_waveforms: # If measure_girf was called before load for this axis
                self.input_waveforms[axis]['params'].update(params)
            else: # Could store them somewhere else or ensure input_waveforms dict exists
                # For now, this assumes measure_girf might not have been called for these axes yet
                pass


        print(f"GIRF calibration loaded. Axes: {list(self.girf_spectra.keys())}. Harmonics loaded for: {list(self.harmonic_components.keys())}")


if __name__ == '__main__':
    print("--- Running GIRFCalibrator Example with Harmonic Analysis ---")

    # Time parameters for dummy data
    dt_val = 4e-6  # 4 us
    num_pts_main = 512 # Number of points for waveforms
    time_vec = np.arange(num_pts_main) * dt_val

    # 1. Initialize Calibrator
    calibrator = GIRFCalibrator(gradient_axes=['Gx', 'Gy'], dt=dt_val)

    # 2. Simulate Measurement Data for Gx
    # Input: a simple sine wave for Gx
    freq_input_gx = 10e3 # 10 kHz
    input_gx_wf = np.sin(2 * np.pi * freq_input_gx * time_vec)

    # Simulated Measured Response: attenuated, phase-shifted, and with a harmonic distortion
    attenuation_gx = 0.9
    phase_shift_gx = np.pi / 6
    measured_gx_wf = attenuation_gx * np.sin(2 * np.pi * freq_input_gx * time_vec - phase_shift_gx)
    # Add a harmonic (e.g., 3rd harmonic of input frequency)
    harmonic_freq_gx = 3 * freq_input_gx
    harmonic_amp_gx = 0.1 * np.max(measured_gx_wf) # 10% of main signal peak
    measured_gx_wf += harmonic_amp_gx * np.sin(2 * np.pi * harmonic_freq_gx * time_vec + np.pi/3)
    measured_gx_wf += 0.02 * np.random.normal(size=num_pts_main) # Add some noise

    calibrator.measure_girf('Gx', input_gx_wf, measured_gx_wf, waveform_params={'dt': dt_val, 'notes': 'Gx test with 3rd harmonic'})

    # 3. Compute GIRF Spectrum & Harmonics for Gx
    try:
        girf_spectrum_gx = calibrator.compute_girf_spectrum('Gx', regularization_factor=1e-5,
                                                            harmonic_height_factor=0.05, harmonic_N_std=2,
                                                            harmonic_distance_factor=0.005) # Smaller distance for potentially closer harmonics
        # print(f"Computed GIRF Spectrum for Gx (first 5 points): {girf_spectrum_gx[:5]}")
        if 'Gx' in calibrator.harmonic_components and calibrator.harmonic_components['Gx']:
            print("\nIdentified Harmonics for Gx (strongest first):")
            for h_comp in calibrator.harmonic_components['Gx'][:5]: # Print top 5
                 print(f"  Freq: {h_comp['freq_hz']/1e3:.1f} kHz, "
                       f"Amp: {np.abs(h_comp['complex_value']):.3e}, "
                       f"Phase: {np.angle(h_comp['complex_value']):.2f} rad")
        else:
            print("No significant harmonics found for Gx based on current detection settings.")

    except ValueError as e:
        print(f"Error computing GIRF for Gx: {e}")

    # 4. Save and Load Calibration
    calibration_file_harmonics = "dummy_girf_calibration_with_harmonics.json"
    try:
        calibrator.save_calibration(calibration_file_harmonics)

        new_calibrator = GIRFCalibrator(dt=dt_val) # Init with dt for consistency if file doesn't have it
        new_calibrator.load_calibration(calibration_file_harmonics)

        print("\n--- Loaded Calibration Data ---")
        if 'Gx' in new_calibrator.girf_spectra:
            print(f"Gx GIRF spectrum loaded (length {len(new_calibrator.girf_spectra['Gx']['spectrum'])}).")
        if 'Gx' in new_calibrator.harmonic_components and new_calibrator.harmonic_components['Gx']:
            print("Gx Harmonics loaded:")
            for h_comp in new_calibrator.harmonic_components['Gx'][:3]:
                 print(f"  Freq: {h_comp['freq_hz']/1e3:.1f} kHz, Amp: {np.abs(h_comp['complex_value']):.3e}")
        else:
            print("No Gx harmonics loaded or found.")

    except Exception as e:
        print(f"Error during save/load test: {e}")
    finally:
        if os.path.exists(calibration_file_harmonics):
            os.remove(calibration_file_harmonics)

    print("\n--- GIRFCalibrator Harmonic Analysis Example Finished ---")
