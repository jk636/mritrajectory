import numpy as np
import json

class GIRFCalibrator:
    def __init__(self, gradient_axes=None):
        """
        Initializes the GIRFCalibrator.

        Args:
            gradient_axes (list of str, optional): The gradient axes to consider (e.g., ['x', 'y', 'z']).
                                                   Defaults to ['x', 'y', 'z'] if None.
        """
        if gradient_axes is None:
            self.gradient_axes = ['x', 'y', 'z']
        else:
            self.gradient_axes = gradient_axes

        self.input_waveforms = {}  # Stores input waveforms per axis: {'x': np.array, ...}
        self.measured_responses = {} # Stores measured responses per axis: {'x': np.array, ...}
        self.girf_spectra = {}  # Stores computed GIRF spectra per axis: {'x': np.array, ...}
        self.waveform_params = {} # Stores optional parameters for waveforms per axis

        print(f"GIRFCalibrator initialized for axes: {self.gradient_axes}")

    def measure_girf(self, axis, input_waveform, measured_response, waveform_params=None):
        """
        Stores the input waveform and measured response for a given gradient axis.

        Args:
            axis (str): The gradient axis (e.g., 'x', 'y', 'z').
            input_waveform (list or np.array): The input gradient waveform data.
            measured_response (list or np.array): The measured system response waveform data.
            waveform_params (dict, optional): Additional parameters related to the waveforms.

        Raises:
            ValueError: If the specified axis is not in the calibrator's configured axes.
        """
        if axis not in self.gradient_axes:
            raise ValueError(f"Axis '{axis}' is not configured for this calibrator. Configured axes: {self.gradient_axes}")

        self.input_waveforms[axis] = np.asarray(input_waveform)
        self.measured_responses[axis] = np.asarray(measured_response)
        if waveform_params:
            self.waveform_params[axis] = waveform_params

        print(f"Measurement data stored for axis '{axis}'. Input length: {len(self.input_waveforms[axis])}, Response length: {len(self.measured_responses[axis])}")

    def compute_girf_spectrum(self, axis, regularization_factor=1e-6):
        """
        Computes the GIRF spectrum for the specified axis.
        GIRF(omega) = FFT(Measured_Response(t)) / FFT(Input_Waveform(t))

        Args:
            axis (str): The gradient axis for which to compute the GIRF.
            regularization_factor (float): Small value added to the denominator
                                           to prevent division by zero.

        Returns:
            np.array: The computed GIRF spectrum (complex-valued).

        Raises:
            ValueError: If input or measured response data for the axis is missing.
        """
        if axis not in self.input_waveforms or axis not in self.measured_responses:
            raise ValueError(f"Input waveform or measured response data not found for axis '{axis}'. Please run measure_girf() first.")

        input_wf = self.input_waveforms[axis]
        measured_resp = self.measured_responses[axis]

        if len(input_wf) == 0 or len(measured_resp) == 0:
            raise ValueError(f"Waveform data for axis '{axis}' is empty.")

        if len(input_wf) != len(measured_resp):
            # For simplicity, we require them to be the same length.
            # One might implement padding/truncation if necessary.
            raise ValueError(f"Input waveform and measured response for axis '{axis}' must have the same length. "
                             f"Got {len(input_wf)} and {len(measured_resp)}.")

        print(f"Computing GIRF spectrum for axis '{axis}'...")

        # Perform Fourier Transform
        fft_input = np.fft.fft(input_wf)
        fft_measured = np.fft.fft(measured_resp)

        # Compute GIRF spectrum with regularization
        # Add regularization_factor to the magnitude of fft_input before division
        denominator = fft_input + regularization_factor
        # A more robust regularization might involve checking where abs(fft_input) is small
        # For example: denominator = fft_input.copy(); small_indices = np.abs(fft_input) < regularization_factor; denominator[small_indices] = regularization_factor

        girf_spectrum = fft_measured / denominator

        self.girf_spectra[axis] = girf_spectrum
        print(f"GIRF spectrum computed for axis '{axis}'. Length: {len(girf_spectrum)}")
        return girf_spectrum

    def save_calibration(self, file_path):
        """
        Saves the computed GIRF spectra to a file (JSON format).
        Converts numpy arrays to lists for JSON serialization.

        Args:
            file_path (str): The path to the file where calibration data will be saved.

        Raises:
            IOError: If there's an error writing the file.
        """
        if not self.girf_spectra:
            print("Warning: No GIRF spectra computed yet. Saving an empty calibration.")
            data_to_save = {}
        else:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = {
                axis: spectrum.tolist() if isinstance(spectrum, np.ndarray) else spectrum
                for axis, spectrum in self.girf_spectra.items()
            }

        calibration_data = {
            "gradient_axes": self.gradient_axes,
            "girf_spectra_complex": data_to_save, # Store complex numbers as [real, imag] pairs or similar
            "waveform_params": self.waveform_params # Also save waveform params if any
        }

        # Custom JSON encoder for complex numbers if arrays are not already converted to list of lists for complex parts
        class ComplexEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.complexfloating):
                    return [obj.real, obj.imag]
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)

        print(f"Saving GIRF calibration data to {file_path}...")
        try:
            with open(file_path, 'w') as f:
                json.dump(calibration_data, f, indent=4, cls=ComplexEncoder)
            print("GIRF calibration saved successfully.")
        except IOError as e:
            print(f"Error saving calibration data to {file_path}: {e}")
            raise

    def load_calibration(self, file_path):
        """
        Loads GIRF spectra from a file (JSON format) into self.girf_spectra.
        Converts lists back to numpy arrays.

        Args:
            file_path (str): The path to the file from which to load calibration data.

        Raises:
            IOError: If there's an error reading the file.
            ValueError: If the file format is incorrect.
        """
        print(f"Loading GIRF calibration data from {file_path}...")
        try:
            with open(file_path, 'r') as f:
                calibration_data = json.load(f)
        except IOError as e:
            print(f"Error loading calibration data from {file_path}: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in {file_path}") from e

        if "girf_spectra_complex" not in calibration_data or "gradient_axes" not in calibration_data:
            raise ValueError(f"File {file_path} does not contain expected 'girf_spectra_complex' or 'gradient_axes' keys.")

        self.gradient_axes = calibration_data.get("gradient_axes", ['x', 'y', 'z'])
        loaded_spectra_raw = calibration_data["girf_spectra_complex"]

        self.girf_spectra = {}
        for axis, spectrum_data in loaded_spectra_raw.items():
            if axis not in self.gradient_axes:
                print(f"Warning: Axis '{axis}' from file is not in configured axes. It will be added.")
                if axis not in self.gradient_axes : self.gradient_axes.append(axis)

            # Handle complex numbers stored as [real, imag] pairs by the custom encoder
            if spectrum_data and isinstance(spectrum_data[0], list) and len(spectrum_data[0]) == 2:
                 self.girf_spectra[axis] = np.array([complex(item[0], item[1]) for item in spectrum_data])
            elif isinstance(spectrum_data, list): # If it was stored as a simple list (e.g. if it was purely real)
                 self.girf_spectra[axis] = np.array(spectrum_data)
            else:
                raise ValueError(f"Unsupported spectrum format for axis '{axis}' in {file_path}")

        self.waveform_params = calibration_data.get("waveform_params", {}) # Load params if present
        print(f"GIRF calibration loaded successfully for axes: {list(self.girf_spectra.keys())}.")


if __name__ == '__main__':
    print("--- Running GIRFCalibrator Example with Actual Logic ---")

    # 1. Initialize Calibrator
    calibrator = GIRFCalibrator(gradient_axes=['Gx', 'Gy']) # Example with Gx, Gy

    # 2. Simulate Measurement Data for Gx
    sampling_rate = 1e6  # 1 MHz
    duration = 1e-3    # 1 ms
    num_points = int(sampling_rate * duration)
    time_vector = np.linspace(0, duration, num_points, endpoint=False)

    # Input: a simple sine wave for Gx
    freq_input_gx = 5e3 # 5 kHz
    input_gx = np.sin(2 * np.pi * freq_input_gx * time_vector)

    # Simulated Measured Response: attenuated and phase-shifted version of input
    attenuation_gx = 0.8
    phase_shift_gx = np.pi / 4 # 45 degrees
    # Simulate a simple linear system effect in time domain (for placeholder response)
    # A real measurement would be from the scanner.
    # This is NOT how a real GIRF would make a system respond, but it's for testing the class.
    measured_gx = attenuation_gx * np.sin(2 * np.pi * freq_input_gx * time_vector - phase_shift_gx)
    # Add some noise
    measured_gx += 0.05 * np.random.normal(size=num_points)

    calibrator.measure_girf('Gx', input_gx, measured_gx, waveform_params={"freq_input_hz": freq_input_gx})

    # 3. Compute GIRF Spectrum for Gx
    try:
        girf_spectrum_gx = calibrator.compute_girf_spectrum('Gx', regularization_factor=1e-5)
        print(f"Computed GIRF Spectrum for Gx (first 5 points): {girf_spectrum_gx[:5]}")
        # The spectrum should ideally represent the system's transfer function (attenuation/phase shift)
        # For the simple sine wave input, the most relevant point is at freq_input_gx.
        # fft_freqs = np.fft.fftfreq(num_points, d=1/sampling_rate)
        # target_idx = np.argmin(np.abs(fft_freqs - freq_input_gx))
        # print(f"GIRF at {freq_input_gx} Hz (approx): {girf_spectrum_gx[target_idx]}")
        # print(f"Expected (approx based on simple model): Gain={attenuation_gx}, Phase={-phase_shift_gx} rad")

    except ValueError as e:
        print(f"Error computing GIRF for Gx: {e}")

    # 4. Save Calibration
    calibration_file = "dummy_girf_calibration_actual.json"
    try:
        calibrator.save_calibration(calibration_file)
    except Exception as e:
        print(f"Error saving calibration: {e}")

    # 5. Load Calibration into a new instance
    print("\n--- Testing Load Calibration ---")
    new_calibrator = GIRFCalibrator(gradient_axes=['Gx']) # Can start with minimal axes
    try:
        new_calibrator.load_calibration(calibration_file)
        print(f"Loaded GIRF spectra axes: {list(new_calibrator.girf_spectra.keys())}")
        if 'Gx' in new_calibrator.girf_spectra:
            # print(f"Loaded Gx GIRF spectrum (first 5 points): {new_calibrator.girf_spectra['Gx'][:5]}")
            # Compare if it's close to the original one
            if np.allclose(new_calibrator.girf_spectra['Gx'], girf_spectrum_gx):
                print("Loaded Gx spectrum matches the original computed spectrum.")
            else:
                print("Warning: Loaded Gx spectrum differs from original.")
        if new_calibrator.waveform_params.get('Gx'):
            print(f"Loaded Gx waveform params: {new_calibrator.waveform_params['Gx']}")

    except Exception as e:
        print(f"Error loading calibration: {e}")

    # Example of trying to compute for an unmeasured axis
    print("\n--- Testing Error Handling ---")
    try:
        calibrator.compute_girf_spectrum('Gy') # Gy data not measured
    except ValueError as e:
        print(f"Caught expected error for 'Gy': {e}")

    print("\n--- GIRFCalibrator Example Finished ---")
