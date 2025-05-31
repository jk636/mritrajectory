import numpy as np
from scipy.signal import find_peaks, iirnotch, savgol_filter
from scipy.ndimage import gaussian_filter1d

def analyze_nominal_gradient_spectrum(nominal_gradient_waveform, dt, sensitive_harmonics_info,
                                      analysis_threshold_factor=0.1, fft_scale_factor=1.0):
    """
    Analyzes the spectrum of a nominal gradient waveform to see if it has significant energy
    at frequencies identified as sensitive (e.g., from GIRF residual harmonics).

    Args:
        nominal_gradient_waveform (np.ndarray): 1D array of a single axis gradient (T/m).
        dt (float): Time step for the gradient waveform (s).
        sensitive_harmonics_info (list of dict): Information about sensitive harmonics.
            Each dict should have at least {'freq_hz': float, 'amp_sensitivity': float}.
            'amp_sensitivity' is the amplitude of the harmonic found in GIRF residuals,
            indicating system resonance or non-linear response at that frequency.
        analysis_threshold_factor (float): Factor to determine if nominal gradient energy
            at a sensitive frequency is significant. Comparison is:
            nominal_amp_at_freq > amp_sensitivity * analysis_threshold_factor.
        fft_scale_factor (float): Scaling factor for FFT magnitudes if needed (e.g., 2/N for one-sided spectrum physical units).
                                   Default 1.0 means direct FFT magnitude.

    Returns:
        list of dict: Analysis results for each sensitive harmonic.
            [{'freq_hz': f, 'nominal_amp_at_freq': amp_found,
              'sensitivity_at_freq': s, 'is_potentially_excited': True/False,
              'threshold_value': calculated_threshold}, ...]
    """
    if not isinstance(nominal_gradient_waveform, np.ndarray) or nominal_gradient_waveform.ndim != 1:
        raise ValueError("nominal_gradient_waveform must be a 1D NumPy array.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not sensitive_harmonics_info:
        return [] # No sensitive frequencies to analyze

    n_points = len(nominal_gradient_waveform)
    if n_points == 0: return []

    grad_fft_complex = np.fft.fft(nominal_gradient_waveform)
    grad_fft_magnitude = np.abs(grad_fft_complex) * fft_scale_factor / n_points # Scale to be like amplitude of sinewave components
                                                                       # (2/N for positive freqs, 1/N for DC/Nyquist)
                                                                       # For simplicity, using 1/N for all, or user provides scale_factor.
                                                                       # Let's use 2/N for positive, 1/N for DC for better physical meaning.
    if n_points > 0:
        grad_fft_magnitude[0] /= 2 # Correct DC component scaling (was scaled by 2/N, should be 1/N)
        if n_points % 2 == 0: # Nyquist frequency if N is even
            grad_fft_magnitude[n_points//2] /=2 # Correct Nyquist scaling

    freqs_hz = np.fft.fftfreq(n_points, d=dt)

    analysis_results = []

    for harmonic_info in sensitive_harmonics_info:
        h_freq = harmonic_info.get('freq_hz')
        h_amp_sensitivity = harmonic_info.get('amp_sensitivity') # This is from GIRFCalibrator harmonic['complex_value'] (abs)
                                                                # which was residual_peak_fft / N_calib
                                                                # So it's already an amplitude.

        if h_freq is None or h_amp_sensitivity is None:
            print(f"Warning: Skipping sensitive harmonic info due to missing keys: {harmonic_info}")
            continue

        # Find magnitude of nominal gradient spectrum at/near h_freq
        # Consider only positive frequencies for nominal spectrum magnitude matching
        if h_freq == 0: # DC component
            idx_at_h_freq = 0
        else: # Positive frequency
            positive_freq_indices = np.where(freqs > 0)[0]
            if not len(positive_freq_indices): # Should not happen if n_points > 1
                 nominal_amp_at_h_freq = 0
            else:
                freqs_positive = freqs[positive_freq_indices]
                grad_fft_mag_positive = grad_fft_magnitude[positive_freq_indices]
                # Find closest positive frequency bin
                idx_in_positive = np.argmin(np.abs(freqs_positive - h_freq))
                # Check if this frequency is reasonably close, otherwise amplitude is likely from sidelobe
                if np.isclose(freqs_positive[idx_in_positive], h_freq, rtol=0.05): # Allow 5% tolerance for freq match
                    nominal_amp_at_h_freq = grad_fft_mag_positive[idx_in_positive]
                else:
                    nominal_amp_at_h_freq = 0.0 # Freq not directly represented, or use interpolation

        threshold_value = h_amp_sensitivity * analysis_threshold_factor
        is_excited = nominal_amp_at_h_freq > threshold_value

        analysis_results.append({
            'freq_hz': h_freq,
            'nominal_amp_at_freq': nominal_amp_at_h_freq,
            'sensitivity_at_freq': h_amp_sensitivity,
            'threshold_value': threshold_value,
            'is_potentially_excited': is_excited
        })

    return analysis_results


def suggest_notch_filter_params(frequency_to_notch_hz, sampling_rate_hz, q_factor=30.0, depth_db=None):
    """
    Suggests parameters for a notch filter and optionally provides filter coefficients.
    Note: `scipy.signal.iirnotch` does not use `depth_db`. It creates a deep notch.
          To control depth, a custom filter design or different function would be needed.

    Args:
        frequency_to_notch_hz (float): Center frequency for the notch (Hz). Must be > 0.
        sampling_rate_hz (float): Sampling rate of the signal to be filtered (Hz).
        q_factor (float): Quality factor ( F_center / Bandwidth ).
        depth_db (float, optional): Desired attenuation (dB, negative). Not used by iirnotch.

    Returns:
        dict: {'type': 'notch', 'center_freq_hz': ..., 'q_factor': ...,
               'bandwidth_hz': ..., 'scipy_coeffs': (b,a) or None}
    """
    if frequency_to_notch_hz <= 0:
        raise ValueError("Notch frequency must be positive.")
    if frequency_to_notch_hz >= sampling_rate_hz / 2:
        raise ValueError("Notch frequency must be less than Nyquist frequency (sampling_rate_hz / 2).")

    bandwidth_hz = frequency_to_notch_hz / q_factor

    try:
        b, a = iirnotch(frequency_to_notch_hz, q_factor, fs=sampling_rate_hz)
        coeffs = (b.tolist(), a.tolist()) # Store as lists for JSON friendliness if needed
    except Exception as e:
        print(f"Warning: Could not generate scipy.signal.iirnotch coefficients: {e}")
        coeffs = None

    return {
        'type': 'notch',
        'center_freq_hz': frequency_to_notch_hz,
        'q_factor': q_factor,
        'bandwidth_hz': bandwidth_hz,
        'depth_db_requested': depth_db, # Store requested, even if not used by iirnotch
        'scipy_coeffs': coeffs
    }


def evaluate_waveform_smoothing(gradient_waveform, dt, smoothing_type='gaussian', params=None):
    """
    Applies smoothing to a gradient waveform and evaluates its effect on slew rate and spectrum.

    Args:
        gradient_waveform (np.ndarray): 1D array of a single axis gradient (T/m).
        dt (float): Time step (s).
        smoothing_type (str): 'gaussian' or 'savitzky_golay'.
        params (dict, optional): Parameters for the smoother.
            - For 'gaussian': {'sigma_ms': float} (sigma in milliseconds).
            - For 'savitzky_golay': {'window_length_ms': float, 'polyorder': int}.

    Returns:
        dict: {'original_max_slew': ..., 'smoothed_max_slew': ...,
               'smoothed_waveform': ..., 'reduction_metrics': {...}}
    """
    if not isinstance(gradient_waveform, np.ndarray) or gradient_waveform.ndim != 1:
        raise ValueError("gradient_waveform must be a 1D NumPy array.")
    if dt <= 0: raise ValueError("dt must be positive.")
    if params is None: params = {}

    # Calculate original max slew rate
    original_slew_rates = np.diff(gradient_waveform, prepend=gradient_waveform[0]) / dt
    original_slew_rates[0] = gradient_waveform[0]/dt # More accurate first point
    original_max_slew = np.max(np.abs(original_slew_rates))

    smoothed_waveform = None
    if smoothing_type == 'gaussian':
        sigma_ms = params.get('sigma_ms', 1.0) # Default 1ms sigma
        sigma_samples = sigma_ms / (dt * 1000.0) # Convert sigma from ms to samples
        if sigma_samples < 0: raise ValueError("sigma_ms must be non-negative.")
        smoothed_waveform = gaussian_filter1d(gradient_waveform, sigma=sigma_samples)
    elif smoothing_type == 'savitzky_golay':
        window_length_ms = params.get('window_length_ms', 5.0) # Default 5ms window
        polyorder = params.get('polyorder', 2)

        window_samples = int(window_length_ms / (dt * 1000.0))
        if window_samples <= polyorder: # window must be > polyorder
            window_samples = polyorder + 1
        if window_samples % 2 == 0: # window must be odd
            window_samples += 1
        if window_samples <= 0 : raise ValueError("Window length must be positive.")

        smoothed_waveform = savgol_filter(gradient_waveform, window_samples, polyorder)
    else:
        raise ValueError(f"Unsupported smoothing_type: {smoothing_type}")

    # Calculate smoothed max slew rate
    smoothed_slew_rates = np.diff(smoothed_waveform, prepend=smoothed_waveform[0]) / dt
    smoothed_slew_rates[0] = smoothed_waveform[0]/dt
    smoothed_max_slew = np.max(np.abs(smoothed_slew_rates))

    # Spectral analysis (conceptual: energy reduction at high frequencies)
    n_points = len(gradient_waveform)
    freqs = np.fft.fftfreq(n_points, d=dt)

    original_fft_mag = np.abs(np.fft.fft(gradient_waveform))
    smoothed_fft_mag = np.abs(np.fft.fft(smoothed_waveform))

    # Example metric: ratio of energy above Nyquist/4
    high_freq_cutoff_idx = np.where(np.abs(freqs) > (freqs.max() / 4))[0] # Consider positive and negative
    if len(high_freq_cutoff_idx) > 0:
        original_hf_energy = np.sum(original_fft_mag[high_freq_cutoff_idx]**2)
        smoothed_hf_energy = np.sum(smoothed_fft_mag[high_freq_cutoff_idx]**2)
        hf_energy_reduction_ratio = (smoothed_hf_energy / original_hf_energy) if original_hf_energy > 1e-9 else 1.0
    else:
        hf_energy_reduction_ratio = 1.0 # No high frequencies to compare or too short signal

    return {
        'original_waveform': gradient_waveform,
        'smoothed_waveform': smoothed_waveform,
        'original_max_slew_T_per_m_per_s': original_max_slew,
        'smoothed_max_slew_T_per_m_per_s': smoothed_max_slew,
        'reduction_metrics': {
            'slew_reduction_factor': smoothed_max_slew / original_max_slew if original_max_slew > 1e-9 else 1.0,
            'high_freq_energy_reduction_ratio': hf_energy_reduction_ratio
        }
    }


if __name__ == '__main__':
    print("--- Running girf.harmonics.py example simulations ---")
    dt_test = 4e-6  # 4 us
    num_samples_test = 1024
    time_test = np.arange(num_samples_test) * dt_test

    # 1. Create a sample gradient waveform (e.g., a sharp pulse + some oscillation)
    grad_wf = np.zeros(num_samples_test)
    grad_wf[100:150] = 0.02 # 20 mT/m pulse
    # Add some high-frequency oscillation (potential harmonic source or target for smoothing)
    grad_wf += 0.002 * np.sin(2 * np.pi * 50e3 * time_test) # 50 kHz oscillation
    grad_wf += 0.001 * np.sin(2 * np.pi * 120e3 * time_test) # 120 kHz oscillation

    # 2. Define some sensitive harmonic info (as if from GIRFCalibrator)
    # Assume GIRFCalibrator output 'complex_value' is Xk/N, so abs() is amplitude A/2 for A*cos
    # For 'amp_sensitivity' here, let's use that A/2 value.
    sensitive_harmonics = [
        {'freq_hz': 50000.0, 'amp_sensitivity': 0.001}, # Matches one oscillation, sensitivity is A/2
        {'freq_hz': 100000.0, 'amp_sensitivity': 0.0005} # A frequency not strongly in grad_wf
    ]

    print("\n--- Testing analyze_nominal_gradient_spectrum ---")
    analysis = analyze_nominal_gradient_spectrum(grad_wf, dt_test, sensitive_harmonics,
                                                 analysis_threshold_factor=0.5, fft_scale_factor=2.0) # Using 2/N for one-sided
    for item in analysis:
        print(f"  Freq: {item['freq_hz']/1e3:.1f} kHz, Nom.Amp: {item['nominal_amp_at_freq']:.2e}, "
              f"Sensitivity: {item['sensitivity_at_freq']:.2e}, Threshold: {item['threshold_value']:.2e}, "
              f"Excited: {item['is_potentially_excited']}")

    # 3. Suggest notch filter for an excited frequency
    print("\n--- Testing suggest_notch_filter_params ---")
    excited_freq_info = next((item for item in analysis if item['is_potentially_excited']), None)
    if excited_freq_info:
        notch_params = suggest_notch_filter_params(excited_freq_info['freq_hz'],
                                                 sampling_rate_hz=1/dt_test, q_factor=20)
        print(f"  Suggested Notch Filter for {excited_freq_info['freq_hz']/1e3:.1f} kHz:")
        print(f"    Q-factor: {notch_params['q_factor']}, Bandwidth: {notch_params['bandwidth_hz']/1e3:.2f} kHz")
        if notch_params['scipy_coeffs']:
            print(f"    SciPy b coeffs: {[f'{x:.3f}' for x in notch_params['scipy_coeffs'][0]]}")
            # print(f"    SciPy a coeffs: {['%.3f' % x for x in notch_params['scipy_coeffs'][1]]}")
    else:
        print("  No excited frequencies found to suggest a notch filter for.")

    # 4. Evaluate waveform smoothing
    print("\n--- Testing evaluate_waveform_smoothing ---")
    # Gaussian smoothing
    gauss_params = {'sigma_ms': 0.01} # 10 us sigma
    smooth_eval_gauss = evaluate_waveform_smoothing(grad_wf, dt_test,
                                                    smoothing_type='gaussian', params=gauss_params)
    print(f"  Gaussian Smoothing (sigma={gauss_params['sigma_ms']}ms):")
    print(f"    Original Max Slew: {smooth_eval_gauss['original_max_slew_T_per_m_per_s']:.1f} T/m/s")
    print(f"    Smoothed Max Slew: {smooth_eval_gauss['smoothed_max_slew_T_per_m_per_s']:.1f} T/m/s")
    print(f"    Slew Reduction Factor: {smooth_eval_gauss['reduction_metrics']['slew_reduction_factor']:.3f}")
    print(f"    HF Energy Reduction Ratio: {smooth_eval_gauss['reduction_metrics']['high_freq_energy_reduction_ratio']:.3f}")

    # Savitzky-Golay smoothing
    savgol_params = {'window_length_ms': 0.05, 'polyorder': 2} # 50 us window
    smooth_eval_savgol = evaluate_waveform_smoothing(grad_wf, dt_test,
                                                     smoothing_type='savitzky_golay', params=savgol_params)
    print(f"\n  Savitzky-Golay Smoothing (window={savgol_params['window_length_ms']}ms, order={savgol_params['polyorder']}):")
    print(f"    Original Max Slew: {smooth_eval_savgol['original_max_slew_T_per_m_per_s']:.1f} T/m/s")
    print(f"    Smoothed Max Slew: {smooth_eval_savgol['smoothed_max_slew_T_per_m_per_s']:.1f} T/m/s")
    print(f"    Slew Reduction Factor: {smooth_eval_savgol['reduction_metrics']['slew_reduction_factor']:.3f}")
    print(f"    HF Energy Reduction Ratio: {smooth_eval_savgol['reduction_metrics']['high_freq_energy_reduction_ratio']:.3f}")

    print("\n--- girf.harmonics.py example simulations finished ---")
