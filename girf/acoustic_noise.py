import numpy as np
# from scipy.fft import fft, fftfreq # Not used in this simplified version yet

def estimate_acoustic_noise_level(gradient_waveforms_dict, dt, weighting_factors=None):
    """
    Estimates a conceptual acoustic noise metric based on gradient waveform characteristics.
    This is a highly simplified model and does not predict actual dBA levels.
    It provides a relative metric based on RMS slew rate and peak gradient magnitudes.

    Args:
        gradient_waveforms_dict (dict): Dictionary of {axis_name: waveform_array (T/m)}.
                                        Waveforms are 1D NumPy arrays.
        dt (float): Time step of the gradient waveforms (s).
        weighting_factors (dict, optional): Placeholder for future weighting factors.
            Example: {'slew_rate_contribution': 0.7, 'gradient_amplitude_contribution': 0.3,
                      'frequency_weighting_func': callable}.
            Currently, fixed internal weights C1, C2 are used for simplicity.

    Returns:
        dict: {'estimated_noise_metric': float,
               'details': {
                   'overall_rms_slew_rate_T_per_m_per_s': float,
                   'overall_max_gradient_T_per_m': float,
                   'per_axis_metrics': {
                       axis: {'max_slew_T_per_m_per_s': float, 'max_grad_T_per_m': float}
                   }
               }}
    """
    if not isinstance(gradient_waveforms_dict, dict) or not gradient_waveforms_dict:
        raise ValueError("gradient_waveforms_dict must be a non-empty dictionary.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if weighting_factors is None: # Use default internal factors for this conceptual model
        # These factors are arbitrary for this simplified model.
        C1_slew = 0.8
        C2_grad = 0.2
    else:
        # Allow user to specify if they have a model, but keep it simple for now.
        C1_slew = weighting_factors.get('slew_rate_contribution', 0.8)
        C2_grad = weighting_factors.get('gradient_amplitude_contribution', 0.2)
        # frequency_weighting_func = weighting_factors.get('frequency_weighting_func')
        # if frequency_weighting_func:
        #     print("Note: Frequency weighting is defined but not yet implemented in this conceptual model.")


    per_axis_metrics = {}
    all_slew_rate_magnitudes_squared = [] # To collect (SR_x^2 + SR_y^2 + SR_z^2) at each time point
    all_gradient_magnitudes = [] # To find overall max gradient

    num_time_points = -1

    # Calculate per-axis metrics and collect data for combined metrics
    for axis, grad_wf in gradient_waveforms_dict.items():
        if not isinstance(grad_wf, np.ndarray) or grad_wf.ndim != 1:
            raise ValueError(f"Waveform for axis '{axis}' must be a 1D NumPy array.")

        if num_time_points == -1:
            num_time_points = len(grad_wf)
        elif len(grad_wf) != num_time_points:
            raise ValueError("All gradient waveforms must have the same length.")

        if num_time_points == 0: # Empty waveform
            per_axis_metrics[axis] = {'max_slew_T_per_m_per_s': 0.0, 'max_grad_T_per_m': 0.0, 'slew_rate_wf_T_per_m_per_s': np.array([])}
            all_gradient_magnitudes.append(0.0) # For overall_max_grad calculation
            continue

        # Calculate slew rate: sr[n] = (g[n] - g[n-1]) / dt, sr[0] = g[0]/dt
        slew_rate_wf = np.zeros_like(grad_wf)
        if num_time_points > 0 : slew_rate_wf[0] = grad_wf[0] / dt
        if num_time_points > 1 : slew_rate_wf[1:] = np.diff(grad_wf) / dt

        per_axis_metrics[axis] = {
            'max_slew_T_per_m_per_s': np.max(np.abs(slew_rate_wf)) if slew_rate_wf.size > 0 else 0.0,
            'max_grad_T_per_m': np.max(np.abs(grad_wf)) if grad_wf.size > 0 else 0.0,
            'slew_rate_wf_T_per_m_per_s': slew_rate_wf # Store for combined RMS calculation
        }
        all_gradient_magnitudes.append(per_axis_metrics[axis]['max_grad_T_per_m'])

    if num_time_points <= 0: # All waveforms were empty
        return {
            'estimated_noise_metric': 0.0,
            'details': {
                'overall_rms_slew_rate_T_per_m_per_s': 0.0,
                'overall_max_gradient_T_per_m': 0.0,
                'per_axis_metrics': per_axis_metrics,
                'notes': "Input waveforms were empty."
            }
        }

    # Calculate overall RMS slew rate (vector magnitude RMS)
    # Sum of squares of slew rates from all axes at each time point
    sum_sr_sq_per_point = np.zeros(num_time_points)
    for axis in gradient_waveforms_dict.keys(): # Iterate using same keys to ensure order if it matters
        if axis in per_axis_metrics and per_axis_metrics[axis]['slew_rate_wf_T_per_m_per_s'].size > 0 :
            sum_sr_sq_per_point += per_axis_metrics[axis]['slew_rate_wf_T_per_m_per_s']**2

    # RMS of the vector magnitude of slew rates
    # vector_slew_magnitude_per_point = np.sqrt(sum_sr_sq_per_point)
    # overall_rms_slew_rate = np.sqrt(np.mean(vector_slew_magnitude_per_point**2))
    # This simplifies to RMS of sum_sr_sq_per_point if N_axes is not factored in.
    # A more common RMS for vector magnitude: sqrt( mean( SRx^2 + SRy^2 + SRz^2 ) )
    overall_rms_slew_rate = np.sqrt(np.mean(sum_sr_sq_per_point))

    overall_max_gradient = np.max(all_gradient_magnitudes) if all_gradient_magnitudes else 0.0

    # Simplified conceptual noise metric
    # Note: Units of overall_rms_slew_rate (T/m/s) and overall_max_gradient (T/m) are different.
    # This linear combination is a very high-level proxy. Normalization might be needed if C1, C2 are not tuned.
    estimated_noise_metric = (C1_slew * overall_rms_slew_rate +
                              C2_grad * overall_max_gradient)

    # Clean up slew_rate_wf from details before returning if too large
    for axis in per_axis_metrics:
        del per_axis_metrics[axis]['slew_rate_wf_T_per_m_per_s']


    return {
        'estimated_noise_metric': float(estimated_noise_metric),
        'details': {
            'overall_rms_slew_rate_T_per_m_per_s': float(overall_rms_slew_rate),
            'overall_max_gradient_T_per_m': float(overall_max_gradient),
            'per_axis_metrics': per_axis_metrics,
            'weighting_factors_used': {'C1_slew': C1_slew, 'C2_grad': C2_grad}
        }
    }

# Placeholder for a simplified A-weighting like function - NOT USED in current estimate
# def simplified_a_weighting(frequencies_hz):
#     """ Very coarse A-weighting approximation. Returns 1.0 for simplicity. """
#     # A real A-weighting curve is complex:
#     # R_A(f) = (12194^2 * f^4) / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) * (f^2 + 12194^2))
#     # A(f) = 20 * log10(R_A(f)) + 2.00 (dB)
#     # For this placeholder, let's assume a very simple band-pass like behavior if needed.
#     weights = np.ones_like(frequencies_hz, dtype=float)
#     # Example: attenuate low and very high frequencies if desired
#     # weights[np.abs(frequencies_hz) < 500] *= 0.5 # Attenuate below 500 Hz
#     # weights[np.abs(frequencies_hz) > 10000] *= 0.5 # Attenuate above 10 kHz
#     return weights


if __name__ == '__main__':
    print("--- Running girf.acoustic_noise.py example simulations ---")

    dt_val = 4e-6  # 4 us
    num_pts = 2048 # Longer waveform for more interesting spectrum if we were to use it
    time_vec = np.arange(num_pts) * dt_val

    # Sample Gradient Waveforms
    # X-axis: Fast switching, high slew rate
    grad_x = np.zeros(num_pts)
    for i in range(50, num_pts - 50, 100):
        grad_x[i : i+25] = 0.035 # 35 mT/m
        grad_x[i+25 : i+50] = -0.035

    # Y-axis: Slower, smoother sine wave
    grad_y = 0.02 * np.sin(2 * np.pi * 500 * time_vec) # 20 mT/m, 500 Hz

    # Z-axis: Relatively quiet
    grad_z = 0.005 * np.sin(2 * np.pi * 100 * time_vec) # 5 mT/m, 100 Hz

    sample_gradients = {'x': grad_x, 'y': grad_y, 'z': grad_z}

    print(f"\n--- Estimating Acoustic Noise (dt = {dt_val*1e6} us) ---")

    # Default weighting
    noise_estimate_default = estimate_acoustic_noise_level(sample_gradients, dt_val)
    print("\nDefault Weighting:")
    print(f"  Estimated Noise Metric: {noise_estimate_default['estimated_noise_metric']:.2f} (arbitrary units)")
    print(f"  Details:")
    print(f"    Overall RMS Slew Rate: {noise_estimate_default['details']['overall_rms_slew_rate_T_per_m_per_s']:.1f} T/m/s")
    print(f"    Overall Max Gradient: {noise_estimate_default['details']['overall_max_gradient_T_per_m']*1000:.1f} mT/m")
    for axis, metrics in noise_estimate_default['details']['per_axis_metrics'].items():
        print(f"    Axis {axis}: Max Grad={metrics['max_grad_T_per_m']*1000:.1f} mT/m, Max Slew={metrics['max_slew_T_per_m_per_s']:.1f} T/m/s")

    # Custom weighting (conceptual, as current model uses fixed C1, C2 internally)
    custom_weights = {'slew_rate_contribution': 0.9, 'gradient_amplitude_contribution': 0.1}
    noise_estimate_custom = estimate_acoustic_noise_level(sample_gradients, dt_val, weighting_factors=custom_weights)
    print("\nCustom Weighting (Conceptual - current model uses fixed internal factors):")
    print(f"  Estimated Noise Metric: {noise_estimate_custom['estimated_noise_metric']:.2f} (arbitrary units)")
    print(f"  (Note: Actual factors used: {noise_estimate_custom['details']['weighting_factors_used']})")

    # Example with an empty waveform to test robustness
    empty_gradients = {'x': np.array([])}
    noise_empty = estimate_acoustic_noise_level(empty_gradients, dt_val)
    print("\nWith Empty Waveform:")
    print(f"  Estimated Noise Metric: {noise_empty['estimated_noise_metric']:.2f}")
    print(f"  Details Notes: {noise_empty['details'].get('notes')}")

    print("\n--- girf.acoustic_noise.py example simulations finished ---")
