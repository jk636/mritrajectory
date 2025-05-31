import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve as scipy_convolve

# Default Gyromagnetic ratio for protons in Hz/T
DEFAULT_GAMMA_PROTON = 42.576e6

# --- Trajectory / Gradient / Slew Rate Conversion Utilities ---

def _ensure_numpy_array(data, num_dims_expected=None, name="data"):
    """Helper to convert input to NumPy array and validate dimensions."""
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data)
        except Exception as e:
            raise TypeError(f"Could not convert {name} to NumPy array: {e}")

    if num_dims_expected is not None and data.ndim != num_dims_expected:
        raise ValueError(f"{name} expected to have {num_dims_expected} dimensions, but got {data.ndim}.")
    return data

def standardize_trajectory_format(trajectory_data, num_spatial_dims=None, target_format='array', default_axis_names=None):
    """
    Standardizes trajectory/gradient/slew data between dictionary and array formats.

    Args:
        trajectory_data (dict or np.ndarray): Input data.
            If dict: {'axis_name': np.array_1D_waveform, ...}
            If np.ndarray: Shape (num_time_points, num_spatial_dims)
        num_spatial_dims (int, optional): Expected number of spatial dimensions. Validated if provided.
        target_format (str): 'array' or 'dict'.
        default_axis_names (list of str, optional): Used if converting array to dict and specific names needed.
                                                 Defaults to ['x', 'y', 'z'] based on num_spatial_dims.

    Returns:
        dict or np.ndarray: Data in the target_format.

    Raises:
        ValueError: For inconsistent data, unsupported formats, or dimension mismatches.
    """
    if target_format not in ['array', 'dict']:
        raise ValueError("target_format must be 'array' or 'dict'.")

    if isinstance(trajectory_data, dict):
        if not trajectory_data: raise ValueError("Input trajectory dictionary is empty.")

        axis_names = sorted(trajectory_data.keys()) # Consistent order
        if num_spatial_dims is not None and len(axis_names) != num_spatial_dims:
            raise ValueError(f"Dictionary has {len(axis_names)} axes, expected {num_spatial_dims}.")

        first_axis_data = _ensure_numpy_array(trajectory_data[axis_names[0]], 1, f"Data for axis {axis_names[0]}")
        num_time_points = len(first_axis_data)

        for axis in axis_names[1:]:
            axis_data = _ensure_numpy_array(trajectory_data[axis], 1, f"Data for axis {axis}")
            if len(axis_data) != num_time_points:
                raise ValueError("Inconsistent lengths of waveforms in trajectory dictionary.")

        if target_format == 'dict':
            return {k: _ensure_numpy_array(v) for k,v in trajectory_data.items()} # Ensure all values are arrays
        else: # target_format == 'array'
            array_data = np.zeros((num_time_points, len(axis_names)))
            for i, axis in enumerate(axis_names):
                array_data[:, i] = trajectory_data[axis]
            return array_data

    elif isinstance(trajectory_data, (list, np.ndarray)):
        array_data = _ensure_numpy_array(trajectory_data, name="Input trajectory array")
        if array_data.ndim == 1: # Assume it's a single axis waveform
            array_data = array_data[:, np.newaxis] # Convert to (T, 1)
        if array_data.ndim != 2:
            raise ValueError("Input NumPy array must be 2D (num_time_points, num_spatial_dims).")

        current_num_dims = array_data.shape[1]
        if num_spatial_dims is not None and current_num_dims != num_spatial_dims:
            raise ValueError(f"Array has {current_num_dims} dimensions, expected {num_spatial_dims}.")

        if target_format == 'array':
            return array_data
        else: # target_format == 'dict'
            if default_axis_names:
                if len(default_axis_names) != current_num_dims:
                    raise ValueError("Length of default_axis_names must match number of array dimensions.")
                axis_names_to_use = default_axis_names
            else: # Generate default names
                if current_num_dims > 3: # Default x,y,z only up to 3D
                    axis_names_to_use = [f'axis_{i}' for i in range(current_num_dims)]
                else:
                    axis_names_to_use = ['x', 'y', 'z'][:current_num_dims]

            dict_data = {axis_names_to_use[i]: array_data[:, i] for i in range(current_num_dims)}
            return dict_data
    else:
        raise TypeError("trajectory_data must be a dictionary or a NumPy array/list.")


def compute_gradient_waveforms(kspace_trajectory, gamma, dt, output_format='dict', input_format_hint=None):
    """
    Computes gradient waveforms from a k-space trajectory.
    g(t) = dk/dt / gamma. Approximated as g[n] = (k[n] - k[n-1]) / (dt * gamma) for n>0,
    and g[0] = k[0] / (dt*gamma) assuming k starts from 0.

    Args:
        kspace_trajectory (dict or np.ndarray): K-space trajectory data (units: m^-1).
            If dict: {'axis': waveform_1D_m_inv, ...}
            If np.ndarray: (num_time_points, num_spatial_dims)
        gamma (float): Gyromagnetic ratio (Hz/T).
        dt (float): Time step / dwell time (s).
        output_format (str): 'dict' or 'array'.
        input_format_hint (str, optional): 'dict' or 'array' if type ambiguity.

    Returns:
        dict or np.ndarray: Gradient waveforms (units: T/m).
    """
    if dt <= 0: raise ValueError("dt must be positive.")
    if gamma == 0: raise ValueError("gamma cannot be zero.")

    # Standardize to array for computation
    k_array = standardize_trajectory_format(kspace_trajectory, target_format='array')

    grad_array = np.zeros_like(k_array)
    # Grad[0] = (k[0] - 0) / (dt*gamma)
    grad_array[0, :] = k_array[0, :] / (dt * gamma)
    # Grad[i] = (k[i] - k[i-1]) / (dt*gamma) for i > 0
    grad_array[1:, :] = np.diff(k_array, axis=0) / (dt * gamma)

    # Determine original axis names if input was dict, for dict output
    default_axes = None
    if isinstance(kspace_trajectory, dict) or (input_format_hint == 'dict'):
        default_axes = sorted(kspace_trajectory.keys()) if isinstance(kspace_trajectory, dict) else None

    return standardize_trajectory_format(grad_array, target_format=output_format, default_axis_names=default_axes)


def compute_slew_rates(gradient_waveforms, dt, output_format='dict', input_format_hint=None):
    """
    Computes slew rates from gradient waveforms.
    sr(t) = dg/dt. Approximated as sr[n] = (g[n] - g[n-1]) / dt for n>0,
    and sr[0] = g[0] / dt assuming g starts from 0.

    Args:
        gradient_waveforms (dict or np.ndarray): Gradient waveform data (units: T/m).
        dt (float): Time step (s).
        output_format (str): 'dict' or 'array'.
        input_format_hint (str, optional): 'dict' or 'array' if type ambiguity.

    Returns:
        dict or np.ndarray: Slew rate waveforms (units: T/m/s).
    """
    if dt <= 0: raise ValueError("dt must be positive.")

    g_array = standardize_trajectory_format(gradient_waveforms, target_format='array')

    sr_array = np.zeros_like(g_array)
    sr_array[0, :] = g_array[0, :] / dt # Slew to reach g[0] from 0
    sr_array[1:, :] = np.diff(g_array, axis=0) / dt

    default_axes = None
    if isinstance(gradient_waveforms, dict) or (input_format_hint == 'dict'):
        default_axes = sorted(gradient_waveforms.keys()) if isinstance(gradient_waveforms, dict) else None

    return standardize_trajectory_format(sr_array, target_format=output_format, default_axis_names=default_axes)


def integrate_trajectory(gradient_waveforms, gamma, dt, initial_k0=None, output_format='dict', input_format_hint=None):
    """
    Integrates gradient waveforms to obtain a k-space trajectory.
    k(t) = integral(g(tau) * gamma * dtau). Approximated as k[n] = sum(g[i] * dt * gamma).

    Args:
        gradient_waveforms (dict or np.ndarray): Gradient waveform data (T/m).
        gamma (float): Gyromagnetic ratio (Hz/T).
        dt (float): Time step (s).
        initial_k0 (dict, np.ndarray, or float, optional): Initial k-space point(s) (m^-1).
            If float, applied to all axes. If dict/array, must match dimensions. Defaults to zero.
        output_format (str): 'dict' or 'array'.
        input_format_hint (str, optional): 'dict' or 'array' if type ambiguity.

    Returns:
        dict or np.ndarray: K-space trajectory (m^-1).
    """
    if dt <= 0: raise ValueError("dt must be positive.")
    if gamma == 0: raise ValueError("gamma cannot be zero.")

    g_array = standardize_trajectory_format(gradient_waveforms, target_format='array')
    num_time_points, num_dims = g_array.shape

    k_deltas = g_array * (dt * gamma)
    k_array = np.cumsum(k_deltas, axis=0)

    if initial_k0 is not None:
        if isinstance(initial_k0, (int, float)):
            initial_k0_array = np.full(num_dims, float(initial_k0))
        else: # dict or array
            initial_k0_standardized = standardize_trajectory_format(initial_k0, num_spatial_dims=num_dims, target_format='array')
            if initial_k0_standardized.shape == (1, num_dims) or initial_k0_standardized.shape == (num_dims,): # single point
                 initial_k0_array = initial_k0_standardized.flatten()
            else: # perhaps full trajectory of initial points? For now, expect single point.
                 raise ValueError("initial_k0, if array, should represent a single starting point (1, N_dims) or (N_dims,).")

        # Adjust trajectory to start from initial_k0.
        # cumsum starts from g[0]*dt*gamma, so k_array[0,:] is the k-space after first step.
        # If initial_k0 is where k should be at time t=0 (BEFORE first grad step),
        # then all points are shifted by initial_k0.
        k_array += initial_k0_array

    default_axes = None
    if isinstance(gradient_waveforms, dict) or (input_format_hint == 'dict'):
        default_axes = sorted(gradient_waveforms.keys()) if isinstance(gradient_waveforms, dict) else None

    return standardize_trajectory_format(k_array, target_format=output_format, default_axis_names=default_axes)


# --- Constraint Checking Utilities ---

def check_gradient_strength(gradient_waveforms, gmax_T_per_m, tolerance=1e-9):
    """Checks if gradient amplitudes exceed Gmax."""
    g_array = standardize_trajectory_format(gradient_waveforms, target_format='array')
    abs_g = np.abs(g_array)
    max_g_found_val = np.max(abs_g)

    is_ok = max_g_found_val <= (gmax_T_per_m + tolerance)

    details = {'max_g_found_T_per_m': float(max_g_found_val), 'gmax_limit_T_per_m': float(gmax_T_per_m)}
    if not is_ok:
        exceeding_indices = np.unravel_index(np.argmax(abs_g), abs_g.shape)
        details['first_exceeding_time_idx'] = int(exceeding_indices[0])
        details['first_exceeding_axis_idx'] = int(exceeding_indices[1])
        details['first_exceeding_value_T_per_m'] = float(abs_g[exceeding_indices])
    return is_ok, float(max_g_found_val), details


def check_slew_rate(slew_rates, smax_T_per_m_per_s, tolerance=1e-9):
    """Checks if slew rates exceed Smax."""
    sr_array = standardize_trajectory_format(slew_rates, target_format='array')
    abs_sr = np.abs(sr_array)
    max_sr_found_val = np.max(abs_sr)

    is_ok = max_sr_found_val <= (smax_T_per_m_per_s + tolerance)

    details = {'max_sr_found_T_per_m_per_s': float(max_sr_found_val), 'smax_limit_T_per_m_per_s': float(smax_T_per_m_per_s)}
    if not is_ok:
        exceeding_indices = np.unravel_index(np.argmax(abs_sr), abs_sr.shape)
        details['first_exceeding_time_idx'] = int(exceeding_indices[0])
        details['first_exceeding_axis_idx'] = int(exceeding_indices[1])
        details['first_exceeding_value_T_per_m_per_s'] = float(abs_sr[exceeding_indices])
    return is_ok, float(max_sr_found_val), details


# --- Waveform Manipulation Utilities ---

def resample_waveform(waveform_1d, target_length, kind='linear'):
    """Resamples a 1D waveform to a target length using interpolation."""
    waveform_1d = _ensure_numpy_array(waveform_1d, 1, "waveform_1d")
    original_length = len(waveform_1d)
    if original_length == target_length:
        return waveform_1d.copy()
    if original_length < 2 : # Need at least 2 points for interp1d
        # If constant value, repeat it. If single point, also repeat.
        # Or based on use case, could be error or zero padding.
        return np.full(target_length, waveform_1d[0] if original_length==1 else 0.0)

    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_length)

    # For complex data, interpolate real and imaginary parts separately
    if np.iscomplexobj(waveform_1d):
        interp_func_real = interp1d(original_indices, waveform_1d.real, kind=kind, fill_value="extrapolate")
        interp_func_imag = interp1d(original_indices, waveform_1d.imag, kind=kind, fill_value="extrapolate")
        resampled_waveform = interp_func_real(target_indices) + 1j * interp_func_imag(target_indices)
    else:
        interp_func = interp1d(original_indices, waveform_1d, kind=kind, fill_value="extrapolate")
        resampled_waveform = interp_func(target_indices)

    return resampled_waveform


def resample_girf_spectra(girf_spectra_dict, target_length_fft, kind='linear'):
    """Resamples all GIRF spectra in a dictionary to a target FFT length."""
    resampled_girf_dict = {}
    for axis, spectrum in girf_spectra_dict.items():
        resampled_girf_dict[axis] = resample_waveform(spectrum, target_length_fft, kind=kind)
    return resampled_girf_dict


# --- Convolution Utility ---

def convolve_signals(signal, kernel, mode='same'):
    """Thin wrapper around scipy.signal.convolve."""
    signal = _ensure_numpy_array(signal, 1, "signal")
    kernel = _ensure_numpy_array(kernel, 1, "kernel")
    return scipy_convolve(signal, kernel, mode=mode)


# --- General Math Utility ---
def absolute_value(x):
    """Computes the absolute value of x (NumPy array, list, or scalar)."""
    return np.abs(np.asarray(x))


if __name__ == '__main__':
    print("--- Running girf.utils.py example simulations ---")
    dt_test = 4e-6  # 4 us
    gamma_test = DEFAULT_GAMMA_PROTON

    # 1. Test standardize_trajectory_format
    print("\n1. Test standardize_trajectory_format:")
    dict_traj = {'x': np.array([1,2,3]), 'y': np.array([4,5,6])}
    array_from_dict = standardize_trajectory_format(dict_traj, target_format='array')
    print(f"  Dict to Array:\n{array_from_dict}")
    dict_from_array = standardize_trajectory_format(array_from_dict, target_format='dict', default_axis_names=['x','y'])
    print(f"  Array to Dict:\n{dict_from_array}")

    # 2. Test k-space <-> gradient <-> slew rate conversions
    print("\n2. Test k-space/gradient/slew conversions (dict format):")
    k_dict = {'Gx': np.array([0.0, 10.0, 20.0, 15.0]), 'Gy': np.array([0.0, 0.0, 5.0, 5.0])} # m^-1

    g_dict = compute_gradient_waveforms(k_dict, gamma_test, dt_test, output_format='dict')
    print(f"  Gradients (T/m) from k_dict:\n  Gx: {g_dict['Gx']}\n  Gy: {g_dict['Gy']}")

    sr_dict = compute_slew_rates(g_dict, dt_test, output_format='dict')
    print(f"  Slew Rates (T/m/s) from g_dict:\n  Gx: {sr_dict['Gx']}\n  Gy: {sr_dict['Gy']}")

    k_recon_dict = integrate_trajectory(g_dict, gamma_test, dt_test, initial_k0={'Gx':0.0, 'Gy':0.0}, output_format='dict')
    print(f"  Reconstructed K-space (m^-1) from g_dict (should be like k_dict):\n  Gx: {k_recon_dict['Gx']}\n  Gy: {k_recon_dict['Gy']}")

    # Test with array format
    print("\n   Test k-space/gradient/slew conversions (array format):")
    k_array_test = standardize_trajectory_format(k_dict, target_format='array')
    g_array_test = compute_gradient_waveforms(k_array_test, gamma_test, dt_test, output_format='array')
    print(f"  Gradients (array) shape: {g_array_test.shape}, Gx[1]={g_array_test[1,0]:.3f}")
    sr_array_test = compute_slew_rates(g_array_test, dt_test, output_format='array')
    print(f"  Slew Rates (array) shape: {sr_array_test.shape}, Gx[1]={sr_array_test[1,0]/1e6:.1f} MT/m/s")


    # 3. Test Constraint Checkers
    print("\n3. Test Constraint Checkers:")
    gmax_limit = 0.05 # T/m
    smax_limit = 200e6 # T/m/s (using large slew for this example grad)

    # Create some gradient data (array format for simplicity in test)
    test_grads_arr = np.array([[0.01, 0.02], [0.06, 0.03], [-0.04, 0.01]]) # T/m
    test_slews_arr = compute_slew_rates(test_grads_arr, dt_test, output_format='array')

    ok_g, max_g, _ = check_gradient_strength(test_grads_arr, gmax_limit)
    print(f"  Grad check (limit {gmax_limit*1000} mT/m): OK={ok_g}, Max Found={max_g*1000:.1f} mT/m")

    ok_sr, max_sr, _ = check_slew_rate(test_slews_arr, smax_limit)
    print(f"  Slew check (limit {smax_limit/1e6} MT/m/s): OK={ok_sr}, Max Found={max_sr/1e6:.1f} MT/m/s")

    # Test with exceeding values
    gmax_tight = 0.03 # T/m
    smax_tight = 50e6  # T/m/s
    ok_g_fail, max_g_f, det_g = check_gradient_strength(test_grads_arr, gmax_tight)
    print(f"  Grad check (limit {gmax_tight*1000} mT/m): OK={ok_g_fail}, Max Found={max_g_f*1000:.1f} mT/m. Details: {det_g if not ok_g_fail else ''}")
    ok_sr_fail, max_sr_f, det_sr = check_slew_rate(test_slews_arr, smax_tight)
    print(f"  Slew check (limit {smax_tight/1e6} MT/m/s): OK={ok_sr_fail}, Max Found={max_sr_f/1e6:.1f} MT/m/s. Details: {det_sr if not ok_sr_fail else ''}")


    # 4. Test Resampling
    print("\n4. Test Resampling:")
    original_wf = np.array([0., 1., 2., 1., 0., 0.5, 1.5])
    resampled_wf = resample_waveform(original_wf, 13) # Resample to 13 points
    print(f"  Original length: {len(original_wf)}, Resampled length: {len(resampled_wf)}")
    # print(f"  Resampled wf: {np.round(resampled_wf, 2)}")

    complex_wf = original_wf + 1j * original_wf[::-1]
    resampled_complex = resample_waveform(complex_wf, 10)
    print(f"  Resampled complex wf length: {len(resampled_complex)}")

    girf_test_dict = {'x': complex_wf, 'y': original_wf.astype(float)}
    resampled_girfs = resample_girf_spectra(girf_test_dict, 20)
    print(f"  Resampled GIRFs: axis 'x' new length {len(resampled_girfs['x'])}, axis 'y' new length {len(resampled_girfs['y'])}")

    # 5. Test Convolution
    print("\n5. Test Convolution:")
    sig = np.repeat([0., 1., 0.], 10)
    kern = np.array([1.,1.,1.]) / 3.
    conv_res = convolve_signals(sig, kern, mode='same')
    # print(f"  Convolved signal (should be smoothed):\n  {np.round(conv_res,2)}")
    print(f"  Convolution result length (same as input signal): {len(conv_res)}")

    # 6. Test Absolute Value
    print("\n6. Test Absolute Value:")
    print(f"  abs(-5.5) = {absolute_value(-5.5)}")
    print(f"  abs([1+2j, -3-4j]) = {absolute_value(np.array([1+2j, -3-4j]))}")

    print("\n--- girf.utils.py example simulations finished ---")
