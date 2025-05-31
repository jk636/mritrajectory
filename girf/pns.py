import numpy as np
from scipy.signal import convolve # For convolution

class PNSModel:
    def __init__(self, pns_thresholds=None, dt=None):
        """
        Initializes the PNSModel.

        Args:
            pns_thresholds (dict, optional): PNS limits and model parameters.
                Example: {'rheobase_T_per_s': 20.0,  # T/s (effective, after spatial consideration)
                          'chronaxie_ms': 0.36,     # ms
                          'max_total_pns_normalized': 0.8} # Max allowed normalized PNS (e.g., 80% of limit)
            dt (float, optional): Time step (sampling interval) in seconds. Essential for nerve response generation.
        """
        default_thresholds = {
            'rheobase_T_per_s': 20.0,  # Effective rheobase in T/s (or T/m/s if slew rate used directly)
            'chronaxie_ms': 0.36,      # Chronaxie in milliseconds
            'max_total_pns_normalized': 0.8 # Target: 80% of perceived stimulation limit
        }
        if pns_thresholds is None:
            self.pns_thresholds = default_thresholds
        else:
            self.pns_thresholds = {**default_thresholds, **pns_thresholds}

        if dt is None:
            # dt is crucial for generating the nerve response filter correctly.
            raise ValueError("'dt' (time step in seconds) must be provided during initialization.")
        self.dt = dt # seconds

        self.slew_rates_history = [] # To store slew_rates_per_axis if needed
        self.pns_values_history = []   # To store computed total_pns_normalized_timeseries

        # Convert chronaxie to seconds for calculations
        self.chronaxie_s = self.pns_thresholds['chronaxie_ms'] / 1000.0 # Convert ms to s

        self.nerve_ir_filter = None # Cached nerve impulse response filter

        print(f"PNSModel initialized. dt={self.dt*1000:.3f} ms, Rheobase={self.pns_thresholds['rheobase_T_per_s']:.1f} T/s (effective), Chronaxie={self.pns_thresholds['chronaxie_ms']:.2f} ms")

    def _generate_nerve_response_function(self, filter_duration_factor=5):
        """
        Generates a nerve impulse response function h(t) = (1/chronaxie) * exp(-t/chronaxie).
        The filter is generated once and cached.

        Args:
            filter_duration_factor (float): Determines filter length relative to chronaxie
                                            (e.g., 5*chronaxie_s).
        Returns:
            np.array: The 1D nerve impulse response filter.
        """
        if self.nerve_ir_filter is not None:
            return self.nerve_ir_filter

        # Duration of the filter: e.g., 5 times chronaxie, or until it decays significantly
        num_filter_points = int(filter_duration_factor * self.chronaxie_s / self.dt)
        if num_filter_points < 1:
            num_filter_points = 1 # Ensure at least one point
            print(f"Warning: PNS filter length is very short ({num_filter_points} point) due to small chronaxie/dt ratio.")

        t_filter = np.arange(num_filter_points) * self.dt # Time vector for the filter

        # Nerve impulse response h(t) = (1/C) * exp(-t/C)
        # This is normalized so that integral(h(t)dt) = 1 (approximately, for long enough duration)
        # The (1/C) factor ensures that for a constant input S over long time, output -> S.
        # So, the output of convolution(SlewRate, h(t)) is an "effective SlewRate" or "perceived SlewRate"
        # which can then be compared to Rheobase.

        # For the convolution, we just need exp(-t/C). The scaling (1/C) can be tricky.
        # Let's use h(t) = exp(-t/C). The result of conv(SR, h(t))*dt will have units SR*time.
        # A common way: filter h(t) = (1/chronaxie_s) * exp(-t / chronaxie_s)
        # Output of convolution: sum( SlewRate[k-m]*h[m] ) * dt_convolution.
        # If scipy.signal.convolve is used, it's sum( SlewRate[k-m]*h[m] ). No dt.

        self.nerve_ir_filter = (1.0 / self.chronaxie_s) * np.exp(-t_filter / self.chronaxie_s)
        # Normalize the filter to have sum=1 if using 'same' mode convolution without dt scaling later.
        # Or, ensure the convolution result is scaled by dt.
        # scipy.signal.convolve(A, B) * dt_A is a common way if B is an IR.
        # If h(t) has unit area, then conv(SR(t), h(t)) has units of SR.
        # Integral of (1/C)exp(-t/C) from 0 to inf is 1. So, sum(filter_coeffs * dt) should be ~1.
        self.nerve_ir_filter *= self.dt # Scale by dt to make sum(filter_coeffs) ~ 1 (discrete integral)
                                        # This makes it an approximation of integral h(t)dt for each tap.
                                        # So output of convolve(SR, filter) is approx. SR_effective.

        # Alternative: if filter is just exp(-t/C), then output_conv = convolve(SR, filter) * (dt/C)
        # Let's stick to filter = (dt/C) * exp(-t/C) for now.
        # This means sum(filter) will be approx 1.

        if np.sum(self.nerve_ir_filter) < 0.1: # Check if filter is too weak due to dt >> chronaxie
            print(f"Warning: PNS nerve filter sum is very small ({np.sum(self.nerve_ir_filter):.3e}). "
                  "This might happen if dt is much larger than chronaxie. PNS prediction might be inaccurate.")

        print(f"Generated nerve response filter of length {num_filter_points} points ({num_filter_points*self.dt*1000:.2f} ms duration). Sum(filter_coeffs)={np.sum(self.nerve_ir_filter):.3f}")
        return self.nerve_ir_filter

    def compute_pns(self, slew_rates_per_axis, axis_weights=None):
        """
        Computes the total normalized PNS activity time series.

        Args:
            slew_rates_per_axis (dict or np.array):
                - If dict: {'axis_name': np.array_slew_waveform_T_per_m_per_s, ...}
                - If np.array: Shape (num_time_points, num_axes). Assumes standard axis order or that
                               axis_weights dict keys can map to columns if provided.
            axis_weights (dict, optional): Relative PNS sensitivity per axis.
                                           {'axis_name': weight, ...}. Defaults to 1.0 for all.

        Returns:
            np.array: Time series of total normalized PNS activity.
        """
        if not isinstance(slew_rates_per_axis, (dict, np.ndarray)):
            raise TypeError("slew_rates_per_axis must be a dictionary or a NumPy array.")

        nerve_filter = self._generate_nerve_response_function()

        processed_slew_rates = {}
        num_time_points = -1

        if isinstance(slew_rates_per_axis, dict):
            for axis, sr_waveform in slew_rates_per_axis.items():
                if not isinstance(sr_waveform, np.ndarray): sr_waveform = np.asarray(sr_waveform)
                processed_slew_rates[axis] = sr_waveform
                if num_time_points == -1: num_time_points = len(sr_waveform)
                elif num_time_points != len(sr_waveform): raise ValueError("Inconsistent slew rate waveform lengths.")
        else: # NumPy array (time_points, num_axes)
            num_time_points, num_axes = slew_rates_per_axis.shape
            # Use generic axis names if no weights provided, or try to match with weights
            for i in range(num_axes):
                axis_name = list(axis_weights.keys())[i] if axis_weights and i < len(axis_weights.keys()) else f'axis_{i}'
                processed_slew_rates[axis_name] = slew_rates_per_axis[:, i]

        if num_time_points == -1 : num_time_points = 0 # Handle empty input

        if axis_weights is None: axis_weights = {}

        sum_sq_weighted_pns_norm = np.zeros(num_time_points + len(nerve_filter) - 1, dtype=np.float64)
        # The convolution output length will be N_sr + N_filter - 1. We should probably use 'same' mode.

        all_axes_pns_activity_normalized = {}

        for axis_name, sr_waveform in processed_slew_rates.items():
            if len(sr_waveform) == 0:
                # If an axis has no slew waveform, its PNS contribution is zero.
                # This might happen if a gradient axis is static.
                pns_activity_axis_normalized = np.zeros(num_time_points, dtype=np.float64)
            else:
                # Convolve slew rate with nerve response filter
                # Output of convolve is sum(SR[k-m]*filter[m]). Units = SR units if sum(filter)=1.
                # mode='full' is default. 'same' might be more convenient.
                effective_dbdt = convolve(sr_waveform, nerve_filter, mode='same')

                # Normalize by rheobase
                pns_activity_axis_normalized = effective_dbdt / self.pns_thresholds['rheobase_T_per_s']

            all_axes_pns_activity_normalized[axis_name] = pns_activity_axis_normalized

            weight = axis_weights.get(axis_name, 1.0) # Default weight is 1.0

            # Accumulate for RSS. Ensure consistent lengths for summation.
            # If using 'same' mode for convolution, lengths match num_time_points.
            if len(pns_activity_axis_normalized) == len(sum_sq_weighted_pns_norm): # Should match if mode='same'
                 sum_sq_weighted_pns_norm += (pns_activity_axis_normalized * weight)**2
            else: # Fallback if lengths don't match (e.g. if mode='full' was used inadvertently)
                target_len = len(sum_sq_weighted_pns_norm)
                current_len = len(pns_activity_axis_normalized)
                if current_len > target_len: # Should not happen with mode='same'
                    pns_activity_axis_normalized = pns_activity_axis_normalized[:target_len]
                elif current_len < target_len and current_len == num_time_points: # if sum_sq initialized too long
                     sum_sq_weighted_pns_norm = sum_sq_weighted_pns_norm[:current_len] # Trim sum_sq

                sum_sq_weighted_pns_norm[:len(pns_activity_axis_normalized)] += (pns_activity_axis_normalized * weight)**2


        total_pns_normalized_timeseries = np.sqrt(sum_sq_weighted_pns_norm)

        self.slew_rates_history.append(processed_slew_rates) # Store the input
        self.pns_values_history.append({"total_normalized": total_pns_normalized_timeseries,
                                        "per_axis_normalized": all_axes_pns_activity_normalized})

        print(f"Computed PNS. Peak total normalized PNS: {np.max(total_pns_normalized_timeseries):.3f} (timeseries length: {len(total_pns_normalized_timeseries)})")
        return total_pns_normalized_timeseries


    def check_limits(self, total_pns_normalized_timeseries=None):
        """
        Checks if computed PNS values exceed predefined limits.

        Args:
            total_pns_normalized_timeseries (np.array, optional): Time series of total normalized PNS.
                If None, uses the last computed PNS result from history.

        Returns:
            tuple: (bool is_compliant, float peak_pns_value)
        """
        if total_pns_normalized_timeseries is None:
            if not self.pns_values_history:
                raise ValueError("No PNS values computed yet. Run compute_pns() first or provide timeseries.")
            # Use the 'total_normalized' from the last entry in history
            total_pns_normalized_timeseries = self.pns_values_history[-1]["total_normalized"]

        if not isinstance(total_pns_normalized_timeseries, np.ndarray) or total_pns_normalized_timeseries.ndim == 0 : # Check if it's an array and not empty
             if isinstance(total_pns_normalized_timeseries, (float, int)): # Scalar value
                  peak_pns_value = float(total_pns_normalized_timeseries)
             else: # Empty or invalid
                  print("Warning: PNS timeseries for limit check is empty or invalid. Assuming non-compliant.")
                  return False, np.inf
        else: # It's an array
            if total_pns_normalized_timeseries.size == 0:
                 print("Warning: PNS timeseries for limit check is empty. Assuming compliant (no stimulation).")
                 return True, 0.0
            peak_pns_value = np.max(total_pns_normalized_timeseries)

        limit = self.pns_thresholds['max_total_pns_normalized']
        is_compliant = peak_pns_value <= limit

        print(f"PNS Limit Check: Peak PNS = {peak_pns_value:.3f}, Limit = {limit:.3f}. Compliant: {is_compliant}")
        return is_compliant, peak_pns_value

    def optimize_slew_rate(self, slew_rates_per_axis, target_pns_limit_factor=None):
        """
        Conceptual: Suggests slew rate scaling if PNS limits are exceeded.
        A real implementation would be an iterative optimization process.

        Args:
            slew_rates_per_axis (dict or np.array): The input slew rates.
            target_pns_limit_factor (float, optional): Fraction of max_total_pns_normalized to target.
                Defaults to self.pns_thresholds['max_total_pns_normalized'] itself.

        Returns:
            dict: {'scaled_slew_rates': dict_of_scaled_slew_rates (or original if compliant),
                   'original_peak_pns': float,
                   'target_pns_limit': float,
                   'scaling_factor': float,
                   'status': str}
        """
        print("Attempting to optimize slew rates (conceptual)...")
        current_pns_ts = self.compute_pns(slew_rates_per_axis) # Uses current axis_weights if any set
        is_compliant, peak_pns = self.check_limits(current_pns_ts)

        if target_pns_limit_factor is None:
            target_limit = self.pns_thresholds['max_total_pns_normalized']
        else: # User provides a factor e.g. 0.9 to aim for 90% of the hard limit
            target_limit = self.pns_thresholds['max_total_pns_normalized'] * target_pns_limit_factor
            # Ensure this user target is not higher than the absolute limit
            target_limit = min(target_limit, self.pns_thresholds['max_total_pns_normalized'])


        scaled_slew_rates = slew_rates_per_axis # By default, return original
        scaling_factor = 1.0
        status = "Compliant"

        if peak_pns > target_limit : # If current peak exceeds the desired target (could be the hard limit or a tighter one)
            status = "Exceeds Target, Scaling Suggested"
            if peak_pns > 0: # Avoid division by zero if peak_pns is somehow zero but still > target_limit (unlikely)
                scaling_factor = target_limit / peak_pns
            else: # peak_pns is zero or negative, but exceeded target (target must be negative, unusual)
                scaling_factor = 1.0 # No change if peak_pns is not positive

            scaling_factor = max(0.0, min(scaling_factor, 1.0)) # Clamp factor between 0 and 1

            print(f"  Peak PNS ({peak_pns:.3f}) > Target ({target_limit:.3f}). Suggesting slew scaling by {scaling_factor:.3f}.")

            if isinstance(slew_rates_per_axis, dict):
                scaled_slew_rates = {
                    axis: sr_vals * scaling_factor
                    for axis, sr_vals in slew_rates_per_axis.items()
                }
            elif isinstance(slew_rates_per_axis, np.ndarray):
                scaled_slew_rates = slew_rates_per_axis * scaling_factor
            else: # Should not happen
                status = "Error: Unknown slew rate data type for scaling."
                scaled_slew_rates = slew_rates_per_axis # Return original on error
                scaling_factor = 1.0
        else:
            status = f"Compliant with Target ({target_limit:.3f})"
            print(f"  Peak PNS ({peak_pns:.3f}) <= Target ({target_limit:.3f}). No scaling needed.")


        return {
            'scaled_slew_rates': scaled_slew_rates,
            'original_peak_pns': peak_pns,
            'target_pns_limit_used': target_limit,
            'scaling_factor_applied': scaling_factor,
            'status': status
        }


if __name__ == '__main__':
    print("--- Running PNSModel Example with Actual Logic ---")

    dt_val = 4e-6  # 4 us
    num_pts = 1000
    time_vector = np.arange(num_pts) * dt_val

    # Example Slew Rates (T/m/s)
    # Axis X: A sharp pulse that might trigger PNS
    slew_x = np.zeros(num_pts)
    slew_x[100:120] = 150  # 150 T/m/s for a short duration (20*4us = 80us)
    slew_x[120:140] = -150

    # Axis Y: A slower, less intense slew rate
    slew_y = np.sin(2 * np.pi * 1000 * time_vector) * 50 # 50 T/m/s @ 1kHz

    # Axis Z: Constant zero slew rate (no PNS contribution)
    slew_z = np.zeros(num_pts)

    slew_rates_data = {
        'Gx': slew_x,
        'Gy': slew_y,
        'Gz': slew_z  # Example with a Z axis
    }

    # Alternative as array:
    # slew_rates_array = np.stack([slew_x, slew_y, slew_z], axis=-1) # (num_pts, 3)

    # Initialize PNSModel
    # Using slightly different thresholds for demo
    custom_thresholds = {
        'rheobase_T_per_s': 25.0,    # Effective Rheobase in T/s (or T/m/s)
        'chronaxie_ms': 0.40,        # Chronaxie in ms
        'max_total_pns_normalized': 0.8 # Target 80% of "perceived limit"
    }
    pns_model = PNSModel(pns_thresholds=custom_thresholds, dt=dt_val)

    # Define axis weights (optional, example values)
    # These depend on body part, coil, etc. Typically z is most sensitive.
    pns_axis_weights = {'Gx': 0.7, 'Gy': 0.7, 'Gz': 1.0}


    # 1. Compute PNS
    try:
        total_pns_ts = pns_model.compute_pns(slew_rates_data, axis_weights=pns_axis_weights)
        # total_pns_ts_from_array = pns_model.compute_pns(slew_rates_array, axis_weights=pns_axis_weights)

        # print(f"Total normalized PNS time series (first 10): {total_pns_ts[:10]}")
        # print(f"Length of PNS timeseries: {len(total_pns_ts)}")

        # 2. Check Limits
        compliant, peak_val = pns_model.check_limits(total_pns_ts) # or pns_model.check_limits() to use last
        print(f"PNS Compliance: {compliant}, Peak PNS Value: {peak_val:.3f}")

        # 3. Optimize Slew Rates (Conceptual)
        if not compliant:
            print("\nAttempting conceptual slew rate optimization...")
            # Target 95% of the 'max_total_pns_normalized' value from thresholds
            optimization_result = pns_model.optimize_slew_rate(slew_rates_data, target_pns_limit_factor=0.95)
            print(f"Optimization Status: {optimization_result['status']}")
            print(f"  Original Peak PNS: {optimization_result['original_peak_pns']:.3f}")
            print(f"  Target PNS Limit Used: {optimization_result['target_pns_limit_used']:.3f}")
            print(f"  Scaling Factor Applied: {optimization_result['scaling_factor_applied']:.3f}")

            # One could re-check PNS with scaled_slew_rates:
            # scaled_sr = optimization_result['scaled_slew_rates']
            # pns_after_scaling_ts = pns_model.compute_pns(scaled_sr, axis_weights=pns_axis_weights)
            # compliant_after, peak_after = pns_model.check_limits(pns_after_scaling_ts)
            # print(f"PNS Compliance after scaling: {compliant_after}, Peak PNS: {peak_after:.3f}")

    except ValueError as e:
        print(f"Error in PNS Model operations: {e}")
    except Exception as e_gen:
        print(f"An unexpected error occurred: {e_gen}")
        import traceback
        traceback.print_exc()

    print("\n--- PNSModel Example Finished ---")
