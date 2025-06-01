import numpy as np
from scipy.signal import convolve # For convolution

from . import utils # For standardize_trajectory_format

class PNSModel:
    def __init__(self, pns_thresholds_config=None, dt=None):
        """
        Initializes the PNSModel.

        Args:
            pns_thresholds_config (dict, optional): PNS limits and model parameters.
                Can contain a 'default' key and axis-specific keys (e.g., 'x', 'y', 'z').
                Example: {
                    'default': {'rheobase_T_per_s': 20.0, 'chronaxie_ms': 0.36, 'max_total_pns_normalized': 0.8},
                    'x': {'rheobase_T_per_s': 22.0, 'chronaxie_ms': 0.38}
                }
            dt (float): Time step (sampling interval) in seconds. Essential.
        """
        default_global_thresholds = {
            'rheobase_T_per_s': 20.0,
            'chronaxie_ms': 0.36,
            'max_total_pns_normalized': 0.8
        }

        if pns_thresholds_config is None:
            self.pns_thresholds = {'default': default_global_thresholds.copy()}
        else:
            self.pns_thresholds = {'default': default_global_thresholds.copy()}
            # Merge provided config: default first, then axis-specific
            if 'default' in pns_thresholds_config:
                self.pns_thresholds['default'].update(pns_thresholds_config['default'])
            for axis, params in pns_thresholds_config.items():
                if axis != 'default':
                    if axis not in self.pns_thresholds: self.pns_thresholds[axis] = {}
                    self.pns_thresholds[axis].update(params)

        if dt is None:
            raise ValueError("'dt' (time step in seconds) must be provided during initialization.")
        self.dt = dt # seconds

        self.slew_rates_history = []
        self.pns_values_history = []
        self.nerve_filter_cache = {} # Cache for nerve filters, key by axis_name

        # Initial print of effective default parameters
        def_thresh = self.pns_thresholds['default']
        print(f"PNSModel initialized. dt={self.dt*1000:.3f} ms. Default thresholds: "
              f"Rheobase={def_thresh['rheobase_T_per_s']:.1f} T/s (effective), "
              f"Chronaxie={def_thresh['chronaxie_ms']:.2f} ms, "
              f"Max PNS Norm={def_thresh['max_total_pns_normalized']:.2f}")


    def _get_axis_param(self, axis_name, param_key):
        """Helper to get axis-specific param, falling back to default."""
        # Try axis-specific first
        if axis_name in self.pns_thresholds and param_key in self.pns_thresholds[axis_name]:
            return self.pns_thresholds[axis_name][param_key]
        # Fallback to default
        return self.pns_thresholds['default'][param_key]

    def _generate_nerve_response_function(self, axis_name='default', filter_duration_factor=5):
        """
        Generates or retrieves from cache a nerve impulse response function for a given axis.
        h(t) = (dt/chronaxie_s) * exp(-t/chronaxie_s).
        """
        filter_key = axis_name
        if filter_key in self.nerve_filter_cache:
            return self.nerve_filter_cache[filter_key]

        chronaxie_ms_axis = self._get_axis_param(axis_name, 'chronaxie_ms')
        chronaxie_s_axis = chronaxie_ms_axis / 1000.0

        num_filter_points = int(filter_duration_factor * chronaxie_s_axis / self.dt)
        if num_filter_points < 1: num_filter_points = 1

        t_filter = np.arange(num_filter_points) * self.dt

        # Using (dt/C) * exp(-t/C) so that sum of filter coeffs is approx. 1.
        # This implies the output of convolve(SR, filter) has units of SR.
        nerve_ir_filter_axis = (self.dt / chronaxie_s_axis) * np.exp(-t_filter / chronaxie_s_axis)

        if np.sum(nerve_ir_filter_axis) < 0.1:
            print(f"Warning: PNS nerve filter sum for axis '{axis_name}' is very small "
                  f"({np.sum(nerve_ir_filter_axis):.3e}). dt/chronaxie ratio might be an issue.")

        self.nerve_filter_cache[filter_key] = nerve_ir_filter_axis
        print(f"Generated nerve response filter for axis '{axis_name}' (Chronaxie={chronaxie_ms_axis:.2f}ms): "
              f"length {num_filter_points} pts, sum={np.sum(nerve_ir_filter_axis):.3f}")
        return nerve_ir_filter_axis

    def compute_pns(self, slew_rates_dict_or_array, axis_weights=None):
        """
        Computes the total normalized PNS activity time series using axis-specific thresholds.
        """
        # Standardize input to dict: {'axis_name': waveform_array, ...}
        # Determine default axis names based on number of dimensions if input is array
        num_axes_input = 0
        if isinstance(slew_rates_dict_or_array, np.ndarray):
            if slew_rates_dict_or_array.ndim == 1: num_axes_input = 1
            elif slew_rates_dict_or_array.ndim == 2: num_axes_input = slew_rates_dict_or_array.shape[1]
        elif isinstance(slew_rates_dict_or_array, dict):
            num_axes_input = len(slew_rates_dict_or_array)

        default_axis_names_for_std = [f'axis_{i}' for i in range(num_axes_input)] \
                                     if num_axes_input > 3 else ['x', 'y', 'z'][:num_axes_input]


        slew_rates_dict = utils.standardize_trajectory_format(
            slew_rates_dict_or_array,
            target_format='dict',
            default_axis_names=default_axis_names_for_std
        )

        if not slew_rates_dict: # Handle empty input after standardization
            print("Warning: No slew rates provided to compute_pns.")
            return np.array([])

        num_time_points = -1
        for axis_name in slew_rates_dict:
            current_len = len(slew_rates_dict[axis_name])
            if num_time_points == -1: num_time_points = current_len
            elif num_time_points != current_len:
                raise ValueError("Inconsistent slew rate waveform lengths across axes.")
        if num_time_points == 0: return np.array([])


        if axis_weights is None: axis_weights = {}
        all_axes_pns_activity_normalized = {}

        # Prepare for RSS: sum_sq_weighted_pns_norm will hold the sum of squares
        # Its length will be num_time_points because convolve mode is 'same'.
        sum_sq_weighted_pns_norm = np.zeros(num_time_points, dtype=np.float64)

        for axis_name, sr_waveform in slew_rates_dict.items():
            nerve_filter_axis = self._generate_nerve_response_function(axis_name=axis_name)

            if len(sr_waveform) == 0:
                pns_activity_axis_normalized = np.zeros(num_time_points, dtype=np.float64)
            else:
                effective_dbdt_axis = convolve(sr_waveform, nerve_filter_axis, mode='same')
                rheobase_axis = self._get_axis_param(axis_name, 'rheobase_T_per_s')
                pns_activity_axis_normalized = effective_dbdt_axis / rheobase_axis

            all_axes_pns_activity_normalized[axis_name] = pns_activity_axis_normalized
            weight = axis_weights.get(axis_name, 1.0)
            sum_sq_weighted_pns_norm += (pns_activity_axis_normalized * weight)**2

        total_pns_normalized_timeseries = np.sqrt(sum_sq_weighted_pns_norm)

        self.slew_rates_history.append(slew_rates_dict)
        self.pns_values_history.append({
            "total_normalized": total_pns_normalized_timeseries,
            "per_axis_normalized": all_axes_pns_activity_normalized
        })

        peak_pns_val = np.max(total_pns_normalized_timeseries) if total_pns_normalized_timeseries.size > 0 else 0.0
        print(f"Computed PNS. Peak total normalized PNS: {peak_pns_val:.3f} (timeseries length: {len(total_pns_normalized_timeseries)})")
        return total_pns_normalized_timeseries


    def check_limits(self, total_pns_normalized_timeseries=None):
        """ Checks if computed PNS values exceed the global 'max_total_pns_normalized' limit. """
        if total_pns_normalized_timeseries is None:
            if not self.pns_values_history:
                raise ValueError("No PNS values computed. Call compute_pns() or provide timeseries.")
            total_pns_normalized_timeseries = self.pns_values_history[-1]["total_normalized"]

        # Ensure it's a NumPy array for np.max and .size
        if not isinstance(total_pns_normalized_timeseries, np.ndarray):
            total_pns_normalized_timeseries = np.array(total_pns_normalized_timeseries)

        if total_pns_normalized_timeseries.size == 0:
            print("Warning: PNS timeseries for limit check is empty. Assuming compliant (no stimulation).")
            return True, 0.0

        peak_pns_value = np.max(total_pns_normalized_timeseries)

        # Global limit is from 'default' settings
        limit = self.pns_thresholds['default']['max_total_pns_normalized']
        is_compliant = peak_pns_value <= limit

        print(f"PNS Limit Check: Peak PNS = {peak_pns_value:.3f}, Global Limit = {limit:.3f}. Compliant: {is_compliant}")
        return is_compliant, peak_pns_value

    def optimize_slew_rate(self, slew_rates_dict_or_array, target_pns_limit_factor=None, axis_weights=None):
        """
        Conceptual: Suggests slew rate scaling if PNS limits are exceeded.
        Uses axis_weights from argument if provided, else from internal if set, else default.
        """
        print("Attempting to optimize slew rates (conceptual)...")
        # Standardize input for compute_pns and for scaling later
        slew_rates_dict = utils.standardize_trajectory_format(
            slew_rates_dict_or_array, target_format='dict',
            default_axis_names=['x','y','z'][:slew_rates_dict_or_array.shape[1] if isinstance(slew_rates_dict_or_array, np.ndarray) and slew_rates_dict_or_array.ndim==2 else 1]
        )

        current_pns_ts = self.compute_pns(slew_rates_dict, axis_weights=axis_weights)
        is_compliant, peak_pns = self.check_limits(current_pns_ts) # Uses global limit from default thresholds

        # Determine target limit for optimization
        default_max_limit = self.pns_thresholds['default']['max_total_pns_normalized']
        if target_pns_limit_factor is None:
            target_limit = default_max_limit
        else:
            target_limit = default_max_limit * target_pns_limit_factor
            target_limit = min(target_limit, default_max_limit) # Cannot target higher than hard limit

        scaled_slew_rates_dict = slew_rates_dict
        scaling_factor = 1.0
        status = "Compliant"

        if peak_pns > target_limit:
            status = "Exceeds Target, Scaling Suggested"
            if peak_pns > 1e-9: # Avoid division by zero or tiny numbers
                scaling_factor = target_limit / peak_pns
            else: # peak_pns is effectively zero but still > target_limit (e.g. target is negative/zero)
                scaling_factor = 1.0

            scaling_factor = max(0.0, min(scaling_factor, 1.0)) # Clamp factor

            print(f"  Peak PNS ({peak_pns:.3f}) > Target ({target_limit:.3f}). Suggesting slew scaling by {scaling_factor:.3f}.")

            scaled_slew_rates_dict = {
                axis: sr_vals * scaling_factor
                for axis, sr_vals in slew_rates_dict.items()
            }
        else:
            status = f"Compliant with Target ({target_limit:.3f})"
            print(f"  Peak PNS ({peak_pns:.3f}) <= Target ({target_limit:.3f}). No scaling needed.")

        # Return in original format if possible (best effort, or always dict)
        # For now, always returns dict as scaled_slew_rates_dict
        output_slew_rates = scaled_slew_rates_dict
        if isinstance(slew_rates_dict_or_array, np.ndarray): # If input was array, try to convert back
            try:
                output_slew_rates = utils.standardize_trajectory_format(scaled_slew_rates_dict, target_format='array',
                                                                       default_axis_names=list(slew_rates_dict.keys()))
            except Exception: # Fallback to dict if conversion fails
                 pass


        return {
            'scaled_slew_rates': output_slew_rates, # Could be dict or array
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
