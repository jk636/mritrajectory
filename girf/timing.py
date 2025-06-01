import numpy as np

# Conceptual Event Data Structures (for doc/reference):
# Event Type: 'rf'
#   {'id': str, 'type': 'rf', 'channel': str,
#    'start_time_s': float, 'duration_s': float, 'frequency_hz': float, 'amplitude_au': float}
# Event Type: 'gradient' (full waveform or segment definition)
#   {'id': str, 'type': 'gradient', 'axis': str,
#    'start_time_s': float, 'duration_s': float, 'target_amplitude_T_per_m': float,
#    'waveform_shape': str ('rect', 'trap', 'sine', etc.), 'ramp_up_s': float (for trap)}
# Event Type: 'gradient_plateau_check' (for verification against a measured/simulated waveform)
#   {'id': str, 'type': 'gradient_plateau_check', 'axis': str,
#    'expected_start_time_s': float, 'expected_duration_s': float,
#    'expected_amplitude_T_per_m': float, 'amplitude_tolerance_percent': float,
#    'timing_tolerance_s': float}
# Event Type: 'synchronization_check' (for verifying relative timing of two events)
#   {'id': str, 'type': 'synchronization_check',
#    'event_A_id': str, 'event_A_timepoint': str ('start', 'center', 'end'),
#    'event_B_id': str, 'event_B_timepoint': str ('start', 'center', 'end'),
#    'max_allowed_delay_s': float}
# Event Type: 'adc_readout'
#   {'id': str, 'type': 'adc', 'start_time_s': float, 'duration_s': float, 'num_samples': int, 'dwell_time_s': float}


def check_gradient_event_timing(gradient_waveform_1d, dt, expected_event):
    """
    Checks if a segment of a gradient waveform matches expected properties (e.g., a plateau).
    Simplified: Checks average amplitude within the expected time window.

    Args:
        gradient_waveform_1d (np.ndarray): 1D array for a single gradient axis (T/m).
        dt (float): Time step of the waveform (s).
        expected_event (dict): A dictionary describing the expected gradient event.
            Required keys for 'gradient_plateau_check':
            - 'id' (str): Event identifier.
            - 'expected_start_time_s' (float)
            - 'expected_duration_s' (float)
            - 'expected_amplitude_T_per_m' (float)
            - 'amplitude_tolerance_percent' (float)

    Returns:
        dict: Analysis result.
            {'event_id': str, 'amplitude_ok': bool,
             'measured_avg_amplitude_T_per_m': float,
             'expected_amplitude_T_per_m': float,
             'details': str}
    """
    if not isinstance(gradient_waveform_1d, np.ndarray) or gradient_waveform_1d.ndim != 1:
        raise ValueError("gradient_waveform_1d must be a 1D NumPy array.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not expected_event or expected_event.get('type') != 'gradient_plateau_check':
        raise ValueError("Invalid expected_event or type not 'gradient_plateau_check'.")

    event_id = expected_event['id']
    exp_start_s = expected_event['expected_start_time_s']
    exp_dur_s = expected_event['expected_duration_s']
    exp_amp_T_per_m = expected_event['expected_amplitude_T_per_m']
    amp_tol_percent = expected_event['amplitude_tolerance_percent']

    # Convert times to sample indices
    start_idx = int(round(exp_start_s / dt))
    end_idx = int(round((exp_start_s + exp_dur_s) / dt)) # Exclusive end index for slicing

    if start_idx < 0 or end_idx > len(gradient_waveform_1d) or start_idx >= end_idx:
        return {
            'event_id': event_id, 'amplitude_ok': False,
            'measured_avg_amplitude_T_per_m': np.nan,
            'expected_amplitude_T_per_m': exp_amp_T_per_m,
            'details': f"Expected time window [{exp_start_s*1e3:.3f}ms - {(exp_start_s+exp_dur_s)*1e3:.3f}ms] "
                       f"(indices [{start_idx}-{end_idx-1}]) is out of waveform bounds [0-{len(gradient_waveform_1d)-1}]."
        }

    plateau_segment = gradient_waveform_1d[start_idx:end_idx]
    if plateau_segment.size == 0:
        return {
            'event_id': event_id, 'amplitude_ok': False,
            'measured_avg_amplitude_T_per_m': np.nan,
            'expected_amplitude_T_per_m': exp_amp_T_per_m,
            'details': "Plateau segment is empty based on calculated indices."
        }

    measured_avg_amp = np.mean(plateau_segment)

    # Check amplitude tolerance
    abs_amp_tolerance = np.abs(exp_amp_T_per_m * (amp_tol_percent / 100.0))
    lower_bound = exp_amp_T_per_m - abs_amp_tolerance
    upper_bound = exp_amp_T_per_m + abs_amp_tolerance

    amplitude_ok = lower_bound <= measured_avg_amp <= upper_bound

    details_msg = (f"Avg amp {measured_avg_amp:.4f} T/m vs exp {exp_amp_T_per_m:.4f} T/m "
                   f"(tol: +/- {abs_amp_tolerance:.4f} T/m).")
    if not amplitude_ok:
        details_msg = "FAIL: " + details_msg
    else:
        details_msg = "PASS: " + details_msg

    # Simplified timing check: For now, we only check amplitude within the *expected* window.
    # A more advanced check would find actual start/end of plateau based on waveform crossing thresholds.
    timing_deviation_info = "Simplified check: Timing deviation not fully assessed."

    return {
        'event_id': event_id,
        'amplitude_ok': amplitude_ok,
        'measured_avg_amplitude_T_per_m': float(measured_avg_amp),
        'expected_amplitude_T_per_m': exp_amp_T_per_m,
        'details': details_msg,
        'timing_deviation_info': timing_deviation_info
    }


def check_event_synchronization(event_A_actual_time_s, event_B_actual_time_s, max_allowed_delay_s,
                                event_A_id="EventA", event_B_id="EventB"):
    """
    Checks if two events are synchronized within a maximum allowed delay.

    Args:
        event_A_actual_time_s (float): Actual time of the specific point on Event A (s).
        event_B_actual_time_s (float): Actual time of the specific point on Event B (s).
        max_allowed_delay_s (float): Maximum allowed absolute delay between the events (s).
        event_A_id (str): Identifier for event A.
        event_B_id (str): Identifier for event B.

    Returns:
        dict: {'is_synchronized': bool, 'actual_delay_s': float, 'details': str}
    """
    if max_allowed_delay_s < 0:
        raise ValueError("max_allowed_delay_s must be non-negative.")

    delay_s = event_A_actual_time_s - event_B_actual_time_s # Signed delay
    abs_delay_s = np.abs(delay_s)
    is_synchronized = abs_delay_s <= max_allowed_delay_s

    details_msg = (f"Delay between {event_A_id} ({event_A_actual_time_s*1e6:.1f}us) and "
                   f"{event_B_id} ({event_B_actual_time_s*1e6:.1f}us) is {delay_s*1e6:.1f}us. "
                   f"Limit: +/- {max_allowed_delay_s*1e6:.1f}us.")
    if not is_synchronized:
        details_msg = "FAIL: " + details_msg
    else:
        details_msg = "PASS: " + details_msg

    return {
        'is_synchronized': is_synchronized,
        'actual_delay_s': float(delay_s), # Keep signed delay for more info
        'abs_delay_s': float(abs_delay_s),
        'max_allowed_delay_s': max_allowed_delay_s,
        'details': details_msg
    }


def process_timing_checks(nominal_events_list, actual_event_data_map=None,
                          gradient_waveforms_dict=None, dt=None):
    """
    Processes a list of nominal timing events and performs specified checks.

    Args:
        nominal_events_list (list of dict): List of event dictionaries defining checks.
        actual_event_data_map (dict, optional): Maps event IDs to their actual measured/derived data.
            Example: {'evt_001': {'start_time_s': 0.00101, 'duration_s': 0.00049}, ...}
        gradient_waveforms_dict (dict, optional): Dict of {axis_name: waveform_array (T/m)}.
        dt (float, optional): Time step for gradient waveforms (s). Required for gradient checks.

    Returns:
        list: List of check result dictionaries.
    """
    results = []
    if actual_event_data_map is None: actual_event_data_map = {}

    for event_check_def in nominal_events_list:
        check_type = event_check_def.get('type')
        event_id = event_check_def.get('id', 'UnknownEvent')

        if check_type == 'gradient_plateau_check':
            axis = event_check_def.get('axis')
            if not gradient_waveforms_dict or axis not in gradient_waveforms_dict:
                results.append({'event_id': event_id, 'status': 'SKIPPED',
                                'reason': f"Gradient waveform for axis '{axis}' not provided."})
                continue
            if dt is None:
                results.append({'event_id': event_id, 'status': 'SKIPPED',
                                'reason': "'dt' not provided for gradient checks."})
                continue

            waveform = gradient_waveforms_dict[axis]
            result = check_gradient_event_timing(waveform, dt, event_check_def)
            results.append(result)

        elif check_type == 'synchronization_check':
            event_A_id = event_check_def.get('event_A_id')
            event_A_tp_key = event_check_def.get('event_A_timepoint', 'start_time_s') # Default to 'start_time_s'
            event_B_id = event_check_def.get('event_B_id')
            event_B_tp_key = event_check_def.get('event_B_timepoint', 'start_time_s')
            max_delay = event_check_def.get('max_allowed_delay_s')

            if not all([event_A_id, event_B_id, max_delay is not None]):
                results.append({'event_id': event_id, 'status': 'SKIPPED',
                                'reason': "Missing required fields for synchronization_check."})
                continue

            actual_A_data = actual_event_data_map.get(event_A_id)
            actual_B_data = actual_event_data_map.get(event_B_id)

            if not actual_A_data or event_A_tp_key not in actual_A_data:
                results.append({'event_id': event_id, 'status': 'SKIPPED',
                                'reason': f"Actual data for '{event_A_tp_key}' of event '{event_A_id}' not found."})
                continue
            if not actual_B_data or event_B_tp_key not in actual_B_data:
                results.append({'event_id': event_id, 'status': 'SKIPPED',
                                'reason': f"Actual data for '{event_B_tp_key}' of event '{event_B_id}' not found."})
                continue

            time_A = actual_A_data[event_A_tp_key]
            time_B = actual_B_data[event_B_tp_key]

            result = check_event_synchronization(time_A, time_B, max_delay, event_A_id, event_B_id)
            results.append({'event_id': event_id, **result}) # Merge event_id with check result

        else:
            results.append({'event_id': event_id, 'status': 'SKIPPED',
                            'reason': f"Unsupported event check type: '{check_type}'."})

    return results


if __name__ == '__main__':
    print("--- Running girf.timing.py example simulations ---")

    # Setup
    dt_val = 4e-6  # 4 us
    num_pts = 1000
    time_vec = np.arange(num_pts) * dt_val

    # Sample Gradient Waveform (e.g., for X-axis)
    grad_x = np.zeros(num_pts)
    grad_x[250:500] = 0.02  # Plateau from 1ms to 2ms (0.02 T/m)
    grad_x[500:750] = 0.015 # Another plateau from 2ms to 3ms (0.015 T/m)
    # Add some noise
    grad_x += np.random.normal(0, 0.0005, num_pts)

    gradient_waveforms = {'x': grad_x, 'y': np.zeros(num_pts)}

    # Sample Actual Event Data (e.g., from a sequence execution log or measurement)
    actual_events = {
        'rf_pulse_1': {'id': 'rf_pulse_1', 'start_time_s': 0.000998, 'duration_s': 0.000502},
        'grad_x_main': {'id': 'grad_x_main', 'start_time_s': 0.001005, 'duration_s': 0.001000,
                        'actual_amplitude_T_per_m': 0.0198}, # Not used by current checks, but good for context
        'adc_readout_1': {'id': 'adc_readout_1', 'start_time_s': 0.001250, 'duration_s': 0.000500}
    }

    # Define Nominal Event Checks
    nominal_checks = [
        {
            'id': 'check_gx_plateau1', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': 0.001, 'expected_duration_s': 0.001,
            'expected_amplitude_T_per_m': 0.02, 'amplitude_tolerance_percent': 10.0
        },
        {
            'id': 'check_gx_plateau2_fail', 'type': 'gradient_plateau_check', 'axis': 'x',
            'expected_start_time_s': 0.002, 'expected_duration_s': 0.001,
            'expected_amplitude_T_per_m': 0.02, 'amplitude_tolerance_percent': 5.0 # This should fail amplitude
        },
        {
            'id': 'sync_rf_grad_start', 'type': 'synchronization_check',
            'event_A_id': 'rf_pulse_1', 'event_A_timepoint': 'start_time_s',
            'event_B_id': 'grad_x_main', 'event_B_timepoint': 'start_time_s',
            'max_allowed_delay_s': 10e-6 # 10 us
        },
        {
            'id': 'sync_grad_adc_fail', 'type': 'synchronization_check',
            'event_A_id': 'grad_x_main', 'event_A_timepoint': 'start_time_s', # grad_x_main starts at 1.005ms
            'event_B_id': 'adc_readout_1', 'event_B_timepoint': 'start_time_s',# adc_readout_1 starts at 1.250ms
            'max_allowed_delay_s': 50e-6 # Delay is 245us, should fail
        }
    ]

    print("\n--- Processing Timing Checks ---")
    check_results = process_timing_checks(nominal_checks,
                                          actual_event_data_map=actual_events,
                                          gradient_waveforms_dict=gradient_waveforms,
                                          dt=dt_val)

    for result in check_results:
        print(f"\nEvent ID: {result.get('event_id', 'N/A')}")
        if 'amplitude_ok' in result:
            print(f"  Amplitude OK: {result['amplitude_ok']}")
            print(f"  Measured Avg: {result.get('measured_avg_amplitude_T_per_m',np.nan):.4f} T/m, "
                  f"Expected: {result.get('expected_amplitude_T_per_m',np.nan):.4f} T/m")
        if 'is_synchronized' in result:
            print(f"  Synchronized: {result['is_synchronized']}")
            print(f"  Actual Delay: {result.get('actual_delay_s',np.nan)*1e6:.1f} us "
                  f"(Limit: +/- {result.get('max_allowed_delay_s',np.nan)*1e6:.1f} us)")
        if 'status' in result and result['status'] == 'SKIPPED':
            print(f"  Status: SKIPPED, Reason: {result.get('reason')}")
        if 'details' in result: print(f"  Details: {result['details']}")

    print("\n--- girf.timing.py example simulations finished ---")
