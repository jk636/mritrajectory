import numpy as np
# from . import utils # If utils are needed, e.g. for slew calculation from gradients

# --- Conceptual Data Structures (for docstrings/reference) ---
# epi_params = {
#     'sequence_type': 'epi_3d', # Must be present for dispatcher
#     'echo_train_length': 64,
#     'time_per_echo_ms': 0.8, # Time to acquire one echo in the ETL
#     't2_star_ms': 50,
#     'max_allowed_etl_t2_star_ratio': 1.0,
#     # For blip checks (conceptual, would need more details on expected k-space steps)
#     'ky_blip_amplitudes': np.array([...]), # Amplitudes or k-space steps
#     'kz_blip_amplitudes': np.array([...]),
#     'blip_consistency_tolerance_percent': 5.0
# }
# spiral_params = {
#     'sequence_type': 'spiral_3d', # Must be present
#     'num_interleaves': 32,
#     'actual_readout_duration_ms': 8.0, # Duration of one spiral interleaf readout
#     'max_readout_duration_ms': 10.0, # Guideline for off-resonance
#     'moments_to_check_for_nulling': (0, 1) # Check M0 and M1
# }

# --- 3D EPI Checks ---

def check_epi_echo_train_length(etl, t2_star_ms=None,
                                max_allowed_etl_t2_star_ratio=None,
                                time_per_echo_ms=None):
    """
    Checks basic properties of EPI echo train length (ETL), optionally against T2*.

    Args:
        etl (int): Echo Train Length.
        t2_star_ms (float, optional): Estimated T2* of tissue (ms).
        max_allowed_etl_t2_star_ratio (float, optional): Max ratio of total ETL duration to T2*.
        time_per_echo_ms (float, optional): Time to acquire one echo in the ETL (ms).

    Returns:
        dict: Check results.
    """
    if not isinstance(etl, int) or etl <= 0:
        return {'etl_ok': False, 'details': 'ETL must be a positive integer.'}

    result = {'etl_ok': True, 'etl_value': etl, 'details': 'Basic ETL > 0 check passed.'}

    if t2_star_ms is not None and \
       max_allowed_etl_t2_star_ratio is not None and \
       time_per_echo_ms is not None:

        if t2_star_ms <=0 or max_allowed_etl_t2_star_ratio <=0 or time_per_echo_ms <=0:
            result['etl_duration_vs_t2star_check'] = 'Skipped'
            result['etl_duration_details'] = 'Invalid T2*, ratio, or time_per_echo provided for duration check.'
            return result

        total_etl_duration_ms = etl * time_per_echo_ms
        limit_ms = t2_star_ms * max_allowed_etl_t2_star_ratio
        is_duration_ok = total_etl_duration_ms <= limit_ms

        result['etl_duration_vs_t2star_ok'] = is_duration_ok
        result['total_etl_duration_ms'] = total_etl_duration_ms
        result['t2_star_limit_ms'] = limit_ms
        if not is_duration_ok:
            result['details'] += f" ETL duration {total_etl_duration_ms:.2f}ms exceeds limit {limit_ms:.2f}ms (T2* based)."
            # Potentially set overall etl_ok to False if this is critical
            # result['etl_ok'] = False
        else:
             result['details'] += f" ETL duration {total_etl_duration_ms:.2f}ms within limit {limit_ms:.2f}ms."
    return result

def check_epi_phase_encoding_blips(ky_blip_amplitudes=None, kz_blip_amplitudes=None,
                                   consistency_tolerance_percent=5.0):
    """
    Conceptual check for consistency of EPI phase encoding blips.
    Placeholder: A real check needs more detailed inputs on expected k-space steps or blip areas.
    """
    # For now, if data is provided, check basic std dev / mean as a mock consistency.
    results = {'details': 'Placeholder: Blip consistency check is conceptual.'}
    overall_ok = True

    for axis_name, blip_amps in [('ky', ky_blip_amplitudes), ('kz', kz_blip_amplitudes)]:
        key_ok = f'blip_consistency_{axis_name}_ok'
        if blip_amps is not None:
            blip_amps = np.asarray(blip_amps)
            if len(blip_amps) > 1 and np.mean(np.abs(blip_amps)) > 1e-9: # Avoid division by zero for all zero blips
                consistency = np.std(np.abs(blip_amps)) / np.mean(np.abs(blip_amps))
                is_consistent = consistency < (consistency_tolerance_percent / 100.0)
                results[key_ok] = is_consistent
                results[f'{axis_name}_blip_mean_abs_amp'] = np.mean(np.abs(blip_amps))
                results[f'{axis_name}_blip_std_dev_abs_amp'] = np.std(np.abs(blip_amps))
                results[f'{axis_name}_blip_consistency_metric'] = consistency
                if not is_consistent:
                    overall_ok = False
                    results['details'] += f" {axis_name} blips show high variation."
            elif len(blip_amps) > 0 : # All zeros or single blip
                 results[key_ok] = True # Consider consistent
                 results[f'{axis_name}_blip_mean_abs_amp'] = np.mean(np.abs(blip_amps))
            else: # Empty list
                 results[key_ok] = True # No blips to be inconsistent
        else:
            results[key_ok] = True # No data provided, considered ok/not applicable

    results['overall_blip_consistency_ok'] = overall_ok
    return results


# --- 3D Spiral Checks ---

def check_spiral_readout_duration(actual_readout_duration_ms, max_readout_duration_ms=None):
    """
    Checks if the spiral readout duration is within an optional maximum limit.
    """
    if actual_readout_duration_ms <= 0:
        return {'readout_duration_ok': False, 'actual_ms': actual_readout_duration_ms,
                'limit_ms': max_readout_duration_ms,
                'details': 'Actual readout duration must be positive.'}

    is_ok = True
    details_msg = f"Actual readout: {actual_readout_duration_ms:.2f}ms."
    if max_readout_duration_ms is not None:
        if max_readout_duration_ms <=0:
            details_msg += " Invalid max_readout_duration_ms provided (must be >0)."
        else:
            is_ok = actual_readout_duration_ms <= max_readout_duration_ms
            details_msg += f" Limit: {max_readout_duration_ms:.2f}ms. Compliance: {is_ok}."

    return {'readout_duration_ok': is_ok, 'actual_ms': actual_readout_duration_ms,
            'limit_ms': max_readout_duration_ms, 'details': details_msg}


def check_spiral_gradient_moment_nulling(gradient_waveforms_dict, dt, moments_to_check=(0,), moment_tolerance=1e-6):
    """
    Checks if specified gradient moments (M0, M1, etc.) are nulled for each axis.
    Moment m = integral(G(t) * t^m dt).
    """
    if not gradient_waveforms_dict or not isinstance(gradient_waveforms_dict, dict):
        return {'moments_ok': False, 'details': "gradient_waveforms_dict must be a non-empty dictionary."}
    if dt <= 0:
        return {'moments_ok': False, 'details': "dt must be positive."}

    overall_moments_ok = True
    moment_details = {}

    for axis, waveform in gradient_waveforms_dict.items():
        waveform = np.asarray(waveform)
        if waveform.ndim != 1 or waveform.size == 0:
            moment_details[axis] = {m: {'value': np.nan, 'is_nulled': False, 'error': 'Invalid waveform data'} for m in moments_to_check}
            overall_moments_ok = False
            continue

        num_points = waveform.shape[0]
        time_vector = np.arange(num_points) * dt
        axis_details = {}

        for moment_order in moments_to_check:
            if moment_order < 0:
                axis_details[moment_order] = {'value': np.nan, 'is_nulled': False, 'error': 'Moment order must be non-negative.'}
                overall_moments_ok = False
                continue

            integrand = waveform * (time_vector ** moment_order)
            moment_val = np.sum(integrand) * dt
            is_nulled = np.abs(moment_val) < moment_tolerance

            axis_details[moment_order] = {'value': float(moment_val), 'is_nulled': is_nulled}
            if not is_nulled:
                overall_moments_ok = False
        moment_details[axis] = axis_details

    return {'overall_moments_ok': overall_moments_ok, 'details': moment_details}


# --- Dispatcher ---

def run_sequence_specific_checks(sequence_params,
                                 # trajectory_data=None, # Not directly used yet by example checks
                                 gradient_waveforms_dict=None,
                                 dt=None):
    """
    Runs sequence-specific checks based on 'sequence_type' in sequence_params.
    """
    seq_type = sequence_params.get('sequence_type')
    results = []
    check_id_counter = 0

    if not seq_type:
        return [{'check_id': 'dispatcher_error', 'error': 'sequence_type not specified in sequence_params.'}]

    if seq_type.lower() == 'epi_3d':
        check_id_counter +=1
        etl_res = check_epi_echo_train_length(
            etl=sequence_params.get('echo_train_length', 0),
            t2_star_ms=sequence_params.get('t2_star_ms'),
            max_allowed_etl_t2_star_ratio=sequence_params.get('max_allowed_etl_t2_star_ratio'),
            time_per_echo_ms=sequence_params.get('time_per_echo_ms')
        )
        results.append({'check_id': f'{seq_type}_etl_check_{check_id_counter}', **etl_res})

        check_id_counter +=1
        blip_res = check_epi_phase_encoding_blips(
            ky_blip_amplitudes=sequence_params.get('ky_blip_amplitudes'),
            kz_blip_amplitudes=sequence_params.get('kz_blip_amplitudes'),
            consistency_tolerance_percent=sequence_params.get('blip_consistency_tolerance_percent', 5.0)
        )
        results.append({'check_id': f'{seq_type}_blip_check_{check_id_counter}', **blip_res})

    elif seq_type.lower() == 'spiral_3d':
        check_id_counter +=1
        duration_res = check_spiral_readout_duration(
            actual_readout_duration_ms=sequence_params.get('actual_readout_duration_ms', 0.0),
            max_readout_duration_ms=sequence_params.get('max_readout_duration_ms')
        )
        results.append({'check_id': f'{seq_type}_duration_check_{check_id_counter}', **duration_res})

        check_id_counter +=1
        if gradient_waveforms_dict and dt:
            moment_res = check_spiral_gradient_moment_nulling(
                gradient_waveforms_dict, dt,
                moments_to_check=sequence_params.get('moments_to_check_for_nulling', (0,))
            )
            results.append({'check_id': f'{seq_type}_moment_check_{check_id_counter}', **moment_res})
        else:
            results.append({'check_id': f'{seq_type}_moment_check_{check_id_counter}',
                            'moments_ok': False, # Or 'SKIPPED'
                            'details': 'Gradient waveforms or dt not provided for moment nulling check.'})
    else:
        results.append({'check_id': 'dispatcher_error',
                        'error': f"Unknown or unsupported sequence_type: '{seq_type}'."})

    return results


if __name__ == '__main__':
    print("--- Running girf.sequence_checks.py example simulations ---")

    # Common parameters
    example_dt = 4e-6 # 4 us
    example_num_points = 1024
    example_time_vector = np.arange(example_num_points) * example_dt

    # --- EPI Example ---
    print("\n--- EPI Sequence Checks Example ---")
    epi_params_example = {
        'sequence_type': 'epi_3d',
        'echo_train_length': 60,
        'time_per_echo_ms': 0.7,
        't2_star_ms': 40,
        'max_allowed_etl_t2_star_ratio': 1.2, # Allow ETL to be 1.2 * T2*
        'ky_blip_amplitudes': np.random.normal(loc=0.001, scale=0.00002, size=59), # 59 blips for 60 echos
        'blip_consistency_tolerance_percent': 10.0
    }
    epi_checks_results = run_sequence_specific_checks(epi_params_example)
    for res in epi_checks_results: print(f"  EPI Check ({res.get('check_id')}): {res}")

    # --- Spiral Example ---
    print("\n--- Spiral Sequence Checks Example ---")
    # Dummy gradient waveforms for spiral moment check
    spiral_gx = np.sin(2 * np.pi * 1e3 * example_time_vector) * 0.02 # 1kHz, 20mT/m peak
    spiral_gy = np.cos(2 * np.pi * 1e3 * example_time_vector) * 0.02
    # Make M0 non-zero for Gx for demonstration
    spiral_gx += 0.00001
    spiral_waveforms = {'x': spiral_gx, 'y': spiral_gy, 'z': np.zeros_like(spiral_gx)}

    spiral_params_example = {
        'sequence_type': 'spiral_3d',
        'actual_readout_duration_ms': (example_num_points * example_dt) * 1000.0, # Full waveform duration
        'max_readout_duration_ms': 5.0, # ms
        'moments_to_check_for_nulling': (0, 1) # Check M0 (area) and M1
    }
    spiral_checks_results = run_sequence_specific_checks(spiral_params_example,
                                                         gradient_waveforms_dict=spiral_waveforms,
                                                         dt=example_dt)
    for res in spiral_checks_results: print(f"  Spiral Check ({res.get('check_id')}): {res}")

    print("\n--- girf.sequence_checks.py example simulations finished ---")
