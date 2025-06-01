import numpy as np
import json

# Attempt to import GIRFCalibrator for loading GIRF data
try:
    from .calibrator import GIRFCalibrator
except ImportError:
    GIRFCalibrator = None

# Attempt to import PNSModel and TrajectoryPredictor
try:
    from .pns import PNSModel
    from .predictor import TrajectoryPredictor
    from . import utils # Import the utils module
except ImportError:
    PNSModel = None
    TrajectoryPredictor = None
    utils = None # Handle if utils cannot be imported (e.g. circular dependency if not careful)

# Gyromagnetic ratio for protons in Hz/T
DEFAULT_GAMMA_PROTON = 42.576e6

class TrajectoryPlanner:
    def __init__(self, girf_spectra=None, constraints=None, dt=None, gamma=DEFAULT_GAMMA_PROTON,
                 pns_model_instance=None, trajectory_predictor_instance=None):
        """
        Initializes the TrajectoryPlanner.

        Args:
            girf_spectra (dict, optional): Pre-loaded GIRF spectra.
            constraints (dict, optional): Hardware and PNS constraints.
                Example: {'Gmax_T_per_m': 0.04, 'Smax_T_per_m_per_s': 150,
                          'PNS_threshold_factor': 0.8, 'pns_coeffs_dbdt': [val1, val2]}
            dt (float, optional): Time step (sampling interval) in seconds. Essential for calculations.
            gamma (float, optional): Gyromagnetic ratio in Hz/T.
            pns_model_instance (PNSModel, optional): An instance of PNSModel.
            trajectory_predictor_instance (TrajectoryPredictor, optional): An instance of TrajectoryPredictor.
        """
        self.girf_spectra = girf_spectra if girf_spectra is not None else {}
        self.constraints = constraints if constraints is not None else {}
        self.optimized_trajectory_kspace = None # Stores final k-space trajectory
        self.nominal_gradients_time = None      # Stores nominal gradients before pre-emphasis
        self.preemphasized_gradients_time = None # Stores gradients after pre-emphasis

        if dt is None:
            raise ValueError("'dt' (time step) must be provided during initialization.")
        self.dt = dt
        self.gamma = gamma

        # Ensure constraints like Gmax_T_per_m, Smax_T_per_m_per_s, etc. are present
        # Defaulting them here if not provided.
        self.constraints.setdefault('Gmax_T_per_m', 0.04)
        self.constraints.setdefault('Smax_T_per_m_per_s', 150)
        # thermal_limits is optional, no default needed here, its presence will be checked by methods.
        self.constraints.setdefault('thermal_limits', {})


        if PNSModel is None and pns_model_instance is None:
            print("Warning: PNSModel class not available and no instance provided. PNS checks will be skipped.")
            self.pns_model = None
        else:
            # If pns_model_instance is provided, use it. Otherwise, create a new one.
            # Pass relevant constraints to PNSModel if creating it here.
            pns_thresholds_from_planner_constraints = self.constraints.get('pns_model_thresholds') # Explicit key
            if not pns_thresholds_from_planner_constraints and 'PNS_threshold_factor' in self.constraints: # Older way
                 pns_thresholds_from_planner_constraints = {'max_total_pns_normalized': self.constraints['PNS_threshold_factor']}

            self.pns_model = pns_model_instance if pns_model_instance else \
                             (PNSModel(pns_thresholds=pns_thresholds_from_planner_constraints, dt=self.dt) if PNSModel else None)


        if TrajectoryPredictor is None and trajectory_predictor_instance is None:
            print("Warning: TrajectoryPredictor class not available and no instance provided. Pre-emphasis may be skipped or limited.")
            self.trajectory_predictor = None
        else:
            self.trajectory_predictor = trajectory_predictor_instance if trajectory_predictor_instance else \
                                       (TrajectoryPredictor(dt=self.dt, gamma=self.gamma, girf_spectra=self.girf_spectra) if TrajectoryPredictor else None)
            if self.trajectory_predictor : # Ensure predictor has the latest GIRF and dt from planner
                if not self.trajectory_predictor.girf_spectra and self.girf_spectra:
                     self.trajectory_predictor.girf_spectra = self.girf_spectra
                if self.trajectory_predictor.dt is None : self.trajectory_predictor.dt = self.dt
                if not self.trajectory_predictor.harmonic_components and hasattr(self, 'harmonic_components') and self.harmonic_components:
                     self.trajectory_predictor.harmonic_components = self.harmonic_components


        print(f"TrajectoryPlanner initialized. dt={self.dt*1e6:.1f}us. Constraints: Gmax={self.constraints['Gmax_T_per_m']*1e3:.1f}mT/m, Smax={self.constraints['Smax_T_per_m_per_s']:.0f}T/m/s" +
              (f", Smax_vec={self.constraints['Smax_vector_T_per_m_per_s']:.0f}T/m/s" if 'Smax_vector_T_per_m_per_s' in self.constraints else ""))


    def load_girf(self, girf_data_or_path):
        """ Loads GIRF spectra, similar to TrajectoryPredictor. """
        if isinstance(girf_data_or_path, dict):
            self.girf_spectra = {k: np.asarray(v) for k, v in girf_data_or_path.items()}
            print("GIRF spectra loaded from dictionary for Planner.")
        elif isinstance(girf_data_or_path, str):
            if GIRFCalibrator is not None:
                try:
                    cal = GIRFCalibrator()
                    cal.load_calibration(girf_data_or_path)
                    self.girf_spectra = cal.girf_spectra
                    print(f"GIRF spectra loaded from file via GIRFCalibrator: {girf_data_or_path}")
                except Exception as e:
                    raise IOError(f"Failed to load GIRF from {girf_data_or_path} using GIRFCalibrator: {e}")
            else: # Fallback direct JSON load (simplified)
                try:
                    with open(girf_data_or_path, 'r') as f: data = json.load(f)
                    raw_spectra = data.get("girf_spectra_complex", {})
                    self.girf_spectra = {
                        axis: np.array([complex(r, i) for r, i in spec_list]) if spec_list and isinstance(spec_list[0], list) else np.asarray(spec_list)
                        for axis, spec_list in raw_spectra.items()
                    }
                    print(f"GIRF spectra loaded directly from JSON (GIRFCalibrator not found): {girf_data_or_path}")
                except Exception as e_json:
                    raise IOError(f"Failed to load GIRF from {girf_data_or_path} (direct JSON): {e_json}")
        else:
            raise TypeError("girf_data_or_path must be a dictionary or a file path string.")

        # Update trajectory_predictor if it exists
        if self.trajectory_predictor:
            self.trajectory_predictor.girf_spectra = self.girf_spectra
            print("Updated TrajectoryPredictor instance with new GIRF spectra.")


    # --- Gradient / K-space Conversion Helpers (adapted simplified versions) ---
    def _kspace_to_gradients(self, trajectory_kspace):
        """ Converts k-space trajectory (T, N_axes) to gradient waveforms (T, N_axes). """
        if not isinstance(trajectory_kspace, np.ndarray): trajectory_kspace = np.asarray(trajectory_kspace)
        if trajectory_kspace.ndim == 1: trajectory_kspace = trajectory_kspace[:, np.newaxis]

        gradients = np.diff(trajectory_kspace, axis=0, prepend=trajectory_kspace[0:1,:]) / (self.gamma * self.dt)
        # Correcting the first point: g[0] = k[0]/(gamma*dt) if k starts from k[0] due to g[0]
        # Or more commonly, g[0] is designed to take k from 0 to k[0].
        # If k[0] is non-zero and we assume k starts at 0, then g[0] is responsible for k[0].
        # k[0] = g[0]*gamma*dt. So g[0] = k[0]/(gamma*dt)
        # The diff with prepend results in g[0] = (k[0]-k[0]) / (gamma*dt) = 0 if k[0] is first point.
        # Let's assume the trajectory input k[0] is the first target k-space point, starting from k=0.
        # So, grad[0] should be k[0]/(gamma*dt)
        # And grad[i] = (k[i]-k[i-1])/(gamma*dt) for i > 0

        actual_gradients = np.zeros_like(trajectory_kspace)
        actual_gradients[0,:] = trajectory_kspace[0,:] / (self.gamma * self.dt) # Grad to reach first k-point from zero
        actual_gradients[1:,:] = np.diff(trajectory_kspace, axis=0) / (self.gamma * self.dt)
        return actual_gradients

    def _gradients_to_kspace(self, gradients_time, initial_kspace_point=None):
        """ Converts gradient waveforms (T, N_axes) to k-space trajectory (T, N_axes). """
        if not isinstance(gradients_time, np.ndarray): gradients_time = np.asarray(gradients_time)
        if gradients_time.ndim == 1: gradients_time = gradients_time[:, np.newaxis]

        k_deltas = gradients_time * self.gamma * self.dt
        trajectory_kspace = np.cumsum(k_deltas, axis=0)
        if initial_kspace_point is not None:
            initial_kspace_point = np.asarray(initial_kspace_point)
            trajectory_kspace += initial_kspace_point - trajectory_kspace[0,:]
        return trajectory_kspace

    def _check_hardware_constraints(self, gradients_time):
        """
        Checks gradient (per-axis), slew rate (per-axis), and optionally vector slew rate limits.
        Returns a consolidated hardware status (bool) and a detailed violation report (dict).
        """
        if utils is None:
            raise ImportError("girf.utils module not imported, cannot perform hardware checks.")

        g_array = utils.standardize_trajectory_format(gradients_time, target_format='array',
                                                     default_axis_names=['x', 'y', 'z'][:gradients_time.shape[1] if hasattr(gradients_time, 'shape') else 1])


        # Get limits from constraints
        gmax_limit = self.constraints.get('Gmax_T_per_m')
        smax_axis_limit = self.constraints.get('Smax_T_per_m_per_s')
        smax_vector_limit = self.constraints.get('Smax_vector_T_per_m_per_s') # Optional
        tolerance = self.constraints.get('tolerance', 1e-9)

        # --- Per-axis Gradient Check ---
        g_ok, max_g_found, g_details_report = utils.check_gradient_strength(
            g_array, gmax_limit, tolerance
        )

        # --- Per-axis Slew Rate Calculation & Check ---
        slew_rates_arr = utils.compute_slew_rates(g_array, self.dt, output_format='array')
        s_axis_ok, max_s_axis_found, s_axis_details_report = utils.check_slew_rate(
            slew_rates_arr, smax_axis_limit, tolerance
        )

        # --- Vector Slew Rate Check (Optional) ---
        s_vec_ok = True # Default to True if not checked
        max_s_vec_found = 0.0
        s_vec_details_report = {'message': 'Not checked or not applicable.'}

        if smax_vector_limit is not None:
            if slew_rates_arr.shape[1] > 1: # Only for multi-axis data
                s_vec_ok, max_s_vec_found, s_vec_details_report = utils.check_vector_slew_rate(
                    slew_rates_arr, smax_vector_limit, tolerance
                )
            else: # 1D data
                s_vec_details_report['message'] = "Vector slew check skipped for 1D gradient data."

        overall_hw_ok = g_ok and s_axis_ok and s_vec_ok

        # Consolidate report
        hw_report = {
            'overall_ok': overall_hw_ok,
            'G_ok': g_ok, 'max_G_found_T_per_m': max_g_found, 'G_details': g_details_report,
            'S_axis_ok': s_axis_ok, 'max_S_axis_found_T_per_m_per_s': max_s_axis_found, 'S_axis_details': s_axis_details_report,
            'S_vector_ok': s_vec_ok, 'max_S_vector_found_T_per_m_per_s': max_s_vec_found, 'S_vector_details': s_vec_details_report
        }
        return overall_hw_ok, hw_report # Return overall status and the detailed report dict

    def _generate_archimedean_spiral(self, num_points, k_max_cycles_per_m, num_revolutions, num_axes=2):
        """ Generates a simple 2D Archimedean spiral k-space trajectory. """
        if num_axes != 2: raise ValueError("Archimedean spiral is 2D.")

        theta = np.linspace(0, num_revolutions * 2 * np.pi, num_points)
        radius_norm = theta / np.max(theta) # Normalized radius [0,1]
        radius = radius_norm * k_max_cycles_per_m # Physical radius in m^-1

        kx = radius * np.cos(theta)
        ky = radius * np.sin(theta)

        return np.stack([kx, ky], axis=-1)


    def design_trajectory(self, traj_type='spiral', design_params=None, target_resolution_mm=None):
        """
        Designs a nominal k-space trajectory. (Conceptual/Simplified)
        """
        print(f"Designing trajectory of type '{traj_type}'...")
        if design_params is None: design_params = {}

        num_points = design_params.get('num_points', 1024)

        # FOV and resolution determine k_max
        # k_max_cycles_per_m = 1 / (2 * target_resolution_m)
        # For now, let's use a fixed k_max if not derivable
        k_max_m_inv = design_params.get('k_max_m_inv', 1 / (2 * 0.001) if target_resolution_mm else 250) # e.g., for 1mm res

        if traj_type == 'spiral':
            num_revs = design_params.get('num_revolutions', 16)
            self.optimized_trajectory_kspace = self._generate_archimedean_spiral(num_points, k_max_m_inv, num_revs)
            self.nominal_gradients_time = self._kspace_to_gradients(self.optimized_trajectory_kspace)
        elif traj_type == 'radial': # Placeholder for radial
            # For radial, one might define number of spokes, points per spoke
            spokes = design_params.get('num_spokes', 128)
            pts_per_spoke = num_points // spokes
            k_radial = np.linspace(0, k_max_m_inv, pts_per_spoke)
            all_spokes_k = []
            for i in range(spokes):
                angle = i * np.pi / spokes
                kx_spoke = k_radial * np.cos(angle)
                ky_spoke = k_radial * np.sin(angle)
                all_spokes_k.append(np.stack([kx_spoke, ky_spoke], axis=-1))
            self.optimized_trajectory_kspace = np.concatenate(all_spokes_k, axis=0)
            self.nominal_gradients_time = self._kspace_to_gradients(self.optimized_trajectory_kspace)
        else:
            raise ValueError(f"Unsupported trajectory type: {traj_type}")

        print(f"Generated nominal {traj_type} trajectory with {self.optimized_trajectory_kspace.shape[0]} points.")
        return self.optimized_trajectory_kspace


    def apply_pre_emphasis(self, nominal_gradients_override=None):
        """
        Applies pre-emphasis to nominal gradients. (Conceptual/Simplified)
        Uses the TrajectoryPredictor's internal logic for actual prediction,
        so this method is more about setting up the *input* to achieve the *desired nominal output*.
        True pre-emphasis (inverse problem) is G_input = IFFT( FFT(G_desired) / GIRF ).
        """
        target_gradients = nominal_gradients_override if nominal_gradients_override is not None else self.nominal_gradients_time
        if target_gradients is None:
            raise ValueError("Nominal gradients not available. Design a trajectory first.")

        if not self.girf_spectra or not self.trajectory_predictor:
            print("Warning: GIRF spectra or TrajectoryPredictor not available. Skipping pre-emphasis, using nominal gradients.")
            self.preemphasized_gradients_time = target_gradients.copy()
            return self.preemphasized_gradients_time

        print("Applying pre-emphasis (simplified inverse GIRF)...")
        num_points, num_axes = target_gradients.shape
        self.preemphasized_gradients_time = np.zeros_like(target_gradients, dtype=np.complex128)

        current_axes_names = getattr(self.trajectory_predictor, 'axes_names', [f'axis_{i}' for i in range(num_axes)])
        if len(current_axes_names) != num_axes: # Fallback if axes_names not set in predictor
            current_axes_names = [f'axis_{i}' for i in range(num_axes)]


        for i in range(num_axes):
            axis_name = current_axes_names[i]
            if axis_name not in self.girf_spectra:
                print(f"Warning: GIRF for axis '{axis_name}' not found. Using nominal gradient for this axis in pre-emphasis.")
                self.preemphasized_gradients_time[:, i] = target_gradients[:, i]
                continue

            desired_grad_axis = target_gradients[:, i]
            girf_spectrum_axis = self.girf_spectra[axis_name]

            fft_desired_grad = np.fft.fft(desired_grad_axis)

            n_fft = len(fft_desired_grad) # Length of FFT of current gradient
            if len(girf_spectrum_axis) < n_fft:
                padding_val = girf_spectrum_axis[-1] if len(girf_spectrum_axis) > 0 else 1.0
                girf_resized = np.pad(girf_spectrum_axis.astype(np.complex128), (0, n_fft - len(girf_spectrum_axis)), 'constant', constant_values=(padding_val,))
            elif len(girf_spectrum_axis) > n_fft:
                girf_resized = girf_spectrum_axis[:n_fft]
            else:
                girf_resized = girf_spectrum_axis.astype(np.complex128) # Ensure complex type

            # Inverse operation: G_input_fft = G_desired_fft / GIRF
            # Add regularization to avoid division by zero/small numbers in GIRF.
            regularization_factor = 1e-6 * np.max(np.abs(girf_resized)) if np.max(np.abs(girf_resized)) > 0 else 1e-6

            # Create regularized GIRF: if |GIRF| is small, replace GIRF^-1 with 0 or a small number.
            # For G_input = G_desired / GIRF, if GIRF is small, G_input blows up.
            # So, where GIRF is small, we want G_input to be small or G_desired to be small.
            # A common approach is Wiener filter like: G_input_fft = G_desired_fft * conj(GIRF) / (|GIRF|^2 + lambda)
            conj_girf = np.conj(girf_resized)
            abs_sq_girf = np.abs(girf_resized)**2

            #fft_preemphasized_grad = fft_desired_grad * conj_girf / (abs_sq_girf + regularization_factor)
            # Simpler: fft_preemphasized_grad = fft_desired_grad / (girf_resized + epsilon)
            # but need to handle small girf_resized values to prevent blowing up input grads.
            # If GIRF element is very small, implies system cannot produce that frequency.
            # So, desired_grad at that frequency should also be zero, or we accept distortion.
            # For pre-emphasis, we want G_actual = G_input * GIRF approx G_desired
            # So G_input approx G_desired / GIRF.
            # If GIRF(f) is near 0, G_input(f) would be huge. This is limited by Gmax/Smax.
            # This is where iterative methods come in.
            # Placeholder: just apply division with regularization

            # Create a copy of girf_resized for safe division
            denominator = girf_resized.copy()
            small_den_mask = np.abs(denominator) < regularization_factor
            denominator[small_den_mask] = regularization_factor # or inf to make output zero

            fft_preemphasized_grad = fft_desired_grad / denominator

            self.preemphasized_gradients_time[:, i] = np.fft.ifft(fft_preemphasized_grad).real

        print("Pre-emphasis applied (simplified).")
        return self.preemphasized_gradients_time


    def verify_constraints(self, gradient_waveforms_to_check, pns_check=True):
        """ Verifies hardware and PNS constraints for given gradient waveforms. """
        hw_ok, hw_violations = self._check_hardware_constraints(gradient_waveforms_to_check)

        pns_ok = True # Assume PNS ok if not checked or no model
        pns_report = {"status": "Not Checked"}

        if pns_check and self.pns_model:
            # PNSModel needs slew rates
            # slew_rates = np.diff(gradient_waveforms_to_check, axis=0) / self.dt
            # To match length and handle first point:
            slew_rates_calc = np.diff(gradient_waveforms_to_check, axis=0, prepend=gradient_waveforms_to_check[0:1,:]) / self.dt
            slew_rates_calc[0,:] = gradient_waveforms_to_check[0,:] / self.dt # Slew from 0 to G[0]

            # PNSModel might expect slew_rates as dict {'axis': array}
            # Assuming gradient_waveforms_to_check is (T, N_axes)
            num_axes_pns = gradient_waveforms_to_check.shape[1]
            # Use generic axis names if specific aren't easily available
            # This part needs careful alignment with PNSModel's expected input structure
            pns_slew_input = {f'axis_{idx}': slew_rates_calc[:, idx] for idx in range(num_axes_pns)}

            try:
        # This depends on PNSModel's API.
        try:
            # PNSModel's compute_pns now takes slew_rates directly.
            pns_values_computed_ts = self.pns_model.compute_pns(pns_slew_input)
            pns_ok, peak_pns_val = self.pns_model.check_limits(pns_values_computed_ts)
            pns_report = {
                "status": "Checked",
                "is_compliant": pns_ok,
                "peak_total_normalized_pns": peak_pns_val,
                # "pns_timeseries": pns_values_computed_ts, # Optionally return full timeseries
                "thresholds_used": self.pns_model.pns_thresholds # Report thresholds from model
            }
        except Exception as e:
            pns_ok = False # Fail safe
            pns_report = {"status": "Error during PNS check", "error_message": str(e)}
            print(f"Error during PNS check: {e}")

    # Combine per-axis and vector slew checks for overall hardware status
    # The original hw_ok from _check_hardware_constraints now contains per-axis G and S plus vector S.
    # We just need to correctly report it.
    # The _check_hardware_constraints should return a more structured report.
    # Let's assume _check_hardware_constraints returns dict like:
    # {'G_ok': bool, 'S_axis_ok': bool, 'S_vec_ok': bool, 'violations': {...}}
    # Then hw_ok would be all(G_ok, S_axis_ok, S_vec_ok)

    # For now, verify_constraints will just pass through the hw_ok from _check_hardware_constraints,
    # which needs to be updated to aggregate these checks.
    # The 'hw_report' from _check_hardware_constraints now contains all hardware check details.

    print(f"Constraint Verification: HW Overall OK: {overall_hw_status}, PNS OK: {pns_ok}")
    return overall_hw_status, pns_ok, {"hw_report": hw_details_report, "pns_report": pns_report}


    def calculate_gradient_duty_cycles(self, gradient_waveforms_dict_or_array,
                                     active_threshold_factor=0.1):
        """
        Calculates the duty cycle for each gradient axis based on an activity threshold.
        Duty cycle: Percentage of time gradient magnitude is >= active_threshold_factor * Gmax_axis.

        Args:
            gradient_waveforms_dict_or_array (dict or np.ndarray):
                Gradient waveforms {'axis': waveform} or array (T, N_axes).
            active_threshold_factor (float): Factor of Gmax above which gradient is "active".

        Returns:
            dict: {axis_name (str): duty_cycle_percent (float)}
        """
        if utils is None:
            raise ImportError("girf.utils module not imported, cannot standardize trajectory format.")

        # Standardize to dict format for easier per-axis Gmax handling
        # Try to get number of axes for default_axis_names
        num_axes = 1
        if isinstance(gradient_waveforms_dict_or_array, dict):
            num_axes = len(gradient_waveforms_dict_or_array)
        elif hasattr(gradient_waveforms_dict_or_array, 'shape'):
            num_axes = gradient_waveforms_dict_or_array.shape[1] if gradient_waveforms_dict_or_array.ndim > 1 else 1

        default_names = ['x','y','z'][:num_axes] if num_axes <=3 else [f'axis_{i}' for i in range(num_axes)]

        grad_waveforms_dict = utils.standardize_trajectory_format(
            gradient_waveforms_dict_or_array, target_format='dict', default_axis_names=default_names
        )

        duty_cycles = {}
        gmax_scalar = self.constraints.get('Gmax_T_per_m') # Overall Gmax

        for axis, waveform in grad_waveforms_dict.items():
            waveform = np.asarray(waveform)
            if waveform.size == 0:
                duty_cycles[axis] = 0.0
                continue

            # Determine Gmax for this specific axis.
            # This is simplified: assumes either a per-axis Gmax_axis (e.g. Gmax_x) is in constraints,
            # or falls back to the global Gmax_T_per_m. A more robust system might require explicit per-axis Gmax.
            gmax_axis = self.constraints.get(f'Gmax_{axis}_T_per_m', gmax_scalar)
            if gmax_axis is None: # Should not happen if Gmax_T_per_m has a default
                 print(f"Warning: Gmax not defined for axis {axis} or globally. Skipping duty cycle calc for this axis.")
                 duty_cycles[axis] = np.nan
                 continue

            threshold = active_threshold_factor * gmax_axis
            active_time_points = np.sum(np.abs(waveform) >= threshold)
            total_time_points = waveform.shape[0]
            duty_cycle_percent = (active_time_points / total_time_points) * 100.0 if total_time_points > 0 else 0.0
            duty_cycles[axis] = duty_cycle_percent

        return duty_cycles

    def check_thermal_limits_conceptual(self, gradient_waveforms_dict_or_array, dt_for_calc=None):
        """
        Conceptual check of thermal limits based on simplified duty cycle.
        Requires 'thermal_limits': {axis: {'max_duty_cycle_percent': val}} in self.constraints.

        Args:
            gradient_waveforms_dict_or_array (dict or np.ndarray): Gradient waveforms.
            dt_for_calc (float, optional): Time step. If None, uses self.dt.

        Returns:
            dict: {'thermal_ok': bool_overall, 'details': report_per_axis}
        """
        # dt is not directly used in this simplified duty cycle calc but kept for API consistency if RMS were used.
        # The active_threshold_factor for duty cycle calculation is fixed internally for now.

        thermal_limits_config = self.constraints.get('thermal_limits')
        if not thermal_limits_config or not isinstance(thermal_limits_config, dict):
            return {'thermal_ok': True, 'details': {}, 'message': 'No thermal_limits defined in constraints.'}

        # Gmax values for each axis are needed for calculate_gradient_duty_cycles's threshold logic
        # This part assumes Gmax is either globally defined or per-axis like Gmax_x, Gmax_y etc.
        # For this conceptual check, calculate_gradient_duty_cycles will use self.constraints['Gmax_T_per_m'] if specific not found.

        calculated_duty_cycles = self.calculate_gradient_duty_cycles(
            gradient_waveforms_dict_or_array,
            active_threshold_factor=self.constraints.get('thermal_duty_cycle_active_threshold_factor', 0.1)
        )

        all_axes_ok = True
        report = {}

        for axis, duty_cycle in calculated_duty_cycles.items():
            if np.isnan(duty_cycle): # Gmax was not found for this axis
                report[axis] = {'duty_cycle_percent': np.nan, 'limit_percent': np.nan,
                                'is_ok': False, 'message': f'Could not calculate duty cycle for axis {axis} (Gmax missing?).'}
                all_axes_ok = False
                continue

            axis_limit_info = thermal_limits_config.get(axis)
            if axis_limit_info and 'max_duty_cycle_percent' in axis_limit_info:
                limit = axis_limit_info['max_duty_cycle_percent']
                is_ok = duty_cycle <= limit
                report[axis] = {'duty_cycle_percent': duty_cycle, 'limit_percent': limit, 'is_ok': is_ok}
                if not is_ok:
                    all_axes_ok = False
            else:
                report[axis] = {'duty_cycle_percent': duty_cycle, 'limit_percent': None,
                                'is_ok': True, 'message': f'No duty cycle limit defined for axis {axis}.'}

        return {'thermal_ok': all_axes_ok, 'details': report}


if __name__ == '__main__':
    print("--- Running TrajectoryPlanner Example with Actual Logic ---")

    # Basic Setup
    dt_val = 4e-6  # 4 us
    constraints_config = {
        'Gmax_T_per_m': 0.040,          # 40 mT/m
        'Smax_T_per_m_per_s': 180,      # 180 T/m/s
        'Smax_vector_T_per_m_per_s': 200, # Vector Slew Max (e.g., for 3D trajectories)
        'pns_model_thresholds': {       # Explicit thresholds for PNSModel
            'rheobase_T_per_s': 20.0,
            'chronaxie_ms': 0.36,
            'max_total_pns_normalized': 0.8
        },
        'tolerance': 1e-7, # Tolerance for hardware constraint checks
        'thermal_limits': { # New thermal limits section
            'x': {'max_duty_cycle_percent': 60},
            'y': {'max_duty_cycle_percent': 50},
            'z': {'max_duty_cycle_percent': 45, 'some_other_thermal_param': 123} # Example other param
        },
        'thermal_duty_cycle_active_threshold_factor': 0.05 # 5% of Gmax to be "active"
    }

    # Dummy GIRF (e.g., identity for now, or simple attenuation)
    num_pts_test = 1024
    dummy_girf_spectra_planner = {
        'axis_0': np.ones(num_pts_test, dtype=np.complex128) * 0.9, # 10% attenuation
        'axis_1': np.ones(num_pts_test, dtype=np.complex128) * 0.85 # 15% attenuation
    }

    # Initialize PNSModel and TrajectoryPredictor if available, else they'll be None
    pns_model_inst = PNSModel(pns_thresholds={'axis_0': 1.0, 'axis_1': 1.0}) if PNSModel else None # Dummy thresholds
    # The predictor needs GIRF spectra. Planner loads it and can pass it.
    predictor_inst = TrajectoryPredictor(dt=dt_val, girf_spectra=dummy_girf_spectra_planner) if TrajectoryPredictor else None

    planner = TrajectoryPlanner(girf_spectra=dummy_girf_spectra_planner,
                                constraints=constraints_config,
                                dt=dt_val,
                                pns_model_instance=pns_model_inst,
                                trajectory_predictor_instance=predictor_inst)

    # 1. Design a nominal trajectory (e.g., spiral)
    design_p = {'num_points': num_pts_test, 'k_max_m_inv': 250, 'num_revolutions': 20}
    nominal_k_traj = planner.design_trajectory(traj_type='spiral', design_params=design_p)
    print(f"Designed nominal trajectory of shape: {nominal_k_traj.shape}")

    # Retrieve nominal gradients computed during design
    nominal_gradients = planner.nominal_gradients_time
    if nominal_gradients is not None:
        print(f"Nominal gradients shape: {nominal_gradients.shape}")

        # 2. Verify constraints for the nominal (un-preemphasized) gradients
        print("\nVerifying constraints for NOMINAL gradients:")
        # The nominal_gradients are stored as an array (T, Ndims)
        hw_ok_nom, pns_ok_nom, report_nom = planner.verify_constraints(nominal_gradients, pns_check=True)
        print(f"Nominal - HW OK: {hw_ok_nom}, PNS OK: {pns_ok_nom}")
        if not hw_ok_nom: print(f"  HW Report: {report_nom['hw_report']}")
        if not pns_ok_nom: print(f"  PNS Report: {report_nom['pns_report']}")

        # 2b. Check thermal limits for nominal gradients
        print("\nChecking conceptual thermal limits for NOMINAL gradients:")
        # Need to pass gradients as dict for current duty cycle calc if Gmax is per axis from dict keys
        nominal_gradients_dict = utils.standardize_trajectory_format(nominal_gradients, target_format='dict',
                                                                    default_axis_names=['x','y','z'][:nominal_gradients.shape[1]])
        thermal_report_nom = planner.check_thermal_limits_conceptual(nominal_gradients_dict)
        print(f"Nominal - Thermal OK: {thermal_report_nom['thermal_ok']}")
        for axis, details in thermal_report_nom.get('details', {}).items():
            print(f"  Axis {axis}: Duty Cycle={details.get('duty_cycle_percent',0):.1f}%, Limit={details.get('limit_percent','N/A')}%, OK={details.get('is_ok')}")


        # 3. Apply pre-emphasis (simplified)
        preemph_gradients = planner.apply_pre_emphasis() # Uses planner.nominal_gradients_time (array)
        print(f"\nPre-emphasized gradients shape: {preemph_gradients.shape}")

        # 4. Verify constraints for the PRE-EMPHASIZED gradients
        print("\nVerifying constraints for PRE-EMPHASIZED gradients:")
        hw_ok_pre, pns_ok_pre, report_pre = planner.verify_constraints(preemph_gradients, pns_check=True)
        print(f"Pre-emphasized - HW OK: {hw_ok_pre}, PNS OK: {pns_ok_pre}")
        if not hw_ok_pre: print(f"  HW Report: {report_pre['hw_report']}")
        if not pns_ok_pre: print(f"  PNS Report: {report_pre['pns_report']}")

        # 4b. Check thermal limits for pre-emphasized gradients
        print("\nChecking conceptual thermal limits for PRE-EMPHASIZED gradients:")
        preemph_gradients_dict = utils.standardize_trajectory_format(preemph_gradients, target_format='dict',
                                                                    default_axis_names=['x','y','z'][:preemph_gradients.shape[1]])
        thermal_report_pre = planner.check_thermal_limits_conceptual(preemph_gradients_dict)
        print(f"Pre-emphasized - Thermal OK: {thermal_report_pre['thermal_ok']}")
        for axis, details in thermal_report_pre.get('details', {}).items():
            print(f"  Axis {axis}: Duty Cycle={details.get('duty_cycle_percent',0):.1f}%, Limit={details.get('limit_percent','N/A')}%, OK={details.get('is_ok')}")


        # 5. Convert pre-emphasized gradients back to k-space (final optimized trajectory)
        final_k_traj_optimized = planner._gradients_to_kspace(preemph_gradients)
        planner.optimized_trajectory_kspace = final_k_traj_optimized # Store it
        print(f"\nFinal optimized k-space trajectory (from pre-emph grads) shape: {final_k_traj_optimized.shape}")

    else:
        print("Could not retrieve nominal gradients from planner.")

    print("\n--- TrajectoryPlanner Example Finished ---")
