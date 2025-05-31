import unittest
import numpy as np
import json
import tempfile
import os

# Adjust path for imports
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from girf.planner import TrajectoryPlanner
from girf.pns import PNSModel # For passing as instance
from girf.predictor import TrajectoryPredictor # For passing as instance
from girf.utils import DEFAULT_GAMMA_PROTON

class TestTrajectoryPlanner(unittest.TestCase):

    def setUp(self):
        self.dt = 4e-6  # seconds
        self.gamma = DEFAULT_GAMMA_PROTON
        self.constraints = {
            'Gmax_T_per_m': 0.040, # T/m
            'Smax_T_per_m_per_s': 180.0, # T/m/s
            'Smax_vector_T_per_m_per_s': 200.0, # T/m/s
            'pns_model_thresholds': {
                'rheobase_T_per_s': 22.0,
                'chronaxie_ms': 0.40,
                'max_total_pns_normalized': 0.85
            },
            'tolerance': 1e-7,
            'thermal_limits': { # Added for thermal tests
                'x': {'max_duty_cycle_percent': 50.0},
                'y': {'max_duty_cycle_percent': 60.0},
                # 'z' axis has no explicit duty cycle limit in this test config
            },
            'thermal_duty_cycle_active_threshold_factor': 0.1 # 10% of Gmax
        }
        self.num_points = 256

        # Dummy GIRF (identity for x, attenuated for y)
        self.dummy_girf_spectra = {
            'x': np.ones(self.num_points, dtype=np.complex128),
            'y': np.ones(self.num_points, dtype=np.complex128)
        }

        # Create a temporary GIRF file
        girf_to_save = {
            "gradient_axes": ["x", "y"],
            "girf_spectra_complex": {
                axis: [[val.real, val.imag] for val in spectrum]
                for axis, spectrum in self.dummy_girf_spectra.items()
            }, "waveform_params": {}
        }
        self.temp_girf_file_handle, self.temp_girf_file_path = tempfile.mkstemp(suffix='.json')
        with open(self.temp_girf_file_path, 'w') as f: json.dump(girf_to_save, f)

        # Instances of dependencies (can be mocked in more complex scenarios)
        self.pns_model_inst = PNSModel(pns_thresholds=self.constraints['pns_model_thresholds'], dt=self.dt)
        self.predictor_inst = TrajectoryPredictor(girf_spectra=self.dummy_girf_spectra, dt=self.dt, gamma=self.gamma)

        self.planner = TrajectoryPlanner(
            girf_spectra=self.dummy_girf_spectra,
            constraints=self.constraints,
            dt=self.dt,
            gamma=self.gamma,
            pns_model_instance=self.pns_model_inst,
            trajectory_predictor_instance=self.predictor_inst
        )

    def tearDown(self):
        os.close(self.temp_girf_file_handle)
        os.remove(self.temp_girf_file_path)

    def test_01_initialization(self):
        self.assertEqual(self.planner.dt, self.dt)
        self.assertEqual(self.planner.constraints['Gmax_T_per_m'], 0.040)
        self.assertIsNotNone(self.planner.pns_model)
        self.assertIsNotNone(self.planner.trajectory_predictor)
        self.assertTrue(np.array_equal(self.planner.girf_spectra['x'], self.dummy_girf_spectra['x']))

        with self.assertRaises(ValueError): # dt not provided
            TrajectoryPlanner(constraints=self.constraints)

    def test_02_load_girf(self):
        new_girf_data = {'x': np.full(self.num_points, 0.5, dtype=np.complex128)}
        self.planner.load_girf(new_girf_data) # From dict
        self.assertTrue(np.array_equal(self.planner.girf_spectra['x'], new_girf_data['x']))
        if self.planner.trajectory_predictor: # Check if predictor was updated
             self.assertTrue(np.array_equal(self.planner.trajectory_predictor.girf_spectra['x'], new_girf_data['x']))

        self.planner.load_girf(self.temp_girf_file_path) # From file
        self.assertTrue(np.allclose(self.planner.girf_spectra['x'], self.dummy_girf_spectra['x']))


    def test_03_kspace_gradient_conversions_internal(self):
        # These are simplified versions, main tests in test_utils.py
        # Test if they run and produce approx correct shapes/values
        k_test = np.zeros((self.num_points, 2))
        k_test[:,0] = np.linspace(0, 100, self.num_points)

        grads = self.planner._kspace_to_gradients(k_test)
        self.assertEqual(grads.shape, k_test.shape)
        # g[0] = k[0]/(g*dt)
        self.assertAlmostEqual(grads[0,0], k_test[0,0]/(self.gamma*self.dt))
        if self.num_points > 1:
            self.assertAlmostEqual(grads[1,0], (k_test[1,0]-k_test[0,0])/(self.gamma*self.dt))

        k_recon = self.planner._gradients_to_kspace(grads, initial_kspace_point=np.array([0.,0.]))
        np.testing.assert_array_almost_equal(k_recon, k_test, decimal=3)


    def test_04_check_hardware_constraints(self):
        # Gradients that are OK
        grads_ok_val = self.constraints['Gmax_T_per_m'] * 0.5
        slews_ok_val = self.constraints['Smax_T_per_m_per_s'] * 0.5

        # Create gradients that result in slews_ok_val: g[0]=0, g[1]=slews_ok_val*dt
        grads_hw_ok = np.zeros((self.num_points, 2))
        grads_hw_ok[1,0] = slews_ok_val * self.dt
        grads_hw_ok[:,:] = np.clip(grads_hw_ok, -grads_ok_val, grads_ok_val) # Ensure Gmax is also met
        # Manually ensure Gmax is met for the specific point we set for slew rate
        # Ensure Gmax is met for the specific point we set for slew rate
        test_grad_val_for_slew = min(grads_ok_val, slews_ok_val * self.dt)
        grads_hw_ok[1,0] = test_grad_val_for_slew
        grads_hw_ok[1,1] = test_grad_val_for_slew / 2


        overall_ok, hw_report = self.planner._check_hardware_constraints(grads_hw_ok)
        self.assertTrue(overall_ok, f"HW constraints failed unexpectedly. Report: {hw_report}")
        self.assertTrue(hw_report['G_ok'], f"Gmax check failed: {hw_report.get('G_details')}")
        self.assertTrue(hw_report['S_axis_ok'], f"Smax_axis check failed: {hw_report.get('S_axis_details')}")
        if self.planner.constraints.get('Smax_vector_T_per_m_per_s') is not None: # Check only if Smax_vector is defined
            self.assertTrue(hw_report['S_vector_ok'], f"Smax_vector check failed: {hw_report.get('S_vector_details')}")


        # Gradients that violate Gmax
        grads_gmax_viol = np.zeros((self.num_points, 2))
        grads_gmax_viol[10,0] = self.constraints['Gmax_T_per_m'] * 1.1
        overall_ok_g, hw_report_g = self.planner._check_hardware_constraints(grads_gmax_viol)
        self.assertFalse(overall_ok_g)
        self.assertFalse(hw_report_g['G_ok'])
        self.assertIn('first_exceeding_value_T_per_m', hw_report_g['G_details'])

        # Gradients that violate Smax (per-axis)
        grads_smax_axis_viol = np.zeros((self.num_points, 2))
        smax_axis = self.constraints['Smax_T_per_m_per_s']
        # Ensure grad value itself is within Gmax to isolate Smax violation
        grads_smax_axis_viol[1,0] = min(self.constraints['Gmax_T_per_m'], smax_axis * 1.1 * self.dt)

        overall_ok_sa, hw_report_sa = self.planner._check_hardware_constraints(grads_smax_axis_viol)
        self.assertFalse(overall_ok_sa)
        self.assertFalse(hw_report_sa['S_axis_ok'])
        self.assertIn('first_exceeding_value_T_per_m_per_s', hw_report_sa['S_axis_details'])

        # Gradients that violate Smax_vector but not necessarily per-axis Smax
        smax_vec = self.constraints.get('Smax_vector_T_per_m_per_s')
        if smax_vec is not None: # Only run this part if Smax_vector is defined
            smax_axis_val = self.constraints['Smax_T_per_m_per_s']
            # Want individual slews < smax_axis_val, but sqrt(slew_x^2 + slew_y^2) > smax_vec
            # Example: smax_axis = 180, smax_vec = 200.
            # slew_x = 150, slew_y = 150. Both < 180. sqrt(150^2+150^2) = 212 > 200.
            # Need to ensure grad values for these slews don't exceed Gmax

            # Target slew for each component:
            # Choose a value that is below smax_axis_val but whose vector sum (if on multiple axes) exceeds smax_vec
            # e.g. each component = smax_vec / sqrt(num_active_axes) * 0.8 (to be under smax_axis if smax_axis is close to smax_vec/sqrt(N))
            # and also smax_vec * 0.8 (if only one component is this large, it must also be < smax_axis)
            # This needs careful construction. Let's try specific values.
            # If smax_axis = 180, smax_vec = 200.
            # Set slew_x = 170, slew_y = 170. Vector = sqrt(170^2+170^2) = 170*sqrt(2) = 240.4 > 200.
            # And 170 < 180 (per-axis Smax).
            slew_comp_val = 170.0
            if slew_comp_val >= smax_axis_val : # Adjust if our chosen val is too high for per-axis
                slew_comp_val = smax_axis_val * 0.95

            # Ensure vector sum will violate if possible with this slew_comp_val
            # (slew_comp_val * sqrt(2) > smax_vec)
            if slew_comp_val * np.sqrt(2) < smax_vec: # if this test setup is not good for the current constraints
                print(f"Skipping Smax_vector specific violation test part as chosen slew_comp_val ({slew_comp_val}) times sqrt(2) is not greater than Smax_vector ({smax_vec})")
            else:
                grads_smax_vec_viol = np.zeros((self.num_points, 2))
                grad_for_slew_comp = min(self.constraints['Gmax_T_per_m'], slew_comp_val * self.dt)

                grads_smax_vec_viol[1,0] = grad_for_slew_comp
                grads_smax_vec_viol[1,1] = grad_for_slew_comp

                overall_ok_sv, hw_report_sv = self.planner._check_hardware_constraints(grads_smax_vec_viol)
                self.assertFalse(overall_ok_sv, f"Smax_vector should be violated. Report: {hw_report_sv}")
                self.assertTrue(hw_report_sv['S_axis_ok'], f"S_axis_ok should be true for this Smax_vector test. Report: {hw_report_sv}")
                self.assertFalse(hw_report_sv['S_vector_ok'], f"S_vector_ok should be false. Report: {hw_report_sv}")
                self.assertIn('first_exceeding_vector_value_T_per_m_per_s', hw_report_sv['S_vector_details'])
        else:
            print("Skipping Smax_vector specific violation test as Smax_vector_T_per_m_per_s not in constraints.")


    def test_05_design_trajectory_spiral(self):
        params = {'num_points': self.num_points, 'k_max_m_inv': 200, 'num_revolutions': 10}
        k_traj_spiral = self.planner.design_trajectory(traj_type='spiral', design_params=params)
        self.assertIsNotNone(k_traj_spiral)
        self.assertEqual(k_traj_spiral.shape, (self.num_points, 2)) # Assuming 2D spiral
        self.assertIsNotNone(self.planner.nominal_gradients_time)
        self.assertEqual(self.planner.nominal_gradients_time.shape, (self.num_points, 2))

    def test_06_design_trajectory_radial(self):
        params_radial = {'num_points': self.num_points, 'k_max_m_inv': 200, 'num_spokes': 32}
        k_traj_radial = self.planner.design_trajectory(traj_type='radial', design_params=params_radial)
        self.assertIsNotNone(k_traj_radial)
        # num_points might be adjusted by radial implementation (num_spokes * pts_per_spoke)
        self.assertTrue(k_traj_radial.shape[0] <= self.num_points) # Or exactly num_spokes * (num_points // num_spokes)
        self.assertEqual(k_traj_radial.shape[1], 2) # Assuming 2D radial
        self.assertIsNotNone(self.planner.nominal_gradients_time)


    def test_07_apply_pre_emphasis_conceptual(self):
        # Design a trajectory first to populate nominal_gradients_time
        self.planner.design_trajectory(traj_type='spiral',
                                       design_params={'num_points': self.num_points, 'k_max_m_inv': 200, 'num_revolutions': 10})

        # With identity GIRF, pre-emphasized should be very close to nominal
        preemph_grads_identity_girf = self.planner.apply_pre_emphasis()
        np.testing.assert_array_almost_equal(preemph_grads_identity_girf, self.planner.nominal_gradients_time, decimal=3)

        # Test with a GIRF that causes attenuation (e.g., 0.5x)
        attenuating_girf = {'x': np.full(self.num_points, 0.5, dtype=np.complex128),
                            'y': np.full(self.num_points, 0.5, dtype=np.complex128)}
        self.planner.load_girf(attenuating_girf) # This also updates predictor's GIRF

        preemph_grads_atten_girf = self.planner.apply_pre_emphasis()
        # Pre-emphasized grads should be roughly nominal_grads / 0.5 = nominal_grads * 2
        # This is a very rough check due to FFTs, regularization, etc.
        # Check magnitudes are larger on average
        self.assertTrue(np.mean(np.abs(preemph_grads_atten_girf))) > np.mean(np.abs(self.planner.nominal_gradients_time)))

        # Test without GIRF (should return nominal)
        self.planner.girf_spectra = {}
        if self.planner.trajectory_predictor: self.planner.trajectory_predictor.girf_spectra = {}
        preemph_no_girf = self.planner.apply_pre_emphasis()
        np.testing.assert_array_equal(preemph_no_girf, self.planner.nominal_gradients_time)


    def test_08_verify_constraints(self):
        # Design a trajectory
        self.planner.design_trajectory(traj_type='spiral',
                                       design_params={'num_points': self.num_points, 'k_max_m_inv': 100, 'num_revolutions': 5}) # smaller kmax for less aggressive grads

        grads_to_check = self.planner.nominal_gradients_time

        # Case 1: Should pass (hopefully, with low k_max)
        # May need to adjust constraints or trajectory params for this to pass reliably
        # Forcing PNS model to be very lenient for this test part
        if self.planner.pns_model: # Check if PNS model was successfully instantiated
            original_pns_thresholds = self.planner.pns_model.pns_thresholds.copy()
            self.planner.pns_model.pns_thresholds['max_total_pns_normalized'] = 10.0 # Very high limit
            self.planner.pns_model.pns_thresholds['rheobase_T_per_s'] = 200.0

        overall_hw_status, pns_ok, report = self.planner.verify_constraints(grads_to_check, pns_check=True)

        self.assertIsInstance(overall_hw_status, bool)
        self.assertIsInstance(pns_ok, bool)
        self.assertIn('hw_report', report)
        self.assertIn('pns_report', report)

        # Example: Check if overall hardware status is reflected in the report
        self.assertEqual(overall_hw_status, report['hw_report']['overall_ok'])

        if self.planner.pns_model:
            self.planner.pns_model.pns_thresholds = original_pns_thresholds # Restore


        # Case 2: Introduce a clear violation for Gmax
        grads_gmax_fail = grads_to_check.copy()
        grads_gmax_fail[self.num_points//2, 0] = self.constraints['Gmax_T_per_m'] * 2.0 # Exceed Gmax

        hw_ok_f, _, report_f = self.planner.verify_constraints(grads_gmax_fail, pns_check=False) # PNS check off
        self.assertFalse(hw_ok_f)
        self.assertFalse(report_f['hw_report']['G_ok'])
        self.assertIn('first_exceeding_value_T_per_m', report_f['hw_report']['G_details'])


    def test_09_verify_constraints_no_pns_model(self):
        # Test verify_constraints when PNS model is not available in planner
        planner_no_pns = TrajectoryPlanner(girf_spectra=self.dummy_girf_spectra,
                                           constraints=self.constraints, dt=self.dt, gamma=self.gamma,
                                           pns_model_instance=None, # Explicitly None
                                           trajectory_predictor_instance=self.predictor_inst)

        planner_no_pns.design_trajectory(traj_type='spiral', design_params={'num_points': self.num_points})
        grads_to_check = planner_no_pns.nominal_gradients_time
        hw_ok, pns_ok, report = planner_no_pns.verify_constraints(grads_to_check, pns_check=True)

        self.assertIsInstance(hw_ok, bool) # HW check should still run
        self.assertTrue(pns_ok) # PNS should be true (or "not applicable") if model is None
        self.assertEqual(report['pns_report']['status'], "Not Checked")

    # --- Tests for Thermal Related Methods ---
    def test_10_calculate_gradient_duty_cycles(self):
        # Create a waveform: 50% of points are >= 0.1 * Gmax
        gmax_val = self.constraints['Gmax_T_per_m']
        active_thresh_factor = self.constraints['thermal_duty_cycle_active_threshold_factor'] # 0.1
        threshold = active_thresh_factor * gmax_val # 0.1 * 0.04 = 0.004

        wf_x = np.zeros(self.num_points)
        wf_x[:self.num_points // 2] = threshold * 1.5 # First half is active
        wf_y = np.ones(self.num_points) * threshold * 0.5 # All points inactive

        grads_dict = {'x': wf_x, 'y': wf_y}
        duty_cycles = self.planner.calculate_gradient_duty_cycles(grads_dict)

        self.assertAlmostEqual(duty_cycles['x'], 50.0)
        self.assertAlmostEqual(duty_cycles['y'], 0.0)

        # Test with Gmax_axis specific (if planner was configured with Gmax_x etc.)
        # For now, it uses the global Gmax_T_per_m from constraints if Gmax_axis not found in constraints.
        # If self.constraints had 'Gmax_x_T_per_m': 0.02, then threshold for x would be 0.002.
        # Let's test that Gmax fallback
        planner_global_gmax = TrajectoryPlanner(constraints={'Gmax_T_per_m': 0.05, 'thermal_duty_cycle_active_threshold_factor': 0.1}, dt=self.dt)
        wf_g_global_test = np.zeros(100)
        wf_g_global_test[:30] = 0.005 # 0.005 is 0.1 * 0.05 (Gmax)
        duty_global = planner_global_gmax.calculate_gradient_duty_cycles({'testax': wf_g_global_test})
        self.assertAlmostEqual(duty_global['testax'], 30.0)


    def test_11_check_thermal_limits_conceptual(self):
        # X: 50% duty, limit 50% -> OK
        # Y: 70% duty, limit 60% -> Fail
        # Z: 30% duty, no limit defined -> OK (as per current logic)
        gmax_val = self.constraints['Gmax_T_per_m']
        active_thresh_factor = self.constraints['thermal_duty_cycle_active_threshold_factor']
        threshold = active_thresh_factor * gmax_val

        wf_x = np.zeros(self.num_points); wf_x[:int(0.50 * self.num_points)] = threshold * 2.0
        wf_y = np.zeros(self.num_points); wf_y[:int(0.70 * self.num_points)] = threshold * 2.0
        wf_z = np.zeros(self.num_points); wf_z[:int(0.30 * self.num_points)] = threshold * 2.0

        grads_for_thermal_test = {'x': wf_x, 'y': wf_y, 'z': wf_z}

        thermal_results = self.planner.check_thermal_limits_conceptual(grads_for_thermal_test)

        self.assertFalse(thermal_results['thermal_ok']) # Overall should be False due to Y
        self.assertTrue(thermal_results['details']['x']['is_ok'])
        self.assertAlmostEqual(thermal_results['details']['x']['duty_cycle_percent'], 50.0)
        self.assertFalse(thermal_results['details']['y']['is_ok'])
        self.assertAlmostEqual(thermal_results['details']['y']['duty_cycle_percent'], 70.0)
        self.assertTrue(thermal_results['details']['z']['is_ok']) # No limit for Z, so OK
        self.assertIn('No duty cycle limit defined for axis z', thermal_results['details']['z']['message'])

        # Test with no thermal limits in config
        constraints_no_thermal = self.constraints.copy()
        del constraints_no_thermal['thermal_limits']
        planner_no_thermal_cfg = TrajectoryPlanner(constraints=constraints_no_thermal, dt=self.dt)
        no_thermal_res = planner_no_thermal_cfg.check_thermal_limits_conceptual(grads_for_thermal_test)
        self.assertTrue(no_thermal_res['thermal_ok'])
        self.assertIn('No thermal_limits defined', no_thermal_res['message'])


if __name__ == '__main__':
    unittest.main()
