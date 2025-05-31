class PNSModel:
    def __init__(self, slew_rates=None, pns_thresholds=None):
        self.slew_rates = slew_rates  # Time-varying slew rates for each gradient axis (e.g., list of lists or dict of lists)
        self.pns_thresholds = pns_thresholds  # PNS thresholds (e.g., dB/dt or specific model parameters)
        self.pns_values = None  # Computed PNS values for the given slew rates

    def compute_pns(self):
        """
        Placeholder for computing PNS values based on slew rates and a PNS model.
        The model could be empirical, based on Maxwell equations, or a more complex physiological model.
        """
        print("Computing PNS values...")
        if self.slew_rates is None:
            raise ValueError("Slew rates not provided. Cannot compute PNS.")
        if self.pns_thresholds is None: # Or specific model parameters
            print("Warning: PNS thresholds or model parameters not provided. Using a dummy calculation.")
            # Fallback to a very simple dummy calculation if no thresholds are set
            # This part would be highly model-dependent.

        # Dummy PNS computation:
        # In reality, this would involve:
        # 1. A specific PNS prediction model (e.g., based on dB/dt, electric field simulation).
        # 2. Applying this model to the input self.slew_rates.
        # For this placeholder, let's simulate PNS as being proportional to the max slew rate,
        # and compare it against a simplified threshold.
        # Assume slew_rates is a dict like {'Gx': [sr1, sr2,...], 'Gy': [...], 'Gz': [...]}

        self.pns_values = {}
        if not isinstance(self.slew_rates, dict):
            print("Warning: Slew rates are not in the expected dictionary format. PNS computation might be incorrect.")
            # Simulate a single PNS value if slew_rates is just a list (e.g. for one axis or combined)
            if isinstance(self.slew_rates, list) and self.slew_rates:
                 max_slew = max(abs(sr) for sr in self.slew_rates if isinstance(sr, (int, float)))
                 # Dummy PNS value - could be a percentage of a fixed threshold
                 self.pns_values["combined"] = max_slew * 0.01 # Arbitrary scaling
            else:
                self.pns_values["combined"] = 0 # Default if format is unexpected
            print("PNS computation complete (simulated).")
            return self.pns_values

        for axis, sr_values in self.slew_rates.items():
            if not sr_values: # Empty list for an axis
                self.pns_values[axis] = 0.0
                continue

            # Simple model: PNS value is proportional to the maximum slew rate on that axis.
            # This is a placeholder and not physically accurate.
            try:
                max_slew_on_axis = max(abs(sr) for sr in sr_values if isinstance(sr, (int, float)))
                # The scaling factor (0.01 here) would come from the PNS model parameters/thresholds
                # For example, if pns_thresholds is a dict of scaling factors per axis:
                # scale_factor = self.pns_thresholds.get(axis, {}).get("scale_factor", 0.01)
                # self.pns_values[axis] = max_slew_on_axis * scale_factor
                self.pns_values[axis] = max_slew_on_axis * 0.01 # Using arbitrary scaling
            except (ValueError, TypeError): # Handles if sr_values is not numeric or empty after filtering
                print(f"Warning: Could not compute PNS for axis {axis} due to invalid slew rate data.")
                self.pns_values[axis] = None

        print("PNS computation complete (simulated).")
        return self.pns_values

    def check_limits(self):
        """
        Placeholder for checking if computed PNS values exceed predefined limits/thresholds.
        """
        print("Checking PNS limits...")
        if self.pns_values is None:
            raise ValueError("PNS values not computed yet. Run compute_pns() first.")
        if self.pns_thresholds is None:
            print("Warning: PNS thresholds not provided. Cannot check limits effectively. Assuming all within limits.")
            return True # Vacuously true if no thresholds to check against

        # Dummy limit check:
        # Assume pns_thresholds is a dictionary mapping axis to its max PNS value.
        # Or it could be a single value if PNS is combined.
        all_within_limits = True

        if not isinstance(self.pns_thresholds, dict): # Simple case: a single global threshold
            global_threshold = self.pns_thresholds if isinstance(self.pns_thresholds, (int, float)) else 1.0 # Default
            for axis, pns_val in self.pns_values.items():
                if pns_val is not None and pns_val > global_threshold:
                    print(f"Warning: PNS value for {axis} ({pns_val:.2f}) exceeds global threshold ({global_threshold:.2f}).")
                    all_within_limits = False
            return all_within_limits

        for axis, pns_val in self.pns_values.items():
            if pns_val is None:
                print(f"Notice: PNS value for axis {axis} could not be determined. Skipping limit check for this axis.")
                continue

            threshold = self.pns_thresholds.get(axis) # Get threshold specific to this axis
            if threshold is None:
                print(f"Warning: No PNS threshold defined for axis {axis}. Cannot check its limit.")
                continue

            if pns_val > threshold:
                print(f"Warning: PNS value for {axis} ({pns_val:.2f}) exceeds threshold ({threshold:.2f}).")
                all_within_limits = False
            else:
                print(f"PNS value for {axis} ({pns_val:.2f}) is within threshold ({threshold:.2f}).")

        if all_within_limits:
            print("All computed PNS values are within defined limits.")
        return all_within_limits

    def optimize_slew_rate(self, target_pns_limit_factor=0.8):
        """
        Placeholder for optimizing/adjusting slew rates to meet PNS limits.
        This is a complex optimization problem, often iterative.
        """
        print("Optimizing slew rates to meet PNS limits (simulated)...")
        if self.slew_rates is None:
            raise ValueError("Initial slew rates not provided.")
        if self.pns_thresholds is None:
            raise ValueError("PNS thresholds needed for optimization.")

        # Dummy optimization:
        # If any PNS value exceeds its threshold (scaled by target_pns_limit_factor),
        # simply scale down all slew rates on that axis by a fixed amount.
        # This is a naive approach. Real optimization is much more involved.

        self.compute_pns() # Ensure PNS values are current

        if not isinstance(self.slew_rates, dict) or not isinstance(self.pns_thresholds, dict):
            print("Warning: Slew rates or PNS thresholds are not in dict format. Skipping dummy optimization.")
            return self.slew_rates

        current_slew_rates = {axis: list(sr_vals) for axis, sr_vals in self.slew_rates.items()} # Make a mutable copy

        for axis, pns_val in self.pns_values.items():
            if pns_val is None: continue

            threshold = self.pns_thresholds.get(axis)
            if threshold is None: continue

            target_pns = threshold * target_pns_limit_factor
            if pns_val > target_pns:
                print(f"PNS for {axis} ({pns_val:.2f}) is above target ({target_pns:.2f}). Adjusting slew rates.")
                # Naive adjustment: scale down slew rates for this axis
                # The factor (e.g. 0.9) should ideally be derived from the PNS model's sensitivity
                scaling_factor = (target_pns / pns_val) if pns_val > 0 else 1.0
                # Ensure scaling factor is not excessively large if target_pns is much higher than pns_val (e.g. pns_val is negative)
                scaling_factor = min(scaling_factor, 1.0) if scaling_factor > 0 else max(scaling_factor, 0.1)


                current_slew_rates[axis] = [sr * scaling_factor for sr in current_slew_rates[axis]]
                print(f"Slew rates for axis {axis} scaled by {scaling_factor:.2f}.")

        self.slew_rates = current_slew_rates
        self.compute_pns() # Recompute PNS with adjusted slew rates
        print("Slew rate optimization step complete (simulated). New PNS values:")
        print(self.pns_values)
        self.check_limits()
        return self.slew_rates


if __name__ == '__main__':
    print("Starting PNS Model Example...")

    # Dummy slew rates (e.g., from a trajectory design, T/m/s)
    # These would typically be time series data for each gradient axis
    dummy_slew_rates = {
        'Gx': [10, 50, 100, 150, 100, 50, 10], # Max 150 T/m/s
        'Gy': [20, 60, 120, 100, 60, 20],    # Max 120 T/m/s
        'Gz': [5, 15, 30, 15, 5]             # Max 30 T/m/s
    }

    # Dummy PNS thresholds (e.g., unitless PNS value, or could be dB/dt limits)
    # This depends heavily on the specific PNS model implemented in compute_pns
    dummy_thresholds = {
        'Gx': 1.2,  # Max allowed PNS value for Gx (using the 0.01 scaling in dummy compute_pns)
        'Gy': 1.0,  # Max allowed PNS value for Gy
        'Gz': 0.4   # Max allowed PNS value for Gz
    }

    pns_model = PNSModel(slew_rates=dummy_slew_rates, pns_thresholds=dummy_thresholds)

    # 1. Compute PNS (simulated)
    pns_values = pns_model.compute_pns()
    print(f"Computed PNS values (simulated): {pns_values}")

    # 2. Check Limits (simulated)
    if pns_values:
        within_limits = pns_model.check_limits()
        print(f"PNS values within limits: {within_limits}")

        # 3. Optimize Slew Rates (simulated)
        if not within_limits:
            print("Attempting to optimize slew rates...")
            optimized_slew_rates = pns_model.optimize_slew_rate(target_pns_limit_factor=0.9)
            # print(f"Optimized slew rates (simulated): {optimized_slew_rates}")
            # print(f"PNS values after optimization: {pns_model.pns_values}")
            # pns_model.check_limits()
        else:
            print("No optimization needed as PNS is within limits.")

    # Example with undefined thresholds to show warning
    print("\nExample with undefined PNS thresholds:")
    pns_model_no_thresh = PNSModel(slew_rates=dummy_slew_rates)
    pns_model_no_thresh.compute_pns()
    pns_model_no_thresh.check_limits() # Should show warnings


    print("PNS Model Example Finished.")
