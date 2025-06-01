# trajgen/optimizers/cost_components.py
"""
Individual cost component functions for trajectory optimization.
These functions take a Trajectory object and relevant parameters,
and return a scalar cost/penalty.
"""
import numpy as np
from trajgen.trajectory import Trajectory # Assuming Trajectory class is in trajgen.trajectory
from typing import Optional

__all__ = [
    'calculate_hardware_penalty',
    'calculate_gradient_roughness_penalty',
    'calculate_pns_proxy_penalty'
]

def calculate_hardware_penalty(
    trajectory: Trajectory,
    grad_limit_Tm_per_m: Optional[float] = None,
    slew_limit_Tm_per_s_per_m: Optional[float] = None, # Assuming T/m/s
    penalty_factor: float = 100.0
) -> float:
    """
    Calculates a penalty based on violations of gradient and slew rate limits.
    Penalty is quadratic for the normalized amount exceeding the limit.

    Args:
        trajectory (Trajectory): The trajectory object to evaluate.
        grad_limit_Tm_per_m (Optional[float]): Maximum gradient limit in T/m.
        slew_limit_Tm_per_s_per_m (Optional[float]): Maximum slew rate limit in T/m/s.
            (Note: prompt used T/m/s/m, but T/m/s is typical for slew rate itself).
        penalty_factor (float): Factor to scale the penalty.

    Returns:
        float: Calculated penalty value.
    """
    cost = 0.0

    if grad_limit_Tm_per_m is not None and grad_limit_Tm_per_m > 0: # Ensure limit is positive
        max_grad_achieved = trajectory.get_max_grad_Tm()
        if max_grad_achieved is not None and max_grad_achieved > grad_limit_Tm_per_m:
            cost += penalty_factor * ((max_grad_achieved - grad_limit_Tm_per_m) / grad_limit_Tm_per_m)**2

    if slew_limit_Tm_per_s_per_m is not None and slew_limit_Tm_per_s_per_m > 0: # Ensure limit is positive
        max_slew_achieved = trajectory.get_max_slew_Tm_per_s() # This is norm of slew vectors
        if max_slew_achieved is not None and max_slew_achieved > slew_limit_Tm_per_s_per_m:
            cost += penalty_factor * ((max_slew_achieved - slew_limit_Tm_per_s_per_m) / slew_limit_Tm_per_s_per_m)**2
    return cost

def calculate_gradient_roughness_penalty(
    trajectory: Trajectory,
    penalty_factor: float = 1.0
) -> float:
    """
    Calculates a penalty based on the roughness of the gradient waveforms.
    Uses the sum of squared magnitudes of the slew rate vectors, normalized by the
    number of slew points. This penalizes rapid changes in gradient throughout
    the trajectory, which can contribute to acoustic noise or vibrations.

    Args:
        trajectory (Trajectory): The trajectory object to evaluate.
        penalty_factor (float): Factor to scale the penalty.

    Returns:
        float: Calculated penalty value.
    """
    gradients = trajectory.get_gradient_waveforms_Tm() # Shape (D, N)
    if gradients is None or gradients.shape[1] < 2: # Need at least 2 gradient points for diff
        return 0.0

    dt = trajectory.dt_seconds
    if dt is None or dt <= 1e-9: # dt must be positive and non-zero
        # Cannot calculate slew rate, or it will be infinite/undefined.
        # Return a large penalty if gradients are not flat, or 0 if they are.
        if np.allclose(gradients, gradients[:,0:1]): # Check if all gradient vectors are the same
            return 0.0
        return penalty_factor * 1e12 # Large penalty if dt is invalid and grads change

    # slew_waveforms shape (D, N-1)
    slew_waveforms = np.diff(gradients, n=1, axis=1) / dt

    if slew_waveforms.size == 0:
        return 0.0

    # Sum of squared magnitudes of slew vectors (Frobenius norm squared of slew matrix, essentially)
    roughness_metric = np.sum(np.linalg.norm(slew_waveforms, axis=0)**2)

    num_slew_points = slew_waveforms.shape[1]
    normalized_roughness = roughness_metric / num_slew_points if num_slew_points > 0 else 0.0

    return penalty_factor * normalized_roughness

def calculate_pns_proxy_penalty(
    trajectory: Trajectory,
    pns_threshold_T_per_s: float = 180.0, # Example threshold for max slew rate norm (T/m/s)
    penalty_factor: float = 10.0
) -> float:
    """
    Calculates a Peripheral Nerve Stimulation (PNS) proxy penalty.
    This simplified model penalizes trajectories where the maximum vector norm of the
    slew rate exceeds a given threshold. True PNS is more complex and depends on
    factors like dB/dt on specific axes, body part, and pulse duration.

    Args:
        trajectory (Trajectory): The trajectory object to evaluate.
        pns_threshold_T_per_s (float): The threshold for the maximum slew rate (T/m/s).
                                       If max achieved slew exceeds this, a penalty is incurred.
        penalty_factor (float): Factor to scale the penalty.

    Returns:
        float: Calculated penalty value.
    """
    cost = 0.0
    if pns_threshold_T_per_s <= 0: # Threshold must be positive
        return 0.0

    max_slew_achieved = trajectory.get_max_slew_Tm_per_s() # This is norm of slew vectors

    if max_slew_achieved is not None and max_slew_achieved > pns_threshold_T_per_s:
        # Penalty for exceeding the threshold, quadratic in normalized excess
        cost = penalty_factor * ((max_slew_achieved - pns_threshold_T_per_s) / pns_threshold_T_per_s)**2
    return cost
