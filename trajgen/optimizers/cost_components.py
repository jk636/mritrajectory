# trajgen/optimizers/cost_components.py
"""
Individual cost component functions for trajectory optimization.
These functions take a Trajectory object and relevant parameters,
and return a scalar cost/penalty.
"""
import numpy as np
from scipy.spatial import Voronoi, ConvexHull, QhullError
from trajgen.trajectory import Trajectory
from typing import Optional, Callable, List

__all__ = [
    'calculate_hardware_penalty',
    'calculate_gradient_roughness_penalty',
    'calculate_pns_proxy_penalty',
    'calculate_signal_decay_penalty',
    'calculate_psf_incoherence_penalty' # New
]

# Helper function (module-level)
def _estimate_voronoi_cell_volume(
    vor: Voronoi,
    point_idx: int,
    trajectory_dim: int,
    k_max_rad_per_m: float,
    fallback_scale_factor: float = 5.0 # Multiplier for k_max^D for infinite cells
) -> float:
    """
    Estimates the volume (or area in 2D) of a Voronoi cell.
    For boundary (infinite) cells, returns a scaled large fallback value.
    For problematic finite cells (e.g., too few vertices for ConvexHull, degenerate),
    returns a small fallback value.
    """
    region_idx = vor.point_region[point_idx]
    region_vertices_indices = vor.regions[region_idx]

    if not region_vertices_indices or -1 in region_vertices_indices: # Infinite or empty region
        # Fallback for infinite cells: estimate based on k_max extent.
        # This is a heuristic, assuming it's a large cell at the periphery.
        # The volume of a sphere/circle of radius k_max is (4/3)pi*k_max^3 or pi*k_max^2.
        # We return a multiple of k_max^D as a rough upper bound.
        if trajectory_dim == 2:
            return fallback_scale_factor * (k_max_rad_per_m**2)
        elif trajectory_dim == 3:
            return fallback_scale_factor * (k_max_rad_per_m**3)
        return float('inf') # Should not happen if dim is 2 or 3

    cell_vertices = vor.vertices[region_vertices_indices]

    # Check for minimum number of vertices to form a simplex in this dimension
    if cell_vertices.shape[0] < trajectory_dim + 1:
        # Not enough vertices (e.g., collinear in 2D, coplanar in 3D for a 3-simplex)
        return 1e-9 * (k_max_rad_per_m**trajectory_dim) # Very small relative volume

    try:
        # ConvexHull computes area for 2D points, volume for 3D points.
        hull = ConvexHull(cell_vertices) # Qhull options 'QJ' could be added for joggling if needed
        return hull.volume
    except QhullError:
        # Handles cases like flat hull (e.g. coplanar points for a 3D hull attempt)
        # or other Qhull issues indicating a degenerate cell.
        # print(f"Warning: QhullError for point_idx {point_idx}. Cell vertices might be degenerate.")
        return 1e-9 * (k_max_rad_per_m**trajectory_dim) # Small volume for degenerate cells
    except Exception:
        # Catch any other errors during hull computation
        # print(f"Warning: Exception during ConvexHull for point_idx {point_idx}.")
        return 1e-8 * (k_max_rad_per_m**trajectory_dim) # Slightly larger small volume


def calculate_psf_incoherence_penalty(
    trajectory: Trajectory,
    k_max_rad_per_m: float,
    target_density_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    penalty_factor: float = 1.0,
    voronoi_qhull_options: str = 'Qbb Qc Qz'
) -> float:
    """
    Calculates a penalty based on k-space coverage incoherence and density variation
    using Voronoi cell analysis. Aims to promote uniform cell sizes (for incoherence)
    and adherence to a target density profile. Also penalizes large empty regions.

    Args:
        trajectory: The Trajectory object.
        k_max_rad_per_m: The expected maximum k-space radius (e.g., from resolution).
                         Used for scaling and boundary cell estimation.
        target_density_func: Optional function that takes k-space radii (rad/m)
                             and returns a desired relative density profile (higher means denser).
        penalty_factor: Multiplier for the final penalty.
        voronoi_qhull_options: Qhull options for Voronoi tessellation. 'Qz' adds point at infinity.

    Returns:
        A float representing the calculated penalty. Lower is better.
    """
    k_points_transposed = trajectory.kspace_points_rad_per_m.T # Voronoi expects (N, Dim)
    num_points, dim = k_points_transposed.shape

    if num_points < dim + 2 or dim not in [2,3]: # Need sufficient points for meaningful Voronoi/Hull
        return penalty_factor * 1000.0 # High fixed penalty for invalid inputs

    try:
        vor = Voronoi(k_points_transposed, qhull_options=voronoi_qhull_options)
    except QhullError:
        return penalty_factor * 1000.0 # High penalty if Voronoi fails (e.g. all points collinear)
    except Exception:
        return penalty_factor * 1000.0

    cell_volumes_list: List[float] = []
    for i in range(num_points):
        vol = _estimate_voronoi_cell_volume(vor, i, dim, k_max_rad_per_m)
        cell_volumes_list.append(vol)

    cell_volumes_np = np.array(cell_volumes_list)

    # Filter out non-finite values and extremely small (degenerate) cell volumes
    valid_cell_mask = np.isfinite(cell_volumes_np) & (cell_volumes_np > 1e-12 * (k_max_rad_per_m**dim))
    finite_cell_volumes = cell_volumes_np[valid_cell_mask]

    incoherence_penalty_val: float
    if finite_cell_volumes.size < num_points * 0.5 or finite_cell_volumes.size < dim + 2:
        incoherence_penalty_val = 100.0 # High if too many cells failed or are degenerate
    elif finite_cell_volumes.size > 0:
        mean_vol = np.mean(finite_cell_volumes)
        var_vol = np.var(finite_cell_volumes)
        # Coefficient of variation squared, penalizes non-uniformity of cell sizes
        incoherence_penalty_val = var_vol / (mean_vol**2 + 1e-12) if mean_vol > 1e-9 else var_vol
    else:
        incoherence_penalty_val = 200.0 # Very high if no valid cells found

    density_penalty_val = 0.0
    if target_density_func is not None and finite_cell_volumes.size > 0:
        radii_all_points = np.linalg.norm(k_points_transposed, axis=1)
        radii_for_finite_cells = radii_all_points[valid_cell_mask]

        if radii_for_finite_cells.size > 0:
            actual_density_approx = 1.0 / (finite_cell_volumes + 1e-12) # Inverse of cell volume
            desired_density_at_points = target_density_func(radii_for_finite_cells)

            # Normalize both densities to sum to 1 for profile comparison
            norm_actual_density = actual_density_approx / (np.sum(actual_density_approx) + 1e-12)
            norm_desired_density = desired_density_at_points / (np.sum(desired_density_at_points) + 1e-12)

            # Mean Squared Error of normalized densities
            density_penalty_val = np.mean((norm_actual_density - norm_desired_density)**2) * 1000 # Scale up MSE
        else:
            density_penalty_val = 50.0 # High if no valid cells to evaluate density

    empty_penalty_val = 0.0
    if vor.vertices.shape[0] > 0:
        min_dist_to_any_k_point_for_each_vertex_list: List[float] = []
        # Consider only Voronoi vertices within a reasonable bound of the k-space extent
        relevant_vertices = vor.vertices[np.linalg.norm(vor.vertices, axis=1) < 2.0 * k_max_rad_per_m]

        if relevant_vertices.shape[0] > 0:
            for vertex in relevant_vertices:
                distances_to_k_points = np.linalg.norm(k_points_transposed - vertex, axis=1)
                min_dist_to_any_k_point_for_each_vertex_list.append(np.min(distances_to_k_points))

        if min_dist_to_any_k_point_for_each_vertex_list:
            max_min_dist = np.max(min_dist_to_any_k_point_for_each_vertex_list)
            avg_spacing_approx = k_max_rad_per_m / (num_points**(1.0/dim) + 1e-9) if num_points > 0 else k_max_rad_per_m

            # Penalize if the largest "hole" (max_min_dist) is significantly larger than average spacing
            if max_min_dist > 1.5 * avg_spacing_approx and avg_spacing_approx > 1e-9: # Threshold factor 1.5
                 empty_penalty_val = ((max_min_dist - 1.5 * avg_spacing_approx) / avg_spacing_approx)**2 * 10 # Scaled
        # else: No relevant Voronoi vertices found or no k-points to compare against, no empty penalty.
    else: # No Voronoi vertices at all (e.g. very few input points)
        empty_penalty_val = 10.0 # Penalize if Voronoi diagram is degenerate

    total_penalty = (0.5 * incoherence_penalty_val +
                     0.3 * density_penalty_val +
                     0.2 * empty_penalty_val)

    return penalty_factor * total_penalty


# --- Existing cost functions below ---
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
        if np.allclose(gradients, gradients[:,0:1]):
            return 0.0
        return penalty_factor * 1e12

    slew_waveforms = np.diff(gradients, n=1, axis=1) / dt

    if slew_waveforms.size == 0:
        return 0.0

    roughness_metric = np.sum(np.linalg.norm(slew_waveforms, axis=0)**2)

    num_slew_points = slew_waveforms.shape[1]
    normalized_roughness = roughness_metric / num_slew_points if num_slew_points > 0 else 0.0

    return penalty_factor * normalized_roughness

def calculate_pns_proxy_penalty(
    trajectory: Trajectory,
    pns_threshold_T_per_s: float = 180.0,
    penalty_factor: float = 10.0
) -> float:
    """
    Calculates a Peripheral Nerve Stimulation (PNS) proxy penalty.
    This simplified model penalizes trajectories where the maximum vector norm of the
    slew rate exceeds a given threshold. True PNS is more complex.

    Args:
        trajectory (Trajectory): The trajectory object to evaluate.
        pns_threshold_T_per_s (float): The threshold for the maximum slew rate (T/m/s).
        penalty_factor (float): Factor to scale the penalty.

    Returns:
        float: Calculated penalty value.
    """
    cost = 0.0
    if pns_threshold_T_per_s <= 0:
        return 0.0

    max_slew_achieved = trajectory.get_max_slew_Tm_per_s()

    if max_slew_achieved is not None and max_slew_achieved > pns_threshold_T_per_s:
        cost = penalty_factor * ((max_slew_achieved - pns_threshold_T_per_s) / pns_threshold_T_per_s)**2
    return cost

def calculate_signal_decay_penalty(
    trajectory: Trajectory,
    T1_ms: Optional[float] = None,
    T2_ms: float = 80.0,
    b_value_s_per_mm2: float = 0.0,
    k_space_weighting_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    is_13c: bool = False,
    penalty_factor: float = 1.0
) -> float:
    """
    Calculates a penalty based on MR signal decay (T1, T2, diffusion)
    over the course of the trajectory. Penalizes loss of signal, weighted by
    k-space location (higher penalty for late high-k points).

    Args:
        trajectory (Trajectory): The Trajectory object.
        T1_ms (Optional[float]): Longitudinal relaxation time (ms). If None or <=0,
                                 T1 decay is ignored unless is_13c is True (then warning).
        T2_ms (float): Transverse relaxation time (ms). If <=0, returns 0 penalty.
        b_value_s_per_mm2 (float): Diffusion b-value in s/mm^2.
        k_space_weighting_func (Optional[Callable[[np.ndarray], np.ndarray]]):
            Optional function that takes k-space radii (rad/m) and returns
            a weight for each point. If None, defaults to radii^2 (2D) or radii^3 (3D).
        is_13c (bool): If True, T1 decay is more strictly considered. If T1_ms is not
                       valid, a warning might be implicitly raised or behavior defined by use.
        penalty_factor (float): Multiplier for the final penalty.

    Returns:
        A float representing the calculated penalty. Higher means more signal loss
        at important (weighted) k-space points or excessive acquisition time.
    """
    if T2_ms <= 0:
        return 0.0

    acquisition_times_ms = trajectory.get_acquisition_times_ms()
    if acquisition_times_ms.size == 0:
        return 0.0

    decay = np.ones_like(acquisition_times_ms, dtype=float)
    decay *= np.exp(-acquisition_times_ms / T2_ms)

    effective_T1_ms = T1_ms
    if is_13c and (T1_ms is None or T1_ms <= 0):
        print("Warning: is_13c is True but T1_ms is not valid for signal_decay_penalty. T1 decay will be ignored.")
        effective_T1_ms = None

    if effective_T1_ms is not None and effective_T1_ms > 0:
        decay *= np.exp(-acquisition_times_ms / effective_T1_ms)

    if b_value_s_per_mm2 > 0:
        ADC_mm2_per_s_approx = 0.001
        b_times_ADC = b_value_s_per_mm2 * ADC_mm2_per_s_approx
        diffusion_decay_factor = np.exp(-b_times_ADC * (acquisition_times_ms / 1000.0))
        decay *= diffusion_decay_factor

    k_points = trajectory.kspace_points_rad_per_m
    if k_points.size == 0:
        return 0.0

    radii_rad_per_m = np.linalg.norm(k_points, axis=0)

    if k_space_weighting_func is not None:
        try:
            current_weights = k_space_weighting_func(radii_rad_per_m)
            if current_weights.shape != radii_rad_per_m.shape:
                print(f"Warning: k_space_weighting_func output shape {custom_weights.shape} "
                      f"does not match radii shape {radii_rad_per_m.shape}. Using default weights.")
                current_weights = radii_rad_per_m**2 if trajectory.get_num_dimensions() < 3 else radii_rad_per_m**3
        except Exception as e:
            print(f"Warning: Error applying k_space_weighting_func: {e}. Using default weights.")
            current_weights = radii_rad_per_m**2 if trajectory.get_num_dimensions() < 3 else radii_rad_per_m**3
    else:
        if trajectory.get_num_dimensions() < 3:
            current_weights = radii_rad_per_m**2
        else:
            current_weights = radii_rad_per_m**3

    signal_loss_penalty_sum = np.sum((1.0 - decay) * current_weights)

    time_penalty = 0.0
    if acquisition_times_ms.size > 0:
        max_acq_time_ms = acquisition_times_ms[-1]

        time_limit_ms_t2_component = 3 * T2_ms

        if is_13c:
            t1_for_limit = effective_T1_ms if (effective_T1_ms is not None and effective_T1_ms > 0) else (5 * T2_ms)
            time_limit_ms_t1_component = 0.5 * t1_for_limit
            time_limit_ms = min(time_limit_ms_t2_component, time_limit_ms_t1_component)
        elif effective_T1_ms is not None and effective_T1_ms > 0:
             time_limit_ms = min(time_limit_ms_t2_component, 2 * effective_T1_ms)
        else:
            time_limit_ms = time_limit_ms_t2_component

        if max_acq_time_ms > time_limit_ms and time_limit_ms > 0:
            time_penalty = 100.0 * ((max_acq_time_ms - time_limit_ms) / time_limit_ms)**2
        elif time_limit_ms <= 0:
            time_penalty = penalty_factor * 1000.0

    total_penalty = signal_loss_penalty_sum + time_penalty
    return penalty_factor * total_penalty

```
