"""
Utilities for Trajectory Manipulation, Reconstruction, and Display
-----------------------------------------------------------------

This module provides utility functions for working with MRI k-space trajectories,
including:
- Applying hardware constraints (gradient/slew limits) to trajectories.
- Performing basic image reconstruction from k-space data using gridding.
- Visualizing trajectories and their associated waveforms (gradients, slew rates).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift # For older scipy, ifftn might be in numpy.fft
from scipy.interpolate import griddata
from typing import Optional, Tuple, Union # Added Optional, Tuple, Union

# Placeholder for Trajectory class import, will be correctly imported when used
from .trajectory import Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T
from typing import List # Added List

__all__ = [
    'constrain_trajectory',
    'reconstruct_image',
    'display_trajectory',
    'stack_2d_trajectory_to_3d',
    'rotate_3d_trajectory' # New function
]


def rotate_3d_trajectory(
    trajectory_3d: Trajectory,
    rotation_matrix: Optional[np.ndarray] = None,
    euler_angles_deg: Optional[Tuple[float, float, float]] = None, # ZYX convention: alpha, beta, gamma
    output_name_prefix: str = "rotated_"
) -> Trajectory:
    """
    Rotates a 3D k-space trajectory using either a provided rotation matrix
    or Euler angles (ZYX convention).

    Args:
        trajectory_3d (Trajectory): The input 3D Trajectory object.
        rotation_matrix (Optional[np.ndarray]): A 3x3 numpy array representing the
            rotation matrix.
        euler_angles_deg (Optional[Tuple[float, float, float]]): A tuple of three
            Euler angles (alpha, beta, gamma) in degrees, representing rotations
            around Z, then new Y, then new X axes respectively.
        output_name_prefix (str): Prefix for the name of the new 3D trajectory.

    Returns:
        Trajectory: A new 3D Trajectory object with rotated k-space points.

    Raises:
        ValueError: If the input trajectory is not 3D, if neither or both
                    rotation_matrix and euler_angles_deg are provided, or
                    if rotation_matrix has an invalid shape.
    """
    if trajectory_3d.get_num_dimensions() != 3:
        raise ValueError("Input trajectory must be 3-dimensional.")

    if rotation_matrix is None and euler_angles_deg is None:
        raise ValueError("Either rotation_matrix or euler_angles_deg must be provided.")
    if rotation_matrix is not None and euler_angles_deg is not None:
        raise ValueError("Provide either rotation_matrix or euler_angles_deg, not both.")

    final_rotation_matrix: np.ndarray

    if euler_angles_deg is not None:
        alpha_rad, beta_rad, gamma_rad = np.deg2rad(euler_angles_deg)

        # Rz(alpha)
        Rz = np.array([
            [np.cos(alpha_rad), -np.sin(alpha_rad), 0],
            [np.sin(alpha_rad),  np.cos(alpha_rad), 0],
            [0,                  0,                 1]
        ])
        # Ry(beta)
        Ry = np.array([
            [np.cos(beta_rad),  0, np.sin(beta_rad)],
            [0,                 1, 0],
            [-np.sin(beta_rad), 0, np.cos(beta_rad)]
        ])
        # Rx(gamma)
        Rx = np.array([
            [1, 0,                  0],
            [0, np.cos(gamma_rad), -np.sin(gamma_rad)],
            [0, np.sin(gamma_rad),  np.cos(gamma_rad)]
        ])
        # ZYX convention: Apply Rx, then Ry, then Rz
        # Combined matrix R = Rz @ Ry @ Rx
        final_rotation_matrix = Rz @ Ry @ Rx
        rotation_info = {'euler_angles_deg_ZYX': euler_angles_deg}

    elif rotation_matrix is not None: # Explicitly use elif for clarity
        if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must be a 3x3 NumPy array.")
        final_rotation_matrix = rotation_matrix
        rotation_info = {'rotation_matrix': rotation_matrix.tolist()} # Store as list for metadata

    k_3d_points = trajectory_3d.kspace_points_rad_per_m # Shape (3, N_points_3d)
    rotated_k_3d_points = final_rotation_matrix @ k_3d_points

    new_name = output_name_prefix + trajectory_3d.name
    new_metadata = trajectory_3d.metadata.copy()
    new_metadata['rotation_applied'] = True
    new_metadata['rotation_details'] = rotation_info

    # Note: If the original trajectory had pre-computed gradients,
    # they are now invalid. The Trajectory class will recompute them if asked.

    return Trajectory(
        name=new_name,
        kspace_points_rad_per_m=rotated_k_3d_points,
        dt_seconds=trajectory_3d.dt_seconds,
        metadata=new_metadata,
        gamma_Hz_per_T=trajectory_3d.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']),
        dead_time_start_seconds=trajectory_3d.dead_time_start_seconds,
        dead_time_end_seconds=trajectory_3d.dead_time_end_seconds
    )


def stack_2d_trajectory_to_3d(
    trajectory_2d: Trajectory,
    num_slices: int,
    slice_separation_rad_per_m: float,
    rotation_angles_deg: Optional[List[float]] = None,
    output_name_prefix: str = "stacked_"
) -> Trajectory:
    """
    Stacks a 2D k-space trajectory along the kz-axis to create a 3D trajectory.
    Optionally applies in-plane rotations to each slice.

    Args:
        trajectory_2d (Trajectory): The input 2D Trajectory object.
        num_slices (int): The number of slices to stack in the kz direction.
        slice_separation_rad_per_m (float): The k-space separation between adjacent slices in rad/m.
        rotation_angles_deg (Optional[List[float]]): A list of in-plane rotation angles
            in degrees, one for each slice. If None, no rotation is applied.
            If the list is shorter than num_slices, the last angle is repeated.
        output_name_prefix (str): Prefix for the name of the new 3D trajectory.

    Returns:
        Trajectory: A new 3D Trajectory object.

    Raises:
        ValueError: If the input trajectory is not 2D, or if num_slices is not positive.
    """
    if trajectory_2d.get_num_dimensions() != 2:
        raise ValueError("Input trajectory must be 2-dimensional.")
    if num_slices <= 0:
        raise ValueError("Number of slices must be positive.")

    k_2d_points = trajectory_2d.kspace_points_rad_per_m  # Shape (2, N_points_2d)
    n_points_2d = trajectory_2d.get_num_points()

    all_3d_points_segments = []

    kz_base_offsets = np.linspace(
        -(num_slices - 1) / 2.0 * slice_separation_rad_per_m,
        (num_slices - 1) / 2.0 * slice_separation_rad_per_m,
        num_slices
    )

    for i_slice in range(num_slices):
        current_k_slice_2d = np.copy(k_2d_points)
        angle_rad = 0.0

        if rotation_angles_deg:
            if i_slice < len(rotation_angles_deg):
                angle_rad = np.deg2rad(rotation_angles_deg[i_slice])
            elif rotation_angles_deg: # Not empty list
                angle_rad = np.deg2rad(rotation_angles_deg[-1])

        if angle_rad != 0.0:
            rot_mat_2d = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            current_k_slice_2d = rot_mat_2d @ current_k_slice_2d

        kx_rotated = current_k_slice_2d[0, :]
        ky_rotated = current_k_slice_2d[1, :]
        kz_values = np.full_like(kx_rotated, kz_base_offsets[i_slice])

        k_slice_3d = np.vstack((kx_rotated, ky_rotated, kz_values)) # Shape (3, N_points_2d)
        all_3d_points_segments.append(k_slice_3d)

    final_k_3d_points = np.concatenate(all_3d_points_segments, axis=1)

    new_name = output_name_prefix + trajectory_2d.name
    new_metadata = trajectory_2d.metadata.copy()
    new_metadata['num_dimensions'] = 3
    new_metadata['original_2d_trajectory_name'] = trajectory_2d.name
    new_metadata['stacking_parameters'] = {
        'num_slices': num_slices,
        'slice_separation_rad_per_m': slice_separation_rad_per_m,
        'rotation_angles_deg': rotation_angles_deg if rotation_angles_deg else [0.0] * num_slices,
        'original_num_points_per_slice': n_points_2d
    }
    # Preserve sequence_params if they exist, but they are specific to the original sequence
    # and might not fully describe the new stacked trajectory's generation method.
    # For now, we just copy all metadata which includes sequence_params.

    return Trajectory(
        name=new_name,
        kspace_points_rad_per_m=final_k_3d_points,
        dt_seconds=trajectory_2d.dt_seconds, # dt remains the same for each point
        metadata=new_metadata,
        gamma_Hz_per_T=trajectory_2d.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']),
        dead_time_start_seconds=trajectory_2d.dead_time_start_seconds, # These apply to the whole acquisition
        dead_time_end_seconds=trajectory_2d.dead_time_end_seconds,
        # sequence_params from original traj is in new_metadata.
        # The new trajectory is not directly a "sequence" from a sequence class anymore.
    )


def constrain_trajectory(
    trajectory_obj: Trajectory,
    max_grad_mT_per_m: float,
    max_slew_Tm_per_s_ms: float,
    dt_s: Optional[float] = None, # Made Optional explicit
    gamma_Hz_per_T: Optional[float] = None, # Made Optional explicit
    num_iterations: int = 2 # Number of iterations for gradient and slew constraints
    ) -> Trajectory:
    """
    Applies simplified gradient and slew rate constraints to a trajectory.

    Note: This is a simplified implementation. True hardware-compliant trajectory
    optimization is a complex iterative process. This version performs a few passes
    of scaling.

    Parameters:
    - trajectory_obj: An instance of the Trajectory class.
    - max_grad_mT_per_m: Maximum gradient amplitude (mT/m).
    - max_slew_Tm_per_s_ms: Maximum slew rate (T/m/s/ms).
    - dt_s: Dwell time in seconds. If None, uses trajectory_obj.dt_seconds.
    - gamma_Hz_per_T: Gyromagnetic ratio. If None, uses trajectory_obj.metadata.
    - num_iterations: Number of iterations to apply constraints.

    Returns:
    - A new Trajectory object with the constrained k-space and updated gradient/slew metadata.
    """

    if not isinstance(trajectory_obj, Trajectory):
        raise TypeError("trajectory_obj must be an instance of the Trajectory class.")

    current_dt_s = dt_s if dt_s is not None else trajectory_obj.dt_seconds
    if current_dt_s is None or current_dt_s <= 0:
        raise ValueError("Dwell time (dt_s) must be positive and provided either as argument or in trajectory_obj.")

    current_gamma_Hz_per_T = gamma_Hz_per_T if gamma_Hz_per_T is not None \
        else trajectory_obj.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
    if current_gamma_Hz_per_T == 0: # Avoid division by zero
        current_gamma_Hz_per_T = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']


    # Convert constraints to standard units
    max_grad_T_per_m = max_grad_mT_per_m / 1000.0
    max_slew_Tm_per_s_s = max_slew_Tm_per_s_ms * 1000.0 # T/m/s/ms * 1000 ms/s = T/m/s^2

    # Work with a copy of k-space data (D, N)
    k_space_rad_per_m = np.copy(trajectory_obj.kspace_points_rad_per_m)

    if k_space_rad_per_m.shape[1] == 0: # No points
        return Trajectory(
            name=trajectory_obj.name + "_constrained_empty",
            kspace_points_rad_per_m=k_space_rad_per_m,
            dt_seconds=current_dt_s,
            metadata=dict(trajectory_obj.metadata), # Copy metadata
            gamma_Hz_per_T=current_gamma_Hz_per_T
        )
    if k_space_rad_per_m.shape[1] == 1: # Single point, no gradients/slew to constrain beyond initial grad=0
        # Calculate gradient assuming it's zero (as it's a single point)
        grad_T_per_m = np.zeros_like(k_space_rad_per_m)
        grad_mag = np.linalg.norm(grad_T_per_m, axis=0)
        if np.any(grad_mag > max_grad_T_per_m):
            # This case should ideally not happen for a single point if grad is truly 0
            # but as a safeguard if initial k-space implies non-zero start.
            # However, get_gradient_waveforms_Tm for N=1 returns zeros.
            pass # No change needed as gradients are zero for a single point.

        return Trajectory(
            name=trajectory_obj.name + "_constrained_single_pt",
            kspace_points_rad_per_m=k_space_rad_per_m, # Original k-space
            gradient_waveforms_Tm=grad_T_per_m, # Zero gradients
            dt_seconds=current_dt_s,
            metadata=dict(trajectory_obj.metadata),
            gamma_Hz_per_T=current_gamma_Hz_per_T
        )

    # Initial gradient calculation from the current k-space
    # (D,N) = d(k)/dt / gamma
    grad_T_per_m = np.gradient(k_space_rad_per_m, current_dt_s, axis=1) / current_gamma_Hz_per_T


    for iteration in range(num_iterations):
        # 1. Constrain Gradients
        grad_mag = np.linalg.norm(grad_T_per_m, axis=0) # Magnitude per time point (N,)

        scale_factors_grad = np.ones_like(grad_mag)
        viol_grad_indices = grad_mag > max_grad_T_per_m
        if np.any(viol_grad_indices):
            scale_factors_grad[viol_grad_indices] = max_grad_T_per_m / grad_mag[viol_grad_indices]

        # Scale down gradient vectors (D,N)
        grad_T_per_m = grad_T_per_m * scale_factors_grad # Broadcasting (D,N) * (N,) -> (D,N)

        # 2. Constrain Slew Rates
        # Slew rate s = dg/dt (D, N-1) for diff, or (D,N) for np.gradient
        # Using np.gradient for slew to keep same shape as grad_T_per_m for easier processing,
        # though edge effects of np.gradient for slew might be less accurate than np.diff.
        # For np.diff, shape would be (D, N-1).

        slew_Tm_per_s_s = np.gradient(grad_T_per_m, current_dt_s, axis=1)
        slew_mag = np.linalg.norm(slew_Tm_per_s_s, axis=0) # (N,)

        scale_factors_slew = np.ones_like(slew_mag)
        viol_slew_indices = slew_mag > max_slew_Tm_per_s_s
        if np.any(viol_slew_indices):
            scale_factors_slew[viol_slew_indices] = max_slew_Tm_per_s_s / slew_mag[viol_slew_indices]

        # How to apply slew scaling?
        # Scaling slew directly is tricky. Common methods involve time stretching or modifying gradient steps.
        # A simple approach: if slew s_i = (g_i - g_{i-1})/dt is too high,
        # we need to reduce |g_i - g_{i-1}|.
        # This means g_i needs to be closer to g_{i-1}.
        # We can try to adjust the gradient differences.

        if np.any(viol_slew_indices):
            # This is a very simplified slew limiting:
            # It scales the gradient *differences* where slew is violated.
            # Then reconstructs gradients. This is not standard but a basic attempt.
            grad_diffs = np.diff(grad_T_per_m, axis=1) # (D, N-1)

            # Apply scaling to differences where slew (calculated from original grads at this iteration) was too high.
            # The viol_slew_indices are (N,). We need to map this to (N-1,) for grad_diffs.
            # A point slew violation s_k means (g_k - g_{k-1}) or similar is too large.
            # We scale grad_diff[:, k] if slew_mag[k] or slew_mag[k+1] is large.
            # For simplicity, let's assume viol_slew_indices (N,) can be used to guide scaling of diffs.
            # We scale the diff that *would result* in the slew violation.
            # If slew_mag[k] (derived from g[k-1], g[k], g[k+1]) is too high,
            # it's often related to diffs (g[k]-g[k-1]) and (g[k+1]-g[k]).

            # Iterate through gradient differences and scale them if the corresponding slew was too high.
            # This is heuristic.
            for k_diff in range(grad_diffs.shape[1]): # 0 to N-2
                # Slew at point k_diff+1 is (g[k_diff+1] - g[k_diff])/dt
                # If slew_mag[k_diff+1] is a violator:
                if scale_factors_slew[k_diff+1] < 1.0:
                     grad_diffs[:, k_diff] *= scale_factors_slew[k_diff+1]
                # Also consider slew_mag[k_diff] if it affects this segment (due to np.gradient behavior)
                # This part is tricky with np.gradient for slew.
                # A more direct np.diff for slew might be easier to reason about here.
                # Let's use np.diff for slew calculation for constraint application:

            slew_via_diff = np.diff(grad_T_per_m, axis=1) / current_dt_s # (D, N-1)
            slew_mag_diff = np.linalg.norm(slew_via_diff, axis=0) # (N-1,)

            viol_slew_indices_diff = slew_mag_diff > max_slew_Tm_per_s_s
            if np.any(viol_slew_indices_diff):
                scale_factors_slew_diff = np.ones_like(slew_mag_diff)
                scale_factors_slew_diff[viol_slew_indices_diff] = max_slew_Tm_per_s_s / slew_mag_diff[viol_slew_indices_diff]

                grad_diffs_scaled = grad_diffs * scale_factors_slew_diff # Scale the N-1 differences

                # Reconstruct gradients from scaled differences
                # g_new[0] = g_orig[0]
                # g_new[i] = g_new[i-1] + diff_scaled[i-1]
                new_grad_T_per_m = np.zeros_like(grad_T_per_m)
                new_grad_T_per_m[:, 0] = grad_T_per_m[:, 0] # Keep first gradient point
                new_grad_T_per_m[:, 1:] = np.cumsum(grad_diffs_scaled, axis=1) + new_grad_T_per_m[:, 0:1]
                grad_T_per_m = new_grad_T_per_m

    # 3. Recalculate k-space from the final constrained gradients
    # k = integral(gamma * G * dt)
    # k_space(t) = gamma * cumsum(G(t') * dt) from t'=0 to t
    # Note: cumsum needs to be careful about initial k-space value if not starting from k=0
    # Assuming k-space starts effectively from k_space_rad_per_m[:,0] before first gradient step,
    # or that k_space_rad_per_m includes the point at k=0 if applicable.
    # The np.gradient for grad_T_per_m gives gradients whose integral reconstructs k_space_rad_per_m.
    # So, integrating the *new* grad_T_per_m should give the new k_space.

    # Initial k-space point:
    # If the original trajectory implies a k-space offset (e.g. not starting at 0,0,0),
    # that offset needs to be preserved or handled.
    # Here, we assume the integration of gradients starts from an initial k-value.
    # Let's assume the *first point* of the original k-space is the starting k-value.
    k_space_start_point = np.copy(trajectory_obj.kspace_points_rad_per_m[:, 0:1])

    # Cumulative sum of (gamma * G * dt)
    k_space_deltas = current_gamma_Hz_per_T * grad_T_per_m[:, 1:] * current_dt_s # (D, N-1)

    final_k_space_rad_per_m = np.zeros_like(k_space_rad_per_m)
    final_k_space_rad_per_m[:, 0:1] = k_space_start_point
    final_k_space_rad_per_m[:, 1:] = k_space_start_point + np.cumsum(current_gamma_Hz_per_T * grad_T_per_m[:, :-1] * current_dt_s, axis=1)
    # A small correction: grad_T_per_m has N points. The deltas are between them.
    # k[i] = k[i-1] + gamma * g[i-1] * dt (if g is taken at start of interval)
    # or k[i] = k[i-1] + gamma * (g[i]+g[i-1])/2 * dt (trapezoidal)
    # Simplest: k_recon = gamma * cumtrapz(grad_T_per_m, dx=current_dt_s, initial=0, axis=1) + k_space_start_point
    # scipy.integrate.cumulative_trapezoid is better
    from scipy.integrate import cumulative_trapezoid

    # We need to ensure the integral of grad_T_per_m matches the shape of k_space_rad_per_m
    # cumulative_trapezoid returns N-1 points.
    k_space_integrated_part = cumulative_trapezoid(
        current_gamma_Hz_per_T * grad_T_per_m,
        dx=current_dt_s,
        axis=1,
        initial=0 # This makes the output start with 0, then integral values
    )
    # The above 'initial=0' means the first point of k_space_integrated_part is 0.
    # So, k_final = k_original_start + k_space_integrated_part
    final_k_space_rad_per_m = k_space_start_point + k_space_integrated_part


    # Create a new Trajectory object
    constrained_traj = Trajectory(
        name=trajectory_obj.name + "_constrained",
        kspace_points_rad_per_m=final_k_space_rad_per_m,
        gradient_waveforms_Tm=grad_T_per_m, # Store the explicitly calculated/constrained gradients
        dt_seconds=current_dt_s,
        metadata=dict(trajectory_obj.metadata), # Copy original metadata
        gamma_Hz_per_T=current_gamma_Hz_per_T,
        dead_time_start_seconds=trajectory_obj.dead_time_start_seconds,
        dead_time_end_seconds=trajectory_obj.dead_time_end_seconds
    )
    # The Trajectory class's _compute_metrics will recalculate slew, etc.
    # Or we can populate them here.
    constrained_traj.metadata['max_grad_constraint_mT_per_m'] = max_grad_mT_per_m
    constrained_traj.metadata['max_slew_constraint_Tm_per_s_ms'] = max_slew_Tm_per_s_ms
    # The actual achieved max grad/slew will be computed by _compute_metrics.

    return constrained_traj


def reconstruct_image(
    kspace_data: np.ndarray,
    trajectory_obj: Trajectory,
    density_comp_weights: Optional[np.ndarray] = None,
    recon_matrix_size: Optional[Tuple[int, ...]] = None,
    oversampling_factor: float = 2.0,
    gridding_method: str = 'linear' # 'nearest', 'linear', 'cubic' (from griddata)
    ) -> np.ndarray:
    """
    Reconstructs an image from k-space data using basic gridding and FFT.

    Parameters:
    - kspace_data (np.ndarray): Complex k-space samples (1D array matching trajectory points).
    - trajectory_obj (Trajectory): The Trajectory object corresponding to kspace_data.
    - density_comp_weights (Optional[np.ndarray]): Density compensation weights.
                                                 If None, uses 'density_compensation_weights_voronoi'
                                                 from trajectory metadata if available, else uniform.
    - recon_matrix_size (Optional[Tuple[int,...]]): Output image matrix size (e.g., (256, 256)).
                                                 If None, defaults to (128, 128) or (128,128,128).
    - oversampling_factor (float): Factor for k-space grid oversampling before FFT.
    - gridding_method (str): Interpolation method for griddata ('nearest', 'linear', 'cubic').

    Returns:
    - np.ndarray: Reconstructed image (absolute magnitude).
    """
    if not isinstance(trajectory_obj, Trajectory):
        raise TypeError("trajectory_obj must be an instance of the Trajectory class.")
    if kspace_data.ndim != 1 or kspace_data.shape[0] != trajectory_obj.get_num_points():
        raise ValueError("kspace_data must be a 1D array with length matching trajectory points.")

    num_dims = trajectory_obj.get_num_dimensions()
    if num_dims not in [2, 3]:
        raise NotImplementedError(f"Reconstruction for {num_dims}D not implemented. Only 2D/3D supported.")

    # Determine reconstruction matrix size
    if recon_matrix_size is None:
        default_size = 128
        recon_matrix_size = tuple([default_size] * num_dims)
        print(f"Warning: recon_matrix_size not provided, defaulting to {recon_matrix_size}.")
    elif len(recon_matrix_size) != num_dims:
        raise ValueError(f"recon_matrix_size tuple length {len(recon_matrix_size)} must match num_dimensions {num_dims}.")

    # Density compensation
    if density_comp_weights is None:
        if 'density_compensation_weights_voronoi' in trajectory_obj.metadata and \
           trajectory_obj.metadata['density_compensation_weights_voronoi'] is not None and \
           trajectory_obj.metadata['density_compensation_weights_voronoi'].size == kspace_data.size:
            density_comp_weights = trajectory_obj.metadata['density_compensation_weights_voronoi']
            print("Using Voronoi density compensation weights from trajectory metadata.")
        else:
            print("Warning: No density compensation weights provided or found. Using uniform weights.")
            density_comp_weights = np.ones_like(kspace_data, dtype=float)

    if density_comp_weights.shape != kspace_data.shape:
        raise ValueError("Shape mismatch between kspace_data and density_comp_weights.")

    weighted_kspace_data = kspace_data * density_comp_weights

    # Prepare k-space grid
    grid_size = [int(s * oversampling_factor) for s in recon_matrix_size]
    k_grid = np.zeros(grid_size, dtype=complex)

    # Scale trajectory points to fit the grid
    # The grid coordinates range from -grid_size/2 to grid_size/2 (approximately)
    # kspace_points_rad_per_m need to be mapped to these grid indices.
    # Max k-space extent from trajectory points:
    # k_max_abs = np.max(np.abs(trajectory_obj.kspace_points_rad_per_m), axis=1) # Max per dimension
    # This is not robust if trajectory doesn't span symmetric k-space.
    # Better: use full extent for scaling.

    # The grid itself represents k-space from -k_max_grid to +k_max_grid after FFT shift.
    # An FFT of an image of FOV implies k-space samples are at 1/FOV spacing.
    # Max k-value represented by grid edge is (grid_size/2) * (1/FOV_grid).
    # For simplicity, normalize trajectory k-space points to [-0.5, 0.5] * grid_size.
    # This assumes the k-space samples cover the Nyquist range for the desired recon_matrix_size.

    k_points_normalized = np.copy(trajectory_obj.kspace_points_rad_per_m) # (D, N)

    # Find max absolute k-value in the trajectory to normalize it to grid dimensions later
    # This ensures that the trajectory points are scaled relative to their own extent.
    max_k_coord_overall = 0
    for dim_idx in range(num_dims):
        dim_max = np.max(np.abs(k_points_normalized[dim_idx, :]))
        if dim_max > max_k_coord_overall:
            max_k_coord_overall = dim_max

    if max_k_coord_overall < 1e-9: # All k-points are zero
        print("Warning: All k-space trajectory points are zero. Reconstruction will be empty.")
    else:
        # Scale factor to map trajectory k-space coords to grid indices
        # We want the max extent of the trajectory to map near the edge of the oversampled grid.
        # Grid indices go from 0 to grid_size-1. Center is at grid_size/2.
        # So, k-space value `k` maps to `k_scaled * (grid_size / (2*k_max_mapped_to_edge)) + grid_center`
        # Let's map max_k_coord_overall to (grid_size/2 - 1) for safety.
        scale_to_grid = (np.array(grid_size) / 2.0 - 1) / max_k_coord_overall

        grid_centers = np.array(grid_size) / 2.0

        # Transpose k_points for griddata: list of (x,y,z) points (N, D)
        points_for_griddata = (k_points_normalized.T * scale_to_grid).T # Scale each dim

        # Shift to grid center: coordinates for griddata should be 0 to grid_size-1
        points_for_griddata = points_for_griddata + grid_centers[:, np.newaxis]

        # Create meshgrid for interpolation points on the k-space grid
        grid_coords = [np.arange(s) for s in grid_size]
        mesh = np.meshgrid(*grid_coords, indexing='ij')
        grid_points_to_interpolate = np.vstack([m.flatten() for m in mesh]).T # (M, D)

        # Perform gridding (interpolation)
        # griddata expects points (N,D), values (N,), and xi (M,D)
        try:
            gridded_values = griddata(
                points_for_griddata.T,          # Sampled k-space trajectory points (N, D)
                weighted_kspace_data,           # Corresponding k-space data values (N,)
                tuple(mesh),                    # Target grid coordinates (D, grid_x, grid_y, [grid_z])
                method=gridding_method,
                fill_value=0.0                  # Fill outside convex hull with zeros
            )
            k_grid = gridded_values
        except Exception as e:
            print(f"Gridding failed: {e}. Falling back to simple assignment (potential data loss).")
            # Fallback: very basic nearest neighbor by rounding (less accurate than griddata's nearest)
            # This is just a crude fallback, not a good gridding method.
            for i in range(trajectory_obj.get_num_points()):
                coords = np.clip(np.round(points_for_griddata[:, i]).astype(int), 0, np.array(grid_size) - 1)
                if num_dims == 2:
                    k_grid[coords[0], coords[1]] += weighted_kspace_data[i]
                elif num_dims == 3:
                    k_grid[coords[0], coords[1], coords[2]] += weighted_kspace_data[i]


    # Perform inverse FFT
    # Shift k-space center to corner for ifft, then shift back
    image_os = ifftshift(ifftn(ifftshift(k_grid)))

    # Crop to final matrix size (remove oversampling)
    # Determine cropping indices
    start_indices = [(os_size - final_size) // 2 for os_size, final_size in zip(grid_size, recon_matrix_size)]
    end_indices = [start + final_size for start, final_size in zip(start_indices, recon_matrix_size)]

    if num_dims == 2:
        image_cropped = image_os[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1]]
    elif num_dims == 3:
        image_cropped = image_os[start_indices[0]:end_indices[0],
                                 start_indices[1]:end_indices[1],
                                 start_indices[2]:end_indices[2]]
    else: # Should not be reached due to earlier check
        image_cropped = np.abs(image_os)


    return np.abs(image_cropped) # Return magnitude image


def display_trajectory(
    trajectory_obj: Trajectory,
    max_points_kspace: int = 2000,
    show_gradients: bool = False,
    show_slew: bool = False,
    fig_size: Optional[Tuple[float, float]] = None,
    max_points_waveforms: int = 2000
    ) -> plt.Figure:
    """
    Displays the k-space trajectory and optionally its gradient and slew rate waveforms.

    Parameters:
    - trajectory_obj (Trajectory): The Trajectory object to display.
    - max_points_kspace (int): Maximum k-space points to plot (passed to trajectory_obj.plot_2d/3d).
    - show_gradients (bool): If True, plot gradient waveforms.
    - show_slew (bool): If True, plot slew rate waveforms.
    - fig_size (Optional[Tuple[float,float]]): Figure size for matplotlib.
    - max_points_waveforms (int): Max points to display for waveform plots (uses slicing).


    Returns:
    - matplotlib.figure.Figure: The figure object containing the plots.
    """
    if not isinstance(trajectory_obj, Trajectory):
        raise TypeError("trajectory_obj must be an instance of the Trajectory class.")

    num_subplots = 1
    if show_gradients: num_subplots += 1
    if show_slew: num_subplots += 1

    if fig_size is None:
        fig_size = (7 * num_subplots, 6) # Adjust figure width based on number of subplots

    # fig, axes = plt.subplots(1, num_subplots, figsize=fig_size, squeeze=False) # squeeze=False ensures axes is always 2D array
    # ax_idx = 0

    fig = plt.figure(figsize=fig_size)
    current_subplot_idx = 0 # For indexing into a conceptual 1xN grid

    # Plot K-space Trajectory
    current_subplot_idx += 1
    if trajectory_obj.get_num_dimensions() >= 3:
        ax_kspace = fig.add_subplot(1, num_subplots, current_subplot_idx, projection='3d')
        trajectory_obj.plot_3d(ax=ax_kspace, title=f"K-space: {trajectory_obj.name}", max_total_points=max_points_kspace)
    elif trajectory_obj.get_num_dimensions() == 2:
        ax_kspace = fig.add_subplot(1, num_subplots, current_subplot_idx)
        trajectory_obj.plot_2d(ax=ax_kspace, title=f"K-space: {trajectory_obj.name}", max_total_points=max_points_kspace)
    else: # 1D
        ax_kspace = fig.add_subplot(1, num_subplots, current_subplot_idx)
        ax_kspace.plot(trajectory_obj.kspace_points_rad_per_m[0, :max_points_kspace], np.zeros(min(trajectory_obj.get_num_points(), max_points_kspace)))
        ax_kspace.set_title(f"K-space (1D): {trajectory_obj.name}")
        ax_kspace.set_xlabel("Kx (rad/m)")

    dt_s = trajectory_obj.dt_seconds if trajectory_obj.dt_seconds is not None else 1.0 # Assume dt=1 for plotting if not set

    # Plot Gradients
    if show_gradients:
        current_subplot_idx += 1
        ax_grad = fig.add_subplot(1, num_subplots, current_subplot_idx)
        gradients = trajectory_obj.get_gradient_waveforms_Tm() # This will compute if not already stored
        if gradients is not None and gradients.size > 0:
            num_grad_dims, num_grad_points = gradients.shape

            # Determine stride for plotting if too many points
            stride = 1
            if num_grad_points > max_points_waveforms:
                stride = num_grad_points // max_points_waveforms
                stride = max(1, stride)

            time_axis = np.arange(0, num_grad_points * dt_s, dt_s)[::stride]
            if len(time_axis) > gradients[:,::stride].shape[1]: # Fix for potential length mismatch if stride makes time_axis shorter
                 time_axis = time_axis[:gradients[:,::stride].shape[1]]


            for d in range(num_grad_dims):
                ax_grad.plot(time_axis, gradients[d, ::stride], label=f"Grad Dim {d}")
            ax_grad.set_title(f"Gradient Waveforms ({gradients[:,::stride].shape[1]} pts shown)")
            ax_grad.set_xlabel("Time (s)" if trajectory_obj.dt_seconds else "Sample Index")
            ax_grad.set_ylabel("Gradient (T/m)")
            ax_grad.legend()
        else:
            ax_grad.text(0.5, 0.5, "Gradients not available or empty.", horizontalalignment='center', verticalalignment='center')
            ax_grad.set_title("Gradient Waveforms")

    # Plot Slew Rates
    if show_slew:
        current_subplot_idx += 1
        ax_slew = fig.add_subplot(1, num_subplots, current_subplot_idx)
        gradients = trajectory_obj.get_gradient_waveforms_Tm() # Get again, could be modified by user
        if gradients is not None and gradients.shape[1] > 1 and trajectory_obj.dt_seconds is not None and trajectory_obj.dt_seconds > 0:
            slew_rates = np.diff(gradients, axis=1) / trajectory_obj.dt_seconds # (D, N-1)
            num_slew_dims, num_slew_points = slew_rates.shape

            stride = 1
            if num_slew_points > max_points_waveforms:
                stride = num_slew_points // max_points_waveforms
                stride = max(1, stride)

            # Time axis for diff is typically for the midpoint or start of interval
            time_axis_slew = np.arange(0, num_slew_points * dt_s, dt_s)[::stride]
            if len(time_axis_slew) > slew_rates[:,::stride].shape[1]:
                 time_axis_slew = time_axis_slew[:slew_rates[:,::stride].shape[1]]


            for d in range(num_slew_dims):
                ax_slew.plot(time_axis_slew, slew_rates[d, ::stride], label=f"Slew Dim {d}")
            ax_slew.set_title(f"Slew Rate Waveforms ({slew_rates[:,::stride].shape[1]} pts shown)")
            ax_slew.set_xlabel("Time (s)")
            ax_slew.set_ylabel("Slew Rate (T/m/s)")
            ax_slew.legend()
        else:
            ax_slew.text(0.5, 0.5, "Slew rates not calculable \n(need >1 grad point and dt_s > 0).",
                         horizontalalignment='center', verticalalignment='center')
            ax_slew.set_title("Slew Rate Waveforms")

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # Basic test/example area that will be removed or moved to actual tests
    print("trajgen/utils.py created and functions added.")

    # Example: Create a dummy trajectory for testing
    # Needs trajgen.trajectory and trajgen.generators to be importable if used here directly.
    # For now, this block is just for basic check of file structure.
    pass
