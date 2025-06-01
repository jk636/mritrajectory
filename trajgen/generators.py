"""
K-Space Trajectory Generation Algorithms
----------------------------------------

This module provides functions for generating various types of k-space trajectories
commonly used in MRI, such as Spiral, Radial, EPI, Cones, and Rosette.

Each function typically takes FOV, resolution, and trajectory-specific parameters
as input and returns a NumPy array of k-space sample coordinates in rad/m.
These functions are generally used by the higher-level `KSpaceTrajectoryGenerator`
class but can also be used directly.
"""
import numpy as np
from typing import Tuple, Union, Optional # Added
from .trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T # Added

__all__ = [
    'generate_spiral_trajectory',
    'generate_radial_trajectory',
    'generate_cones_trajectory',
    'generate_epi_trajectory',
    'generate_rosette_trajectory',
    'generate_tpi_trajectory',
    'generate_propeller_blade_trajectory',
    'generate_wave_caipi_trajectory',
    'generate_drunken_spiral_trajectory' # New
]

# Existing imports:
# import numpy as np
# from typing import Tuple, Union, Optional
# from .trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T


def generate_drunken_spiral_trajectory(
    fov_mm: Union[float, Tuple[float, float]],
    resolution_mm: Union[float, Tuple[float, float]],
    num_points: int,
    dt_seconds: float,
    base_spiral_turns: float = 5.0,
    perturbation_amplitude_factor: float = 0.1, # Relative to k_max scaled by sqrt(num_points)
    density_sigma_factor: float = 0.25, # Sigma for Gaussian decay of noise weight, relative to k_max
    max_grad_Tm_per_m: Optional[float] = None,
    max_slew_Tm_per_s_per_m: Optional[float] = None, # Note: Slew here is T/m/s/m, not T/m/s (as in Trajectory class)
                                                # For consistency, let's assume this will be T/m/s
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
    num_smoothing_iterations: int = 3,
    smoothing_kernel_size: int = 5
) -> np.ndarray:
    """
    Generates a 2D "drunken" spiral k-space trajectory.

    This trajectory starts with a base spiral and adds weighted random perturbations.
    It then iteratively attempts to smooth the trajectory to meet optional
    gradient and slew rate constraints.

    Parameters:
    - fov_mm: Field of view in mm.
    - resolution_mm: Desired resolution in mm.
    - num_points (int): Total number of k-space points for the trajectory.
    - dt_seconds (float): Dwell time (time between k-space samples) in seconds.
    - base_spiral_turns (float): Number of turns for the underlying Archimedean spiral.
    - perturbation_amplitude_factor (float): Factor scaling the random noise amplitude.
                                           The actual perturbation scale is relative to k_max/sqrt(num_points).
    - density_sigma_factor (float): Sigma for the Gaussian weighting of noise,
                                    relative to k_max. Noise is stronger at k-space center.
    - max_grad_Tm_per_m (Optional[float]): Maximum gradient amplitude constraint (T/m).
    - max_slew_Tm_per_s_per_m (Optional[float]): Maximum slew rate constraint (T/m/s).
                                               (Prompt used T/m/s/m, but T/m/s is more standard for slew).
                                               This implementation will assume T/m/s for this variable.
    - gamma_Hz_per_T (float): Gyromagnetic ratio.
    - num_smoothing_iterations (int): Number of iterations to apply smoothing if constraints are violated.
    - smoothing_kernel_size (int): Size of the moving average filter kernel for smoothing. Must be odd.

    Returns:
    - np.ndarray: K-space points of shape (2, num_points) in rad/m.
    """
    num_dimensions = 2

    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive.")
    if smoothing_kernel_size <= 0 or smoothing_kernel_size % 2 == 0:
        raise ValueError("smoothing_kernel_size must be positive and odd.")

    if isinstance(fov_mm, (int, float)):
        fov_mm_tuple: Tuple[float, ...] = (float(fov_mm), float(fov_mm))
    elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == num_dimensions:
        fov_mm_tuple = tuple(map(float, fov_mm))
    else:
        raise ValueError(f"fov_mm must be a number or a tuple/list of length {num_dimensions}.")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm_tuple: Tuple[float, ...] = (float(resolution_mm), float(resolution_mm))
    elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == num_dimensions:
        resolution_mm_tuple = tuple(map(float, resolution_mm))
    else:
        raise ValueError(f"resolution_mm must be a number or a tuple/list of length {num_dimensions}.")

    resolution_m = np.array(resolution_mm_tuple) / 1000.0
    k_max_rad_per_m = np.min(1.0 / (2.0 * resolution_m))

    # Base Spiral
    r_base = np.linspace(0, k_max_rad_per_m, num_points, endpoint=True)
    theta_base = np.linspace(0, base_spiral_turns * 2 * np.pi, num_points, endpoint=True)

    # Noise Generation
    noise_kx = np.random.normal(0, 1, num_points)
    noise_ky = np.random.normal(0, 1, num_points)

    # Density Weighting for Noise
    radial_distance_normalized_sq = (r_base / (k_max_rad_per_m + 1e-9))**2
    weight = np.exp(-radial_distance_normalized_sq / (density_sigma_factor**2 + 1e-9))

    # Perturbation Scaling
    perturb_scale = perturbation_amplitude_factor * (k_max_rad_per_m / (np.sqrt(num_points) + 1e-9))

    # Apply Perturbation
    kx_perturbed = r_base * np.cos(theta_base) + weight * noise_kx * perturb_scale
    ky_perturbed = r_base * np.sin(theta_base) + weight * noise_ky * perturb_scale

    k_points = np.vstack((kx_perturbed, ky_perturbed)) # Shape (2, num_points)

    if num_points == 1: # No gradients or slew for a single point
        return k_points

    # Iterative Constraint Application & Smoothing
    for i_iter in range(num_smoothing_iterations):
        if k_points.shape[1] < 2 : # Need at least 2 points for gradient/slew
             break

        gradients = np.gradient(k_points, dt_seconds, axis=1) / gamma_Hz_per_T

        # Slew rate calculation: np.diff results in N-1 points. Pad to N for norm calculation.
        if k_points.shape[1] > 1:
            slew_rates_diff = np.diff(gradients, axis=1) / dt_seconds
             # Pad the last slew rate to maintain shape, or use np.gradient for slew too
            slew_rates = np.pad(slew_rates_diff, ((0,0),(0,1)), mode='edge') # Pad with edge value
        else:
            slew_rates = np.zeros_like(gradients)


        grad_norm = np.linalg.norm(gradients, axis=0)
        slew_norm = np.linalg.norm(slew_rates, axis=0)

        max_achieved_grad = np.max(grad_norm) if grad_norm.size > 0 else 0
        max_achieved_slew = np.max(slew_norm) if slew_norm.size > 0 else 0

        grad_ok = (max_grad_Tm_per_m is None) or (max_achieved_grad <= max_grad_Tm_per_m)
        # Assuming max_slew_Tm_per_s_per_m is actually T/m/s for this variable name
        slew_ok = (max_slew_Tm_per_s_per_m is None) or (max_achieved_slew <= max_slew_Tm_per_s_per_m)

        if grad_ok and slew_ok:
            print(f"Drunken spiral: Constraints met at iteration {i_iter+1}.")
            break

        if i_iter < num_smoothing_iterations -1 : # Don't smooth on the last iteration if still not met
            if k_points.shape[1] >= smoothing_kernel_size : # Ensure enough points for convolution
                smoothing_filter = np.ones(smoothing_kernel_size) / smoothing_kernel_size
                k_points[0,:] = np.convolve(k_points[0,:], smoothing_filter, mode='same')
                k_points[1,:] = np.convolve(k_points[1,:], smoothing_filter, mode='same')
            else:
                # Not enough points to smooth with this kernel, maybe break or use smaller kernel?
                print(f"Drunken spiral: Not enough points ({k_points.shape[1]}) for smoothing kernel size {smoothing_kernel_size} at iter {i_iter+1}. Stopping smoothing.")
                break

        if i_iter == num_smoothing_iterations - 1 and not (grad_ok and slew_ok):
            warning_msg = "Drunken spiral: Constraints not fully met after smoothing iterations."
            if not grad_ok and max_grad_Tm_per_m is not None:
                warning_msg += f" Max grad: {max_achieved_grad:.2f} T/m (Limit: {max_grad_Tm_per_m:.2f} T/m)."
            if not slew_ok and max_slew_Tm_per_s_per_m is not None:
                warning_msg += f" Max slew: {max_achieved_slew:.2f} T/m/s (Limit: {max_slew_Tm_per_s_per_m:.2f} T/m/s)."
            print(warning_msg)

    return k_points


def generate_tpi_trajectory(
    fov_mm: Union[float, Tuple[float, float, float]],
    resolution_mm: Union[float, Tuple[float, float, float]],
    num_twists: int,
    points_per_segment: int,
    cone_angle_deg: float,
    spiral_turns_per_twist: float,
    undersampling_factor: float = 1.0, # Placeholder, not directly used in this basic TPI
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
) -> np.ndarray:
    """
    Generates a 3D Twisted Projection Imaging (TPI) k-space trajectory.

    TPI involves tracing spiral paths on the surface of cones, where the cones
    themselves are distributed azimuthally.

    Parameters:
    - fov_mm: Field of view in mm (e.g., 256 or (256,256,256)).
    - resolution_mm: Resolution in mm (e.g., 1.0 or (1.0,1.0,1.0)).
    - num_twists (int): Number of twisted projection arms/segments.
    - points_per_segment (int): Number of k-space points along each twisted segment.
    - cone_angle_deg (float): Angle of the cone (half-angle) w.r.t. kz-axis (0 to 90 degrees).
                              0 degrees is a line along kz, 90 degrees is a 2D spiral in kx-ky.
    - spiral_turns_per_twist (float): Number of spiral turns along one projection/twist on the cone surface.
    - undersampling_factor (float): Overall undersampling factor (currently a placeholder).
    - gamma_Hz_per_T (float): Gyromagnetic ratio.

    Returns:
    - np.ndarray: K-space points of shape (3, num_twists * points_per_segment) in rad/m.
    """
    num_dimensions = 3 # TPI is inherently 3D

    if isinstance(fov_mm, (int, float)):
        fov_mm_tuple: Tuple[float, ...] = tuple([float(fov_mm)] * num_dimensions)
    elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == num_dimensions:
        fov_mm_tuple = tuple(map(float, fov_mm))
    else:
        raise ValueError(f"fov_mm must be a number or a tuple/list of length {num_dimensions}.")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm_tuple: Tuple[float, ...] = tuple([float(resolution_mm)] * num_dimensions)
    elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == num_dimensions:
        resolution_mm_tuple = tuple(map(float, resolution_mm))
    else:
        raise ValueError(f"resolution_mm must be a number or a tuple/list of length {num_dimensions}.")

    if not (0 <= cone_angle_deg <= 90):
        raise ValueError("cone_angle_deg must be between 0 and 90 degrees.")
    if num_twists <= 0 or points_per_segment <= 0:
        raise ValueError("num_twists and points_per_segment must be positive.")
    if spiral_turns_per_twist <= 0:
        raise ValueError("spiral_turns_per_twist must be positive.")

    resolution_m = np.array(resolution_mm_tuple) / 1000.0
    # Use the minimum resolution (largest k_max) to define the overall extent
    k_max_rad_per_m = np.min(1.0 / (2.0 * resolution_m))

    all_k_points_list = []

    # Golden angle increment for distributing twists azimuthally
    golden_angle_increment_rad = np.pi * (3.0 - np.sqrt(5.0))

    for i_twist in range(num_twists):
        twist_azimuth_rad = (i_twist * golden_angle_increment_rad) % (2 * np.pi)

        # The spiral radius extends along the surface of the cone.
        # k_max_rad_per_m is the maximum extent from the origin.
        # The length of the spiral path on the cone surface can go up to k_max_rad_per_m.
        if points_per_segment == 1:
            k_spiral_radius_on_surface = np.array([k_max_rad_per_m])
        else:
            k_spiral_radius_on_surface = np.linspace(0, k_max_rad_per_m, points_per_segment, endpoint=True)

        # Angle for the spiral path on the unfolded cone surface
        phi_spiral = np.linspace(0, spiral_turns_per_twist * 2 * np.pi, points_per_segment, endpoint=False if points_per_segment > 1 else True)

        # Map spiral points to the cone surface (local coordinates before twist rotation)
        cos_cone_angle = np.cos(np.deg2rad(cone_angle_deg))
        sin_cone_angle = np.sin(np.deg2rad(cone_angle_deg))

        kz_local = k_spiral_radius_on_surface * cos_cone_angle
        k_xy_plane_radius_local = k_spiral_radius_on_surface * sin_cone_angle

        kx_cone = k_xy_plane_radius_local * np.cos(phi_spiral)
        ky_cone = k_xy_plane_radius_local * np.sin(phi_spiral)

        # Rotate this segment by the twist's azimuthal angle around the main Z-axis
        cos_twist_az = np.cos(twist_azimuth_rad)
        sin_twist_az = np.sin(twist_azimuth_rad)

        kx_final = kx_cone * cos_twist_az - ky_cone * sin_twist_az
        ky_final = kx_cone * sin_twist_az + ky_cone * cos_twist_az
        kz_final = kz_local # kz is invariant under Z-axis rotation

        segment_k_points = np.vstack((kx_final, ky_final, kz_final)) # Shape (3, points_per_segment)
        all_k_points_list.append(segment_k_points)

    if not all_k_points_list:
        return np.zeros((3, 0))

    # Concatenate points from all twists
    final_k_points = np.concatenate(all_k_points_list, axis=1) # Shape (3, num_twists * points_per_segment)

    # Note: Undersampling factor is not explicitly used here yet,
    # it could modify num_twists or points_per_segment before generation,
    # or affect spiral density (turns) which is implicitly handled by spiral_turns_per_twist.
    # For this basic version, it's a placeholder.

    return final_k_points


def generate_wave_caipi_trajectory(
    fov_mm: Union[float, Tuple[float, float]],
    resolution_mm: Union[float, Tuple[float, float]],
    num_echoes: int,
    points_per_echo: int,
    wave_amplitude_mm: float,
    wave_frequency_cycles_per_fov_readout: float,
    wave_phase_offset_rad: float = 0.0,
    epi_type: str = 'flyback', # Affects readout reversal, not wave itself directly here
    phase_encode_direction: str = 'y', # 'y': kx readout, ky phase; 'x': ky readout, kx phase
    undersampling_factor_pe: float = 1.0, # Undersampling in phase-encode direction
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
) -> np.ndarray:
    """
    Generates a 2D Wave-CAIPI k-space trajectory (based on EPI).

    The "Wave" component adds a sinusoidal k-space offset to the phase-encode coordinate,
    modulated along the readout direction.

    Parameters:
    - fov_mm: Field of view in mm for the underlying EPI (e.g., (256, 256)).
    - resolution_mm: Resolution in mm for the underlying EPI (e.g., (2, 2)).
    - num_echoes (int): Total number of phase-encode lines before undersampling.
    - points_per_echo (int): Number of readout points per echo.
    - wave_amplitude_mm (float): Amplitude of the FOV shift in mm caused by the wave gradient,
                                 applied along the phase-encode axis.
    - wave_frequency_cycles_per_fov_readout (float): Number of full sine wave cycles
                                                     across the readout FOV.
    - wave_phase_offset_rad (float): Phase offset for the sine wave modulation.
    - epi_type (str): 'flyback' or 'gradient_recalled' (for EPI echo behavior).
    - phase_encode_direction (str): 'y' (kx readout, ky phase) or 'x' (ky readout, kx phase).
    - undersampling_factor_pe (float): Undersampling factor for phase-encode lines (e.g., 2.0 for R=2).
    - gamma_Hz_per_T (float): Gyromagnetic ratio.

    Returns:
    - np.ndarray: K-space points of shape (2, num_acquired_echoes * points_per_echo) in rad/m.
    """
    num_dimensions = 2

    if isinstance(fov_mm, (int, float)):
        fov_mm_tuple: Tuple[float, ...] = (float(fov_mm), float(fov_mm))
    elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == num_dimensions:
        fov_mm_tuple = tuple(map(float, fov_mm))
    else:
        raise ValueError(f"fov_mm must be a number or a tuple/list of length {num_dimensions}.")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm_tuple: Tuple[float, ...] = (float(resolution_mm), float(resolution_mm))
    elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == num_dimensions:
        resolution_mm_tuple = tuple(map(float, resolution_mm))
    else:
        raise ValueError(f"resolution_mm must be a number or a tuple/list of length {num_dimensions}.")

    if num_echoes <= 0 or points_per_echo <= 0:
        raise ValueError("num_echoes and points_per_echo must be positive.")
    if undersampling_factor_pe < 1.0:
        raise ValueError("undersampling_factor_pe must be >= 1.0.")

    fov_m = np.array(fov_mm_tuple) / 1000.0
    resolution_m = np.array(resolution_mm_tuple) / 1000.0

    readout_dim_idx_val = 0 if phase_encode_direction == 'y' else 1
    phase_dim_idx_val = 1 if phase_encode_direction == 'y' else 0

    k_max_readout = 1.0 / (2.0 * resolution_m[readout_dim_idx_val])
    k_max_phase_encode = 1.0 / (2.0 * resolution_m[phase_dim_idx_val])
    delta_k_phase = 1.0 / fov_m[phase_dim_idx_val]

    # Base k-space coordinates for one echo line (readout direction)
    kx_line_template = np.linspace(-k_max_readout, k_max_readout, points_per_echo, endpoint=True)

    # Wave component calculation
    # Normalize kx_line_template to [0, 1] for wave modulation
    kx_normalized_for_wave = (kx_line_template - (-k_max_readout)) / (2 * k_max_readout)

    # Convert wave_amplitude_mm to k-space shift amplitude in rad/m
    k_shift_amplitude_rad_per_m = (wave_amplitude_mm / (fov_mm_tuple[phase_dim_idx_val] / 2.0)) * k_max_phase_encode
    if fov_mm_tuple[phase_dim_idx_val] == 0:
        k_shift_amplitude_rad_per_m = 0.0

    wave_k_offset_readout = k_shift_amplitude_rad_per_m * np.sin(
        2 * np.pi * wave_frequency_cycles_per_fov_readout * kx_normalized_for_wave + wave_phase_offset_rad
    )

    acquired_echo_k_segments = []
    num_acquired_echoes = 0

    for i_true_echo in range(num_echoes):
        if undersampling_factor_pe > 1 and (i_true_echo % int(round(undersampling_factor_pe)) != 0) :
            continue

        num_acquired_echoes += 1

        ky_val_base = -k_max_phase_encode + i_true_echo * delta_k_phase

        current_kx_readout = np.copy(kx_line_template)
        current_wave_k_offset = np.copy(wave_k_offset_readout)
        if epi_type == 'gradient_recalled' and (num_acquired_echoes -1) % 2 != 0:
            current_kx_readout = current_kx_readout[::-1]
            current_wave_k_offset = current_wave_k_offset[::-1]

        if phase_encode_direction == 'y':
            kx_final_echo = current_kx_readout
            ky_final_echo = ky_val_base + current_wave_k_offset
            acquired_echo_k_segments.append(np.vstack((kx_final_echo, ky_final_echo)))
        else:
            ky_final_echo = current_kx_readout
            kx_final_echo = ky_val_base + current_wave_k_offset
            acquired_echo_k_segments.append(np.vstack((kx_final_echo, ky_final_echo)))

    if not acquired_echo_k_segments:
        return np.zeros((2, 0))

    final_k_points = np.concatenate(acquired_echo_k_segments, axis=1)
    return final_k_points


def generate_propeller_blade_trajectory(
    fov_mm: Union[float, Tuple[float, float]],
    resolution_mm: Union[float, Tuple[float, float]],
    num_blades: int,
    lines_per_blade: int,
    points_per_line: int,
    blade_rotation_angle_increment_deg: float,
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
) -> np.ndarray:
    """
    Generates a 2D PROPELLER/BLADE k-space trajectory.
    Each "blade" is a small Cartesian k-space acquisition (a strip)
    that is rotated around the k-space center.

    Parameters:
    - fov_mm: Field of view for a single blade/strip in mm (e.g., (256, 32) or 256 for isotropic blade FOV).
              The first component is along the readout (points_per_line),
              the second along the phase-encode direction of the blade (lines_per_blade).
    - resolution_mm: Resolution within a single blade in mm (e.g., (1, 4) or 1 for isotropic blade res).
    - num_blades (int): Number of blades to acquire.
    - lines_per_blade (int): Number of phase-encode lines within each blade.
    - points_per_line (int): Number of readout points along each line within a blade.
    - blade_rotation_angle_increment_deg (float): Angle in degrees to rotate each successive blade.
    - gamma_Hz_per_T (float): Gyromagnetic ratio (currently for consistency, not direct use in k-space calc).

    Returns:
    - np.ndarray: K-space points of shape (2, num_blades * lines_per_blade * points_per_line) in rad/m.
    """
    num_dimensions = 2

    if isinstance(fov_mm, (int, float)):
        fov_mm_tuple: Tuple[float, ...] = (float(fov_mm), float(fov_mm)) # Assume square if single val
    elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == num_dimensions:
        fov_mm_tuple = tuple(map(float, fov_mm))
    else:
        raise ValueError(f"fov_mm must be a number or a tuple/list of length {num_dimensions}.")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm_tuple: Tuple[float, ...] = (float(resolution_mm), float(resolution_mm))
    elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == num_dimensions:
        resolution_mm_tuple = tuple(map(float, resolution_mm))
    else:
        raise ValueError(f"resolution_mm must be a number or a tuple/list of length {num_dimensions}.")

    if num_blades <= 0 or lines_per_blade <= 0 or points_per_line <= 0:
        raise ValueError("num_blades, lines_per_blade, and points_per_line must be positive.")

    fov_m = np.array(fov_mm_tuple) / 1000.0
    resolution_m = np.array(resolution_mm_tuple) / 1000.0

    # k_max for the blade dimensions
    # k_max_x_blade is along the readout direction of the blade (points_per_line)
    # k_max_y_blade is along the phase-encode direction of the blade (lines_per_blade)
    k_max_x_blade = 1.0 / (2.0 * resolution_m[0])
    # For phase direction, k_max is (N_pe-1)/2 * delta_k_pe
    # delta_ky_blade = 1.0 / fov_m[1] (FOV in phase direction of blade)
    # k_max_y_blade_extent = (lines_per_blade - 1) / 2.0 * delta_ky_blade
    # This means ky_vals_blade will span from -k_max_y_blade_extent to +k_max_y_blade_extent

    delta_ky_blade = 1.0 / fov_m[1] # k-space step between phase-encode lines in a blade

    all_k_points_list = []

    # Create kx values for a single line (readout direction of the blade)
    # These are centered at 0 before rotation.
    kx_vals_line = np.linspace(-k_max_x_blade, k_max_x_blade, points_per_line, endpoint=True)

    # Create ky values for the phase-encode lines within a blade
    # These are also centered at 0 before rotation.
    if lines_per_blade == 1:
        ky_vals_blade = np.array([0.0])
    else:
        ky_vals_blade = np.linspace(
            -(lines_per_blade - 1) / 2.0 * delta_ky_blade,
            (lines_per_blade - 1) / 2.0 * delta_ky_blade,
            lines_per_blade,
            endpoint=True
        )

    # Generate points for one blade, centered at origin
    blade_k_points_local_list = []
    for i_line in range(lines_per_blade):
        kx_current_line = kx_vals_line
        ky_current_line = np.full_like(kx_vals_line, ky_vals_blade[i_line])
        blade_k_points_local_list.append(np.vstack((kx_current_line, ky_current_line)))

    blade_k_points_local = np.concatenate(blade_k_points_local_list, axis=1) # Shape (2, lines_per_blade * points_per_line)


    for i_blade in range(num_blades):
        current_rotation_rad = np.deg2rad(i_blade * blade_rotation_angle_increment_deg)

        cos_rot = np.cos(current_rotation_rad)
        sin_rot = np.sin(current_rotation_rad)

        rot_mat_2d = np.array([
            [cos_rot, -sin_rot],
            [sin_rot,  cos_rot]
        ])

        # Rotate a copy of the local blade points
        rotated_blade_k_points = rot_mat_2d @ blade_k_points_local
        all_k_points_list.append(rotated_blade_k_points)

    if not all_k_points_list: # Should not happen if num_blades > 0
        return np.zeros((2, 0))

    final_k_points = np.concatenate(all_k_points_list, axis=1)

    return final_k_points


def generate_spiral_trajectory(
    fov_mm,
    resolution_mm,
    num_dimensions=2,
    num_interleaves=1,
    points_per_interleaf=1024,
    undersampling_factor=1.0,
    spiral_type='archimedean',
    density_transition_radius_factor: Optional[float] = None, # Factor of k_max for transition
    density_factor_at_center: Optional[float] = None,     # How much denser at center vs periphery
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
    ):
    """
    Generates a spiral k-space trajectory.

    Parameters:
    - fov_mm (tuple or float): Field of view in millimeters (e.g., (256, 256) for 2D, 256 for isotropic 2D/3D).
    - resolution_mm (tuple or float): Desired resolution in millimeters (e.g., (1, 1) for 2D, 1 for isotropic 2D/3D).
    - num_dimensions (int): 2 or 3.
    - num_interleaves (int): Number of spiral interleaves.
    - points_per_interleaf (int): Number of k-space points per interleaf.
    - undersampling_factor (float): Factor for undersampling (e.g., 1.0 for fully sampled).
    - spiral_type (str): 'archimedean'. 'goldenangle' is a placeholder.
    - density_transition_radius_factor (Optional[float]): Factor of k_max (0-1) where density transition occurs.
    - density_factor_at_center (Optional[float]): Factor by which center is denser than periphery (e.g., 2.0).
    - gamma_Hz_per_T (float): Gyromagnetic ratio.

    Returns:
    - np.ndarray: K-space points of shape (num_dimensions, num_interleaves * points_per_interleaf) in rad/m.
                  Raises ValueError for invalid parameters.
    """

    if num_dimensions not in [2, 3]:
        raise ValueError("num_dimensions must be 2 or 3.")
    if not (isinstance(fov_mm, (int, float, tuple, list)) and isinstance(resolution_mm, (int, float, tuple, list))):
        raise ValueError("fov_mm and resolution_mm must be numbers or tuples/lists.")

    # Normalize fov and resolution to tuples
    if isinstance(fov_mm, (int, float)):
        fov_mm = tuple([fov_mm] * num_dimensions)
    elif len(fov_mm) != num_dimensions:
        raise ValueError(f"fov_mm tuple length must match num_dimensions ({num_dimensions}).")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm = tuple([resolution_mm] * num_dimensions)
    elif len(resolution_mm) != num_dimensions:
        raise ValueError(f"resolution_mm tuple length must match num_dimensions ({num_dimensions}).")

    if points_per_interleaf <= 0:
        raise ValueError("points_per_interleaf must be positive.")
    if num_interleaves <= 0:
        raise ValueError("num_interleaves must be positive.")
    if undersampling_factor <= 0:
        raise ValueError("undersampling_factor must be positive.")

    # Convert mm to m for calculations
    fov_m = np.array(fov_mm) / 1000.0
    resolution_m = np.array(resolution_mm) / 1000.0

    # Calculate maximum k-space extent (rad/m)
    k_max_rad_per_m = 1.0 / (2.0 * resolution_m)

    # Parameter validation for new density params
    if density_transition_radius_factor is not None and not (0 <= density_transition_radius_factor <= 1.0):
        raise ValueError("density_transition_radius_factor must be between 0.0 and 1.0.")
    if density_factor_at_center is not None and density_factor_at_center < 0:
        raise ValueError("density_factor_at_center must be non-negative.")

    use_variable_density = (density_transition_radius_factor is not None and
                            density_factor_at_center is not None and
                            density_factor_at_center != 1.0 and # No change if factor is 1
                            density_transition_radius_factor > 1e-6) # Avoid division by zero if used as denominator

    total_points = num_interleaves * points_per_interleaf
    all_k_points = np.zeros((num_dimensions, total_points))

    if spiral_type.lower() == 'archimedean':
        if num_dimensions == 2:
            k_max_xy = np.min(k_max_rad_per_m[:2])

            for i in range(num_interleaves):
                start_angle = (2 * np.pi * i) / (num_interleaves * undersampling_factor)
                # Using a simplified model where k_radius grows linearly with sample index for each interleaf
                # And angle advances based on a set number of turns.

                phi_max_turns = 10 / np.sqrt(undersampling_factor) # Base number of turns

                # Radius array (linear for now, angular velocity will be modulated)
                radii_n = np.linspace(0, k_max_xy, points_per_interleaf, endpoint=True)

                if use_variable_density and points_per_interleaf > 1:
                    r_transition = density_transition_radius_factor * k_max_xy

                    # Calculate density weighting for each point based on its radius
                    # density_weight_n > 1 at center if DFC > 1
                    density_weights_n = 1.0 + (density_factor_at_center - 1.0) * \
                                        np.exp(-(radii_n / r_transition)**2)

                    inv_density_weights_n = 1.0 / density_weights_n

                    # Calculate base_d_phi such that total angle is phi_max_turns * 2 * pi
                    sum_inv_density_weights = np.sum(inv_density_weights_n)
                    if sum_inv_density_weights < 1e-9: # Avoid division by zero if all weights are huge
                        base_d_phi = (phi_max_turns * 2 * np.pi) / points_per_interleaf
                    else:
                        base_d_phi = (phi_max_turns * 2 * np.pi) / sum_inv_density_weights

                    d_phi_n = base_d_phi * inv_density_weights_n
                    phi = np.cumsum(d_phi_n) - d_phi_n[0] # Start phi from 0
                else: # Uniform density
                    phi = np.linspace(0, phi_max_turns * 2 * np.pi, points_per_interleaf, endpoint=True if points_per_interleaf==1 else False)

                if points_per_interleaf == 1:
                    current_k_radius = np.array([k_max_xy]) # Radius for the single point is k_max_xy
                else:
                    current_k_radius = radii_n # Use the linearly spaced radii

                current_angles = start_angle + phi

                start_idx = i * points_per_interleaf
                end_idx = start_idx + points_per_interleaf

                all_k_points[0, start_idx:end_idx] = current_k_radius * np.cos(current_angles)
                all_k_points[1, start_idx:end_idx] = current_k_radius * np.sin(current_angles)

        elif num_dimensions == 3:
            # Stack-of-spirals with variable density in-plane
            if len(k_max_rad_per_m) < 3 or len(fov_m) < 3:
                raise ValueError("For 3D, fov_mm and resolution_mm must provide 3 components.")

            k_max_xy = np.min(k_max_rad_per_m[:2])
            num_kz_slices = num_interleaves

            kz_values = np.linspace(-k_max_rad_per_m[2], k_max_rad_per_m[2], num_kz_slices, endpoint=True)
            if num_kz_slices == 1: kz_values = np.array([0.0])

            points_per_slice = points_per_interleaf

            for i in range(num_kz_slices):
                start_angle_planar = 0 # No planar interleaving for simple stack-of-spirals here

                phi_max_turns = 10 / np.sqrt(undersampling_factor)
                radii_n_slice = np.linspace(0, k_max_xy, points_per_slice, endpoint=True)

                if use_variable_density and points_per_slice > 1:
                    r_transition_slice = density_transition_radius_factor * k_max_xy
                    density_weights_n_slice = 1.0 + (density_factor_at_center - 1.0) * \
                                              np.exp(-(radii_n_slice / r_transition_slice)**2)
                    inv_density_weights_n_slice = 1.0 / density_weights_n_slice
                    sum_inv_density_weights_slice = np.sum(inv_density_weights_n_slice)

                    if sum_inv_density_weights_slice < 1e-9:
                         base_d_phi_slice = (phi_max_turns * 2 * np.pi) / points_per_slice
                    else:
                         base_d_phi_slice = (phi_max_turns * 2 * np.pi) / sum_inv_density_weights_slice

                    d_phi_n_slice = base_d_phi_slice * inv_density_weights_n_slice
                    phi_slice = np.cumsum(d_phi_n_slice) - d_phi_n_slice[0]
                else: # Uniform density
                    phi_slice = np.linspace(0, phi_max_turns * 2 * np.pi, points_per_slice, endpoint=True if points_per_slice==1 else False)

                if points_per_slice == 1:
                    current_k_radius_slice = np.array([k_max_xy])
                else:
                    current_k_radius_slice = radii_n_slice

                current_angles = start_angle_planar + phi_slice

                start_idx = i * points_per_slice
                end_idx = start_idx + points_per_slice

                all_k_points[0, start_idx:end_idx] = current_k_radius_slice * np.cos(current_angles)
                all_k_points[1, start_idx:end_idx] = current_k_radius_slice * np.sin(current_angles)
                all_k_points[2, start_idx:end_idx] = kz_values[i]
        else:
            raise NotImplementedError(f"Spiral generation for {num_dimensions}D not implemented with Archimedean type.")

    elif spiral_type.lower() == 'goldenangle':
        raise NotImplementedError("Golden angle spiral_type is not yet implemented.")
    else:
        raise ValueError(f"Unknown spiral_type: {spiral_type}. Supported types: 'archimedean'.")

    return all_k_points


def generate_cones_trajectory(
    fov_mm: Union[float, Tuple[float, ...]],
    resolution_mm: Union[float, Tuple[float, ...]],
    num_cones: int,
    points_per_cone: int,
    cone_angle_deg: float,
    undersampling_factor: float = 1.0, # Affects spiral density on cone surface
    rotation_angle_increment_deg: Optional[float] = None, # Azimuthal rotation between cones
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
    ) -> np.ndarray:
    """
    Generates a 3D Cones k-space trajectory.
    Each cone has a spiral trajectory traced on its surface.

    Parameters:
    - fov_mm (Union[float, Tuple[float,...]]): Field of view in mm (e.g., (200,200,200) or 200 for isotropic).
    - resolution_mm (Union[float, Tuple[float,...]]): Resolution in mm (e.g., (1,1,1) or 1 for isotropic).
    - num_cones (int): Number of cones.
    - points_per_cone (int): Number of k-space points along the spiral path on each cone.
    - cone_angle_deg (float): The half-angle of the cone with respect to the kz-axis (0 to 90 degrees).
                               0 degrees would be a line along kz, 90 degrees a 2D spiral in kx-ky plane.
    - undersampling_factor (float): Factor for undersampling the spiral on the cone surface.
    - rotation_angle_increment_deg (Optional[float]): Azimuthal rotation (around kz) between successive cones.
                                                    If None, uses golden angle increment.
    - gamma_Hz_per_T (float): Gyromagnetic ratio.

    Returns:
    - np.ndarray: K-space points of shape (3, num_cones * points_per_cone) in rad/m.
    """
    num_dimensions = 3 # Cones are inherently 3D

    if isinstance(fov_mm, (int, float)):
        fov_mm_tuple: Tuple[float, ...] = tuple([float(fov_mm)] * num_dimensions)
    elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == num_dimensions:
        fov_mm_tuple = tuple(map(float, fov_mm))
    else:
        raise ValueError(f"fov_mm must be a number or a tuple/list of length {num_dimensions}.")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm_tuple: Tuple[float, ...] = tuple([float(resolution_mm)] * num_dimensions)
    elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == num_dimensions:
        resolution_mm_tuple = tuple(map(float, resolution_mm))
    else:
        raise ValueError(f"resolution_mm must be a number or a tuple/list of length {num_dimensions}.")

    if not (0 < cone_angle_deg < 90): # Strictly between 0 and 90 for a cone shape
        raise ValueError("cone_angle_deg must be between 0 and 90 degrees (exclusive).")
    if num_cones <= 0 or points_per_cone <= 0:
        raise ValueError("num_cones and points_per_cone must be positive.")
    if undersampling_factor <= 0:
        raise ValueError("undersampling_factor must be positive.")

    resolution_m = np.array(resolution_mm_tuple) / 1000.0
    # k_max for the whole 3D volume, typically use the minimum resolution component for this
    # This k_max refers to the max radius in k-space if it were a sphere.
    k_max_rad_per_m_sphere = np.min(1.0 / (2.0 * resolution_m))

    cone_angle_rad = np.deg2rad(cone_angle_deg)

    k_max_on_cone_surface = k_max_rad_per_m_sphere

    total_k_points = np.zeros((3, num_cones * points_per_cone))

    if rotation_angle_increment_deg is None:
        azimuthal_increment_rad = np.pi * (3.0 - np.sqrt(5.0))
    else:
        azimuthal_increment_rad = np.deg2rad(rotation_angle_increment_deg)

    current_cone_rotation_angle = 0.0

    for i_cone in range(num_cones):
        phi_max_turns = 10 / np.sqrt(undersampling_factor)
        phi_spiral = np.linspace(0, phi_max_turns * 2 * np.pi, points_per_cone, endpoint=False)

        if points_per_cone == 1:
            k_spiral_radius_on_surface = np.array([k_max_on_cone_surface])
        else:
            k_spiral_radius_on_surface = np.linspace(0, k_max_on_cone_surface, points_per_cone, endpoint=True)

        kz_coords = k_spiral_radius_on_surface * np.cos(cone_angle_rad)
        k_xy_plane_radius = k_spiral_radius_on_surface * np.sin(cone_angle_rad)

        kx_prime = k_xy_plane_radius * np.cos(phi_spiral)
        ky_prime = k_xy_plane_radius * np.sin(phi_spiral)

        if num_cones > 1:
            cos_rot = np.cos(current_cone_rotation_angle)
            sin_rot = np.sin(current_cone_rotation_angle)
            kx_final = kx_prime * cos_rot - ky_prime * sin_rot
            ky_final = kx_prime * sin_rot + ky_prime * cos_rot
        else:
            kx_final = kx_prime
            ky_final = ky_prime

        start_idx = i_cone * points_per_cone
        end_idx = start_idx + points_per_cone

        total_k_points[0, start_idx:end_idx] = kx_final
        total_k_points[1, start_idx:end_idx] = ky_final
        total_k_points[2, start_idx:end_idx] = kz_coords

        current_cone_rotation_angle += azimuthal_increment_rad

    return total_k_points


def generate_rosette_trajectory(
    fov_mm: Union[float, Tuple[float, float]],
    resolution_mm: Union[float, Tuple[float, float]],
    num_petals: int, # Number of major petals (e.g., n_L in k(phi)=k_max*sin(n_L*phi)*exp(i*n_S*phi))
    total_points: int, # Total number of points for the entire trajectory
    num_radial_cycles: int, # Number of times the radius oscillates from 0 to k_max and back (e.g., n_S)
    k_max_rosette_factor: float = 1.0, # Scales k_max relative to Nyquist
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
    ) -> np.ndarray:
    """
    Generates a 2D Rosette k-space trajectory.
    Equation: k(t) = k_max_eff * sin(omega_radial * t) * [cos(omega_petal * t), sin(omega_petal * t)]

    Parameters:
    - fov_mm: Field of view in mm.
    - resolution_mm: Resolution in mm.
    - num_petals (int): Number of major petals (controls rotational frequency omega_petal).
    - total_points (int): Total number of k-space points for the trajectory.
    - num_radial_cycles (int): Number of full cycles for the radial component (sin(omega_radial*t))
                               over the course of t=[0, 2*pi]. This controls lobes on petals.
    - k_max_rosette_factor (float): Factor (0 to 1.0) to scale the actual k_max used for the rosette
                                   relative to Nyquist limit.
    - gamma_Hz_per_T (float): Gyromagnetic ratio.

    Returns:
    - np.ndarray: K-space points of shape (2, total_points) in rad/m.
    """
    num_dimensions = 2 # Rosette is typically 2D

    if isinstance(fov_mm, (int, float)):
        fov_mm_tuple: Tuple[float, float] = (float(fov_mm), float(fov_mm))
    elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == num_dimensions:
        fov_mm_tuple = tuple(map(float, fov_mm))
    else:
        raise ValueError(f"fov_mm must be a number or a tuple/list of length {num_dimensions}.")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm_tuple: Tuple[float, float] = (float(resolution_mm), float(resolution_mm))
    elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == num_dimensions:
        resolution_mm_tuple = tuple(map(float, resolution_mm))
    else:
        raise ValueError(f"resolution_mm must be a number or a tuple/list of length {num_dimensions}.")

    if num_petals <= 0:
        raise ValueError("num_petals must be positive.")
    if total_points <= 0:
        raise ValueError("total_points must be positive.")
    if num_radial_cycles <= 0:
        raise ValueError("num_radial_cycles must be positive.")
    if not (0.0 < k_max_rosette_factor <= 1.0):
        raise ValueError("k_max_rosette_factor must be between 0 (exclusive) and 1 (inclusive).")

    resolution_m = np.array(resolution_mm_tuple) / 1000.0

    # Max k-space extent based on Nyquist, then scaled by factor
    k_nyquist = 1.0 / (2.0 * np.min(resolution_m)) # Use min resolution for overall k_max
    k_max_eff = k_nyquist * k_max_rosette_factor

    # Parameter 't' spans [0, 2*pi] for one full definition of the pattern
    # The density of points is determined by 'total_points' over this [0, 2*pi] range.
    t = np.linspace(0, 2 * np.pi, total_points, endpoint=False)

    omega_petal = float(num_petals)  # Rotational frequency for petals
    omega_radial = float(num_radial_cycles) # Frequency for radial oscillations (lobes)

    # k(t) = k_max_eff * sin(omega_radial * t) * [cos(omega_petal * t), sin(omega_petal * t)]
    radial_component = k_max_eff * np.sin(omega_radial * t)

    kx = radial_component * np.cos(omega_petal * t)
    ky = radial_component * np.sin(omega_petal * t)

    k_space_points = np.vstack((kx, ky)) # Shape (2, total_points)

    return k_space_points


def generate_epi_trajectory(
    fov_mm: Union[float, Tuple[float, float]],
    resolution_mm: Union[float, Tuple[float, float]],
    num_echoes: int, # Number of lines in k-space (phase-encode steps)
    points_per_echo: int, # Number of points along each echo (readout direction)
    ramp_sample_percentage: float = 0.1, # Percentage of points_per_echo for gradient ramps
    epi_type: str = 'flyback', # 'flyback' or 'gradient_recalled'
    phase_encode_direction: str = 'y', # 'y' (kx readout, ky phase) or 'x' (ky readout, kx phase)
    acquire_every_other_line: bool = False, # For partial Fourier or accelerated EPI
    gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
    ) -> np.ndarray:
    """
    Generates a 2D Echo Planar Imaging (EPI) k-space trajectory.

    Parameters:
    - fov_mm (Union[float, Tuple[float,float]]): Field of view in mm (e.g., (256,256) or 256 for isotropic).
    - resolution_mm (Union[float, Tuple[float,float]]): Resolution in mm (e.g., (1,1) or 1 for isotropic).
    - num_echoes (int): Number of echoes (lines in k-space, typically phase-encode steps).
    - points_per_echo (int): Number of points along each echo (readout direction).
    - ramp_sample_percentage (float): Percentage of points_per_echo for gradient ramps at start/end of each line (0 to 0.5).
    - epi_type (str): 'flyback' or 'gradient_recalled'.
    - phase_encode_direction (str): 'y' (kx readout, ky phase steps) or 'x' (ky readout, kx phase steps).
    - acquire_every_other_line (bool): If True, skips every other phase-encode line.
    - gamma_Hz_per_T (float): Gyromagnetic ratio.

    Returns:
    - np.ndarray: K-space points of shape (2, num_acquired_echoes * points_per_echo) in rad/m.
    """
    num_dimensions = 2 # EPI is 2D

    if isinstance(fov_mm, (int, float)):
        fov_mm_tuple: Tuple[float, float] = (float(fov_mm), float(fov_mm))
    elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == num_dimensions:
        fov_mm_tuple = tuple(map(float, fov_mm))
    else:
        raise ValueError(f"fov_mm must be a number or a tuple/list of length {num_dimensions}.")

    if isinstance(resolution_mm, (int, float)):
        resolution_mm_tuple: Tuple[float, float] = (float(resolution_mm), float(resolution_mm))
    elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == num_dimensions:
        resolution_mm_tuple = tuple(map(float, resolution_mm))
    else:
        raise ValueError(f"resolution_mm must be a number or a tuple/list of length {num_dimensions}.")

    if not (0 <= ramp_sample_percentage < 0.5):
        raise ValueError("ramp_sample_percentage must be between 0 and 0.5 (exclusive of 0.5).")
    if num_echoes <= 0 or points_per_echo <= 0:
        raise ValueError("num_echoes and points_per_echo must be positive.")
    if epi_type not in ['flyback', 'gradient_recalled']:
        raise ValueError("epi_type must be 'flyback' or 'gradient_recalled'.")
    if phase_encode_direction not in ['x', 'y']:
        raise ValueError("phase_encode_direction must be 'x' or 'y'.")

    resolution_m = np.array(resolution_mm_tuple) / 1000.0

    # Determine k_max and delta_k for readout and phase-encode directions
    # k_max = 1 / (2 * resolution)
    # delta_k = 1 / FOV

    if phase_encode_direction == 'y': # Readout along kx, phase steps along ky
        k_max_readout = 1.0 / (2.0 * resolution_m[0])
        delta_k_readout = 1.0 / fov_mm_tuple[0] # This is actually pixel size in k-space along readout

        k_max_phase = 1.0 / (2.0 * resolution_m[1])
        delta_k_phase = 1.0 / (fov_mm_tuple[1] / 1000.0) # k-space step per phase line
        readout_dim_idx, phase_dim_idx = 0, 1
    else: # Readout along ky, phase steps along kx
        k_max_readout = 1.0 / (2.0 * resolution_m[1])
        delta_k_readout = 1.0 / fov_mm_tuple[1]

        k_max_phase = 1.0 / (2.0 * resolution_m[0])
        delta_k_phase = 1.0 / (fov_mm_tuple[0] / 1000.0)
        readout_dim_idx, phase_dim_idx = 1, 0

    num_ramp_points = int(points_per_echo * ramp_sample_percentage)
    num_flat_top_points = points_per_echo - 2 * num_ramp_points
    if num_flat_top_points <= 0:
        raise ValueError("ramp_sample_percentage is too high, no flat top points remaining.")

    # Create a single readout line template (from -k_max_readout to +k_max_readout)
    line_k_readout = np.zeros(points_per_echo)
    if num_ramp_points > 0:
        line_k_readout[:num_ramp_points] = np.linspace(-k_max_readout, -k_max_readout * (num_flat_top_points / points_per_echo), num_ramp_points, endpoint=False) # Simplified ramp
        line_k_readout[num_ramp_points + num_flat_top_points:] = np.linspace(k_max_readout * (num_flat_top_points / points_per_echo), k_max_readout, num_ramp_points, endpoint=True) # Simplified ramp

    # Flat top part: should span most of -k_max to +k_max
    # A more realistic flat top for EPI would ensure the area under flat top corresponds to k_max.
    # For now, simple linear space over the flat top portion.
    # If ramp points exist, the flat top is smaller.
    # Let's make the flat top go from -k_max_readout to k_max_readout, and ramps eat into it.
    # This means the actual k_max reached on flat top is k_max_readout.

    # Simpler definition of one echo line:
    # Ramp up, flat top, ramp down.
    # Total points: points_per_echo
    # Ramp points at each end: num_ramp_points
    # Flat top points: points_per_echo - 2 * num_ramp_points

    # k-space points for one echo from -k_max_readout to +k_max_readout
    k_readout_flat = np.linspace(-k_max_readout, k_max_readout, num_flat_top_points)

    if num_ramp_points > 0:
        k_ramp_up = np.linspace(-k_max_readout * (1 + ramp_sample_percentage*2), -k_max_readout, num_ramp_points, endpoint=False) # approximate
        k_ramp_down = np.linspace(k_max_readout, k_max_readout * (1 + ramp_sample_percentage*2), num_ramp_points, endpoint=False) # approximate
        # This ramp definition is not great. Let's use a simpler one:
        # The line_k_readout should just be a linspace from -k_max_readout to k_max_readout.
        # The ramps are implicitly handled by how gradients would be designed to trace this.
        # For k-space points themselves, they just go from -k_max to +k_max.
        line_k_readout = np.linspace(-k_max_readout, k_max_readout, points_per_echo, endpoint=True)


    actual_echoes_to_acquire = []
    for i_echo in range(num_echoes):
        if acquire_every_other_line and (i_echo % 2 != 0):
            continue
        actual_echoes_to_acquire.append(i_echo)

    num_acquired_echoes = len(actual_echoes_to_acquire)
    if num_acquired_echoes == 0:
        return np.zeros((num_dimensions, 0))

    total_k_points = np.zeros((num_dimensions, num_acquired_echoes * points_per_echo))

    current_acquired_echo_idx = 0
    for i_echo_true_index in range(num_echoes): # Iterate through all potential echo indices

        if acquire_every_other_line and (i_echo_true_index % 2 != 0):
            continue # Skip this line

        # Phase encoding for this line
        # Lines are typically from -k_max_phase to +k_max_phase
        # Or 0 to k_max_phase if single quadrant
        # Standard EPI covers full k-space, so from -(N_pe/2)*delta_k_phase to +(N_pe/2-1)*delta_k_phase
        k_phase_step_val = -k_max_phase + i_echo_true_index * delta_k_phase

        current_line_k_readout = np.copy(line_k_readout)
        if epi_type == 'gradient_recalled' and (current_acquired_echo_idx % 2 != 0):
            current_line_k_readout = current_line_k_readout[::-1] # Reverse direction for gradient recalled

        start_idx = current_acquired_echo_idx * points_per_echo
        end_idx = start_idx + points_per_echo

        total_k_points[readout_dim_idx, start_idx:end_idx] = current_line_k_readout
        total_k_points[phase_dim_idx, start_idx:end_idx] = k_phase_step_val

        current_acquired_echo_idx += 1

    return total_k_points


def generate_radial_trajectory(
    num_spokes,
    points_per_spoke,
    num_dimensions=2,
    fov_mm=None,
    resolution_mm=None,
    projection_angle_increment='golden_angle',
    gamma_Hz_per_T=42.576e6 # For future use
    ):
    """
    Generates a radial k-space trajectory.

    Parameters:
    - num_spokes (int): Number of radial spokes.
    - points_per_spoke (int): Number of k-space points per spoke.
    - num_dimensions (int): 2 or 3.
    - fov_mm (tuple or float, optional): Field of view in mm. Used with resolution_mm to set k_max.
    - resolution_mm (tuple or float, optional): Resolution in mm. Used with fov_mm to set k_max.
    - projection_angle_increment (str or float): 'golden_angle' or a fixed angle in degrees for 2D.
                                                 For 3D, 'golden_angle' uses a 3D generalization.
    - gamma_Hz_per_T (float): Gyromagnetic ratio.

    Returns:
    - np.ndarray: K-space points of shape (num_dimensions, num_spokes * points_per_spoke) in rad/m
                  or normalized units if fov/resolution not provided.
                  Raises ValueError for invalid parameters.
    """

    if num_dimensions not in [2, 3]:
        raise ValueError("num_dimensions must be 2 or 3.")
    if num_spokes <= 0 or points_per_spoke <= 0:
        raise ValueError("num_spokes and points_per_spoke must be positive.")

    k_max_val = 0.5 * points_per_spoke
    # scale_to_rad_per_m = False # Not explicitly used, k_max_val directly holds value in target units

    if fov_mm is not None and resolution_mm is not None:
        if not (isinstance(fov_mm, (int, float, tuple, list)) and isinstance(resolution_mm, (int, float, tuple, list))):
            raise ValueError("fov_mm and resolution_mm must be numbers or tuples/lists when provided.")

        _fov_mm_tuple = fov_mm
        if isinstance(fov_mm, (int, float)):
            _fov_mm_tuple = tuple([fov_mm] * num_dimensions)
        elif len(fov_mm) != num_dimensions:
            raise ValueError(f"fov_mm tuple length must match num_dimensions ({num_dimensions}).")

        _res_mm_tuple = resolution_mm
        if isinstance(resolution_mm, (int, float)):
            _res_mm_tuple = tuple([resolution_mm] * num_dimensions)
        elif len(resolution_mm) != num_dimensions:
            raise ValueError(f"resolution_mm tuple length must match num_dimensions ({num_dimensions}).")

        resolution_m = np.array(_res_mm_tuple) / 1000.0
        k_max_val = 1.0 / (2.0 * resolution_m[0])
        # scale_to_rad_per_m = True
    elif fov_mm is not None or resolution_mm is not None: # XOR condition
        raise ValueError("Both fov_mm and resolution_mm must be provided together, or neither.")

    total_points = num_spokes * points_per_spoke
    all_k_points = np.zeros((num_dimensions, total_points))

    # spoke_template_k = np.linspace(0, k_max_val, points_per_spoke, endpoint=True)
    if points_per_spoke == 1:
        spoke_template_k = np.array([k_max_val])
    else:
        spoke_template_k = np.linspace(0, k_max_val, points_per_spoke, endpoint=True)


    if num_dimensions == 2:
        if projection_angle_increment == 'golden_angle':
            golden_angle_rad = np.pi * (3.0 - np.sqrt(5.0))
        elif isinstance(projection_angle_increment, (int, float)):
            golden_angle_rad = np.deg2rad(projection_angle_increment)
        else:
            raise ValueError("projection_angle_increment must be 'golden_angle' or a number (degrees).")

        current_angle = 0.0
        for i in range(num_spokes):
            start_idx = i * points_per_spoke
            end_idx = start_idx + points_per_spoke

            all_k_points[0, start_idx:end_idx] = spoke_template_k * np.cos(current_angle)
            all_k_points[1, start_idx:end_idx] = spoke_template_k * np.sin(current_angle)

            current_angle += golden_angle_rad
            # No need to wrap current_angle, cos and sin handle periodicity.

    elif num_dimensions == 3:
        if projection_angle_increment == 'golden_angle':
            for i in range(num_spokes):
                idx_float = float(i)
                num_spokes_float = float(num_spokes)

                h_k = -1.0 + (2.0 * idx_float) / (num_spokes_float -1.0) if num_spokes > 1 else 0.0
                # Ensure h_k is within [-1, 1] for arccos due to potential float precision issues
                h_k = np.clip(h_k, -1.0, 1.0)
                theta_k = np.arccos(h_k)

                phi_k = 0.0
                # A common formulation for Fibonacci spiral / spherical golden angle
                # Using (sqrt(5)-1)/2 which is approx 0.618 (inverse of golden ratio)
                # Or directly use 2*pi / PHI^2 where PHI = (1+sqrt(5))/2
                # Another simple way: phi_k = (i * angle_increment_3d_azimuthal) % (2*pi)
                # The Saff and Kuijlaars paper suggests a specific method for phi_k related to h_k or i.
                # Let's use a simpler, common one: iterate phi by a golden angle based increment
                # Or use the one that depends on i and golden ratio directly:
                # phi_k = (i * 2.0 * np.pi / ((1.0 + np.sqrt(5.0))/2.0)**2 ) % (2.0*np.pi) # Approx 2.399 rad ~137.5 deg
                # This is the spherical golden angle.
                spherical_golden_angle_rad = 2.0 * np.pi / (((1.0 + np.sqrt(5.0))/2.0)**2)
                phi_k = (i * spherical_golden_angle_rad) % (2.0 * np.pi)

                # The choice of (theta_k, phi_k) generation method for "golden_angle" 3D can vary.
                # The one above (h_k for theta_k, iterative golden angle for phi_k) is one approach.
                # Saff & Kuijlaars: theta_k = arccos(h_k), phi_k = (phi_{k-1} + 3.6/sqrt(N_spokes*(1-h_k^2))) mod 2pi
                # This is complex. Let's stick to a simpler Fibonacci spiral variant for theta and phi.
                # For point i out of N:
                #   theta_i = arccos(1 - 2*i / (N-1))  (for i = 0 to N-1)
                #   phi_i = 2 * pi * i / PHI   (where PHI is golden ratio)

                # Re-evaluating the 3D golden angle based on common practice:
                # (This is a widely cited method for distributing points on a sphere)
                # Source: "Construction of a Regular Triangulation of the Sphere" - E. B. Saff and A. B. J. Kuijlaars
                # Simplified for point index i from 0 to N-1:
                # cos_theta = 1 - (2 * (i + 0.5)) / N  (avoids poles being exactly hit)
                # theta = arccos(cos_theta)
                # phi = 2 * pi * (i + 0.5) / PHI  (PHI = (1+sqrt(5))/2)

                cos_theta = 1.0 - (2.0 * (idx_float + 0.5)) / num_spokes_float
                cos_theta = np.clip(cos_theta, -1.0, 1.0) # Ensure valid input for arccos
                theta_k = np.arccos(cos_theta)

                golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
                phi_k = (2.0 * np.pi * (idx_float + 0.5)) / golden_ratio
                phi_k %= (2.0 * np.pi)


                start_idx = i * points_per_spoke
                end_idx = start_idx + points_per_spoke

                kx_dir = np.sin(theta_k) * np.cos(phi_k)
                ky_dir = np.sin(theta_k) * np.sin(phi_k)
                kz_dir = np.cos(theta_k)

                all_k_points[0, start_idx:end_idx] = spoke_template_k * kx_dir
                all_k_points[1, start_idx:end_idx] = spoke_template_k * ky_dir
                all_k_points[2, start_idx:end_idx] = spoke_template_k * kz_dir

        elif isinstance(projection_angle_increment, (tuple, list)) and len(projection_angle_increment) == num_spokes:
            if not all(isinstance(pair, (tuple, list)) and len(pair) == 2 for pair in projection_angle_increment):
                raise ValueError("Each element in projection_angle_increment list for 3D must be a (theta, phi) pair.")

            for i in range(num_spokes):
                theta_k, phi_k = projection_angle_increment[i] # Assuming radians
                start_idx = i * points_per_spoke
                end_idx = start_idx + points_per_spoke

                kx_dir = np.sin(theta_k) * np.cos(phi_k)
                ky_dir = np.sin(theta_k) * np.sin(phi_k)
                kz_dir = np.cos(theta_k)

                all_k_points[0, start_idx:end_idx] = spoke_template_k * kx_dir
                all_k_points[1, start_idx:end_idx] = spoke_template_k * ky_dir
                all_k_points[2, start_idx:end_idx] = spoke_template_k * kz_dir
        else:
            raise ValueError("For 3D radial, projection_angle_increment must be 'golden_angle' "
                             "or a list/tuple of (theta, phi) radian pairs for each spoke.")
    else:
        raise NotImplementedError(f"Radial generation for {num_dimensions}D not implemented.")

    return all_k_points
