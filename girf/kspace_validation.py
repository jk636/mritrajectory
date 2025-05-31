import numpy as np
# For advanced density (Voronoi), one might import:
# from scipy.spatial import Voronoi

def get_kspace_extents(kspace_trajectory_array):
    """
    Calculates the minimum and maximum k-space extents for each dimension.

    Args:
        kspace_trajectory_array (np.ndarray): K-space trajectory data.
            Shape (num_points, num_dimensions), units: m^-1.

    Returns:
        dict: {'k_min_per_axis': np.array, 'k_max_per_axis': np.array,
               'k_range_per_axis': np.array, 'k_center_per_axis': np.array}
    """
    if not isinstance(kspace_trajectory_array, np.ndarray) or kspace_trajectory_array.ndim != 2:
        raise ValueError("kspace_trajectory_array must be a 2D NumPy array (num_points, num_dimensions).")
    if kspace_trajectory_array.shape[0] == 0: # Empty trajectory
        num_dims = kspace_trajectory_array.shape[1]
        nan_array = np.full(num_dims, np.nan)
        return {
            'k_min_per_axis': nan_array, 'k_max_per_axis': nan_array,
            'k_range_per_axis': nan_array, 'k_center_per_axis': nan_array
        }

    k_min = np.min(kspace_trajectory_array, axis=0)
    k_max = np.max(kspace_trajectory_array, axis=0)
    k_range = k_max - k_min
    k_center = k_min + k_range / 2.0

    return {
        'k_min_per_axis': k_min,
        'k_max_per_axis': k_max,
        'k_range_per_axis': k_range,
        'k_center_per_axis': k_center
    }


def check_kspace_coverage_binary_grid(kspace_trajectory_array, fov_m, matrix_size, kspace_center_offset_m_inv=None):
    """
    Evaluates k-space coverage by mapping trajectory points to a binary grid.

    Args:
        kspace_trajectory_array (np.ndarray): (N_points, N_dims), units: m^-1.
        fov_m (tuple/list/np.ndarray): Field of View for each dimension (m).
        matrix_size (tuple/list/np.ndarray): Matrix size for each dimension.
        kspace_center_offset_m_inv (tuple/list/np.ndarray, optional): Offset of k-space center (m^-1)
            from (0,0,0) if not already centered. This offset is *added* to trajectory points
            before mapping to grid.

    Returns:
        dict: {'coverage_percentage': float, 'num_cells_hit': int,
               'total_grid_cells': int, 'k_space_grid_binary': np.ndarray (boolean)}
    """
    if not isinstance(kspace_trajectory_array, np.ndarray) or kspace_trajectory_array.ndim != 2:
        raise ValueError("kspace_trajectory_array must be a 2D NumPy array.")

    num_dims = kspace_trajectory_array.shape[1]
    fov_m = np.asarray(fov_m)
    matrix_size = np.asarray(matrix_size, dtype=int)

    if len(fov_m) != num_dims or len(matrix_size) != num_dims:
        raise ValueError("fov_m and matrix_size must match trajectory dimensions.")
    if np.any(fov_m <= 0) or np.any(matrix_size <= 0):
        raise ValueError("fov_m and matrix_size elements must be positive.")

    k_traj_to_grid = kspace_trajectory_array.copy()
    if kspace_center_offset_m_inv is not None:
        kspace_center_offset_m_inv = np.asarray(kspace_center_offset_m_inv)
        if len(kspace_center_offset_m_inv) != num_dims:
            raise ValueError("kspace_center_offset_m_inv must match trajectory dimensions.")
        k_traj_to_grid += kspace_center_offset_m_inv

    # dk is the k-space step per pixel/voxel for the specified FOV and matrix size (units: m^-1)
    dk_per_dim = 1.0 / fov_m

    # Normalize k-space coordinates by dk. k_normalized represents k-coords in units of "grid cells from center".
    # A k-space point at +/- (matrix_size / 2) * dk would be at the edge of k-space.
    # k_normalized = k_traj_to_grid / dk_per_dim # This is k_traj * fov

    # Map to grid indices:
    # index = (k_physical / k_max_physical_of_grid) * (matrix_size/2) + matrix_size/2
    # k_max_physical_of_grid for a dimension = (matrix_size_dim / 2) * dk_dim
    # So, index = (k_physical / ( (matrix_size_dim/2) * dk_dim) ) * (matrix_size_dim/2) + matrix_size_dim/2
    # index = (k_physical / dk_dim) + matrix_size_dim/2
    # index = k_physical * fov_dim + matrix_size_dim/2

    k_grid_indices = np.zeros_like(k_traj_to_grid, dtype=float)
    for dim in range(num_dims):
        k_grid_indices[:, dim] = k_traj_to_grid[:, dim] * fov_m[dim] + matrix_size[dim] / 2.0

    k_grid_indices = np.round(k_grid_indices).astype(int)

    k_binary_grid = np.zeros(matrix_size, dtype=bool)

    for indices_per_point in k_grid_indices:
        # Clip indices to be within grid boundaries
        valid_point = True
        clipped_indices = []
        for dim_idx in range(num_dims):
            idx = indices_per_point[dim_idx]
            if not (0 <= idx < matrix_size[dim_idx]):
                valid_point = False
                break
            clipped_indices.append(idx)

        if valid_point:
            k_binary_grid[tuple(clipped_indices)] = True

    num_cells_hit = np.sum(k_binary_grid)
    total_cells = np.prod(matrix_size)
    coverage_percentage = (num_cells_hit / total_cells) * 100.0 if total_cells > 0 else 0.0

    return {
        'coverage_percentage': coverage_percentage,
        'num_cells_hit': num_cells_hit,
        'total_grid_cells': total_cells,
        'k_space_grid_binary': k_binary_grid
    }


def calculate_kspace_density_map_simple(kspace_trajectory_array, fov_m, matrix_size_density_map, kspace_center_offset_m_inv=None):
    """
    Calculates a simple k-space density map by counting hits per grid cell.

    Args:
        kspace_trajectory_array (np.ndarray): (N_points, N_dims), units: m^-1.
        fov_m (tuple/list/np.ndarray): FOV for each dimension (m) for density map scaling.
        matrix_size_density_map (tuple/list/np.ndarray): Matrix size for the density map.
        kspace_center_offset_m_inv (tuple/list/np.ndarray, optional): Offset of k-space center (m^-1).

    Returns:
        dict: {'density_map': np.ndarray (int), 'min_density': int, 'max_density': int,
               'mean_density_in_hit_cells': float, 'std_density_in_hit_cells': float}
    """
    if not isinstance(kspace_trajectory_array, np.ndarray) or kspace_trajectory_array.ndim != 2:
        raise ValueError("kspace_trajectory_array must be a 2D NumPy array.")

    num_dims = kspace_trajectory_array.shape[1]
    fov_m = np.asarray(fov_m)
    matrix_size_density_map = np.asarray(matrix_size_density_map, dtype=int)

    if len(fov_m) != num_dims or len(matrix_size_density_map) != num_dims:
        raise ValueError("fov_m and matrix_size_density_map must match trajectory dimensions.")

    k_traj_to_grid = kspace_trajectory_array.copy()
    if kspace_center_offset_m_inv is not None:
        k_traj_to_grid += np.asarray(kspace_center_offset_m_inv)

    k_grid_indices = np.zeros_like(k_traj_to_grid, dtype=float)
    for dim in range(num_dims):
        k_grid_indices[:, dim] = k_traj_to_grid[:, dim] * fov_m[dim] + matrix_size_density_map[dim] / 2.0
    k_grid_indices = np.round(k_grid_indices).astype(int)

    k_density_grid = np.zeros(matrix_size_density_map, dtype=int)

    for indices_per_point in k_grid_indices:
        valid_point = True
        clipped_indices = []
        for dim_idx in range(num_dims):
            idx = indices_per_point[dim_idx]
            if not (0 <= idx < matrix_size_density_map[dim_idx]):
                valid_point = False; break
            clipped_indices.append(idx)

        if valid_point:
            k_density_grid[tuple(clipped_indices)] += 1

    # Analyze density
    if k_density_grid.size > 0:
        min_density = np.min(k_density_grid)
        max_density = np.max(k_density_grid)
        hit_cells = k_density_grid[k_density_grid > 0]
        if hit_cells.size > 0:
            mean_density_hit_cells = np.mean(hit_cells)
            std_density_hit_cells = np.std(hit_cells)
        else: # No cells were hit
            mean_density_hit_cells = 0.0
            std_density_hit_cells = 0.0
    else: # Empty grid (e.g. matrix_size was 0 in a dimension)
        min_density = max_density = mean_density_hit_cells = std_density_hit_cells = 0

    return {
        'density_map': k_density_grid,
        'min_density_overall': min_density,
        'max_density_overall': max_density,
        'mean_density_in_hit_cells': mean_density_hit_cells,
        'std_density_in_hit_cells': std_density_hit_cells,
        'num_hit_cells': len(hit_cells) if 'hit_cells' in locals() else 0
    }


def analyze_kspace_point_distribution_basic(kspace_trajectory_array, num_bins=50):
    """
    Calculates basic distribution metrics: histogram of radii and consecutive point distances.

    Args:
        kspace_trajectory_array (np.ndarray): (N_points, N_dims), units: m^-1.
        num_bins (int): Number of bins for histograms.

    Returns:
        dict: {'radii_stats': {...}, 'consecutive_point_distance_stats': {...}}
    """
    if not isinstance(kspace_trajectory_array, np.ndarray) or kspace_trajectory_array.ndim != 2:
        raise ValueError("kspace_trajectory_array must be a 2D NumPy array.")
    if kspace_trajectory_array.shape[0] < 2: # Need at least 2 points for diffs
        return {
            'radii_stats': {'mean':np.nan, 'std':np.nan, 'min':np.nan, 'max':np.nan, 'histogram': (np.array([]), np.array([]))},
            'consecutive_point_distance_stats': {'mean':np.nan, 'std':np.nan, 'min':np.nan, 'max':np.nan, 'histogram': (np.array([]), np.array([]))}
        }

    num_dims = kspace_trajectory_array.shape[1]
    radii_stats = {}

    if num_dims > 1:
        radii = np.sqrt(np.sum(kspace_trajectory_array**2, axis=1))
    else: # 1D case, radii are just absolute k-values
        radii = np.abs(kspace_trajectory_array[:,0])

    hist_radii, bin_edges_radii = np.histogram(radii, bins=num_bins)
    radii_stats = {
        'mean': np.mean(radii), 'std': np.std(radii),
        'min': np.min(radii), 'max': np.max(radii),
        'histogram_counts': hist_radii.tolist(),
        'histogram_bin_edges': bin_edges_radii.tolist()
    }

    diffs = np.linalg.norm(np.diff(kspace_trajectory_array, axis=0), axis=1)
    hist_diffs, bin_edges_diffs = np.histogram(diffs, bins=num_bins)
    consecutive_dist_stats = {
        'mean': np.mean(diffs), 'std': np.std(diffs),
        'min': np.min(diffs), 'max': np.max(diffs),
        'histogram_counts': hist_diffs.tolist(),
        'histogram_bin_edges': bin_edges_diffs.tolist()
    }

    return {
        'radii_stats': radii_stats,
        'consecutive_point_distance_stats': consecutive_dist_stats
    }


if __name__ == '__main__':
    print("--- Running girf.kspace_validation.py example simulations ---")

    # --- Sample 2D Spiral Trajectory ---
    num_points_2d = 1000
    k_max_2d = 250 # m^-1
    revolutions_2d = 15
    theta = np.linspace(0, revolutions_2d * 2 * np.pi, num_points_2d)
    radius = np.linspace(0, k_max_2d, num_points_2d)
    kx_2d = radius * np.cos(theta)
    ky_2d = radius * np.sin(theta)
    traj_2d = np.stack([kx_2d, ky_2d], axis=-1)

    print("\n--- 2D Spiral Trajectory Analysis ---")
    extents_2d = get_kspace_extents(traj_2d)
    print(f"  Extents: Min={extents_2d['k_min_per_axis']}, Max={extents_2d['k_max_per_axis']}, Range={extents_2d['k_range_per_axis']}")

    fov_2d = (0.2, 0.2) # 20cm FOV
    matrix_2d = (64, 64)
    coverage_2d = check_kspace_coverage_binary_grid(traj_2d, fov_2d, matrix_2d)
    print(f"  Coverage: {coverage_2d['coverage_percentage']:.2f}% ({coverage_2d['num_cells_hit']}/{coverage_2d['total_grid_cells']} cells)")
    # Optional: plt.imshow(coverage_2d['k_space_grid_binary'].T, origin='lower'); plt.title("2D K-space Coverage"); plt.show()

    density_map_res_2d = (32, 32) # Lower resolution for density map visualization
    density_2d = calculate_kspace_density_map_simple(traj_2d, fov_2d, density_map_res_2d)
    print(f"  Density Map (res {density_map_res_2d}): Min={density_2d['min_density_overall']}, Max={density_2d['max_density_overall']}, "
          f"Mean(hit)={density_2d['mean_density_in_hit_cells']:.2f}, Std(hit)={density_2d['std_density_in_hit_cells']:.2f}")
    # Optional: plt.imshow(density_2d['density_map'].T, origin='lower', cmap='hot'); plt.title("2D Density Map"); plt.colorbar(); plt.show()

    distribution_2d = analyze_kspace_point_distribution_basic(traj_2d)
    print(f"  Radii Stats (2D): Mean={distribution_2d['radii_stats']['mean']:.2f}, Std={distribution_2d['radii_stats']['std']:.2f}")
    print(f"  Consecutive Dist Stats (2D): Mean={distribution_2d['consecutive_point_distance_stats']['mean']:.2f}, Std={distribution_2d['consecutive_point_distance_stats']['std']:.2f}")

    # --- Sample 3D Radial Trajectory (Stack-of-Stars like projection) ---
    num_points_per_spoke_3d = 64
    num_spokes_per_plane_3d = 32
    num_planes_3d = 16 # Along kz
    k_max_xy_3d = 200 # m^-1
    k_max_z_3d = 100 # m^-1

    all_spokes_3d = []
    kz_planes = np.linspace(-k_max_z_3d, k_max_z_3d, num_planes_3d)
    k_radial_profile = np.linspace(0, k_max_xy_3d, num_points_per_spoke_3d)

    for kz_val in kz_planes:
        for i in range(num_spokes_per_plane_3d):
            angle = i * np.pi / num_spokes_per_plane_3d
            kx_spoke = k_radial_profile * np.cos(angle)
            ky_spoke = k_radial_profile * np.sin(angle)
            kz_spoke = np.full_like(kx_spoke, kz_val)
            all_spokes_3d.append(np.stack([kx_spoke, ky_spoke, kz_spoke], axis=-1))
    traj_3d = np.concatenate(all_spokes_3d, axis=0)
    print(f"\n--- 3D Radial Trajectory Analysis (Shape: {traj_3d.shape}) ---")

    extents_3d = get_kspace_extents(traj_3d)
    print(f"  Extents: Min={np.round(extents_3d['k_min_per_axis'],1)}, Max={np.round(extents_3d['k_max_per_axis'],1)}")

    fov_3d = (0.25, 0.25, 0.25)
    matrix_3d = (32, 32, 32) # Coarse grid for faster example
    coverage_3d = check_kspace_coverage_binary_grid(traj_3d, fov_3d, matrix_3d)
    print(f"  Coverage (3D): {coverage_3d['coverage_percentage']:.2f}% ({coverage_3d['num_cells_hit']}/{coverage_3d['total_grid_cells']} cells)")

    density_map_res_3d = (16, 16, 16)
    density_3d = calculate_kspace_density_map_simple(traj_3d, fov_3d, density_map_res_3d)
    print(f"  Density Map (3D, res {density_map_res_3d}): Min={density_3d['min_density_overall']}, Max={density_3d['max_density_overall']}, "
          f"Mean(hit)={density_3d['mean_density_in_hit_cells']:.2f}")

    distribution_3d = analyze_kspace_point_distribution_basic(traj_3d)
    print(f"  Radii Stats (3D): Mean={distribution_3d['radii_stats']['mean']:.2f}")
    print(f"  Consecutive Dist Stats (3D): Mean={distribution_3d['consecutive_point_distance_stats']['mean']:.2f}")

    print("\n--- girf.kspace_validation.py example simulations finished ---")
