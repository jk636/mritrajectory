import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
from scipy.spatial.qhull import QhullError
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


# Gyromagnetic ratios for common nuclei in Hz/T
# Source: Mostly based on standard values, e.g., NIST, textbook tables.
COMMON_NUCLEI_GAMMA_HZ_PER_T = {
    '1H': 42.576e6,     # Proton
    '13C': 10.705e6,    # Carbon-13
    '31P': 17.235e6,    # Phosphorus-31
    '19F': 40.052e6,    # Fluorine-19
    '23Na': 11.262e6,   # Sodium-23
    '129Xe': 11.777e6,  # Xenon-129 (hyperpolarized gas MRI)
    '2H': 6.536e6,      # Deuterium
    '7Li': 16.546e6,     # Lithium-7
}


class Trajectory:
    """
    Container for a k-space trajectory and associated data.
    Includes additional trajectory metrics like max PNS, max slew, FOV, and resolution.
    """
    def __init__(self, name, kspace_points_rad_per_m, 
                 gradient_waveforms_Tm=None, dt_seconds=None, 
                 metadata=None, gamma_Hz_per_T=COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds=0.0, dead_time_end_seconds=0.0):
        """
        Args:
            name (str): Trajectory name/description.
            kspace_points_rad_per_m (np.ndarray): [D, N] or [N, D] k-space coordinates in rad/m.
            gradient_waveforms_Tm (np.ndarray, optional): [D, N] or [N, D] gradient waveforms in T/m.
                                                         If None, will be computed on demand.
            dt_seconds (float, optional): Dwell/sample time in seconds. Required if gradients are to be computed.
            metadata (dict, optional): Additional information.
            gamma_Hz_per_T (float, optional): Gyromagnetic ratio. Defaults to 1H.
            dead_time_start_seconds (float, optional): Dead time at the beginning of the trajectory. Defaults to 0.0.
            dead_time_end_seconds (float, optional): Dead time at the end of the trajectory. Defaults to 0.0.
        """
        self.name = name
        self.kspace_points_rad_per_m = np.array(kspace_points_rad_per_m)
        self.gradient_waveforms_Tm = np.array(gradient_waveforms_Tm) if gradient_waveforms_Tm is not None else None
        self.dt_seconds = dt_seconds
        self.metadata = metadata or {}
        
        self.dead_time_start_seconds = dead_time_start_seconds
        self.dead_time_end_seconds = dead_time_end_seconds

        if 'gamma_Hz_per_T' not in self.metadata:
             self.metadata['gamma_Hz_per_T'] = gamma_Hz_per_T
        
        self._update_dead_time_metadata() 
        self._compute_metrics() 

    def _update_dead_time_metadata(self):
        self.metadata['dead_time_start_seconds'] = self.dead_time_start_seconds
        self.metadata['dead_time_end_seconds'] = self.dead_time_end_seconds
        if self.dt_seconds is not None and self.dt_seconds > 0:
            self.dead_time_start_samples = self.dead_time_start_seconds / self.dt_seconds
            self.dead_time_end_samples = self.dead_time_end_seconds / self.dt_seconds
            self.metadata['dead_time_start_samples'] = self.dead_time_start_samples
            self.metadata['dead_time_end_samples'] = self.dead_time_end_samples
        else:
            self.dead_time_start_samples = None
            self.dead_time_end_samples = None
            self.metadata['dead_time_start_samples'] = None
            self.metadata['dead_time_end_samples'] = None

    def get_gradient_waveforms_Tm(self) -> Optional[np.ndarray]:
        if self.gradient_waveforms_Tm is not None:
            return self.gradient_waveforms_Tm
        if self.dt_seconds is None or self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            return None
        k_data = np.array(self.kspace_points_rad_per_m)
        D = self.get_num_dimensions()
        N = self.get_num_points()
        k_for_gradient = k_data
        if k_data.ndim == 2 and k_data.shape[0] == N and k_data.shape[1] == D: 
            if N != 0 and D != 0 : k_for_gradient = k_data.T
        elif k_data.ndim == 1 and N == k_data.shape[0] and D == 1: 
            k_for_gradient = k_data.reshape(1, N)
        elif k_data.ndim == 2 and k_data.shape[0] == D and k_data.shape[1] == N: 
            pass 
        elif k_data.ndim == 1 and D == k_data.shape[0] and N == 1: 
             k_for_gradient = k_data.reshape(D, 1)
        else:
            if N==0 or D==0: 
                self.gradient_waveforms_Tm = np.array([]).reshape(D,N) 
                return self.gradient_waveforms_Tm
            return None
        gamma = self.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
        if gamma == 0: gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
        computed_gradients = None
        if k_for_gradient.shape[-1] < 2: 
            computed_gradients = np.zeros_like(k_for_gradient)
        else:
            try:
                computed_gradients = np.gradient(k_for_gradient, self.dt_seconds, axis=-1) / gamma
            except ValueError: 
                if hasattr(k_for_gradient, 'shape'):
                    computed_gradients = np.zeros_like(k_for_gradient)
                else:
                    return None 
        self.gradient_waveforms_Tm = computed_gradients
        return self.gradient_waveforms_Tm

    def _calculate_slew_rate(self):
        gradients = self.get_gradient_waveforms_Tm()
        if gradients is not None and gradients.size > 0 and self.dt_seconds is not None and gradients.shape[-1] > 1:
            slew = np.diff(gradients, axis=-1) / self.dt_seconds
            max_slew_rate_Tm_per_s = np.max(np.linalg.norm(slew, axis=0))
            self.metadata['max_slew_rate_Tm_per_s'] = max_slew_rate_Tm_per_s
        else:
            self.metadata['max_slew_rate_Tm_per_s'] = None if gradients is None else 0.0

    def _calculate_pns(self):
        gradients = self.get_gradient_waveforms_Tm()
        if gradients is not None and gradients.size > 0 and self.dt_seconds is not None:
            abs_grad_sum = np.sum(np.abs(gradients), axis=0)
            self.metadata['pns_max_abs_gradient_sum_xyz'] = np.max(abs_grad_sum)
            if gradients.shape[-1] > 1:
                slew = np.diff(gradients, axis=-1) / self.dt_seconds
                abs_slew_sum = np.sum(np.abs(slew), axis=0)
                self.metadata['pns_max_abs_slew_sum_xyz'] = np.max(abs_slew_sum)
            else: 
                self.metadata['pns_max_abs_slew_sum_xyz'] = 0.0
        else:
            self.metadata['pns_max_abs_gradient_sum_xyz'] = None
            self.metadata['pns_max_abs_slew_sum_xyz'] = None

    def _calculate_fov(self):
        if self.kspace_points_rad_per_m is not None and self.kspace_points_rad_per_m.size > 0:
            points = self.kspace_points_rad_per_m
            if points.shape[0] > points.shape[1] and points.shape[0] > 3 : 
                 points = points.T
            k_extent_rad_per_m = np.max(np.abs(points), axis=-1)
            fov_m = 1 / (2 * k_extent_rad_per_m + 1e-9)
            self.metadata['fov_estimate_m'] = fov_m.tolist()
            self.metadata['fov_estimate_mm'] = (fov_m * 1e3).tolist()
        else:
            self.metadata['fov_estimate_m'] = None
            self.metadata['fov_estimate_mm'] = None

    def _calculate_resolution(self):
        if self.kspace_points_rad_per_m is not None and self.kspace_points_rad_per_m.size > 0:
            points = self.kspace_points_rad_per_m
            if points.shape[0] > points.shape[1] and points.shape[0] > 3 :
                 points = points.T
            max_k_radius_rad_per_m = np.max(np.linalg.norm(points, axis=0))
            if max_k_radius_rad_per_m > 1e-9: # Avoid division by zero for empty k-space
                resolution_m_overall = 1 / (2 * max_k_radius_rad_per_m + 1e-9)
                self.metadata['resolution_overall_estimate_m'] = resolution_m_overall
                self.metadata['resolution_overall_estimate_mm'] = resolution_m_overall * 1e3
            else:
                self.metadata['resolution_overall_estimate_m'] = None
                self.metadata['resolution_overall_estimate_mm'] = None
        else:
            self.metadata['resolution_overall_estimate_m'] = None
            self.metadata['resolution_overall_estimate_mm'] = None

    def _compute_metrics(self):
        self._calculate_slew_rate()
        self._calculate_pns()
        self._calculate_fov()
        self._calculate_resolution()

    def get_duration_seconds(self) -> Optional[float]:
        if self.dt_seconds is None: 
            return None
        num_points = self.get_num_points()
        sampling_duration = num_points * self.dt_seconds
        total_duration = self.dead_time_start_seconds + sampling_duration + self.dead_time_end_seconds
        return total_duration

    def get_max_grad_Tm(self) -> Optional[float]:
        gradients = self.get_gradient_waveforms_Tm()
        if gradients is not None and gradients.size > 0:
            return np.max(np.linalg.norm(gradients, axis=0))
        return None

    def get_max_slew_Tm_per_s(self) -> Optional[float]:
        if 'max_slew_rate_Tm_per_s' in self.metadata and self.metadata['max_slew_rate_Tm_per_s'] is not None:
            return self.metadata['max_slew_rate_Tm_per_s']
        gradients = self.get_gradient_waveforms_Tm()
        if gradients is not None and gradients.size > 0 and self.dt_seconds is not None and gradients.shape[-1] > 1:
            slew = np.diff(gradients, axis=-1) / self.dt_seconds
            return np.max(np.linalg.norm(slew, axis=0))
        return None

    def get_num_points(self) -> int:
        if self.kspace_points_rad_per_m is None: return 0
        if self.kspace_points_rad_per_m.shape[0] <= 3 or self.kspace_points_rad_per_m.shape[0] < self.kspace_points_rad_per_m.shape[1]:
            return self.kspace_points_rad_per_m.shape[1]
        return self.kspace_points_rad_per_m.shape[0]

    def get_num_dimensions(self) -> int:
        if self.kspace_points_rad_per_m is None: return 0
        if self.kspace_points_rad_per_m.shape[0] <= 3 or self.kspace_points_rad_per_m.shape[0] < self.kspace_points_rad_per_m.shape[1]:
            return self.kspace_points_rad_per_m.shape[0]
        return self.kspace_points_rad_per_m.shape[1]

    def export(self, filename, filetype=None):
        if filetype is None:
            if filename.endswith('.csv'): filetype = 'csv'
            elif filename.endswith('.npy'): filetype = 'npy'
            elif filename.endswith('.npz'): filetype = 'npz'
            else: filetype = 'txt'
        
        points_to_export = np.array(self.kspace_points_rad_per_m)
        kspace_was_DN_and_transposed = False
        if points_to_export.ndim == 2 and \
           (points_to_export.shape[0] <= 3 or points_to_export.shape[0] < points_to_export.shape[1]) and \
           points_to_export.shape[0] == self.get_num_dimensions() and \
           points_to_export.shape[1] == self.get_num_points():
            points_original_ήταν_DN = True # Original k-space was D,N
            points_to_export = points_to_export.T 
            kspace_was_DN_and_transposed = True

        gradients_from_getter = self.get_gradient_waveforms_Tm()
        gradients_to_export = gradients_from_getter
        if gradients_from_getter is not None and gradients_from_getter.ndim == 2:
            if kspace_was_DN_and_transposed: # k-space was D,N and transposed to N,D
                gradients_to_export = gradients_from_getter.T # Gradients are D,N, so transpose to N,D
            elif points_to_export.ndim == 2 and \
                 points_to_export.shape[0] == self.get_num_points() and \
                 points_to_export.shape[1] == self.get_num_dimensions(): # k-space was N,D
                 if gradients_from_getter.shape[0] == self.get_num_dimensions() and \
                    gradients_from_getter.shape[1] == self.get_num_points(): # Gradients are D,N
                    gradients_to_export = gradients_from_getter.T

        if points_to_export.ndim == 1 and gradients_to_export is not None and gradients_to_export.ndim == 2:
            if gradients_to_export.shape[0] == 1 : 
                gradients_to_export = gradients_to_export.reshape(gradients_to_export.shape[1])
            elif gradients_to_export.shape[1] == 1: 
                gradients_to_export = gradients_to_export.reshape(gradients_to_export.shape[0])

        if filetype == 'csv':
            np.savetxt(filename, points_to_export, delimiter=',')
        elif filetype == 'npy':
            np.save(filename, points_to_export)
        elif filetype == 'npz':
            save_dict = {'kspace_points_rad_per_m': points_to_export,
                         'dt_seconds': self.dt_seconds,
                         'metadata': self.metadata}
            if gradients_to_export is not None:
                save_dict['gradient_waveforms_Tm'] = gradients_to_export
            np.savez(filename, **save_dict)
        elif filetype == 'txt':
            np.savetxt(filename, points_to_export)
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")

    @classmethod
    def import_from(cls, filename):
        if filename.endswith('.csv') or filename.endswith('.txt'):
            points = np.loadtxt(filename, delimiter=',' if filename.endswith('.csv') else None)
            return cls(name=filename, kspace_points_rad_per_m=points)
        elif filename.endswith('.npy'):
            points = np.load(filename)
            return cls(name=filename, kspace_points_rad_per_m=points)
        elif filename.endswith('.npz'):
            data = np.load(filename, allow_pickle=True)
            points_key = 'kspace_points_rad_per_m' if 'kspace_points_rad_per_m' in data else 'points' if 'points' in data else 'kspace'
            gradients_key = 'gradient_waveforms_Tm' if 'gradient_waveforms_Tm' in data else 'gradients'
            dt_key = 'dt_seconds' if 'dt_seconds' in data else 'dt'
            points = data[points_key]
            gradients_data = data.get(gradients_key)
            gradients = np.array(gradients_data) if gradients_data is not None else None
            dt_data = data.get(dt_key)
            dt = dt_data.item() if dt_data is not None and hasattr(dt_data, 'item') else dt_data
            metadata_raw = data.get('metadata')
            metadata_dict = {}
            if metadata_raw is not None:
                try:
                    metadata_dict = metadata_raw.item() if hasattr(metadata_raw, 'item') and callable(metadata_raw.item) else dict(metadata_raw)
                except (TypeError, AttributeError, ValueError): 
                    if isinstance(metadata_raw, dict): metadata_dict = metadata_raw
                    else: metadata_dict = {'raw_metadata': metadata_raw} if metadata_raw is not None else {}
            gamma_from_file = metadata_dict.get('gamma_Hz_per_T')
            dts_s_from_file = metadata_dict.get('dead_time_start_seconds', 0.0)
            dte_s_from_file = metadata_dict.get('dead_time_end_seconds', 0.0)
            traj_instance = cls(name=filename, kspace_points_rad_per_m=points, 
                               gradient_waveforms_Tm=gradients, dt_seconds=dt, 
                               metadata=metadata_dict, 
                               gamma_Hz_per_T=gamma_from_file if gamma_from_file is not None else COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                               dead_time_start_seconds=dts_s_from_file,
                               dead_time_end_seconds=dte_s_from_file)
            return traj_instance
        else:
            raise ValueError(f"Unsupported filetype or extension for: {filename}")

    def calculate_voronoi_density(self, force_recompute=False, qhull_options=None):
        if not force_recompute and 'voronoi_cell_sizes' in self.metadata and \
           self.metadata.get('voronoi_calculation_status') == "Success":
            return self.metadata['voronoi_cell_sizes']
        for key in ['voronoi_cell_sizes', '_cached_voronoi_object', 
                    '_cached_voronoi_unique_points', '_cached_voronoi_unique_indices', 
                    '_cached_voronoi_cell_sizes_unique', 'voronoi_calculation_status']:
            self.metadata.pop(key, None)
        self.metadata['voronoi_calculation_status'] = "Not Attempted"
        if self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            self.metadata['voronoi_calculation_status'] = "Error: K-space data is empty or None."
            self.metadata['voronoi_cell_sizes'] = None
            return None

        k_data_original = np.array(self.kspace_points_rad_per_m)
        D = self.get_num_dimensions()
        N = self.get_num_points()

        # Ensure k_data is in (N, D) format
        points_ND = k_data_original
        if k_data_original.ndim == 2 and k_data_original.shape[0] == D and k_data_original.shape[1] == N and D != N:
            points_ND = k_data_original.T  # Transpose if it's (D, N)
        elif k_data_original.ndim == 1 and D == 1:
            points_ND = k_data_original.reshape(-1, 1) # Reshape if it's 1D (N,)
        elif k_data_original.ndim == 1 and N == 1 and D > 1: # Single point (D,)
             points_ND = k_data_original.reshape(1, D)
        # Add other necessary shape assertions or conversions if any typical cases are missed.
        # For example, if input could be (D,) for a single point.
        
        if points_ND.shape[0] != N or points_ND.shape[1] != D:
             self.metadata['voronoi_calculation_status'] = f"Error: Ambiguous k-space data shape {k_data_original.shape} for (N,D) format. N={N}, D={D}. Processed shape: {points_ND.shape}"
             self.metadata['voronoi_cell_sizes'] = None
             return None
        
        if D not in [2, 3]: # compute_voronoi_density also checks this, but good to have early exit.
            self.metadata['voronoi_calculation_status'] = f"Error: Voronoi calculation only supported for 2D/3D (D={D})."
            self.metadata['voronoi_cell_sizes'] = None
            return None

        # Normalization: Scale points_ND to [-0.5, 0.5] for each dimension
        # This is important for compute_voronoi_density, especially with "periodic" boundaries.
        # The compute_voronoi_density itself has a check, but it's better practice to prepare data here.
        normalized_points = np.copy(points_ND)
        for dim_idx in range(D):
            min_val, max_val = np.min(points_ND[:, dim_idx]), np.max(points_ND[:, dim_idx])
            if (max_val - min_val) > 1e-9: # Avoid division by zero if all points are the same
                normalized_points[:, dim_idx] = (points_ND[:, dim_idx] - min_val) / (max_val - min_val) - 0.5
            else: # All points in this dimension are the same, center them at 0 if not already
                # Or, if min_val == max_val, then (points - min_val) is 0. So result is -0.5.
                # Better to set to 0.0 if they are all same.
                normalized_points[:, dim_idx] = 0.0


        try:
            # Call the new function. For previous behavior, "clipped" is more appropriate.
            # If "periodic" is desired as a new default for this class method, change boundary_type.
            # qhull_options from the method signature can be passed through.
            density_weights = compute_voronoi_density(
                trajectory=normalized_points, 
                boundary_type="clipped", # Or make this a parameter of calculate_voronoi_density
                qhull_options=qhull_options 
            )
            
            if density_weights is not None and density_weights.shape[0] == N:
                self.metadata['voronoi_cell_sizes'] = density_weights
                self.metadata['voronoi_calculation_status'] = "Success"
                # Note: The new function compute_voronoi_density does not directly return
                # the Voronoi object, unique points, etc., so those specific cache items
                # (_cached_voronoi_object, etc.) from the old implementation are not filled here.
                # If plotting or direct access to the Voronoi object is needed later via this class method,
                # compute_voronoi_density would need to be modified to return 'vor'
                # and then it could be stored in self.metadata['_cached_voronoi_object'].
                # For now, only cell sizes (density_weights) are stored.
                # The plot_voronoi method might need adjustment if it relied on these caches.
                # However, plot_voronoi already recalculates if cache is missing, so it might be okay.
                # Let's clear old specific cache keys if they exist, as they are not from this run.
                self.metadata.pop('_cached_voronoi_object', None)
                self.metadata.pop('_cached_voronoi_unique_points', None)
                self.metadata.pop('_cached_voronoi_cell_sizes_unique', None)
                self.metadata.pop('_cached_voronoi_unique_indices', None)

                return density_weights
            else:
                raise ValueError("compute_voronoi_density returned unexpected result.")

        except ValueError as e: # Catch errors from compute_voronoi_density (e.g., bad ndim, empty traj)
            self.metadata['voronoi_calculation_status'] = f"Error: Value error during Voronoi computation: {e}"
            self.metadata['voronoi_cell_sizes'] = None
            return None
        except QhullError as e: # Should be caught by compute_voronoi_density, but as a fallback
            self.metadata['voronoi_calculation_status'] = f"Error: QhullError during Voronoi: {e}"
            self.metadata['voronoi_cell_sizes'] = None
            return None
        except Exception as e: 
            self.metadata['voronoi_calculation_status'] = f"Error: Unexpected error during Voronoi: {e}"
            self.metadata['voronoi_cell_sizes'] = None
            return None

    def plot_3d(self, max_total_points=2000, max_interleaves=None, 
                  interleaf_stride=1, point_stride=1, 
                  title=None, ax=None, figure=None, plot_style='.-'):
        if self.get_num_dimensions() < 3:
            print(f"Trajectory '{self.name}' is not 3D. Use a 2D plot method or ensure data is 3D.")
            if ax is not None: return ax 
            return None
        if self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            print(f"Trajectory '{self.name}' has no k-space points to plot.")
            if ax is not None: return ax
            return None
        if ax is None:
            if figure is None: fig = plt.figure(figsize=(10, 8))
            else: fig = figure
            ax = fig.add_subplot(111, projection='3d')
        elif not isinstance(ax, Axes3D):
            print("Warning: Provided 'ax' is not a 3D projection. Recreating as 3D.")
            fig = ax.get_figure(); fig.clf(); ax = fig.add_subplot(111, projection='3d')
        k_data = np.array(self.kspace_points_rad_per_m) 
        D_actual, n_total_points = k_data.shape
        if D_actual < 3: return ax # Should have been caught by get_num_dimensions earlier
        
        n_interleaves_known, n_points_per_interleaf_known = None, None
        if 'interleaf_structure' in self.metadata and isinstance(self.metadata['interleaf_structure'], tuple) and len(self.metadata['interleaf_structure']) == 2:
            n_il_temp, n_pts_temp = self.metadata['interleaf_structure']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp
        
        if n_interleaves_known is None and 'generator_params' in self.metadata and self.metadata['generator_params'] is not None:
            gp = self.metadata['generator_params']
            # Try to get num_arms or num_spokes or similar from generator_params
            n_il_gen_keys = ['num_arms', 'n_interleaves', 'num_spokes', 'n_shots']
            n_il_gen = None
            for key in n_il_gen_keys:
                if key in gp and isinstance(gp[key], int) and gp[key] > 0:
                    n_il_gen = gp[key]
                    break
            if gp.get('traj_type') == 'stackofspirals' and gp.get('n_stacks') is not None and n_il_gen is not None: 
                 n_il_gen *= gp.get('n_stacks')

            if n_il_gen and n_total_points > 0 and n_total_points % n_il_gen == 0:
                n_interleaves_known, n_points_per_interleaf_known = n_il_gen, n_total_points // n_il_gen
        
        if n_interleaves_known is None and 'kx_shape' in self.metadata and isinstance(self.metadata['kx_shape'], tuple) and len(self.metadata['kx_shape']) == 2:
            n_il_temp, n_pts_temp = self.metadata['kx_shape']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp
        
        plot_segments = []
        if n_interleaves_known and n_points_per_interleaf_known:
            try:
                reshaped_k = k_data.reshape((D_actual, n_interleaves_known, n_points_per_interleaf_known))
                il_stride = max(1, interleaf_stride)
                il_indices = np.arange(0, n_interleaves_known, il_stride)
                if max_interleaves is not None and max_interleaves > 0: il_indices = il_indices[:max_interleaves]
                if len(il_indices) == 0:
                    ax.set_title(title if title else f"3D K-space: {self.name} (No data selected)"); return ax
                
                sub_k_il = reshaped_k[:, il_indices, :]
                pt_stride = max(1, point_stride)
                sub_k_il_pts = sub_k_il[:, :, ::pt_stride]
                
                current_plot_pts = sub_k_il_pts.shape[1] * sub_k_il_pts.shape[2]
                if max_total_points is not None and current_plot_pts > max_total_points and sub_k_il_pts.shape[1] > 0:
                    pts_per_il_budget = max(1, max_total_points // sub_k_il_pts.shape[1])
                    if sub_k_il_pts.shape[2] > pts_per_il_budget:
                        add_stride = max(1, int(np.ceil(sub_k_il_pts.shape[2] / pts_per_il_budget)))
                        sub_k_il_pts = sub_k_il_pts[:, :, ::add_stride]
                for i in range(sub_k_il_pts.shape[1]): plot_segments.append(sub_k_il_pts[:, i, :])
            except ValueError: # If reshape fails, fallback to plotting as a single segment
                plot_segments = [k_data[:, ::max(1,point_stride)]]
        else: # No interleaf info, plot as single segment
            plot_segments = [k_data[:, ::max(1,point_stride)]]

        total_pts_shown = 0
        for seg_idx, seg in enumerate(plot_segments):
            seg_stride = 1
            if max_total_points is not None and (total_pts_shown + seg.shape[1]) > max_total_points:
                remaining_budget = max_total_points - total_pts_shown
                if remaining_budget <= 0 and len(plot_segments) > 1: break 
                seg_stride = max(1, int(np.ceil(seg.shape[1] / max(1,remaining_budget)))) if remaining_budget > 0 else seg.shape[1]
            
            final_seg = seg[:, ::seg_stride]
            if final_seg.shape[1] > 0:
                ax.plot(final_seg[0,:], final_seg[1,:], final_seg[2,:], plot_style, markersize=2, label=f"Interleaf {seg_idx}" if len(plot_segments)>1 else None)
                total_pts_shown += final_seg.shape[1]
            if max_total_points is not None and total_pts_shown >= max_total_points and len(plot_segments) > 1: break
        
        if total_pts_shown == 0 and n_total_points > 0: print("Warning: No points plotted. Adjust subsampling.")
        ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)"); ax.set_zlabel("Kz (rad/m)")
        ax.set_title(title if title else f"3D K-space: {self.name} ({total_pts_shown} pts shown)")
        return ax

    def plot_2d(self, max_total_points=10000, max_interleaves=None, 
                interleaf_stride=1, point_stride=1, 
                title=None, ax=None, figure=None, plot_style='.-', legend_on=False):
        """
        Plots the 2D k-space trajectory (Kx vs Ky).
        """
        if self.get_num_dimensions() < 2:
            print(f"Trajectory '{self.name}' is not at least 2D. Cannot produce 2D plot.")
            if ax is not None: return ax
            return None
        if self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            print(f"Trajectory '{self.name}' has no k-space points to plot.")
            if ax is not None: return ax
            return None

        if ax is None:
            if figure is None: fig = plt.figure(figsize=(8, 7))
            else: fig = figure
            ax = fig.add_subplot(111)
        
        # Ensure k_data is (D,N) for processing
        k_data_raw = np.array(self.kspace_points_rad_per_m)
        D_actual = self.get_num_dimensions()
        n_total_points = self.get_num_points()

        if k_data_raw.shape[0] == n_total_points and k_data_raw.shape[1] == D_actual and D_actual != n_total_points :
            k_data = k_data_raw.T # Convert N,D to D,N
        elif k_data_raw.shape[0] == D_actual and k_data_raw.shape[1] == n_total_points:
            k_data = k_data_raw # Already D,N
        elif k_data_raw.ndim == 1 and D_actual == 1 and k_data_raw.shape[0] == n_total_points: # (N,) for 1D
             print(f"Trajectory '{self.name}' is 1D. Cannot produce 2D plot of Kx vs Ky.")
             if ax is not None: return ax
             return None
        else: # Try to infer if D_actual is correct
            if k_data_raw.shape[0] == D_actual : k_data = k_data_raw
            elif k_data_raw.shape[1] == D_actual : k_data = k_data_raw.T
            else:
                print(f"Warning: Could not unambiguously reshape k-space data {k_data_raw.shape} to ({D_actual}, {n_total_points}). Plot may be incorrect.")
                k_data = k_data_raw # Proceed with caution


        n_interleaves_known, n_points_per_interleaf_known = None, None
        if 'interleaf_structure' in self.metadata and isinstance(self.metadata['interleaf_structure'], tuple) and len(self.metadata['interleaf_structure']) == 2:
            n_il_temp, n_pts_temp = self.metadata['interleaf_structure']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp
        
        if n_interleaves_known is None and 'generator_params' in self.metadata and self.metadata['generator_params'] is not None:
            gp = self.metadata['generator_params']
            n_il_gen_keys = ['num_arms', 'n_interleaves', 'num_spokes', 'n_shots']
            n_il_gen = None
            for key in n_il_gen_keys:
                if key in gp and isinstance(gp[key], int) and gp[key] > 0:
                    n_il_gen = gp[key]
                    break
            # Special handling for stackofspirals if it implies multiple 2D planes are overlaid
            # For now, assume 'n_stacks' for stackofspirals means interleaves are multiplied
            if gp.get('traj_type') == 'stackofspirals' and gp.get('n_stacks') is not None and n_il_gen is not None:
                 n_il_gen *= gp.get('n_stacks')

            if n_il_gen and n_total_points > 0 and n_total_points % n_il_gen == 0:
                n_interleaves_known, n_points_per_interleaf_known = n_il_gen, n_total_points // n_il_gen
        
        if n_interleaves_known is None and 'kx_shape' in self.metadata and isinstance(self.metadata['kx_shape'], tuple) and len(self.metadata['kx_shape']) == 2:
            n_il_temp, n_pts_temp = self.metadata['kx_shape']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp

        plot_segments = []
        labels = []
        if n_interleaves_known and n_points_per_interleaf_known and D_actual > 0:
            try:
                # Reshape to (Dimensions, Interleaves, Points_per_Interleaf)
                reshaped_k = k_data[:2, :].reshape((2, n_interleaves_known, n_points_per_interleaf_known)) # Take only first 2 dims for plotting
                
                il_stride = max(1, interleaf_stride)
                il_indices_to_plot = np.arange(0, n_interleaves_known, il_stride)
                if max_interleaves is not None and max_interleaves > 0: 
                    il_indices_to_plot = il_indices_to_plot[:max_interleaves]

                if len(il_indices_to_plot) == 0:
                    ax.set_title(title if title else f"2D K-space: {self.name} (No data selected)"); return ax
                
                sub_k_il = reshaped_k[:, il_indices_to_plot, :]
                pt_stride = max(1, point_stride)
                sub_k_il_pts = sub_k_il[:, :, ::pt_stride]
                
                current_plot_pts = sub_k_il_pts.shape[1] * sub_k_il_pts.shape[2] # num_il_plotted * pts_per_il_plotted
                
                if max_total_points is not None and current_plot_pts > max_total_points and sub_k_il_pts.shape[1] > 0:
                    # Further reduce points per interleaf if total points still exceed max_total_points
                    pts_per_il_budget = max(1, max_total_points // sub_k_il_pts.shape[1])
                    if sub_k_il_pts.shape[2] > pts_per_il_budget:
                        additional_stride = max(1, int(np.ceil(sub_k_il_pts.shape[2] / pts_per_il_budget)))
                        sub_k_il_pts = sub_k_il_pts[:, :, ::additional_stride]

                for i_idx, original_il_idx in enumerate(il_indices_to_plot):
                    plot_segments.append(sub_k_il_pts[:, i_idx, :])
                    labels.append(f"Interleaf {original_il_idx}")

            except ValueError: # Fallback if reshape fails
                plot_segments = [k_data[:2, ::max(1,point_stride)]]
                labels = [None]
        else: # No interleaf info or 1D data (already handled, but for safety)
            plot_segments = [k_data[:2, ::max(1,point_stride)]] # Take first 2 dims
            labels = [None]

        total_pts_shown = 0
        for i, seg in enumerate(plot_segments):
            seg_stride = 1
            # This subsampling logic might be redundant if already handled by max_total_points logic above
            if max_total_points is not None and (total_pts_shown + seg.shape[1]) > max_total_points:
                remaining_budget = max_total_points - total_pts_shown
                if remaining_budget <= 0 and len(plot_segments) > 1: break 
                seg_stride = max(1, int(np.ceil(seg.shape[1] / max(1,remaining_budget)))) if remaining_budget > 0 else seg.shape[1]
            
            final_seg = seg[:, ::seg_stride]
            if final_seg.shape[1] > 0:
                ax.plot(final_seg[0,:], final_seg[1,:], plot_style, markersize=3, label=labels[i] if legend_on and labels[i] else None)
                total_pts_shown += final_seg.shape[1]
            if max_total_points is not None and total_pts_shown >= max_total_points and len(plot_segments) > 1: break
        
        if total_pts_shown == 0 and n_total_points > 0: print("Warning: No points plotted for 2D. Adjust subsampling.")
        
        ax.set_xlabel("Kx (rad/m)")
        ax.set_ylabel("Ky (rad/m)")
        ax.set_title(title if title else f"2D K-space: {self.name} ({total_pts_shown} pts shown)")
        ax.axis('equal') # Aspect ratio to be equal
        if legend_on and any(labels): ax.legend()
        
        return ax

    def plot_voronoi(self, title=None, show_points=True, show_vertices=False, 
                     color_by_area=True, cmap='viridis', ax=None, figure=None, 
                     line_colors='gray', line_width=1.0, **kwargs_voronoi_plot_2d):
        if self.get_num_dimensions() < 3:
            print(f"Trajectory '{self.name}' is not 3D. Use a 2D plot method or ensure data is 3D.")
            if ax is not None: return ax 
            return None
        if self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            print(f"Trajectory '{self.name}' has no k-space points to plot.")
            if ax is not None: return ax
            return None
        if ax is None:
            if figure is None: fig = plt.figure(figsize=(10, 8))
            else: fig = figure
            ax = fig.add_subplot(111, projection='3d')
        elif not isinstance(ax, Axes3D):
            print("Warning: Provided 'ax' is not a 3D projection. Recreating as 3D.")
            fig = ax.get_figure(); fig.clf(); ax = fig.add_subplot(111, projection='3d')
        k_data = np.array(self.kspace_points_rad_per_m) 
        D_actual, n_total_points = k_data.shape
        if D_actual < 3: return ax # Should have been caught by get_num_dimensions earlier
        
        n_interleaves_known, n_points_per_interleaf_known = None, None
        if 'interleaf_structure' in self.metadata and isinstance(self.metadata['interleaf_structure'], tuple) and len(self.metadata['interleaf_structure']) == 2:
            n_il_temp, n_pts_temp = self.metadata['interleaf_structure']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp
        
        if n_interleaves_known is None and 'generator_params' in self.metadata and self.metadata['generator_params'] is not None:
            gp = self.metadata['generator_params']
            # Try to get num_arms or num_spokes or similar from generator_params
            n_il_gen_keys = ['num_arms', 'n_interleaves', 'num_spokes', 'n_shots']
            n_il_gen = None
            for key in n_il_gen_keys:
                if key in gp and isinstance(gp[key], int) and gp[key] > 0:
                    n_il_gen = gp[key]
                    break
            if gp.get('traj_type') == 'stackofspirals' and gp.get('n_stacks') is not None and n_il_gen is not None: 
                 n_il_gen *= gp.get('n_stacks')

            if n_il_gen and n_total_points > 0 and n_total_points % n_il_gen == 0:
                n_interleaves_known, n_points_per_interleaf_known = n_il_gen, n_total_points // n_il_gen
        
        if n_interleaves_known is None and 'kx_shape' in self.metadata and isinstance(self.metadata['kx_shape'], tuple) and len(self.metadata['kx_shape']) == 2:
            n_il_temp, n_pts_temp = self.metadata['kx_shape']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp
        
        plot_segments = []
        if n_interleaves_known and n_points_per_interleaf_known:
            try:
                reshaped_k = k_data.reshape((D_actual, n_interleaves_known, n_points_per_interleaf_known))
                il_stride = max(1, interleaf_stride)
                il_indices = np.arange(0, n_interleaves_known, il_stride)
                if max_interleaves is not None and max_interleaves > 0: il_indices = il_indices[:max_interleaves]
                if len(il_indices) == 0:
                    ax.set_title(title if title else f"3D K-space: {self.name} (No data selected)"); return ax
                
                sub_k_il = reshaped_k[:, il_indices, :]
                pt_stride = max(1, point_stride)
                sub_k_il_pts = sub_k_il[:, :, ::pt_stride]
                
                current_plot_pts = sub_k_il_pts.shape[1] * sub_k_il_pts.shape[2]
                if max_total_points is not None and current_plot_pts > max_total_points and sub_k_il_pts.shape[1] > 0:
                    pts_per_il_budget = max(1, max_total_points // sub_k_il_pts.shape[1])
                    if sub_k_il_pts.shape[2] > pts_per_il_budget:
                        add_stride = max(1, int(np.ceil(sub_k_il_pts.shape[2] / pts_per_il_budget)))
                        sub_k_il_pts = sub_k_il_pts[:, :, ::add_stride]
                for i in range(sub_k_il_pts.shape[1]): plot_segments.append(sub_k_il_pts[:, i, :])
            except ValueError: # If reshape fails, fallback to plotting as a single segment
                plot_segments = [k_data[:, ::max(1,point_stride)]]
        else: # No interleaf info, plot as single segment
            plot_segments = [k_data[:, ::max(1,point_stride)]]

        total_pts_shown = 0
        for seg_idx, seg in enumerate(plot_segments):
            seg_stride = 1
            if max_total_points is not None and (total_pts_shown + seg.shape[1]) > max_total_points:
                remaining_budget = max_total_points - total_pts_shown
                if remaining_budget <= 0 and len(plot_segments) > 1: break 
                seg_stride = max(1, int(np.ceil(seg.shape[1] / max(1,remaining_budget)))) if remaining_budget > 0 else seg.shape[1]
            
            final_seg = seg[:, ::seg_stride]
            if final_seg.shape[1] > 0:
                ax.plot(final_seg[0,:], final_seg[1,:], final_seg[2,:], plot_style, markersize=2, label=f"Interleaf {seg_idx}" if len(plot_segments)>1 else None)
                total_pts_shown += final_seg.shape[1]
            if max_total_points is not None and total_pts_shown >= max_total_points and len(plot_segments) > 1: break
        
        if total_pts_shown == 0 and n_total_points > 0: print("Warning: No points plotted. Adjust subsampling.")
        ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)"); ax.set_zlabel("Kz (rad/m)")
        ax.set_title(title if title else f"3D K-space: {self.name} ({total_pts_shown} pts shown)")
        return ax

    def plot_2d(self, max_total_points=10000, max_interleaves=None, 
                interleaf_stride=1, point_stride=1, 
                title=None, ax=None, figure=None, plot_style='.-', legend_on=False):
        """
        Plots the 2D k-space trajectory (Kx vs Ky).
        """
        if self.get_num_dimensions() < 2:
            print(f"Trajectory '{self.name}' is not at least 2D. Cannot produce 2D plot.")
            if ax is not None: return ax
            return None
        if self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            print(f"Trajectory '{self.name}' has no k-space points to plot.")
            if ax is not None: return ax
            return None

        if ax is None:
            if figure is None: fig = plt.figure(figsize=(8, 7))
            else: fig = figure
            ax = fig.add_subplot(111)
        
        # Ensure k_data is (D,N) for processing
        k_data_raw = np.array(self.kspace_points_rad_per_m)
        D_actual = self.get_num_dimensions()
        n_total_points = self.get_num_points()

        if k_data_raw.shape[0] == n_total_points and k_data_raw.shape[1] == D_actual and D_actual != n_total_points :
            k_data = k_data_raw.T # Convert N,D to D,N
        elif k_data_raw.shape[0] == D_actual and k_data_raw.shape[1] == n_total_points:
            k_data = k_data_raw # Already D,N
        elif k_data_raw.ndim == 1 and D_actual == 1 and k_data_raw.shape[0] == n_total_points: # (N,) for 1D
             print(f"Trajectory '{self.name}' is 1D. Cannot produce 2D plot of Kx vs Ky.")
             if ax is not None: return ax
             return None
        else: # Try to infer if D_actual is correct
            if k_data_raw.shape[0] == D_actual : k_data = k_data_raw
            elif k_data_raw.shape[1] == D_actual : k_data = k_data_raw.T
            else:
                print(f"Warning: Could not unambiguously reshape k-space data {k_data_raw.shape} to ({D_actual}, {n_total_points}). Plot may be incorrect.")
                k_data = k_data_raw # Proceed with caution


        n_interleaves_known, n_points_per_interleaf_known = None, None
        if 'interleaf_structure' in self.metadata and isinstance(self.metadata['interleaf_structure'], tuple) and len(self.metadata['interleaf_structure']) == 2:
            n_il_temp, n_pts_temp = self.metadata['interleaf_structure']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp
        
        if n_interleaves_known is None and 'generator_params' in self.metadata and self.metadata['generator_params'] is not None:
            gp = self.metadata['generator_params']
            n_il_gen_keys = ['num_arms', 'n_interleaves', 'num_spokes', 'n_shots']
            n_il_gen = None
            for key in n_il_gen_keys:
                if key in gp and isinstance(gp[key], int) and gp[key] > 0:
                    n_il_gen = gp[key]
                    break
            # Special handling for stackofspirals if it implies multiple 2D planes are overlaid
            # For now, assume 'n_stacks' for stackofspirals means interleaves are multiplied
            if gp.get('traj_type') == 'stackofspirals' and gp.get('n_stacks') is not None and n_il_gen is not None:
                 n_il_gen *= gp.get('n_stacks')

            if n_il_gen and n_total_points > 0 and n_total_points % n_il_gen == 0:
                n_interleaves_known, n_points_per_interleaf_known = n_il_gen, n_total_points // n_il_gen
        
        if n_interleaves_known is None and 'kx_shape' in self.metadata and isinstance(self.metadata['kx_shape'], tuple) and len(self.metadata['kx_shape']) == 2:
            n_il_temp, n_pts_temp = self.metadata['kx_shape']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp

        plot_segments = []
        labels = []
        if n_interleaves_known and n_points_per_interleaf_known and D_actual > 0:
            try:
                # Reshape to (Dimensions, Interleaves, Points_per_Interleaf)
                reshaped_k = k_data[:2, :].reshape((2, n_interleaves_known, n_points_per_interleaf_known)) # Take only first 2 dims for plotting
                
                il_stride = max(1, interleaf_stride)
                il_indices_to_plot = np.arange(0, n_interleaves_known, il_stride)
                if max_interleaves is not None and max_interleaves > 0: 
                    il_indices_to_plot = il_indices_to_plot[:max_interleaves]

                if len(il_indices_to_plot) == 0:
                    ax.set_title(title if title else f"2D K-space: {self.name} (No data selected)"); return ax
                
                sub_k_il = reshaped_k[:, il_indices_to_plot, :]
                pt_stride = max(1, point_stride)
                sub_k_il_pts = sub_k_il[:, :, ::pt_stride]
                
                current_plot_pts = sub_k_il_pts.shape[1] * sub_k_il_pts.shape[2] # num_il_plotted * pts_per_il_plotted
                
                if max_total_points is not None and current_plot_pts > max_total_points and sub_k_il_pts.shape[1] > 0:
                    # Further reduce points per interleaf if total points still exceed max_total_points
                    pts_per_il_budget = max(1, max_total_points // sub_k_il_pts.shape[1])
                    if sub_k_il_pts.shape[2] > pts_per_il_budget:
                        additional_stride = max(1, int(np.ceil(sub_k_il_pts.shape[2] / pts_per_il_budget)))
                        sub_k_il_pts = sub_k_il_pts[:, :, ::additional_stride]

                for i_idx, original_il_idx in enumerate(il_indices_to_plot):
                    plot_segments.append(sub_k_il_pts[:, i_idx, :])
                    labels.append(f"Interleaf {original_il_idx}")

            except ValueError: # Fallback if reshape fails
                plot_segments = [k_data[:2, ::max(1,point_stride)]]
                labels = [None]
        else: # No interleaf info or 1D data (already handled, but for safety)
            plot_segments = [k_data[:2, ::max(1,point_stride)]] # Take first 2 dims
            labels = [None]

        total_pts_shown = 0
        for i, seg in enumerate(plot_segments):
            seg_stride = 1
            # This subsampling logic might be redundant if already handled by max_total_points logic above
            if max_total_points is not None and (total_pts_shown + seg.shape[1]) > max_total_points:
                remaining_budget = max_total_points - total_pts_shown
                if remaining_budget <= 0 and len(plot_segments) > 1: break 
                seg_stride = max(1, int(np.ceil(seg.shape[1] / max(1,remaining_budget)))) if remaining_budget > 0 else seg.shape[1]
            
            final_seg = seg[:, ::seg_stride]
            if final_seg.shape[1] > 0:
                ax.plot(final_seg[0,:], final_seg[1,:], plot_style, markersize=3, label=labels[i] if legend_on and labels[i] else None)
                total_pts_shown += final_seg.shape[1]
            if max_total_points is not None and total_pts_shown >= max_total_points and len(plot_segments) > 1: break
        
        if total_pts_shown == 0 and n_total_points > 0: print("Warning: No points plotted for 2D. Adjust subsampling.")
        
        ax.set_xlabel("Kx (rad/m)")
        ax.set_ylabel("Ky (rad/m)")
        ax.set_title(title if title else f"2D K-space: {self.name} ({total_pts_shown} pts shown)")
        ax.axis('equal') # Aspect ratio to be equal
        if legend_on and any(labels): ax.legend()
        
        return ax

    def plot_voronoi(self, title=None, show_points=True, show_vertices=False, 
                     color_by_area=True, cmap='viridis', ax=None, figure=None, 
                     line_colors='gray', line_width=1.0, **kwargs_voronoi_plot_2d):
        num_dims = self.get_num_dimensions()
        if num_dims == 3:
            print("3D Voronoi cell plotting is not yet implemented. Displaying 3D k-space points instead.")
            return self.plot_3d(title=title if title else f"K-space points for {self.name} (3D)", 
                                ax=ax, figure=figure, plot_style='.')
        if num_dims != 2:
            print(f"Voronoi plotting is currently optimized for 2D. Trajectory '{self.name}' is {num_dims}D.")
            if ax is not None: return ax
            return None
        _ = self.calculate_voronoi_density(force_recompute=False) 
        status = self.metadata.get('voronoi_calculation_status')
        vor_obj = self.metadata.get('_cached_voronoi_object')
        points_for_voronoi = self.metadata.get('_cached_voronoi_unique_points')
        cell_areas_for_unique_points = self.metadata.get('_cached_voronoi_cell_sizes_unique')
        if status != "Success" or vor_obj is None or points_for_voronoi is None or cell_areas_for_unique_points is None:
            print(f"Voronoi data not available or calculation failed for {self.name}. Status: {status}")
            if ax is not None: return ax
            return None
        if ax is None:
            if figure is None: fig = plt.figure(figsize=(8, 7))
            else: fig = figure
            ax = fig.add_subplot(111)
        
        voronoi_plot_kwargs = {k: v for k, v in kwargs_voronoi_plot_2d.items() 
                               if k not in ['point_size', 'points_color', 'vertex_size', 'vertices_color']}
        voronoi_plot_2d(vor_obj, ax=ax, show_points=False, show_vertices=False, 
                        line_colors=line_colors, line_width=line_width, **voronoi_plot_kwargs)

        if color_by_area:
            finite_cell_areas = cell_areas_for_unique_points[np.isfinite(cell_areas_for_unique_points)]
            if len(finite_cell_areas) > 0:
                vmin = np.percentile(finite_cell_areas, 1) if len(finite_cell_areas) > 2 else np.min(finite_cell_areas)
                vmax = np.percentile(finite_cell_areas, 99) if len(finite_cell_areas) > 2 else np.max(finite_cell_areas)
                if vmin == vmax: vmax = vmin + 1e-9 
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                cmap_obj = plt.get_cmap(cmap)
                patches = []
                patch_colors_normalized = []
                for i, region_idx in enumerate(vor_obj.point_region):
                    region_vertices_indices = vor_obj.regions[region_idx]
                    if -1 not in region_vertices_indices and region_vertices_indices: 
                        polygon_vertices = vor_obj.vertices[region_vertices_indices]
                        patches.append(Polygon(polygon_vertices, closed=True))
                        area = cell_areas_for_unique_points[i] 
                        if np.isfinite(area): patch_colors_normalized.append(norm(area))
                        else: patch_colors_normalized.append(norm(vmin)) 
                if patches:
                    p_collection = PatchCollection(patches, cmap=cmap_obj, alpha=0.6) 
                    p_collection.set_array(np.array(patch_colors_normalized))
                    ax.add_collection(p_collection)
                    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([]) 
                    plt.colorbar(sm, ax=ax, label='Voronoi Cell Area', orientation='vertical', fraction=0.046, pad=0.04)
        if show_points:
            point_size = kwargs_voronoi_plot_2d.get('point_size', 5); points_color = kwargs_voronoi_plot_2d.get('points_color', 'black')
            ax.plot(points_for_voronoi[:,0], points_for_voronoi[:,1], 'o', markersize=point_size, color=points_color) 
        if show_vertices:
            vertex_size = kwargs_voronoi_plot_2d.get('vertex_size',3); vertices_color = kwargs_voronoi_plot_2d.get('vertices_color', 'red')
            if vor_obj.vertices.size > 0 : 
                ax.plot(vor_obj.vertices[:,0], vor_obj.vertices[:,1], 's', markersize=vertex_size, color=vertices_color)
        ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)")
        ax.set_title(title if title else f"Voronoi Diagram: {self.name}"); ax.axis('equal') 
        return ax

    def summary(self):
        """
        Print a detailed summary of the trajectory, including its properties and calculated metrics.
        """
        # ... (implementation of summary method) ...
        pass


def display_trajectory(trajectory_obj: Trajectory, 
                       plot_type: str = "2D", 
                       **kwargs) -> Optional[plt.Axes]:
    """
    Displays a k-space trajectory using either 2D or 3D plotting methods
    from the Trajectory class.

    Args:
        trajectory_obj (Trajectory): A Trajectory object.
        plot_type (str, optional): Type of plot, "2D" or "3D". Defaults to "2D".
        **kwargs: Additional keyword arguments to pass to the underlying
                  plot methods (e.g., title, ax, figure, max_total_points).

    Returns:
        Optional[plt.Axes]: The Matplotlib axis object returned by the plotting method,
                            or None if plotting failed or method doesn't return an axis.
    
    Raises:
        TypeError: If trajectory_obj is not an instance of the Trajectory class.
        ValueError: If plot_type is not "2D" or "3D".
    """
    if not isinstance(trajectory_obj, Trajectory):
        raise TypeError(f"trajectory_obj must be an instance of Trajectory, got {type(trajectory_obj)}.")
    
    if plot_type not in ["2D", "3D"]:
        raise ValueError(f"plot_type must be '2D' or '3D', got '{plot_type}'.")

    ax_returned = None
    if plot_type == "2D":
        # The plot_2d method was added in the previous step.
        ax_returned = trajectory_obj.plot_2d(**kwargs)
            
    elif plot_type == "3D":
        ax_returned = trajectory_obj.plot_3d(**kwargs)
    
    # The responsibility of plt.show() is typically left to the caller/user script,
    # especially when an axis can be passed in for embedding.
    # So, we just return the axis.
    
    return ax_returned
    """
    Displays a k-space trajectory using either 2D or 3D plotting methods
    from the Trajectory class.

    Args:
        trajectory_obj (Trajectory): A Trajectory object.
        plot_type (str, optional): Type of plot, "2D" or "3D". Defaults to "2D".
        **kwargs: Additional keyword arguments to pass to the underlying
                  plot methods (e.g., title, ax, figure, max_total_points).

    Returns:
        Optional[plt.Axes]: The Matplotlib axis object returned by the plotting method,
                            or None if plotting failed or method doesn't return an axis.
    
    Raises:
        TypeError: If trajectory_obj is not an instance of the Trajectory class.
        ValueError: If plot_type is not "2D" or "3D".
    """
    if not isinstance(trajectory_obj, Trajectory):
        raise TypeError(f"trajectory_obj must be an instance of Trajectory, got {type(trajectory_obj)}.")
    
    if plot_type not in ["2D", "3D"]:
        raise ValueError(f"plot_type must be '2D' or '3D', got '{plot_type}'.")

    ax_returned = None
    if plot_type == "2D":
        # Check if plot_2d method exists, which it should after this subtask
        if hasattr(trajectory_obj, 'plot_2d') and callable(trajectory_obj.plot_2d):
            ax_returned = trajectory_obj.plot_2d(**kwargs)
        else:
            # This case should ideally not be reached if plot_2d is implemented correctly.
            print(f"Error: Trajectory object {trajectory_obj.name} does not have a plot_2d method.")
            # Fallback or raise error? For now, print error and return None.
            return None 
            
    elif plot_type == "3D":
        ax_returned = trajectory_obj.plot_3d(**kwargs)
    
    # Ensure the plot is shown if no external axis was provided
    # This is a common behavior for display functions.
    # Only call plt.show() if 'ax' was not in kwargs, meaning we created the figure/ax.
    if 'ax' not in kwargs and ax_returned is not None:
        if hasattr(ax_returned, 'get_figure') and callable(ax_returned.get_figure):
             fig = ax_returned.get_figure()
             if fig is not None: # Ensure figure exists
                # Check if we are in an interactive environment (e.g. Jupyter)
                # This is a simple check, more robust checks might be needed.
                if not hasattr(plt, 'isinteractive') or not plt.isinteractive():
                    # Only call show if not interactive, or if specifically desired.
                    # For now, let's assume show is desired if ax is not passed.
                    # However, typical library functions return the axis and let user call show().
                    # Let's stick to returning the axis. The user can call plt.show().
                    pass # plt.show() # Potentially call plt.show() here.
    
    return ax_returned


def compute_density_compensation(trajectory: np.ndarray, 
                                 method: str = "voronoi", 
                                 existing_voronoi: Optional[Any] = None, 
                                 dt_seconds: Optional[float] = None, 
                                 gamma_Hz_per_T: Optional[float] = None) -> np.ndarray:
    """
    Computes density compensation factors for a given k-space trajectory.

    Args:
        trajectory (np.ndarray): K-space trajectory. 
                                 Can be complex (kx + i*ky, e.g., (num_arms, num_samples)) 
                                 or real (num_points, D).
        method (str, optional): Method for density calculation. 
                                Either "voronoi" or "pipe". Defaults to "voronoi".
        existing_voronoi (Optional[Any], optional): Pre-computed Voronoi object. 
                                                    Used if method is "voronoi". Defaults to None.
        dt_seconds (Optional[float], optional): Sampling dwell time. 
                                                May be needed for "pipe" method. Defaults to None.
        gamma_Hz_per_T (Optional[float], optional): Gyromagnetic ratio. 
                                                    May be needed for "pipe" method. Defaults to None.

    Returns:
        np.ndarray: Density compensation weights, reshaped to match the original 
                    trajectory's sample structure.
    """
    original_shape = trajectory.shape
    if np.iscomplexobj(trajectory):
        # Reshape to (num_total_points, 1) then convert to (num_total_points, 2)
        # Handles (A,S) or (S) complex inputs
        if trajectory.ndim == 1: # Single interleaf/shot
             trajectory_reshaped = trajectory.reshape(-1, 1)
        else: # Multiple interleaves/shots
             trajectory_reshaped = trajectory.flatten().reshape(-1,1)
        
        points_real = np.concatenate(
            (trajectory_reshaped.real, trajectory_reshaped.imag), axis=1
        )
        if points_real.shape[0] == 1 and points_real.shape[1] == 2 and original_shape == (1,): # Special case: single complex number
            pass # points_real is already (1,2)
        elif original_shape == points_real.shape: # e.g. input was (N,2) complex - unlikely but handle
             pass # original_shape is already N,2, points_real is N,2
        elif len(original_shape) == 2 and original_shape[0] * original_shape[1] == points_real.shape[0]: # (A,S) -> (A*S, 2)
            pass
        elif len(original_shape) == 1 and original_shape[0] == points_real.shape[0]: # (S) -> (S,2)
            pass

        # For output reshaping: if input was (A,S), we want output (A,S)
        # if input was (S), we want output (S)
        # points_real is currently (N_total, 2)
        # density_weights will be (N_total,)
        # So, final reshape should be original_shape
        output_reshape_target = original_shape

    elif trajectory.ndim == 1 and trajectory.shape[0] > 0: # Real 1D array (e.g. [k1, k2, k3...])
        points_real = trajectory.reshape(-1, 1) # Treat as (N, 1)
        output_reshape_target = original_shape # Output should be (N,)
    elif trajectory.ndim == 2: # Real (N, D)
        points_real = trajectory
        # Output should be (N,) if original was (N,D)
        output_reshape_target = (original_shape[0],) if len(original_shape) > 1 else original_shape
    else:
        raise ValueError(f"Unsupported trajectory shape: {original_shape}")

    num_points, num_dims = points_real.shape
    
    if num_points == 0:
        return np.array([]).reshape(output_reshape_target)

    if method == "voronoi":
        density_weights = np.zeros(num_points)
        
        # Use unique points for Voronoi calculation
        unique_pts, unique_indices = np.unique(points_real, axis=0, return_inverse=True)

        if unique_pts.shape[0] < num_dims + 1:
            # Not enough unique points for Voronoi in this dimension
            # Fallback to uniform weighting for these points
            # This case should ideally be handled by the caller or have a more sophisticated fallback
            density_weights.fill(1.0)
        else:
            vor = None
            if existing_voronoi is not None:
                # TODO: Add checks to ensure existing_voronoi is compatible 
                # (e.g., same points, same dimensionality)
                # For now, assume it's compatible if provided.
                vor = existing_voronoi
            else:
                try:
                    # Default qhull options from Trajectory.calculate_voronoi_density
                    qhull_options = 'Qbb Qc Qz' if num_dims > 1 else 'Qbb Qc Qx' # Qx for 1D
                    if num_dims == 1: # Voronoi needs at least 2D points, effectively
                                      # For 1D data, we can simulate this by adding a zero dimension
                        unique_pts_for_voronoi = np.hstack((unique_pts, np.zeros((unique_pts.shape[0], 1))))
                        vor = Voronoi(unique_pts_for_voronoi, qhull_options=qhull_options)
                    else:
                        vor = Voronoi(unique_pts, qhull_options=qhull_options)
                except QhullError as e:
                    # Fallback if Voronoi computation fails
                    density_weights.fill(1.0) 
                    # Consider logging a warning here: f"Voronoi computation failed: {e}. Falling back to uniform weights."
                except Exception as e: # Other potential errors
                    density_weights.fill(1.0)
                    # Consider logging a warning here: f"Error during Voronoi: {e}. Falling back to uniform weights."

            if vor is not None:
                cell_volumes_unique = np.zeros(unique_pts.shape[0])
                for i in range(unique_pts.shape[0]):
                    region_idx = vor.point_region[i]
                    vertex_indices = vor.regions[region_idx]

                    if -1 in vertex_indices or not vertex_indices:
                        cell_volumes_unique[i] = np.inf
                    else:
                        try:
                            if num_dims == 1:
                                # For 1D, "volume" is length. Vertices are (x,0).
                                # Find min/max x of the segment defined by vertices.
                                region_vertices = vor.vertices[vertex_indices, 0]
                                cell_volumes_unique[i] = np.max(region_vertices) - np.min(region_vertices)
                                if cell_volumes_unique[i] < 0: # Should not happen if vertices are sorted
                                    cell_volumes_unique[i] = np.inf # If malformed
                            elif num_dims == 2:
                                region_vertices = vor.vertices[vertex_indices]
                                if region_vertices.shape[0] < num_dims + 1: # Need at least D+1 points for hull
                                    cell_volumes_unique[i] = 0.0 # Or handle as error/inf
                                    continue
                                hull = ConvexHull(region_vertices, qhull_options='QJ') # QJ for joggling
                                cell_volumes_unique[i] = hull.volume # area for 2D
                            elif num_dims == 3:
                                region_vertices = vor.vertices[vertex_indices]
                                if region_vertices.shape[0] < num_dims + 1:
                                    cell_volumes_unique[i] = 0.0
                                    continue
                                hull = ConvexHull(region_vertices, qhull_options='QJ')
                                cell_volumes_unique[i] = hull.volume
                            else: # Should have been caught earlier or by Voronoi itself
                                cell_volumes_unique[i] = 1.0 # Fallback for >3D
                        except QhullError: # If ConvexHull fails for a region
                            cell_volumes_unique[i] = np.inf # Treat as infinite or very large
                        except Exception: # Other errors during hull computation
                            cell_volumes_unique[i] = np.inf
                
                # Handle infinite values
                finite_volumes = cell_volumes_unique[np.isfinite(cell_volumes_unique)]
                if len(finite_volumes) > 0:
                    median_finite_volume = np.median(finite_volumes)
                    if median_finite_volume == 0 and np.any(cell_volumes_unique == 0): # Avoid replacing inf with 0 if some are actual 0s
                         # If median is zero, but other finite non-zero volumes exist, use mean of those
                         non_zero_finite_volumes = finite_volumes[finite_volumes > 1e-9] # Use a small epsilon
                         if len(non_zero_finite_volumes) > 0:
                             median_finite_volume = np.median(non_zero_finite_volumes)
                         else: # All finite volumes are zero or very close to it
                             median_finite_volume = 1.0 # Fallback if all finite are zero
                    
                    cell_volumes_unique[np.isinf(cell_volumes_unique)] = median_finite_volume
                    # Also replace NaNs if any occurred, e.g. from QhullError in ConvexHull
                    cell_volumes_unique[np.isnan(cell_volumes_unique)] = median_finite_volume
                    if np.all(cell_volumes_unique == 0) or median_finite_volume == 0 : # If all cells ended up zero
                        cell_volumes_unique.fill(1.0) # Fallback to uniform for unique points

                elif np.any(np.isinf(cell_volumes_unique)): # All are infinite or mix of inf/nan
                    cell_volumes_unique.fill(1.0) # Fallback: if all regions were infinite or failed
                
                # Map back to original points
                density_weights = cell_volumes_unique[unique_indices]
                if np.all(density_weights == 0): # Final check if all weights are zero
                    density_weights.fill(1.0)

        # Ensure no zero weights if num_points > 0, as this will cause issues in normalization
        if num_points > 0 and np.all(density_weights == 0):
            density_weights.fill(1.0)
        elif num_points > 0 and np.sum(density_weights) < 1e-12: # If sum is extremely small
             density_weights.fill(1.0)


    elif method == "pipe":
        if num_dims != 2:
            raise ValueError(f"Pipe method is only supported for 2D trajectories. Got {num_dims}D.")
        
        # Density is proportional to the radius from k-space center.
        # points_real is (num_points, 2)
        density_weights = np.sqrt(points_real[:, 0]**2 + points_real[:, 1]**2)
        
        # Handle potential zero radius at k-space center if it causes issues downstream.
        # For instance, if weights are used as divisors.
        # A common approach is to set the weight at k=0 to a small value or average of neighbors.
        # For now, direct proportionality is implemented. If k=0, weight is 0.
        if density_weights.shape[0] != num_points: # Should not happen
            density_weights = np.ones(num_points) # Fallback

    else:
        raise ValueError(f"Unknown density compensation method: {method}")

    # Normalization using the new helper function
    density_weights = normalize_density_weights(density_weights)

    # Reshape to original structure
    # If original was (A,S) complex, points_real was (A*S, 2), density_weights is (A*S,). Reshape to (A,S).
    # If original was (S) complex, points_real was (S,2), density_weights is (S,). Reshape to (S).
    # If original was (N,D) real, points_real was (N,D), density_weights is (N,). Reshape to (N,).
    # If original was (N,) real (treated as (N,1)), points_real was (N,1), density_weights is (N,). Reshape to (N,).
    return density_weights.reshape(output_reshape_target)


def create_periodic_points(trajectory: np.ndarray, ndim: int) -> np.ndarray:
    """
    Creates replicated points for periodic boundary conditions.

    Args:
        trajectory (np.ndarray): K-space points, shape (num_points, ndim).
                                 Assumed to be normalized to [-0.5, 0.5].
        ndim (int): Number of dimensions (2 or 3).

    Returns:
        np.ndarray: Extended trajectory with replicated points,
                    shape (num_points * 3^ndim, ndim).
    """
    if not (2 <= ndim <= 3):
        raise ValueError("Number of dimensions must be 2 or 3.")

    num_points = trajectory.shape[0]
    if trajectory.shape[1] != ndim:
        raise ValueError(f"Trajectory shape {trajectory.shape} inconsistent with ndim {ndim}")

    offsets = [-1, 0, 1]
    
    if ndim == 2:
        extended_trajectory = np.zeros((num_points * 9, 2))
        idx = 0
        for point in trajectory:
            for i in offsets:
                for j in offsets:
                    extended_trajectory[idx, :] = point + np.array([i, j])
                    idx += 1
    elif ndim == 3:
        extended_trajectory = np.zeros((num_points * 27, 3))
        idx = 0
        for point in trajectory:
            for i in offsets:
                for j in offsets:
                    for k in offsets:
                        extended_trajectory[idx, :] = point + np.array([i, j, k])
                        idx += 1
    else: # Should be caught by the initial check
        raise ValueError("Number of dimensions must be 2 or 3.")
        
    return extended_trajectory


def compute_cell_area(voronoi: Voronoi, point_index: int, ndim: int, max_points_per_cell: int = 1000) -> float:
    """
    Computes the area/volume of a Voronoi cell.

    Args:
        voronoi (scipy.spatial.Voronoi): The Voronoi object.
        point_index (int): Index of the point in voronoi.points.
        ndim (int): Number of dimensions (2 or 3).
        max_points_per_cell (int): Max vertices for ConvexHull (for safety, though often not strictly needed).

    Returns:
        float: Area (2D) or Volume (3D) of the cell. Returns 0.0 for infinite or invalid cells.
    """
    if point_index < 0 or point_index >= len(voronoi.point_region):
        # This case should ideally not be reached if point_index is always valid
        return 0.0

    region_index = voronoi.point_region[point_index]
    if region_index == -1: # Should not happen with Qhull 'Qc' but good check
        return 0.0

    region_vertices_indices = voronoi.regions[region_index]

    if not region_vertices_indices or -1 in region_vertices_indices:
        return 0.0  # Infinite region

    cell_vertices = voronoi.vertices[region_vertices_indices]

    if cell_vertices.shape[0] < ndim + 1:
        # Not enough vertices to form a simplex in this dimension
        return 0.0
    
    # Optional: Truncate if too many vertices (max_points_per_cell)
    # This is more of a safeguard; ConvexHull itself can be slow with extremely many points,
    # but typical Voronoi cells from reasonably distributed points are fine.
    # if cell_vertices.shape[0] > max_points_per_cell:
    #     # This part is tricky - how to select which points? Randomly? First N?
    #     # For now, let ConvexHull handle it or raise error if it's an issue.
    #     # Consider logging a warning if cell_vertices.shape[0] is very large.
    #     pass

    try:
        # QJ (joggle) is often helpful for robustness in ConvexHull
        hull = ConvexHull(cell_vertices, qhull_options='QJ' if ndim > 1 else None)
        return hull.volume  # .volume gives area in 2D, volume in 3D
    except QhullError:
        return 0.0  # Collinear/coplanar points, or other Qhull issue
    except Exception: # Any other error during hull computation
        return 0.0


def compute_voronoi_density(trajectory: np.ndarray, 
                            boundary_type: str = "periodic", 
                            max_points_per_cell: int = 1000, 
                            qhull_options: Optional[str] = None) -> np.ndarray:
    """
    Computes Voronoi-based density compensation weights for a k-space trajectory.

    Args:
        trajectory (np.ndarray): K-space coordinates, shape (num_points, ndim).
                                 Expected to be normalized to [-0.5, 0.5] for 'periodic'.
        boundary_type (str, optional): "periodic" or "clipped". Defaults to "periodic".
        max_points_per_cell (int, optional): Max vertices for ConvexHull in compute_cell_area.
                                             Defaults to 1000.
        qhull_options (Optional[str], optional): Qhull options for Voronoi. 
                                                 Defaults to 'Qbb Qc Qz'.

    Returns:
        np.ndarray: Density compensation weights, shape (num_points,).
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("Trajectory must be a NumPy array.")
    if trajectory.ndim != 2:
        raise ValueError(f"Trajectory must be 2D (num_points, ndim), got shape {trajectory.shape}")

    num_points, ndim = trajectory.shape

    if ndim not in [2, 3]:
        raise ValueError(f"Number of dimensions (ndim={ndim}) must be 2 or 3.")
    if num_points == 0:
        return np.array([])

    # Step 1: Validate and Preprocess (Normalization)
    # Check if data is roughly in [-0.5, 0.5] range. If not, scale it.
    # This is a simple min-max scaling. More sophisticated normalization might be needed
    # depending on the expected input distribution.
    processed_trajectory = np.copy(trajectory)
    needs_normalization = False
    for dim_idx in range(ndim):
        min_val, max_val = np.min(processed_trajectory[:, dim_idx]), np.max(processed_trajectory[:, dim_idx])
        if min_val < -0.55 or max_val > 0.55 or (max_val - min_val) > 1.1: # Added margin for float precision
            needs_normalization = True
            break
    
    if needs_normalization:
        # print("Normalizing trajectory for compute_voronoi_density") # Optional: for debugging
        for dim_idx in range(ndim):
            min_val, max_val = np.min(processed_trajectory[:, dim_idx]), np.max(processed_trajectory[:, dim_idx])
            if (max_val - min_val) > 1e-9: # Avoid division by zero if all points are the same
                processed_trajectory[:, dim_idx] = (processed_trajectory[:, dim_idx] - min_val) / (max_val - min_val) - 0.5
            else: # All points in this dimension are the same, center them at 0 if not already
                processed_trajectory[:, dim_idx] = 0.0 - np.mean(processed_trajectory[:, dim_idx])


    # Step 2: Handle Boundary Conditions
    original_indices_map = None # To map extended_trajectory points back to original
    if boundary_type == "periodic":
        if not needs_normalization and (np.any(processed_trajectory < -0.5) or np.any(processed_trajectory > 0.5)):
            # Warning or error if periodic is chosen but data is not in range and wasn't auto-normalized well
            # This might indicate an issue with the input data's scale assumption.
            # For now, proceed, but this could be a point of failure.
            pass # print("Warning: Periodic boundary chosen but data seems outside [-0.5, 0.5] even after check.")

        extended_trajectory = create_periodic_points(processed_trajectory, ndim)
        
        # Create a map: extended_trajectory_idx -> original_point_idx
        num_replicas = 3**ndim
        original_indices_map = np.repeat(np.arange(num_points), num_replicas)

    elif boundary_type == "clipped":
        extended_trajectory = processed_trajectory
        original_indices_map = np.arange(num_points) # Direct mapping
    else:
        raise ValueError(f"Unknown boundary_type: {boundary_type}. Must be 'periodic' or 'clipped'.")

    # Step 3: Compute Voronoi Tessellation
    default_qhull_options = 'Qbb Qc Qz' # Common robust options
    # For 3D, Qx (exact pre-merges) can sometimes help if default struggles, but default is usually fine.
    # if ndim == 3 and qhull_options is None: qhull_options = 'Qbb Qc Qz Qx'
    
    actual_qhull_options = qhull_options if qhull_options is not None else default_qhull_options
    
    if extended_trajectory.shape[0] < ndim + 1:
        # Not enough points for Voronoi. Fallback to uniform.
        return np.ones(num_points) / num_points if num_points > 0 else np.array([])

    try:
        vor = Voronoi(extended_trajectory, qhull_options=actual_qhull_options)
    except QhullError as e:
        # If Voronoi fails (e.g. all points collinear in 2D, or coplanar in 3D after extension)
        # Fallback to uniform weights for the original points.
        # print(f"QhullError during Voronoi: {e}. Falling back to uniform weights.") # Optional debug
        return np.ones(num_points) / num_points if num_points > 0 else np.array([])
    except Exception as e: # Other errors
        # print(f"Unexpected error during Voronoi: {e}. Falling back to uniform weights.") # Optional debug
        return np.ones(num_points) / num_points if num_points > 0 else np.array([])


    density_weights = np.zeros(num_points)

    # Step 4: Calculate Density for Each Original Point
    if boundary_type == "periodic":
        # The central cell (non-shifted) for each original point is the one to sum up.
        # In create_periodic_points, the original point is part of the replicas
        # (specifically, when offsets are all 0).
        # We need to find the index in `extended_trajectory` that corresponds to the *original, unshifted*
        # version of each point to ensure we are calculating the area of the *central* cell.
        
        # The `create_periodic_points` function adds points systematically.
        # If point `j` in `processed_trajectory` is `p_j`, then in `extended_trajectory`,
        # the block of `3^ndim` points corresponding to `p_j` starts at `j * 3^ndim`.
        # The replica `p_j + (0,0)` or `p_j + (0,0,0)` is at a specific offset within that block.
        # For 2D: offsets: [-1,0,1] x [-1,0,1]. (0,0) is the 5th element (idx 4) in 3x3 grid.
        # For 3D: offsets: [-1,0,1] x [-1,0,1] x [-1,0,1]. (0,0,0) is the 14th (idx 13) in 3x3x3 grid.
        
        offset_idx_for_original = 0
        if ndim == 2: offset_idx_for_original = 3*1 + 1 # (0,0) offset for i=0, j=0 assuming order is i, then j
        elif ndim == 3: offset_idx_for_original = (3*3)*1 + 3*1 + 1 # (0,0,0) offset

        for original_pt_idx in range(num_points):
            # Find the index in `extended_trajectory` corresponding to the non-shifted version of original_pt_idx
            # This is the point that was `processed_trajectory[original_pt_idx]` before extension.
            # Its index in `extended_trajectory` is `original_pt_idx * (3^ndim) + offset_for_zero_shift`.
            
            # Example: if create_periodic_points generates points for p0, then for p1, etc.
            # For p0: p0+(-1,-1), p0+(-1,0), p0+(-1,1), p0+(0,-1), p0+(0,0), p0+(0,1), ...
            # The index for p0+(0,0) is `0 * 9 + 4 = 4` in 2D.
            # The index for p1+(0,0) is `1 * 9 + 4 = 13` in 2D.
            
            # The `compute_cell_area` function takes an index from `vor.points`.
            # `vor.points` is `extended_trajectory`.
            idx_in_extended = original_pt_idx * (3**ndim) + offset_idx_for_original
            
            # Sanity check: the point in vor.points at idx_in_extended should be very close to processed_trajectory[original_pt_idx]
            # if not np.allclose(vor.points[idx_in_extended], processed_trajectory[original_pt_idx]):
            #     print(f"Warning: Mismatch for point {original_pt_idx}. Expected {processed_trajectory[original_pt_idx]}, got {vor.points[idx_in_extended]}")

            density_weights[original_pt_idx] = compute_cell_area(vor, idx_in_extended, ndim, max_points_per_cell)

    else: # "clipped"
        for pt_idx in range(num_points):
            # Here, extended_trajectory is just processed_trajectory, so indices match.
            density_weights[pt_idx] = compute_cell_area(vor, pt_idx, ndim, max_points_per_cell)

    # Step 5: Handle Invalid or Zero-Density Points
    valid_weights_mask = np.isfinite(density_weights) & (density_weights > 1e-12) # Avoid tiny values being "valid"
    
    if np.sum(valid_weights_mask) == 0: # All weights are zero, NaN, or Inf
        if num_points > 0:
            density_weights.fill(1.0 / num_points)
        # else density_weights is already empty, which is correct
    elif np.sum(valid_weights_mask) < num_points: # Some weights are invalid
        median_valid_weight = np.median(density_weights[valid_weights_mask])
        if median_valid_weight < 1e-12 or not np.isfinite(median_valid_weight): # Median is also bad
            density_weights.fill(1.0 / num_points if num_points > 0 else 1.0)
        else:
            invalid_mask = ~valid_weights_mask
            density_weights[invalid_mask] = median_valid_weight
            # Recheck if any assigned median_valid_weight made the sum zero (e.g. if median was negative, though unlikely for area)
            if np.sum(density_weights) < 1e-12 and num_points > 0 :
                 density_weights.fill(1.0 / num_points)


    # Step 6: Normalize Density Weights using the new helper function
    density_weights = normalize_density_weights(density_weights)

    return density_weights


def normalize_density_weights(density_weights: np.ndarray) -> np.ndarray:
    """
    Normalizes density compensation weights.

    Args:
        density_weights (np.ndarray): Raw density compensation weights.

    Returns:
        np.ndarray: Normalized density compensation weights.
    """
    if not isinstance(density_weights, np.ndarray):
        raise TypeError("Input density_weights must be a NumPy array.")

    if density_weights.size == 0:
        return np.array([]) # Return empty for empty input

    sum_weights = np.sum(density_weights)

    # Using 1e-12 as a threshold for "very close to zero"
    # This was used in compute_density_compensation and compute_voronoi_density
    if np.abs(sum_weights) < 1e-12:
        # Fallback to uniform weights
        # This handles cases where all weights are zero or sum to a negligible amount.
        # The shape of the output should match the input.
        uniform_value = 1.0 / density_weights.size
        return np.full(density_weights.shape, uniform_value)
    else:
        return density_weights / sum_weights


def generate_spiral_trajectory(num_arms: int, 
                               num_samples_per_arm: int, 
                               fov_m: float, 
                               max_k_rad_per_m: Optional[float] = None, 
                               num_revolutions: Optional[float] = None,
                               name: str = "spiral",
                               dt_seconds: float = 4e-6) -> Trajectory:
    """
    Generates a 2D spiral k-space trajectory.

    Args:
        num_arms (int): Number of spiral interleaves.
        num_samples_per_arm (int): Number of k-space samples per interleaf.
        fov_m (float): Field of view in meters. Used if max_k_rad_per_m is not given.
        max_k_rad_per_m (Optional[float], optional): Maximum k-space radius in rad/m. 
                                                     If None, defaults to np.pi / fov_m.
        num_revolutions (Optional[float], optional): Number of revolutions for each spiral arm.
                                                     If None, defaults to k_max / (pi / (fov_m / (num_arms * 0.5)))
                                                     which is a heuristic for coverage. A higher value means tighter spiral.
                                                     Or, more simply, relate to num_arms, e.g. 10-20.
                                                     Let's try a simpler default like num_arms, or a fixed sensible value like 10.
                                                     Let's use a default of `num_arms` if not specified, as per one of the suggestions.
                                                     A common choice is also related to kmax and desired resolution.
                                                     Let's refine to: default to `max_k_rad_per_m / (np.pi / (fov_m / num_arms)) / 2.0`
                                                     This is roughly `k_max / (k_space_dist_between_arms_at_edge / 2)`.
                                                     A simpler default: Let num_revolutions be related to num_arms, e.g., 5 * num_arms / (2*np.pi)
                                                     Or just make it a fixed number if not specified, e.g. 10.
                                                     Let's use a simpler default for now: num_revolutions = num_arms if not specified.
                                                     A common reference (e.g. Pipe et al.) uses alpha in k = A * t * exp(i*alpha*t).
                                                     Let's use the angle formulation: angle_offset + revolutions * 2 * pi * t_sample
                                                     Defaulting num_revolutions to num_arms might be too dense for many arms.
                                                     Let's try a default like 10, or make it dependent on other params.
                                                     A sensible default for num_revolutions could be k_max / (delta_k_Nyquist_step).
                                                     delta_k_Nyquist_step = 1/fov_m.
                                                     So, num_revolutions = k_max * fov_m.
                                                     If num_revolutions is None, let's try: num_revolutions = max_k_rad_per_m * fov_m / (2 * np.pi)
        name (str, optional): Name for the trajectory. Defaults to "spiral".
        dt_seconds (float, optional): Dwell time for the trajectory. Defaults to 4e-6.

    Returns:
        Trajectory: The generated spiral trajectory.
    """

    if max_k_rad_per_m is None:
        k_max = np.pi / fov_m
    else:
        k_max = max_k_rad_per_m

    if num_revolutions is None:
        # Heuristic: aim for arms to be somewhat spaced out at k_max
        # Width of k-space covered by one arm revolution group: num_arms * (1/fov_m) approx.
        # So, num_revolutions to fill k_max: k_max / (1/fov_m) = k_max * fov_m
        # Each revolution is 2*pi radians.
        num_revs = k_max * fov_m / (2 * np.pi) # Revolutions to reach k_max with steps of Nyquist
        # This num_revs is for a single arm to cover the FOV.
        # The term `巻き数` (num_revolutions) in the prompt is per arm.
        # Let's use a fixed default if not provided, e.g., 10, or relate to coverage.
        # A common way: num_revolutions = k_max * fov_m / (2 * np.pi * sqrt_num_arms_factor)
        # For simplicity, let's use a default that might be reasonable for typical parameters.
        # num_revolutions = k_max / (num_arms * (1/fov_m)) # Number of turns for each arm to fill its angular sector
        num_revolutions = k_max * fov_m / (2 * np.pi * 0.5) # Aim for arms to be roughly Nyquist spaced at edge
        # This might still be too large. Let's try a simpler default:
        # Based on the prompt `angle = angle_offset + 2 * PI * t * num_arms` used in example, this implies num_arms revolutions.
        # Let's use that interpretation for `num_revolutions` if not specified, i.e. each arm does `num_arms` turns.
        # This can be very dense. Let's try a more common fixed value or simpler heuristic.
        # Heuristic: number of samples / (some factor, e.g. 100)
        # num_revolutions = num_samples_per_arm / 100.0 # Example
        # Let's adopt the formula from the prompt's discussion:
        # angle = angle_offset + revolutions * 2 * np.pi * t_sample
        # Where `revolutions` is the variable we are setting.
        # Let's make the default `num_revolutions` such that the spiral arms are roughly
        # separated by `1/fov_m` at `k_max`.
        # Circumference at k_max = 2 * pi * k_max.
        # Space per arm = (2 * pi * k_max) / num_arms.
        # We want this space to be roughly `N_rev * (1/fov_m)`.
        # No, that's not right. The angular increment determines turns.
        # Let's use a simpler default from common spirals: num_revolutions = k_max * sqrt(num_samples_per_arm) / (2*pi) (from Berstein Handbook)
        # This is complex. Let's use the prompt's `angle = angle_offset + 2 * PI * t * num_arms` as a guide for the total angular sweep.
        # If `num_revolutions` refers to the `巻き数` in the prompt, then it's the factor for `2 * pi * t_sample`.
        # The prompt's example `angle = angle_offset + 2 * PI * t * num_arms` means "巻き数" = `num_arms`.
        # Let's use this interpretation: if `num_revolutions` is not given, it implies this "num_arms" factor for turns.
        # So, if `num_revolutions` parameter is given, it overrides this.
        # Let's make default revolutions a fixed number like 10 for now if not set.
        effective_revolutions = num_revolutions if num_revolutions is not None else 10.0


    all_k_points_complex = []

    for j in range(num_arms):
        angle_offset = j * (2 * np.pi / num_arms)
        arm_k_points_complex = []
        for s in range(num_samples_per_arm):
            if num_samples_per_arm == 1:
                t_sample = 1.0 # Avoid division by zero, place point at k_max
            else:
                t_sample = s / (num_samples_per_arm - 1) # Normalized time [0, 1]
            
            current_radius = t_sample * k_max
            # Using the formula: current_angle = angle_offset + revolutions * 2 * np.pi * t_sample
            current_angle = angle_offset + effective_revolutions * 2 * np.pi * t_sample
            
            kx = current_radius * np.cos(current_angle)
            ky = current_radius * np.sin(current_angle)
            arm_k_points_complex.append(kx + 1j * ky)
        all_k_points_complex.extend(arm_k_points_complex)

    k_points_complex_array = np.array(all_k_points_complex) # Shape: (num_arms * num_samples_per_arm,)

    # Convert to (2, N) real array
    kx_all = k_points_complex_array.real
    ky_all = k_points_complex_array.imag
    k_space_points_rad_m = np.vstack((kx_all, ky_all)) # Shape: (2, num_arms * num_samples_per_arm)

    metadata = {
        'generator_params': {
            'traj_type': 'spiral',
            'num_arms': num_arms,
            'num_samples_per_arm': num_samples_per_arm,
            'fov_m': fov_m,
            'max_k_rad_per_m_input': max_k_rad_per_m, # Store what was passed
            'k_max_calculated_rad_m': k_max,
            'num_revolutions_effective': effective_revolutions,
            'dt_seconds_input': dt_seconds
        },
        'interleaf_structure': (num_arms, num_samples_per_arm) # For Trajectory class plotting
    }

    return Trajectory(name=name, 
                      kspace_points_rad_per_m=k_space_points_rad_m, 
                      dt_seconds=dt_seconds,
                      metadata=metadata)


def generate_radial_trajectory(num_spokes: int, 
                               num_samples_per_spoke: int, 
                               fov_m: float, 
                               max_k_rad_per_m: Optional[float] = None, 
                               use_golden_angle: bool = True, 
                               name: str = "radial",
                               dt_seconds: float = 4e-6) -> Trajectory:
    """
    Generates a 2D radial k-space trajectory.

    Args:
        num_spokes (int): Number of radial spokes.
        num_samples_per_spoke (int): Number of samples along each spoke.
        fov_m (float): Field of view in meters.
        max_k_rad_per_m (Optional[float], optional): Maximum k-space radius. 
                                                     If None, defaults to np.pi / fov_m.
        use_golden_angle (bool, optional): If True, use golden angle increment. 
                                           Otherwise, use uniform angular spacing. Defaults to True.
        name (str, optional): Name for the trajectory. Defaults to "radial".
        dt_seconds (float, optional): Dwell time for the trajectory. Defaults to 4e-6.

    Returns:
        Trajectory: The generated radial trajectory.
    """

    if max_k_rad_per_m is None:
        k_max = np.pi / fov_m
    else:
        k_max = max_k_rad_per_m

    all_k_points_complex = []
    
    # Create the k-space sample points along one spoke (from -k_max to k_max)
    # If num_samples_per_spoke is 1, it should be at k=0.
    if num_samples_per_spoke == 1:
        k_radii = np.array([0.0])
    else:
        k_radii = np.linspace(-k_max, k_max, num_samples_per_spoke)

    for j in range(num_spokes):
        if use_golden_angle:
            angle = j * np.pi * (3.0 - np.sqrt(5.0))
        else:
            # Ensure that spokes cover 0 to pi (or 0 to 2pi depending on convention)
            # Standard radial often covers 0 to pi, as -k_max to k_max covers the other half.
            angle = j * np.pi / num_spokes 
            # If spokes are defined from 0 to k_max, then angles should go 0 to 2pi.
            # Given k_radii go from -k_max to k_max, 0 to pi for angles is fine.

        spoke_k_points_complex = []
        for k_val in k_radii:
            kx = k_val * np.cos(angle)
            ky = k_val * np.sin(angle)
            spoke_k_points_complex.append(kx + 1j * ky)
        all_k_points_complex.extend(spoke_k_points_complex)

    k_points_complex_array = np.array(all_k_points_complex) # Shape: (num_spokes * num_samples_per_spoke,)

    # Convert to (2, N) real array
    kx_all = k_points_complex_array.real
    ky_all = k_points_complex_array.imag
    k_space_points_rad_m = np.vstack((kx_all, ky_all)) # Shape: (2, num_spokes * num_samples_per_spoke)

    metadata = {
        'generator_params': {
            'traj_type': 'radial',
            'num_spokes': num_spokes,
            'num_samples_per_spoke': num_samples_per_spoke,
            'fov_m': fov_m,
            'max_k_rad_per_m_input': max_k_rad_per_m,
            'k_max_calculated_rad_m': k_max,
            'use_golden_angle': use_golden_angle,
            'dt_seconds_input': dt_seconds
        },
        'interleaf_structure': (num_spokes, num_samples_per_spoke)
    }

    return Trajectory(name=name,
                      kspace_points_rad_per_m=k_space_points_rad_m,
                      dt_seconds=dt_seconds,
                      metadata=metadata)


def generate_golden_angle_3d_trajectory(num_points: int,
                                        fov_m: float | tuple[float, float, float] | list[float],
                                        max_k_rad_per_m: Optional[float | tuple[float, float, float] | list[float]] = None,
                                        name: str = "golden_angle_3d",
                                        dt_seconds: float = 4e-6) -> Trajectory:
    """
    Generates a 3D k-space trajectory using the Golden Angle (Phyllotaxis spiral) method.

    This method distributes points approximately uniformly on the surface of spheres
    of varying radii, or on ellipsoids if max_k_rad_per_m is anisotropic.

    Args:
        num_points (int): Total number of k-space points to generate.
        fov_m (float | tuple/list of 3 floats): Field of View in meters.
            If a single float, isotropic FOV is assumed.
            If a tuple/list of 3 floats (fov_x, fov_y, fov_z), anisotropic FOV is used.
        max_k_rad_per_m (Optional float | tuple/list of 3 floats): Maximum k-space extent
            in rad/m. If a single float, isotropic k-max is assumed.
            If a tuple/list (kmax_x, kmax_y, kmax_z), anisotropic k-max is applied.
            If None, it's calculated as np.pi / fov_m for each dimension.
        name (str, optional): Name for the trajectory. Defaults to "golden_angle_3d".
        dt_seconds (float, optional): Dwell time for the trajectory. Defaults to 4e-6.

    Returns:
        Trajectory: The generated 3D Golden Angle trajectory.
    """
    if num_points <= 0:
        k_space_points = np.empty((3, 0))
        k_max_calculated_rad_m_xyz_final = (0.0, 0.0, 0.0)
        gen_params = {
            'traj_type': 'golden_angle_3d',
            'num_points': num_points,
            'fov_m_input': fov_m,
            'max_k_rad_per_m_input': max_k_rad_per_m,
            'dt_seconds_input': dt_seconds
        }
        metadata = {
            'generator_params': gen_params,
            'k_max_calculated_rad_m_xyz': k_max_calculated_rad_m_xyz_final
        }
        return Trajectory(name=name,
                          kspace_points_rad_per_m=k_space_points,
                          dt_seconds=dt_seconds,
                          metadata=metadata)

    # Determine k_max for each dimension
    if max_k_rad_per_m is None:
        if isinstance(fov_m, (float, int)):
            k_max_x = k_max_y = k_max_z = np.pi / float(fov_m)
        elif isinstance(fov_m, (list, tuple)) and len(fov_m) == 3:
            k_max_x = np.pi / float(fov_m[0])
            k_max_y = np.pi / float(fov_m[1])
            k_max_z = np.pi / float(fov_m[2])
        else:
            raise ValueError("fov_m must be a float or a list/tuple of 3 floats.")
    else:
        if isinstance(max_k_rad_per_m, (float, int)):
            k_max_x = k_max_y = k_max_z = float(max_k_rad_per_m)
        elif isinstance(max_k_rad_per_m, (list, tuple)) and len(max_k_rad_per_m) == 3:
            k_max_x = float(max_k_rad_per_m[0])
            k_max_y = float(max_k_rad_per_m[1])
            k_max_z = float(max_k_rad_per_m[2])
        else:
            raise ValueError("max_k_rad_per_m must be a float or a list/tuple of 3 floats, or None.")

    k_max_calculated_rad_m_xyz_final = (k_max_x, k_max_y, k_max_z)

    # Golden Angle constants
    inc = np.pi * (3. - np.sqrt(5.))  # Golden angle increment
    
    k_space_points = np.zeros((3, num_points))

    for i in range(num_points):
        # Normalized radius, ensuring points fill the volume
        # (i + 0.5) / num_points ensures r_norm ranges from near 0 to near 1
        r_norm = np.power((i + 0.5) / num_points, 1./3.)

        # Spherical coordinates based on Phyllotaxis spiral
        # k corresponds to z-coordinate before scaling (normalized from -1 to 1)
        # Offset ensures points are symmetric around the equator for z
        offset = 2. / num_points if num_points > 0 else 0.
        k_z_norm_sphere = ((i * offset) - 1.) + (offset / 2.) 
        if abs(k_z_norm_sphere) > 1: # Clamp due to potential floating point issues for i=0 or i=N-1
            k_z_norm_sphere = np.sign(k_z_norm_sphere)

        theta = np.arccos(k_z_norm_sphere)  # Polar angle (from z-axis)
        phi = (i % num_points) * inc       # Azimuthal angle, wraps around due to % (though i is already 0 to N-1)
                                           # More standard: phi = i * inc

        # Convert spherical (normalized radius, theta, phi) to Cartesian (normalized)
        # These are coordinates on a unit sphere, scaled by r_norm
        kx_norm = r_norm * np.sin(theta) * np.cos(phi)
        ky_norm = r_norm * np.sin(theta) * np.sin(phi)
        kz_norm = r_norm * np.cos(theta) # which is r_norm * k_z_norm_sphere (if theta derived from k_z_norm_sphere)

        # Scale by anisotropic k_max
        k_space_points[0, i] = kx_norm * k_max_x
        k_space_points[1, i] = ky_norm * k_max_y
        k_space_points[2, i] = kz_norm * k_max_z
        
    gen_params = {
        'traj_type': 'golden_angle_3d',
        'num_points': num_points,
        'fov_m_input': fov_m,
        'max_k_rad_per_m_input': max_k_rad_per_m, # Store what was passed
        'dt_seconds_input': dt_seconds
    }
    
    metadata = {
        'generator_params': gen_params,
        'k_max_calculated_rad_m_xyz': k_max_calculated_rad_m_xyz_final
    }

    return Trajectory(name=name,
                      kspace_points_rad_per_m=k_space_points,
                      dt_seconds=dt_seconds,
                      metadata=metadata)


def constrain_trajectory(trajectory: Trajectory, 
                         max_gradient_Tm_per_m: float, 
                         max_slew_rate_Tm_per_m_per_s: float, 
                         dt_seconds: Optional[float] = None) -> Trajectory:
    """
    Constrains a k-space trajectory based on maximum gradient and slew rate limits.

    Args:
        trajectory (Trajectory): The input Trajectory object.
        max_gradient_Tm_per_m (float): Maximum gradient amplitude (T/m).
        max_slew_rate_Tm_per_m_per_s (float): Maximum slew rate (T/m/s).
        dt_seconds (Optional[float], optional): Sampling dwell time. 
                                                If None, tries to get from trajectory.dt_seconds.
                                                Raises ValueError if unavailable.

    Returns:
        Trajectory: A new Trajectory object with the constrained k-space points.
    """
    # 1. Get k_points_rad_m from trajectory.kspace_points_rad_per_m
    k_points_orig_rad_m = np.array(trajectory.kspace_points_rad_per_m)
    
    # Ensure k_points is (D, N)
    # The Trajectory class stores kspace_points_rad_per_m.
    # get_num_dimensions() and get_num_points() should give consistent D and N.
    # If k_points_orig_rad_m is (N,D), transpose it.
    # A common internal representation for Trajectory seems to be (D,N) for calculations.
    D_orig = trajectory.get_num_dimensions()
    N_orig = trajectory.get_num_points()

    if k_points_orig_rad_m.shape[0] == N_orig and k_points_orig_rad_m.shape[1] == D_orig and D_orig != N_orig:
        k_points_rad_m = k_points_orig_rad_m.T # Convert (N,D) to (D,N)
    elif k_points_orig_rad_m.shape[0] == D_orig and k_points_orig_rad_m.shape[1] == N_orig:
        k_points_rad_m = k_points_orig_rad_m # Already (D,N)
    elif k_points_orig_rad_m.ndim == 1 and D_orig == 1 and k_points_orig_rad_m.shape[0] == N_orig: # (N,) for 1D
        k_points_rad_m = k_points_orig_rad_m.reshape(1, N_orig) # Convert to (1,N)
    elif k_points_orig_rad_m.ndim == 1 and N_orig == 1 and k_points_orig_rad_m.shape[0] == D_orig: # (D,) for 1 point
        k_points_rad_m = k_points_orig_rad_m.reshape(D_orig, 1) # Convert to (D,1)
    else:
        # Attempt to infer if shape is ambiguous but D <=3
        if k_points_orig_rad_m.shape[0] <=3 and k_points_orig_rad_m.shape[1] == N_orig : # D,N
             k_points_rad_m = k_points_orig_rad_m
        elif k_points_orig_rad_m.shape[1] <=3 and k_points_orig_rad_m.shape[0] == N_orig : #N,D
             k_points_rad_m = k_points_orig_rad_m.T
        else:
            raise ValueError(f"Cannot unambiguously determine (D,N) format from shape {k_points_orig_rad_m.shape} for D={D_orig}, N={N_orig}")


    num_dims, num_total_points = k_points_rad_m.shape

    if num_total_points == 0:
        return Trajectory(name=trajectory.name + "_constrained_empty", 
                          kspace_points_rad_per_m=np.array([]), 
                          dt_seconds=trajectory.dt_seconds, 
                          metadata=trajectory.metadata)
    if num_total_points == 1: # Single point trajectory, no constraints apply to segments
        return Trajectory(name=trajectory.name + "_constrained_single_pt", 
                          kspace_points_rad_per_m=k_points_rad_m.copy(), 
                          dt_seconds=trajectory.dt_seconds, 
                          metadata=trajectory.metadata)


    # 2. Get dt_s
    dt_s = dt_seconds if dt_seconds is not None else trajectory.dt_seconds
    if dt_s is None or dt_s <= 0:
        raise ValueError("Dwell time (dt_seconds) must be positive and available.")

    # 3. Get gamma
    gamma = trajectory.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
    if gamma == 0: gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'] # Avoid division by zero

    # 4. Initialize k_points_constrained_rad_m
    k_points_constrained_rad_m = np.zeros_like(k_points_rad_m)
    k_points_constrained_rad_m[:, 0] = k_points_rad_m[:, 0]

    # 5. Initialize prev_grad_Tm
    prev_grad_Tm = np.zeros(num_dims)

    # 6. Loop i from 1 to num_total_points - 1
    for i in range(1, num_total_points):
        current_k_constrained_prev_sample = k_points_constrained_rad_m[:, i-1]
        target_k_original_curr_sample = k_points_rad_m[:, i]
        
        dk_target_rad_m = target_k_original_curr_sample - current_k_constrained_prev_sample
        grad_target_Tm = dk_target_rad_m / (gamma * dt_s)
        
        norm_grad_target = np.linalg.norm(grad_target_Tm)
        
        if norm_grad_target > max_gradient_Tm_per_m:
            grad_actual_after_grad_limit_Tm = grad_target_Tm * (max_gradient_Tm_per_m / norm_grad_target)
        else:
            grad_actual_after_grad_limit_Tm = grad_target_Tm
            
        slew_needed_Tm_s = (grad_actual_after_grad_limit_Tm - prev_grad_Tm) / dt_s
        norm_slew_needed = np.linalg.norm(slew_needed_Tm_s)
        
        if norm_slew_needed > max_slew_rate_Tm_per_m_per_s:
            # Scale the slew vector to the max slew rate
            allowed_slew_Tm_s = slew_needed_Tm_s * (max_slew_rate_Tm_per_m_per_s / norm_slew_needed)
            # The actual gradient is the previous gradient plus the allowed slew over dt
            grad_actual_final_Tm = prev_grad_Tm + allowed_slew_Tm_s * dt_s
        else:
            grad_actual_final_Tm = grad_actual_after_grad_limit_Tm
            
        dk_actual_rad_m = grad_actual_final_Tm * gamma * dt_s
        k_points_constrained_rad_m[:, i] = current_k_constrained_prev_sample + dk_actual_rad_m
        
        prev_grad_Tm = grad_actual_final_Tm

    # 7. Create a new Trajectory object
    new_metadata = trajectory.metadata.copy()
    new_metadata['constraints'] = {
        'max_gradient_Tm_per_m': max_gradient_Tm_per_m,
        'max_slew_rate_Tm_per_m_per_s': max_slew_rate_Tm_per_m_per_s,
        'dt_seconds_constraint_pass': dt_s
    }
    new_metadata['original_trajectory_name'] = trajectory.name
    
    # Ensure output k-space points have same orientation as input if it was (N,D)
    final_k_points_output = k_points_constrained_rad_m
    if k_points_orig_rad_m.shape[0] == N_orig and k_points_orig_rad_m.shape[1] == D_orig and D_orig != N_orig : # Input was N,D
        final_k_points_output = k_points_constrained_rad_m.T


    return Trajectory(name=trajectory.name + "_constrained",
                      kspace_points_rad_per_m=final_k_points_output,
                      dt_seconds=dt_s, # Use the dt_s that was used for calculations
                      metadata=new_metadata,
                      gamma_Hz_per_T=gamma) # Pass the gamma used


def reconstruct_image(kspace_data: np.ndarray, 
                      trajectory_obj: Trajectory, 
                      grid_size: tuple[int, int], 
                      density_comp_method: Optional[str] = "voronoi", 
                      verbose: bool = False) -> np.ndarray:
    """
    Reconstructs a 2D image from non-Cartesian k-space data using regridding.

    Args:
        kspace_data (np.ndarray): Complex NumPy array of k-space data. 
                                  Shape should be compatible with the trajectory 
                                  (e.g., (num_total_samples,) or (num_arms, num_samples_per_arm)).
        trajectory_obj (Trajectory): A Trajectory object containing the k-space coordinates.
        grid_size (tuple[int, int]): Tuple (Ny, Nx) specifying the desired output image matrix size.
        density_comp_method (Optional[str], optional): Method for density compensation 
                                                       ('voronoi', 'pipe', or None). 
                                                       If None, no density compensation. Defaults to "voronoi".
        verbose (bool, optional): If True, print status messages. Defaults to False.

    Returns:
        np.ndarray: Reconstructed magnitude image, shape (Ny, Nx).
    """

    if verbose: print("Starting image reconstruction...")

    # 1. Validate Inputs
    num_traj_points = trajectory_obj.get_num_points()
    if kspace_data.size != num_traj_points:
        raise ValueError(f"kspace_data size ({kspace_data.size}) does not match "
                         f"trajectory points ({num_traj_points}).")

    if trajectory_obj.get_num_dimensions() != 2:
        raise ValueError("Image reconstruction currently only supports 2D trajectories.")

    if not (isinstance(grid_size, tuple) and len(grid_size) == 2 and
            isinstance(grid_size[0], int) and isinstance(grid_size[1], int) and
            grid_size[0] > 0 and grid_size[1] > 0):
        raise ValueError("grid_size must be a tuple of 2 positive integers (Ny, Nx).")

    if verbose: print("Input validation passed.")

    # 2. Prepare K-space Data and Coordinates
    k_data_flat = kspace_data.flatten()

    # Get k-space coordinates, ensure (N, 2) shape for griddata
    # Trajectory.kspace_points_rad_per_m can be (D,N) or (N,D)
    # Trajectory.get_num_dimensions() is D, Trajectory.get_num_points() is N
    k_points_raw = np.array(trajectory_obj.kspace_points_rad_per_m)
    if k_points_raw.shape[0] == trajectory_obj.get_num_dimensions() and \
       k_points_raw.shape[1] == trajectory_obj.get_num_points(): # (D, N)
        k_points_for_griddata = k_points_raw.T # Need (N, D)
    elif k_points_raw.shape[0] == trajectory_obj.get_num_points() and \
         k_points_raw.shape[1] == trajectory_obj.get_num_dimensions(): # (N, D)
        k_points_for_griddata = k_points_raw
    elif k_points_raw.ndim == 1 and trajectory_obj.get_num_dimensions() == 1 and \
         k_points_raw.shape[0] == trajectory_obj.get_num_points(): # (N,) for 1D - error already caught
        raise ValueError("Trajectory is 1D, but 2D is required for reconstruction.")
    else: # Try to infer if D=2
        if k_points_raw.shape[1] == 2: # Assumed (N,2)
             k_points_for_griddata = k_points_raw
        elif k_points_raw.shape[0] == 2: # Assumed (2,N)
             k_points_for_griddata = k_points_raw.T
        else:
            raise ValueError(f"Could not prepare k-space coordinates of shape {k_points_raw.shape} into (N,2) format.")

    kx_traj = k_points_for_griddata[:, 0]
    ky_traj = k_points_for_griddata[:, 1]

    if verbose: print(f"K-space data and coordinates prepared. Trajectory points: {num_traj_points}")

    # 3. Apply Density Compensation (Optional)
    if density_comp_method is not None:
        if verbose: print(f"Applying density compensation using method: {density_comp_method}...")
        try:
            # compute_density_compensation expects trajectory points (N,D) or complex (A,S)
            # k_points_for_griddata is (N,2) which is suitable.
            weights = compute_density_compensation(
                trajectory=k_points_for_griddata, # Pass the (N,2) real k-space points
                method=density_comp_method,
                dt_seconds=trajectory_obj.dt_seconds,
                gamma_Hz_per_T=trajectory_obj.metadata.get('gamma_Hz_per_T')
            )
            # weights will be (N,). k_data_flat is (N,).
            k_data_flat = k_data_flat * weights.flatten() # Ensure weights are also flat
            if verbose: print("Density compensation applied.")
        except Exception as e:
            if verbose: print(f"Error during density compensation: {e}. Proceeding without.")
            # Optionally re-raise or handle more gracefully
    
    # 4. Regrid K-space Data to Cartesian Grid
    if verbose: print("Regridding k-space data...")
    
    k_max_x = np.max(np.abs(kx_traj))
    k_max_y = np.max(np.abs(ky_traj))

    # Handle cases where k_max_x or k_max_y might be zero (e.g. line trajectory)
    if k_max_x == 0: k_max_x = np.pi # Default if no extent in x
    if k_max_y == 0: k_max_y = np.pi # Default if no extent in y

    kx_cart_1d = np.linspace(-k_max_x, k_max_x, grid_size[1]) # Nx points
    ky_cart_1d = np.linspace(-k_max_y, k_max_y, grid_size[0]) # Ny points
    kxx_cart, kyy_cart = np.meshgrid(kx_cart_1d, ky_cart_1d)

    points_for_interp = np.vstack((kx_traj, ky_traj)).T # Shape (num_total_samples, 2)
    
    gridded_k_data = griddata(points_for_interp, k_data_flat, 
                              (kxx_cart, kyy_cart), method='linear', fill_value=0.0)
    
    if verbose: print("K-space data regridded.")

    # 5. Inverse FFT
    if verbose: print("Performing inverse FFT...")
    # Shift zero frequency to center for IFFT
    k_space_shifted = np.fft.ifftshift(gridded_k_data)
    # 2D Inverse FFT
    image_complex_shifted = np.fft.ifft2(k_space_shifted)
    # Shift zero frequency component back to the center of the image
    image_complex = np.fft.fftshift(image_complex_shifted)
    if verbose: print("Inverse FFT performed.")

    # 6. Final Image (Magnitude)
    image_magnitude = np.abs(image_complex)
    if verbose: print("Image reconstruction complete.")
    
    return image_magnitude


class GIRF:
    """
    Represents a Gradient Impulse Response Function (GIRF).

    Attributes:
        h_t_x (np.ndarray): Impulse response for the X-axis.
        h_t_y (np.ndarray): Impulse response for the Y-axis.
        h_t_z (np.ndarray): Impulse response for the Z-axis.
        dt_girf (float): Time resolution of the GIRF data in seconds.
        name (Optional[str]): Optional name for the GIRF profile.
    """
    def __init__(self, h_t_x: np.ndarray, h_t_y: np.ndarray, h_t_z: np.ndarray, 
                 dt_girf: float, name: Optional[str] = None):
        """
        Initializes the GIRF object.

        Args:
            h_t_x (np.ndarray): 1D NumPy array for X-axis impulse response.
            h_t_y (np.ndarray): 1D NumPy array for Y-axis impulse response.
            h_t_z (np.ndarray): 1D NumPy array for Z-axis impulse response.
            dt_girf (float): Time resolution of GIRF data in seconds.
            name (Optional[str], optional): Name for the GIRF profile. Defaults to None.
        
        Raises:
            ValueError: If dt_girf is not positive or if h_t arrays are not 1D.
        """
        if dt_girf <= 0:
            raise ValueError("dt_girf must be positive.")

        self.h_t_x = np.asarray(h_t_x)
        self.h_t_y = np.asarray(h_t_y)
        self.h_t_z = np.asarray(h_t_z)
        
        if self.h_t_x.ndim != 1:
            raise ValueError("h_t_x must be a 1D array.")
        if self.h_t_y.ndim != 1:
            raise ValueError("h_t_y must be a 1D array.")
        if self.h_t_z.ndim != 1:
            raise ValueError("h_t_z must be a 1D array.")
            
        self.dt_girf = float(dt_girf)
        self.name = name if name is not None else "CustomGIRF"

    def __repr__(self) -> str:
        """
        Returns a string representation of the GIRF object.
        """
        name_str = f"'{self.name}'" if self.name else "None"
        return (f"GIRF(name={name_str}, dt_girf={self.dt_girf:.2e}, "
                f"x_len={len(self.h_t_x)}, y_len={len(self.h_t_y)}, z_len={len(self.h_t_z)})")

    @classmethod
    def from_files(cls, filepath_x: str, filepath_y: str, filepath_z: str, 
                   dt_girf: float, name: Optional[str] = None) -> 'GIRF':
        """
        Loads GIRF data from .npy files for X, Y, and Z axes.

        Args:
            filepath_x (str): Path to the .npy file for X-axis GIRF.
            filepath_y (str): Path to the .npy file for Y-axis GIRF.
            filepath_z (str): Path to the .npy file for Z-axis GIRF.
            dt_girf (float): Time resolution of the GIRF data in seconds.
            name (Optional[str], optional): Name for the GIRF profile. 
                                           If None, a default name is generated from filepaths. 
                                           Defaults to None.

        Returns:
            GIRF: An instance of the GIRF class.

        Raises:
            FileNotFoundError: If any of the specified .npy files do not exist.
            ValueError: If dt_girf is not positive, or if files are not valid .npy 
                        or do not contain 1D arrays.
        """
        if dt_girf <= 0:
            raise ValueError("dt_girf must be positive.")

        try:
            h_t_x = np.load(filepath_x)
            h_t_y = np.load(filepath_y)
            h_t_z = np.load(filepath_z)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load GIRF data: {e.filename} not found.") from e
        except Exception as e: # Catches other np.load errors (e.g. bad format)
            raise ValueError(f"Error loading GIRF data from .npy files: {e}")

        if h_t_x.ndim != 1:
            raise ValueError(f"h_t_x from {filepath_x} must be a 1D array, got shape {h_t_x.shape}.")
        if h_t_y.ndim != 1:
            raise ValueError(f"h_t_y from {filepath_y} must be a 1D array, got shape {h_t_y.shape}.")
        if h_t_z.ndim != 1:
            raise ValueError(f"h_t_z from {filepath_z} must be a 1D array, got shape {h_t_z.shape}.")

        if name is None:
            # Generate a default name based on common prefix of filenames
            try:
                import os
                bn_x = os.path.splitext(os.path.basename(filepath_x))[0]
                bn_y = os.path.splitext(os.path.basename(filepath_y))[0]
                bn_z = os.path.splitext(os.path.basename(filepath_z))[0]
                # A simple common name strategy:
                # If all basenames are identical (e.g. girf_x.npy, girf_y.npy, girf_z.npy -> girf)
                # or if they share a common prefix that makes sense.
                # Example: "systemA_girf_x.npy", "systemA_girf_y.npy" -> "systemA_girf"
                # This can be tricky. Let's try a simple approach:
                # Remove common suffixes like _x, _y, _z if they exist
                name_candidate_x = bn_x[:-2] if bn_x.lower().endswith(('_x', '-x')) else bn_x
                name_candidate_y = bn_y[:-2] if bn_y.lower().endswith(('_y', '-y')) else bn_y
                name_candidate_z = bn_z[:-2] if bn_z.lower().endswith(('_z', '-z')) else bn_z

                if name_candidate_x == name_candidate_y == name_candidate_z:
                    default_name = name_candidate_x
                else: # Fallback if names are too different
                    default_name = f"GIRF_{bn_x[:5]}" 
            except Exception: # In case path operations fail for any reason
                default_name = "GIRF_from_files"
            final_name = default_name
        else:
            final_name = name
            
        return cls(h_t_x, h_t_y, h_t_z, dt_girf, name=final_name)


def apply_girf_convolution(gradient_waveform_1d: np.ndarray, 
                           girf_h_t_1d: np.ndarray, 
                           dt_gradient: float, 
                           dt_girf: float) -> np.ndarray:
    """
    Applies a Gradient Impulse Response Function (GIRF) to a gradient waveform
    via convolution, handling resampling of the GIRF if necessary.

    Args:
        gradient_waveform_1d (np.ndarray): The 1D input gradient waveform.
        girf_h_t_1d (np.ndarray): The 1D GIRF impulse response.
        dt_gradient (float): Time resolution (dt) of the gradient_waveform_1d in seconds.
        dt_girf (float): Time resolution (dt) of the girf_h_t_1d in seconds.

    Returns:
        np.ndarray: The 1D gradient waveform convolved with the (potentially resampled) GIRF.
                    Returns an empty array if either input waveform is empty.

    Raises:
        ValueError: If inputs are not 1D NumPy arrays, or if dt values are not positive.
    """
    # Input Validation
    if not isinstance(gradient_waveform_1d, np.ndarray) or gradient_waveform_1d.ndim != 1:
        raise ValueError("gradient_waveform_1d must be a 1D NumPy array.")
    if not isinstance(girf_h_t_1d, np.ndarray) or girf_h_t_1d.ndim != 1:
        raise ValueError("girf_h_t_1d must be a 1D NumPy array.")

    if dt_gradient <= 0:
        raise ValueError("dt_gradient must be positive.")
    if dt_girf <= 0:
        raise ValueError("dt_girf must be positive.")

    if gradient_waveform_1d.size == 0 or girf_h_t_1d.size == 0:
        return np.array([])

    girf_h_t_to_use = girf_h_t_1d

    # Resampling GIRF (if dt_gradient != dt_girf)
    if not np.isclose(dt_gradient, dt_girf):
        # Create time vector for the original GIRF
        t_girf_original = np.arange(len(girf_h_t_1d)) * dt_girf
        
        original_girf_duration = (len(girf_h_t_1d) -1) * dt_girf if len(girf_h_t_1d) > 1 else 0.0
        
        # Ensure num_target_samples is at least 1 if original GIRF had some duration
        # and avoid issues if original_girf_duration is zero.
        if original_girf_duration <= 0: # Handles empty or single-point original GIRF effectively
            num_target_samples = len(girf_h_t_1d) # Preserve length if duration is zero (e.g. single point)
        else:
            num_target_samples = int(round(original_girf_duration / dt_gradient)) + 1
        
        if num_target_samples <= 0 : num_target_samples = 1 # Ensure at least one sample for target time vector

        t_girf_target = np.arange(num_target_samples) * dt_gradient
        
        if len(t_girf_original) == 0: 
             girf_h_t_resampled = np.array([])
        elif len(t_girf_original) == 1: 
             # For a single point GIRF, replicate its value across the new time extent
             # or place it as a scaled delta if that's desired.
             # np.interp with single point xp, fp will extrapolate if t_girf_target is outside.
             # A common way to think of a single point GIRF h[0] at dt_girf is as a delta function
             # with area h[0]*dt_girf. When resampled to dt_gradient, it should be a delta
             # h_res[0]*dt_gradient = h[0]*dt_girf => h_res[0] = h[0]*dt_girf/dt_gradient
             # However, np.interp will just give h[0] at target points that match t_girf_original[0].
             # Let's use a simpler approach: if GIRF is 1 point, it's a delta, sum is its value.
             # Resampled should also be a delta-like shape (e.g. first point) scaled to preserve sum.
             girf_h_t_resampled = np.zeros_like(t_girf_target)
             if girf_h_t_resampled.size > 0:
                girf_h_t_resampled[0] = girf_h_t_1d[0] # Initial value, will be scaled by sum preservation
        else:
             girf_h_t_resampled = np.interp(t_girf_target, t_girf_original, girf_h_t_1d)

        # Normalization to preserve the sum (integral approximation)
        sum_original_girf = np.sum(girf_h_t_1d)
        sum_resampled_girf = np.sum(girf_h_t_resampled)

        if not np.isclose(sum_resampled_girf, 0.0):
            girf_h_t_resampled = girf_h_t_resampled * (sum_original_girf / sum_resampled_girf)
        elif np.isclose(sum_resampled_girf, 0.0) and not np.isclose(sum_original_girf, 0.0):
            # Resampled sum is zero but original wasn't. This implies an issue.
            # For instance, if girf_h_t_resampled became all zeros due to interpolation.
            # In this case, the scaled GIRF remains all zeros. No further action needed.
            pass
            
        girf_h_t_to_use = girf_h_t_resampled
    
    if girf_h_t_to_use.size == 0:
        # This can happen if the original GIRF was empty, or if resampling an extremely short GIRF
        # resulted in an empty array (e.g. if num_target_samples was 0 and not caught).
        # The initial check `gradient_waveform_1d.size == 0 or girf_h_t_1d.size == 0` handles empty original.
        # If resampling leads to an empty girf_h_t_to_use, convolving with it would be like convolving with zero.
        return np.zeros_like(gradient_waveform_1d)

    convolved_gradient = np.convolve(gradient_waveform_1d, girf_h_t_to_use, mode='same')
    
    return convolved_gradient