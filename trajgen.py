import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
from scipy.spatial.qhull import QhullError
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
            return None
        k_data_original = np.array(self.kspace_points_rad_per_m)
        D = self.get_num_dimensions()
        N = self.get_num_points()
        points_ND = k_data_original
        if k_data_original.shape[0] == D and k_data_original.shape[1] == N and D != N : 
            points_ND = k_data_original.T
        elif k_data_original.ndim == 1 and D == 1: 
             points_ND = k_data_original.reshape(-1,1)
        elif points_ND.shape[0] != N or points_ND.shape[1] != D :
            if points_ND.shape[0] != N or points_ND.shape[1] != D:
                 self.metadata['voronoi_calculation_status'] = f"Error: Ambiguous k-space data shape {k_data_original.shape} for (N,D) format. N={N}, D={D}."
                 return None
        if N < D + 1:
            self.metadata['voronoi_calculation_status'] = f"Error: Not enough points ({N}) for Voronoi in {D}D (need at least {D+1})."
            return None
        if D not in [2, 3]:
            self.metadata['voronoi_calculation_status'] = f"Error: Voronoi calculation only supported for 2D/3D (D={D})."
            return None
        unique_points, unique_indices = np.unique(points_ND, axis=0, return_inverse=True)
        if unique_points.shape[0] < D + 1:
            self.metadata['voronoi_calculation_status'] = f"Error: Not enough unique points ({unique_points.shape[0]}) for Voronoi in {D}D."
            return None
        cell_sizes_unique = np.full(unique_points.shape[0], np.nan)
        try:
            default_qhull_options = 'Qbb Qc Qz' 
            vor = Voronoi(unique_points, qhull_options=qhull_options if qhull_options is not None else default_qhull_options)
            for i in range(unique_points.shape[0]):
                region_idx = vor.point_region[i]
                vertex_indices = vor.regions[region_idx]
                if -1 in vertex_indices or not vertex_indices: 
                    cell_sizes_unique[i] = np.inf
                    continue
                region_vertices = vor.vertices[vertex_indices]
                if region_vertices.shape[0] < D + 1: 
                    cell_sizes_unique[i] = 0.0 
                    continue
                try:
                    hull_qhull_options = 'QJ' if D > 1 else None 
                    current_hull = ConvexHull(region_vertices, qhull_options=hull_qhull_options)
                    cell_sizes_unique[i] = current_hull.volume 
                except QhullError: 
                    cell_sizes_unique[i] = np.nan 
                except Exception: 
                    cell_sizes_unique[i] = np.nan
            cell_sizes = cell_sizes_unique[unique_indices]
            self.metadata['voronoi_cell_sizes'] = cell_sizes 
            self.metadata['_cached_voronoi_object'] = vor    
            self.metadata['_cached_voronoi_unique_points'] = unique_points 
            self.metadata['_cached_voronoi_cell_sizes_unique'] = cell_sizes_unique 
            self.metadata['_cached_voronoi_unique_indices'] = unique_indices 
            self.metadata['voronoi_calculation_status'] = "Success"
            return cell_sizes
        except QhullError as e:
            self.metadata['voronoi_calculation_status'] = f"Error: QhullError during Voronoi: {e}"
            return None
        except Exception as e: 
            self.metadata['voronoi_calculation_status'] = f"Error: Unexpected error during Voronoi: {e}"
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
        if D_actual < 3: return ax
        n_interleaves_known, n_points_per_interleaf_known = None, None
        if 'interleaf_structure' in self.metadata and isinstance(self.metadata['interleaf_structure'], tuple) and len(self.metadata['interleaf_structure']) == 2:
            n_il_temp, n_pts_temp = self.metadata['interleaf_structure']
            if isinstance(n_il_temp, int) and isinstance(n_pts_temp, int) and n_il_temp * n_pts_temp == n_total_points:
                n_interleaves_known, n_points_per_interleaf_known = n_il_temp, n_pts_temp
        if n_interleaves_known is None and 'generator_params' in self.metadata:
            gp = self.metadata['generator_params']
            n_il_gen = gp.get('n_interleaves')
            if gp.get('traj_type') == 'stackofspirals' and gp.get('n_stacks') is not None: n_il_gen *= gp.get('n_stacks')
            if n_il_gen and n_il_gen > 0 and n_total_points > 0 and n_total_points % n_il_gen == 0:
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
            except ValueError: plot_segments = [k_data[:, ::max(1,point_stride)]]
        else:
            plot_segments = [k_data[:, ::max(1,point_stride)]]

        total_pts_shown = 0
        for seg in plot_segments:
            seg_stride = 1
            if max_total_points is not None and (total_pts_shown + seg.shape[1]) > max_total_points:
                remaining_budget = max_total_points - total_pts_shown
                if remaining_budget <= 0 and len(plot_segments) > 1: break 
                seg_stride = max(1, int(np.ceil(seg.shape[1] / max(1,remaining_budget)))) if remaining_budget > 0 else seg.shape[1] # Show at least 1 pt if budget allows
            
            final_seg = seg[:, ::seg_stride]
            if final_seg.shape[1] > 0:
                ax.plot(final_seg[0,:], final_seg[1,:], final_seg[2,:], plot_style, markersize=2)
                total_pts_shown += final_seg.shape[1]
            if max_total_points is not None and total_pts_shown >= max_total_points and len(plot_segments) > 1: break
        
        if total_pts_shown == 0 and n_total_points > 0: print("Warning: No points plotted. Adjust subsampling.")
        ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)"); ax.set_zlabel("Kz (rad/m)")
        ax.set_title(title if title else f"3D K-space: {self.name} ({total_pts_shown} pts shown)")
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
>>>>>>> REPLACE
