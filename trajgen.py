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
COMMON_NUCLEI_GAMMA_HZ_PER_T = {
    '1H': 42.576e6, '13C': 10.705e6, '31P': 17.235e6, '19F': 40.052e6,
    '23Na': 11.262e6, '129Xe': 11.777e6, '2H': 6.536e6, '7Li': 16.546e6,
}

class Trajectory:
    def __init__(self, name, kspace_points_rad_per_m, 
                 gradient_waveforms_Tm=None, dt_seconds=None, 
                 metadata=None, gamma_Hz_per_T=COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds=0.0, dead_time_end_seconds=0.0):
        self.name = name
        _kspace_input_arr = np.array(kspace_points_rad_per_m)
        _kspace_points_oriented = _kspace_input_arr
        k_was_transposed = False

        if _kspace_input_arr.ndim == 1:
            _kspace_points_oriented = _kspace_input_arr.reshape(1, -1)
        elif _kspace_input_arr.ndim == 2:
            d0, d1 = _kspace_input_arr.shape
            if d0 == 0 or d1 == 0:
                _kspace_points_oriented = _kspace_input_arr
            elif d0 == 2 and d1 == 1: _kspace_points_oriented = _kspace_input_arr
            elif d0 == 3 and d1 == 2: _kspace_points_oriented = _kspace_input_arr.T; k_was_transposed = True
            elif d0 <= 3 and (d1 > 3 or d0 < d1):
                _kspace_points_oriented = _kspace_input_arr
            elif d1 <= 3 and (d0 > 3 or d1 < d0):
                _kspace_points_oriented = _kspace_input_arr.T; k_was_transposed = True
            elif d0 <= 3 and d1 <= 3:
                if d0 < d1 : _kspace_points_oriented = _kspace_input_arr
                else: _kspace_points_oriented = _kspace_input_arr
            else:
                if d0 < d1: _kspace_points_oriented = _kspace_input_arr
                else: _kspace_points_oriented = _kspace_input_arr.T; k_was_transposed = True
        else:
            _squeezed = np.squeeze(_kspace_input_arr)
            if _squeezed.ndim == 1: _kspace_points_oriented = _squeezed.reshape(1,-1)
            elif _squeezed.ndim == 2:
                d0, d1 = _squeezed.shape
                if d0 == 0 or d1 == 0: _kspace_points_oriented = _squeezed
                elif d0 == 2 and d1 == 1: _kspace_points_oriented = _squeezed
                elif d0 == 3 and d1 == 2: _kspace_points_oriented = _squeezed.T; k_was_transposed = True
                elif d0 <= 3 and (d1 > 3 or d0 < d1): _kspace_points_oriented = _squeezed
                elif d1 <= 3 and (d0 > 3 or d1 < d0): _kspace_points_oriented = _squeezed.T; k_was_transposed = True
                elif d0 <= 3 and d1 <= 3:
                    if d0 < d1: _kspace_points_oriented = _squeezed
                    else: _kspace_points_oriented = _squeezed
                else:
                    if d0 < d1: _kspace_points_oriented = _squeezed
                    else: _kspace_points_oriented = _squeezed.T; k_was_transposed = True
            else: raise ValueError(f"Unsupported k-space input shape: {_kspace_input_arr.shape}")

        self.kspace_points_rad_per_m = _kspace_points_oriented

        if self.kspace_points_rad_per_m.size == 0:
            d0_oriented = self.kspace_points_rad_per_m.shape[0]
            d1_oriented = self.kspace_points_rad_per_m.shape[1]
            if d0_oriented == 0 and d1_oriented > 0: self._D = d1_oriented
            elif d1_oriented == 0 and d0_oriented > 0: self._D = d0_oriented
            elif d0_oriented == 1 and d1_oriented == 0 : self._D = 1
            else: self._D = 0
            self._N = 0
        else:
            self._D = self.kspace_points_rad_per_m.shape[0]
            self._N = self.kspace_points_rad_per_m.shape[1]

        _gradient_input_arr = np.array(gradient_waveforms_Tm) if gradient_waveforms_Tm is not None else None
        if _gradient_input_arr is not None:
            g_oriented = _gradient_input_arr
            if _gradient_input_arr.ndim == 1:
                if self._D == 1 and self._N == _gradient_input_arr.shape[0]: g_oriented = _gradient_input_arr.reshape(1, -1)
                elif self._N == 1 and self._D == _gradient_input_arr.shape[0]: g_oriented = _gradient_input_arr.reshape(-1, 1)
                else: raise ValueError(f"1D gradient shape {_gradient_input_arr.shape} incompatible with k-space D={self._D}, N={self._N}")

            if g_oriented.ndim == 2:
                g_d0, g_d1 = g_oriented.shape
                if g_d0 == self._D and g_d1 == self._N: self.gradient_waveforms_Tm = g_oriented
                elif g_d0 == self._N and g_d1 == self._D: self.gradient_waveforms_Tm = g_oriented.T
                else: raise ValueError(f"Gradient shape ({g_d0},{g_d1}) incompatible with k-space ({self._D},{self._N})")
            else: raise ValueError(f"Unsupported gradient_waveforms_Tm shape: {g_oriented.shape}")
        else: self.gradient_waveforms_Tm = None

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
            self.metadata['dead_time_start_samples'] = self.dead_time_start_seconds / self.dt_seconds
            self.metadata['dead_time_end_samples'] = self.dead_time_end_seconds / self.dt_seconds
        else:
            self.metadata['dead_time_start_samples'] = None
            self.metadata['dead_time_end_samples'] = None

    def get_gradient_waveforms_Tm(self) -> Optional[np.ndarray]:
        if self.gradient_waveforms_Tm is not None:
            if self.gradient_waveforms_Tm.ndim == 2 and self._D is not None and self._N is not None and \
               self.gradient_waveforms_Tm.shape[0] == self._N and \
               self.gradient_waveforms_Tm.shape[1] == self._D and \
               self._D != self._N :
                 self.gradient_waveforms_Tm = self.gradient_waveforms_Tm.T
            return self.gradient_waveforms_Tm

        if self.kspace_points_rad_per_m is None or self._D == 0 :
            return None
        if self._N == 0:
            self.gradient_waveforms_Tm = np.empty((self._D, 0))
            return self.gradient_waveforms_Tm
        if self.dt_seconds is None or self.dt_seconds <= 0:
            return None

        k_data = self.kspace_points_rad_per_m

        gamma = self.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
        if gamma == 0: gamma = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']

        if self._N < 2:
            computed_gradients = np.zeros_like(k_data)
        else:
            try:
                computed_gradients = np.gradient(k_data, self.dt_seconds, axis=1) / gamma
            except ValueError: 
                computed_gradients = np.zeros_like(k_data)
        self.gradient_waveforms_Tm = computed_gradients
        return self.gradient_waveforms_Tm

    def _calculate_slew_rate(self):
        gradients = self.get_gradient_waveforms_Tm()
        if gradients is not None and gradients.size > 0 and self.dt_seconds is not None and self.dt_seconds > 0 and self._N > 1 :
            slew = np.diff(gradients, axis=1) / self.dt_seconds
            self.metadata['max_slew_rate_Tm_per_s'] = np.max(np.linalg.norm(slew, axis=0))
        else:
            self.metadata['max_slew_rate_Tm_per_s'] = 0.0 if self._N > 0 and gradients is not None and gradients.size > 0 else None

    def _calculate_pns(self):
        if self._N == 0 :
            self.metadata['pns_max_abs_gradient_sum_xyz'] = None
            self.metadata['pns_max_abs_slew_sum_xyz'] = None
            return

        gradients = self.get_gradient_waveforms_Tm()

        if gradients is None :
            self.metadata['pns_max_abs_gradient_sum_xyz'] = None
            self.metadata['pns_max_abs_slew_sum_xyz'] = None
            return

        self.metadata['pns_max_abs_gradient_sum_xyz'] = np.max(np.sum(np.abs(gradients), axis=0))

        if self._N < 2 or self.dt_seconds is None or self.dt_seconds <= 0:
            self.metadata['pns_max_abs_slew_sum_xyz'] = 0.0
        else:
            slew = np.diff(gradients, axis=1) / self.dt_seconds
            self.metadata['pns_max_abs_slew_sum_xyz'] = np.max(np.sum(np.abs(slew), axis=0))

    def _calculate_fov(self):
        if self._N > 0 and self._D > 0:
            points = self.kspace_points_rad_per_m
            k_extent_rad_per_m = np.max(np.abs(points), axis=1)

            if np.all(k_extent_rad_per_m < 1e-9):
                self.metadata['fov_estimate_m'] = None
                self.metadata['fov_estimate_mm'] = None
            else:
                fov_m_values = []
                for k_ext in k_extent_rad_per_m:
                    if k_ext < 1e-9: fov_m_values.append(np.inf)
                    else: fov_m_values.append(1 / (2 * k_ext + 1e-9))
                self.metadata['fov_estimate_m'] = fov_m_values
                self.metadata['fov_estimate_mm'] = [f*1e3 if np.isfinite(f) else f for f in fov_m_values]
        else:
            self.metadata['fov_estimate_m'] = None
            self.metadata['fov_estimate_mm'] = None

    def _calculate_resolution(self):
        if self._N > 0 and self._D > 0:
            points = self.kspace_points_rad_per_m
            max_k_radius_rad_per_m = np.max(np.linalg.norm(points, axis=0))
            if max_k_radius_rad_per_m < 1e-9:
                self.metadata['resolution_overall_estimate_m'] = None
                self.metadata['resolution_overall_estimate_mm'] = None
            else:
                res_m = 1 / (2 * max_k_radius_rad_per_m + 1e-9)
                self.metadata['resolution_overall_estimate_m'] = res_m
                self.metadata['resolution_overall_estimate_mm'] = res_m * 1e3
        else:
            self.metadata['resolution_overall_estimate_m'] = None
            self.metadata['resolution_overall_estimate_mm'] = None

    def _compute_metrics(self):
        self._calculate_slew_rate(); self._calculate_pns()
        self._calculate_fov(); self._calculate_resolution()

    def get_duration_seconds(self) -> Optional[float]:
        if self.dt_seconds is None: return None
        return self.dead_time_start_seconds + (self._N * self.dt_seconds) + self.dead_time_end_seconds

    def get_max_grad_Tm(self) -> Optional[float]:
        gradients = self.get_gradient_waveforms_Tm()
        return np.max(np.linalg.norm(gradients, axis=0)) if gradients is not None and gradients.size > 0 else None

    def get_max_slew_Tm_per_s(self) -> Optional[float]:
        return self.metadata.get('max_slew_rate_Tm_per_s')

    def get_num_points(self) -> int: return self._N
    def get_num_dimensions(self) -> int: return self._D

    def export(self, filename, filetype=None):
        if filetype is None:
            filetype = filename.split('.')[-1].lower() if '.' in filename else 'txt'
        
        points_to_export = self.kspace_points_rad_per_m.T if self._N > 0 and self._D > 0 else np.empty((0,self._D))
        if self._D == 0 : points_to_export = np.empty((0,0))

        gradients_from_getter = self.get_gradient_waveforms_Tm()
        gradients_to_export = None
        if gradients_from_getter is not None and gradients_from_getter.size > 0:
            if gradients_from_getter.shape[0] == self._D and gradients_from_getter.shape[1] == self._N:
                gradients_to_export = gradients_from_getter.T
            elif gradients_from_getter.shape[0] == self._N and gradients_from_getter.shape[1] == self._D :
                gradients_to_export = gradients_from_getter

        if filetype == 'csv': np.savetxt(filename, points_to_export, delimiter=',')
        elif filetype == 'npy': np.save(filename, points_to_export)
        elif filetype == 'npz':
            save_dict = {'kspace_points_rad_per_m': points_to_export, 'dt_seconds': self.dt_seconds, 'metadata': self.metadata}
            if gradients_to_export is not None: save_dict['gradient_waveforms_Tm'] = gradients_to_export
            np.savez(filename, **save_dict)
        elif filetype == 'txt': np.savetxt(filename, points_to_export)
        else: raise ValueError(f"Unsupported filetype: {filetype}")

    @classmethod
    def import_from(cls, filename):
        filetype = filename.split('.')[-1].lower() if '.' in filename else 'txt'
        points, gradients, dt, metadata_dict = None, None, None, {}

        if filetype in ['csv', 'txt']:
            points = np.loadtxt(filename, delimiter=',' if filetype == 'csv' else None)
            if points.ndim == 0 and points.size == 1: points = points.reshape(1,1)
            elif points.ndim == 1 and points.size > 0 : points = points.reshape(-1,1)
            elif points.size == 0 : points = np.empty((0,0))
        elif filetype == 'npy':
            points = np.load(filename)
            if points.ndim == 0 and points.size == 1: points = points.reshape(1,1)
            elif points.ndim == 1 and points.size > 0 : points = points.reshape(-1,1)
            elif points.size == 0 : points = np.empty((0,0))
        elif filetype == 'npz':
            data = np.load(filename, allow_pickle=True)
            points = data.get('kspace_points_rad_per_m', data.get('points', data.get('kspace')))
            gradients = data.get('gradient_waveforms_Tm', data.get('gradients'))
            if gradients is not None: gradients = np.array(gradients)
            dt_data = data.get('dt_seconds', data.get('dt'))
            dt = dt_data.item() if dt_data is not None and hasattr(dt_data, 'item') else dt_data
            metadata_raw = data.get('metadata')
            if metadata_raw is not None:
                try: metadata_dict = metadata_raw.item() if hasattr(metadata_raw, 'item') and callable(metadata_raw.item) else dict(metadata_raw)
                except: metadata_dict = {'raw_metadata': metadata_raw} if not isinstance(metadata_raw, dict) else metadata_raw
        else: raise ValueError(f"Unsupported filetype: {filetype}")
        
        if points is None: points = np.empty((0,0))

        return cls(name=filename, kspace_points_rad_per_m=points,
                   gradient_waveforms_Tm=gradients, dt_seconds=dt, metadata=metadata_dict,
                   gamma_Hz_per_T=metadata_dict.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']),
                   dead_time_start_seconds=metadata_dict.get('dead_time_start_seconds', 0.0),
                   dead_time_end_seconds=metadata_dict.get('dead_time_end_seconds', 0.0))

    def calculate_voronoi_density(self, force_recompute=False, qhull_options=None):
        self.metadata['voronoi_calculation_status'] = "Skipped: compute_voronoi_density not available"
        self.metadata['voronoi_cell_sizes'] = None
        return None

    def plot_3d(self, max_total_points=2000, max_interleaves=None, 
                interleaf_stride=1, point_stride=1, 
                title=None, ax=None, figure=None, plot_style='.-'):
        if self._D < 3:
            print(f"Trajectory '{self.name}' is not 3D. Use plot_2d or ensure data is 3D.")
            return ax if ax is not None else None
        if self._N == 0:
            print(f"Trajectory '{self.name}' has no k-space points to plot.")
            return ax if ax is not None else None
        if ax is None:
            fig = figure if figure else plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        elif not isinstance(ax, Axes3D):
             fig = ax.get_figure(); fig.clf(); ax = fig.add_subplot(111, projection='3d')

        k_data = self.kspace_points_rad_per_m

        effective_point_stride = max(1, point_stride)
        num_pts_after_stride = self._N // effective_point_stride if effective_point_stride > 0 else self._N
        num_pts_to_plot = min(num_pts_after_stride, max_total_points)

        final_stride = effective_point_stride
        if num_pts_to_plot < num_pts_after_stride and num_pts_to_plot > 0:
            final_stride = self._N // num_pts_to_plot
        final_stride = max(1, final_stride) # Ensure stride is at least 1

        plot_k_data = k_data[:, ::final_stride] if self._N > 0 else k_data
        final_plot_title = title if title else f"3D K-space: {self.name}"

        if 'interleaf_structure' in self.metadata and max_interleaves is not None and self._N > 0 :
            num_il_orig, pts_per_il_orig = self.metadata.get('interleaf_structure', (1, self._N))
            if num_il_orig > 0 and pts_per_il_orig > 0 and num_il_orig * pts_per_il_orig == self._N :
                num_ils_to_plot = min(max_interleaves, num_il_orig // max(1,interleaf_stride))
                plotted_pts_count = 0
                actual_lines_plotted = 0
                for i in range(num_ils_to_plot):
                    il_idx = i * interleaf_stride
                    start = il_idx * pts_per_il_orig
                    end = start + pts_per_il_orig

                    current_segment_full = k_data[:, start:end]
                    current_num_pts_in_seg_after_stride = current_segment_full.shape[1] // effective_point_stride if effective_point_stride > 0 else current_segment_full.shape[1]
                    current_num_pts_to_plot_this_il = min(current_num_pts_in_seg_after_stride, max_total_points - plotted_pts_count)

                    current_final_il_stride = effective_point_stride
                    if current_num_pts_to_plot_this_il < current_num_pts_in_seg_after_stride and current_num_pts_to_plot_this_il > 0:
                         current_final_il_stride = current_segment_full.shape[1] // current_num_pts_to_plot_this_il
                    current_final_il_stride = max(1, current_final_il_stride)


                    current_segment_plot = current_segment_full[:, ::current_final_il_stride]

                    if current_segment_plot.shape[1] > 0:
                         ax.plot(current_segment_plot[0,:], current_segment_plot[1,:], current_segment_plot[2,:], plot_style)
                         plotted_pts_count += current_segment_plot.shape[1]
                         actual_lines_plotted +=1
                    if plotted_pts_count >= max_total_points: break
                final_plot_title = f"{final_plot_title} ({plotted_pts_count} pts shown)"
            else:
                 ax.plot(plot_k_data[0,:], plot_k_data[1,:], plot_k_data[2,:], plot_style)
                 final_plot_title = f"{final_plot_title} ({plot_k_data.shape[1]} pts shown)"
        else:
             ax.plot(plot_k_data[0,:], plot_k_data[1,:], plot_k_data[2,:], plot_style)
             final_plot_title = f"{final_plot_title} ({plot_k_data.shape[1]} pts shown)"

        ax.set_xlabel("Kx"); ax.set_ylabel("Ky"); ax.set_zlabel("Kz")
        ax.set_title(final_plot_title)
        return ax

    def plot_2d(self, max_total_points=10000, max_interleaves=None, 
                interleaf_stride=1, point_stride=1, 
                title=None, ax=None, figure=None, plot_style='.-', legend_on=False):
        if self._D < 2:
            print(f"Trajectory '{self.name}' is not at least 2D.")
            return ax if ax is not None else None
        if self._N == 0:
            print(f"Trajectory '{self.name}' has no k-space points to plot.")
            return ax if ax is not None else None
        if ax is None:
            fig = figure if figure else plt.figure()
            ax = fig.add_subplot(111)
        
        k_data = self.kspace_points_rad_per_m

        effective_point_stride = max(1, point_stride)
        num_pts_after_stride = self._N // effective_point_stride if effective_point_stride > 0 else self._N
        num_pts_to_plot = min(num_pts_after_stride, max_total_points)

        final_stride = effective_point_stride
        if num_pts_to_plot < num_pts_after_stride and num_pts_to_plot > 0:
            final_stride = self._N // num_pts_to_plot
        final_stride = max(1, final_stride)

        plot_k_data = k_data[:, ::final_stride] if self._N > 0 else k_data
        final_plot_title = title if title else f"2D K-space: {self.name}"
        actual_lines_plotted = 0

        if 'interleaf_structure' in self.metadata and max_interleaves is not None and self._N > 0:
            num_il_orig, pts_per_il_orig = self.metadata.get('interleaf_structure', (1, self._N))
            if num_il_orig > 0 and pts_per_il_orig > 0 and num_il_orig * pts_per_il_orig == self._N :
                num_ils_to_plot = min(max_interleaves, num_il_orig // max(1,interleaf_stride))
                plotted_pts_count = 0
                for i in range(num_ils_to_plot):
                    il_idx = i * interleaf_stride
                    start = il_idx * pts_per_il_orig
                    end = start + pts_per_il_orig

                    current_segment_full = k_data[:, start:end]
                    current_num_pts_in_seg_after_stride = current_segment_full.shape[1] // effective_point_stride if effective_point_stride > 0 else current_segment_full.shape[1]
                    current_num_pts_to_plot_this_il = min(current_num_pts_in_seg_after_stride, max_total_points - plotted_pts_count)

                    current_final_il_stride = effective_point_stride
                    if current_num_pts_to_plot_this_il < current_num_pts_in_seg_after_stride and current_num_pts_to_plot_this_il > 0:
                         current_final_il_stride = current_segment_full.shape[1] // current_num_pts_to_plot_this_il
                    current_final_il_stride = max(1, current_final_il_stride)

                    current_segment_plot = current_segment_full[:, ::current_final_il_stride]

                    if current_segment_plot.shape[1] > 0:
                        ax.plot(current_segment_plot[0,:], current_segment_plot[1,:], plot_style, label=f"IL {il_idx}" if legend_on else None)
                        plotted_pts_count += current_segment_plot.shape[1]
                        actual_lines_plotted +=1
                    if plotted_pts_count >= max_total_points: break
                final_plot_title = f"{final_plot_title} ({plotted_pts_count} pts shown)"
                if legend_on and actual_lines_plotted > 0 : ax.legend()
            else:
                 ax.plot(plot_k_data[0,:], plot_k_data[1,:], plot_style)
                 final_plot_title = f"{final_plot_title} ({plot_k_data.shape[1]} pts shown)"
        else:
             ax.plot(plot_k_data[0,:], plot_k_data[1,:], plot_style)
             final_plot_title = f"{final_plot_title} ({plot_k_data.shape[1]} pts shown)"

        ax.set_xlabel("Kx"); ax.set_ylabel("Ky")
        ax.set_title(final_plot_title)
        ax.axis('equal')
        return ax

    def plot_voronoi(self, title=None, **kwargs):
        print(f"plot_voronoi for {self.name}: Basic 2D/3D plot shown as Voronoi not fully implemented in this pass.")
        if self._D == 2 and self._N > 3: return self.plot_2d(title=title or "Voronoi (fallback to 2D plot)")
        elif self._D ==3 and self._N > 4: return self.plot_3d(title=title or "Voronoi (fallback to 3D plot)")
        return None
        
    def summary(self):
        print(f"Trajectory Summary: {self.name}")
        print(f"  Dimensions (D): {self.get_num_dimensions()}")
        print(f"  Points (N): {self.get_num_points()}")
        if self.kspace_points_rad_per_m is not None:
             print(f"  K-space shape (D,N): {self.kspace_points_rad_per_m.shape}")
        print(f"  Dwell time (s): {self.dt_seconds}")
        print(f"  Duration (s): {self.get_duration_seconds()}")
        for k,v in self.metadata.items():
            if k not in ['fov_estimate_mm', 'resolution_overall_estimate_mm', 'dead_time_start_samples', 'dead_time_end_samples']:
                 print(f"  Metadata '{k}': {v}")
        pass

def normalize_density_weights(density_weights: np.ndarray) -> np.ndarray:
    """Normalizes density compensation weights."""
    if not isinstance(density_weights, np.ndarray):
        raise TypeError("Input density_weights must be a NumPy array.")
    if density_weights.size == 0:
        return np.array([])
    sum_weights = np.sum(density_weights)
    if np.abs(sum_weights) < 1e-12:
        return np.full(density_weights.shape, 1.0 / density_weights.size if density_weights.size > 0 else 1.0)
    return density_weights / sum_weights

def create_periodic_points(trajectory: np.ndarray, ndim: int) -> np.ndarray:
    """Creates replicated points for periodic boundary conditions."""
    if not (2 <= ndim <= 3):
        raise ValueError("Number of dimensions must be 2 or 3.")
    if trajectory.ndim != 2 or trajectory.shape[1] != ndim: # Expect (N,D)
        raise ValueError(f"Trajectory shape {trajectory.shape} inconsistent with ndim {ndim}")
    
    num_points = trajectory.shape[0]
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
    return extended_trajectory

def compute_cell_area(voronoi: Voronoi, point_index: int, ndim: int) -> float:
    """Computes the area/volume of a Voronoi cell."""
    region_index = voronoi.point_region[point_index]
    if region_index == -1 or not voronoi.regions[region_index]:
        return np.inf # Infinite region or no vertices
    
    cell_vertices_indices = voronoi.regions[region_index]
    if -1 in cell_vertices_indices: # Infinite region
        return np.inf

    cell_vertices = voronoi.vertices[cell_vertices_indices]
    if cell_vertices.shape[0] < ndim + 1: # Not enough vertices
        return 0.0
    try:
        hull = ConvexHull(cell_vertices, qhull_options='QJ') # Joggle for robustness
        return hull.volume
    except QhullError: # Collinear/coplanar points
        return 0.0
    except Exception:
        return np.inf # Other errors, treat as problematic

def compute_voronoi_density(trajectory: np.ndarray, 
                            boundary_type: str = "periodic", 
                            qhull_options: Optional[str] = None) -> np.ndarray:
    if not isinstance(trajectory, np.ndarray): raise TypeError("Trajectory must be a NumPy array.")
    if trajectory.ndim != 2: raise ValueError(f"Trajectory must be 2D (num_points, ndim), got shape {trajectory.shape}")

    num_points, ndim = trajectory.shape
    if ndim not in [2, 3]: raise ValueError(f"Number of dimensions (ndim={ndim}) must be 2 or 3.")
    if num_points == 0: return np.array([])
    if num_points == 1: return np.array([1.0])

    processed_trajectory = np.copy(trajectory)
    for dim_idx in range(ndim): # Normalize to [-0.5, 0.5] for periodic
        min_val, max_val = np.min(processed_trajectory[:, dim_idx]), np.max(processed_trajectory[:, dim_idx])
        if (max_val - min_val) > 1e-9:
            processed_trajectory[:, dim_idx] = (processed_trajectory[:, dim_idx] - min_val) / (max_val - min_val) - 0.5
        else: processed_trajectory[:, dim_idx] = 0.0
    
    unique_pts, unique_indices = np.unique(processed_trajectory, axis=0, return_inverse=True)
    if unique_pts.shape[0] < ndim + 1: # Not enough unique points for Voronoi
        return normalize_density_weights(np.ones(num_points)) # Fallback to uniform

    final_points_for_voronoi = processed_trajectory
    if boundary_type == "periodic":
        final_points_for_voronoi = create_periodic_points(unique_pts, ndim)
    elif boundary_type != "clipped":
        raise ValueError(f"Unknown boundary_type: {boundary_type}. Must be 'periodic' or 'clipped'.")

    default_qhull_options = 'Qbb Qc Qz'
    try:
        vor = Voronoi(final_points_for_voronoi, qhull_options=qhull_options if qhull_options is not None else default_qhull_options)
    except QhullError:
        return normalize_density_weights(np.ones(num_points))

    cell_volumes_unique = np.zeros(unique_pts.shape[0])
    for i in range(unique_pts.shape[0]):
        # Find index of original unique point in Voronoi points list
        # For 'clipped', it's direct. For 'periodic', it's the central replica.
        idx_in_vor_pts = i
        if boundary_type == "periodic":
            offset_idx_for_original = (3**ndim) // 2 # Index of the (0,0,0) offset point
            idx_in_vor_pts = i * (3**ndim) + offset_idx_for_original

        cell_volumes_unique[i] = compute_cell_area(vor, idx_in_vor_pts, ndim)

    finite_volumes = cell_volumes_unique[np.isfinite(cell_volumes_unique)]
    if len(finite_volumes) > 0:
        median_finite_volume = np.median(finite_volumes)
        if median_finite_volume == 0: median_finite_volume = np.mean(finite_volumes) if np.mean(finite_volumes) > 0 else 1.0
        cell_volumes_unique[np.isinf(cell_volumes_unique)] = median_finite_volume
        cell_volumes_unique[np.isnan(cell_volumes_unique)] = median_finite_volume
        if np.all(cell_volumes_unique < 1e-9): cell_volumes_unique.fill(1.0)
    else: # All cells were infinite or NaN
        cell_volumes_unique.fill(1.0)

    density_weights = cell_volumes_unique[unique_indices]
    return normalize_density_weights(density_weights)

def compute_density_compensation(trajectory: np.ndarray,
                                 method: str = "voronoi",
                                 existing_voronoi: Optional[Any] = None, # Not used by current voronoi path
                                 dt_seconds: Optional[float] = None, # Not used by current methods
                                 gamma_Hz_per_T: Optional[float] = None) -> np.ndarray: # Not used
    original_shape = trajectory.shape
    if np.iscomplexobj(trajectory):
        points_real = np.vstack((trajectory.real.flatten(), trajectory.imag.flatten())).T
        output_reshape_target = original_shape
    elif trajectory.ndim == 1:
        points_real = trajectory.reshape(-1, 1)
        output_reshape_target = original_shape
    elif trajectory.ndim == 2:
        points_real = trajectory
        output_reshape_target = (original_shape[0],)
    else: raise ValueError(f"Unsupported trajectory shape: {original_shape}")

    if points_real.size == 0: return np.array([]).reshape(output_reshape_target if output_reshape_target else (0,))

    if method == "voronoi":
        if points_real.shape[1] not in [2,3]: # compute_voronoi_density supports 2D/3D
            # Fallback for 1D (or other) to uniform, or could raise error
            return normalize_density_weights(np.ones(points_real.shape[0])).reshape(output_reshape_target)
        density_weights = compute_voronoi_density(points_real, boundary_type="clipped") # Default to clipped
    elif method == "pipe":
        if points_real.shape[1] != 2: raise ValueError("Pipe method is only supported for 2D trajectories.")
        radii = np.linalg.norm(points_real, axis=1)
        if np.allclose(radii, 0): # All points at origin
            density_weights = np.ones(points_real.shape[0])
        else:
            density_weights = radii
    else: raise ValueError(f"Unknown density compensation method: {method}")
    
    return normalize_density_weights(density_weights).reshape(output_reshape_target)
