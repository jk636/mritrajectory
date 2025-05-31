"""
Defines the core `Trajectory` class and related k-space helper functions.

This module provides the `Trajectory` class, which encapsulates k-space data,
gradient waveforms, and associated metadata for an MRI trajectory. It also includes
functions for calculating density compensation weights (e.g., Voronoi-based) and
other k-space utilities.
"""
import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
from scipy.spatial.qhull import QhullError
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

__all__ = [
    'Trajectory',
    'COMMON_NUCLEI_GAMMA_HZ_PER_T',
    'normalize_density_weights',
    'create_periodic_points',
    'compute_cell_area',
    'compute_voronoi_density',
    'compute_density_compensation'
]

# Gyromagnetic ratios for common nuclei in Hz/T
COMMON_NUCLEI_GAMMA_HZ_PER_T = {
    '1H': 42.576e6, '13C': 10.705e6, '31P': 17.235e6, '19F': 40.052e6,
    '23Na': 11.262e6, '129Xe': 11.777e6, '2H': 6.536e6, '7Li': 16.546e6,
}

class Trajectory:
    """
    Represents an MRI k-space trajectory.

    This class stores k-space coordinates, gradient waveforms (optional),
    timing information, and various metadata. It provides methods for
    calculating trajectory properties, density compensation, plotting,
    and import/export.

    Attributes:
        name (str): Name of the trajectory.
        kspace_points_rad_per_m (np.ndarray): K-space sample coordinates in rad/m,
                                             shape (D, N) where D is dimensions, N is points.
        gradient_waveforms_Tm (Optional[np.ndarray]): Gradient waveforms in T/m,
                                                     shape (D, N). Can be computed if not provided.
        dt_seconds (Optional[float]): Dwell time (time between k-space samples) in seconds.
        metadata (Dict[str, Any]): Dictionary to store various metadata associated
                                   with the trajectory.
        gamma_Hz_per_T (float): Gyromagnetic ratio in Hz/T.
        dead_time_start_seconds (float): Dead time at the beginning of the sequence.
        dead_time_end_seconds (float): Dead time at the end of the sequence.
        _D (int): Number of spatial dimensions.
        _N (int): Number of k-space points.
    """
    def __init__(self, name: str,
                 kspace_points_rad_per_m: np.ndarray,
                 gradient_waveforms_Tm: Optional[np.ndarray] = None,
                 dt_seconds: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds: float = 0.0,
                 dead_time_end_seconds: float = 0.0):
        """
        Initializes a Trajectory object.

        Args:
            name (str): A name for the trajectory.
            kspace_points_rad_per_m (np.ndarray): Array of k-space points (rad/m).
                The input shape is flexible (e.g., (N,D) or (D,N)) and will be
                oriented to (D,N) internally. For 1D data, it can be a flat array.
            gradient_waveforms_Tm (Optional[np.ndarray]): Gradient waveforms (T/m).
                If not provided, they can be computed from k-space points if dt_seconds
                and gamma_Hz_per_T are available. Shape (D,N) or (N,D).
            dt_seconds (Optional[float]): Time step between k-space samples (s).
            metadata (Optional[Dict[str, Any]]): Additional metadata.
            gamma_Hz_per_T (float): Gyromagnetic ratio (Hz/T). Defaults to '1H'.
            dead_time_start_seconds (float): Dead time before k-space acquisition (s).
            dead_time_end_seconds (float): Dead time after k-space acquisition (s).
        """
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
        """Updates metadata dictionary with dead time information."""
        self.metadata['dead_time_start_seconds'] = self.dead_time_start_seconds
        self.metadata['dead_time_end_seconds'] = self.dead_time_end_seconds
        if self.dt_seconds is not None and self.dt_seconds > 0:
            self.metadata['dead_time_start_samples'] = self.dead_time_start_seconds / self.dt_seconds
            self.metadata['dead_time_end_samples'] = self.dead_time_end_seconds / self.dt_seconds
        else:
            self.metadata['dead_time_start_samples'] = None
            self.metadata['dead_time_end_samples'] = None

    def get_gradient_waveforms_Tm(self) -> Optional[np.ndarray]:
        """
        Returns the gradient waveforms in T/m, ensuring shape (D, N).

        If waveforms were provided at initialization, they are returned (potentially
        transposed if detected to be (N,D)). Otherwise, they are computed from
        k-space points if `dt_seconds` and `gamma_Hz_per_T` are available.
        The computed gradients are stored in `self.gradient_waveforms_Tm` for future calls.

        Returns:
            Optional[np.ndarray]: Gradient waveforms in T/m (D,N), or None if not computable.
        """
        if self.gradient_waveforms_Tm is not None:
            if self.gradient_waveforms_Tm.ndim == 2 and self._D is not None and self._N is not None and \
               self.gradient_waveforms_Tm.shape[0] == self._N and \
               self.gradient_waveforms_Tm.shape[1] == self._D and \
               self._D != self._N : # Check if it's transposed and not ambiguous (e.g. square matrix N=D)
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
        """
        Calculates and stores PNS-related metrics in metadata.

        Metrics calculated:
        - `pns_max_abs_gradient_sum_xyz`: Max of the sum of absolute gradient values across dimensions.
        - `pns_max_abs_slew_sum_xyz`: Max of the sum of absolute slew values across dimensions.
        """
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
        """
        Estimates and stores overall spatial resolution in metadata.

        Resolution is estimated as `1 / (2 * max_k_radius)`, where `max_k_radius`
        is the maximum distance from the k-space origin.
        Stored in `metadata['resolution_overall_estimate_m']` and `metadata['resolution_overall_estimate_mm']`.
        """
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
        """Calculates the total duration of the trajectory including dead times."""
        if self.dt_seconds is None: return None
        return self.dead_time_start_seconds + (self._N * self.dt_seconds) + self.dead_time_end_seconds

    def get_max_grad_Tm(self) -> Optional[float]:
        """Returns the maximum gradient amplitude in T/m across all points and dimensions."""
        gradients = self.get_gradient_waveforms_Tm()
        return np.max(np.linalg.norm(gradients, axis=0)) if gradients is not None and gradients.size > 0 else None

    def get_max_slew_Tm_per_s(self) -> Optional[float]:
        """Returns the maximum slew rate in T/m/s from metadata `max_slew_rate_Tm_per_s`."""
        return self.metadata.get('max_slew_rate_Tm_per_s')

    def get_num_points(self) -> int:
        """Returns the number of k-space points (N)."""
        return self._N

    def get_num_dimensions(self) -> int:
        """Returns the number of spatial dimensions (D)."""
        return self._D

    def export(self, filename: str, filetype: Optional[str] = None) -> None:
        """
        Exports trajectory data to a file.

        Supported filetypes: 'csv', 'npy', 'npz', 'txt'.
        If filetype is None, it's inferred from the filename extension.
        NPZ format includes k-space, gradients (if available), dt_seconds, and metadata.
        Other formats save only k-space points (transposed to N,D).

        Args:
            filename (str): The name of the file to save.
            filetype (Optional[str]): The type of file to save ('csv', 'npy', 'npz', 'txt').

        Raises:
            ValueError: If the filetype is unsupported.
        """
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
    def import_from(cls, filename: str) -> 'Trajectory':
        """
        Imports trajectory data from a file.

        Supported filetypes: 'csv', 'npy', 'npz', 'txt'.
        Filetype is inferred from the filename extension.
        NPZ files can contain k-space, gradients, dt_seconds, and metadata.
        Other formats load k-space points (assumed N,D) and use default for other parameters.

        Args:
            filename (str): The name of the file to load.

        Returns:
            Trajectory: A new Trajectory object.

        Raises:
            ValueError: If the filetype is unsupported.
            FileNotFoundError: If the file does not exist (via underlying load functions).
        """
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

    def calculate_voronoi_density(self, force_recompute: bool = False,
                                  qhull_options: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Calculates Voronoi-based density compensation weights.

        The k-space points (D,N) are transposed to (N,D) as expected by
        `compute_voronoi_density`. The results are stored in metadata:
        `self.metadata['density_compensation_weights_voronoi']` and
        `self.metadata['voronoi_cell_sizes']` (currently same as weights).
        The status is stored in `self.metadata['voronoi_calculation_status']`.

        Args:
            force_recompute (bool): If True, recalculates even if results exist.
            qhull_options (Optional[str]): Options for Qhull (Voronoi calculation).

        Returns:
            Optional[np.ndarray]: Computed density weights, or None on error/empty.
                                  Returns an empty array for 0-point trajectories.
        """
        if not force_recompute and self.metadata.get('voronoi_calculation_status') == "Success":
            return self.metadata.get('density_compensation_weights_voronoi')

        self.metadata['voronoi_calculation_status'] = "Starting..."
        self.metadata['voronoi_cell_sizes'] = None
        self.metadata['density_compensation_weights_voronoi'] = None

        if self._N == 0:
            self.metadata['voronoi_calculation_status'] = "Skipped: No k-space points"
            return np.array([])

        # compute_voronoi_density expects (N, D)
        # self.kspace_points_rad_per_m is (D, N)
        kspace_points_nd = self.kspace_points_rad_per_m.T

        if kspace_points_nd.shape[1] == 0: # Should be caught by self._N == 0, but as a safeguard
             self.metadata['voronoi_calculation_status'] = "Skipped: K-space points have no dimensions"
             return np.array([])

        try:
            # Assuming compute_voronoi_density is available in the global scope
            # and handles different dimensions appropriately.
            density_weights = compute_voronoi_density(
                kspace_points_nd,
                qhull_options=qhull_options
            )

            if density_weights is not None and density_weights.size == self._N:
                self.metadata['voronoi_cell_sizes'] = density_weights # Original request, might be different from true "cell sizes"
                self.metadata['density_compensation_weights_voronoi'] = density_weights
                self.metadata['voronoi_calculation_status'] = "Success"
                return density_weights
            elif density_weights is not None and density_weights.size != self._N:
                self.metadata['voronoi_calculation_status'] = f"Error: Mismatch in returned weights size ({density_weights.size}) and number of points ({self._N})"
                return None
            else: # density_weights is None
                 self.metadata['voronoi_calculation_status'] = "Error: compute_voronoi_density returned None"
                 return None

        except Exception as e:
            self.metadata['voronoi_calculation_status'] = f"Error: {str(e)}"
            return None

    def plot_3d(self, max_total_points: int = 2000,
                max_interleaves: Optional[int] = None,
                interleaf_stride: int = 1,
                point_stride: int = 1,
                title: Optional[str] = None,
                ax: Optional[Axes3D] = None,
                figure: Optional[plt.Figure] = None,
                plot_style: str = '.-') -> Optional[Axes3D]:
        """
        Plots a 3D trajectory.

        Args:
            max_total_points (int): Max points to display overall after applying strides.
            max_interleaves (Optional[int]): Max interleaves to plot if 'interleaf_structure'
                                           in metadata.
            interleaf_stride (int): Stride for plotting interleaves.
            point_stride (int): Stride for plotting points within an interleaf/trajectory.
            title (Optional[str]): Plot title.
            ax (Optional[Axes3D]): Matplotlib 3D Axes to plot on. If None, a new one is created.
                                   If a non-3D Axes is passed, it's cleared and replaced by a 3D one.
            figure (Optional[plt.Figure]): Matplotlib Figure to use if `ax` is None.
            plot_style (str): Plotting style string (e.g., '.-', 'o').

        Returns:
            Optional[Axes3D]: The Matplotlib 3D Axes object used for plotting, or None if not plotted
                             (e.g., trajectory is not 3D or has no points).
        """
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

    def plot_2d(self, max_total_points: int = 10000,
                max_interleaves: Optional[int] = None,
                interleaf_stride: int = 1,
                point_stride: int = 1,
                title: Optional[str] = None,
                ax: Optional[plt.Axes] = None,
                figure: Optional[plt.Figure] = None,
                plot_style: str = '.-',
                legend_on: bool = False) -> Optional[plt.Axes]:
        """
        Plots a 2D trajectory (or the first two dimensions of a higher-D trajectory).

        Args:
            max_total_points (int): Max points to display overall after applying strides.
            max_interleaves (Optional[int]): Max interleaves to plot if 'interleaf_structure'
                                           in metadata.
            interleaf_stride (int): Stride for plotting interleaves.
            point_stride (int): Stride for plotting points within an interleaf/trajectory.
            title (Optional[str]): Plot title.
            ax (Optional[plt.Axes]): Matplotlib Axes to plot on. If None, a new one is created.
            figure (Optional[plt.Figure]): Matplotlib Figure to use if `ax` is None.
            plot_style (str): Plotting style string (e.g., '.-', 'o').
            legend_on (bool): If True and interleaves are plotted, a legend is shown.

        Returns:
            Optional[plt.Axes]: The Matplotlib Axes object used for plotting, or None if not plotted
                                (e.g., trajectory is not 2D or has no points).
        """
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

    def plot_voronoi(self, title: Optional[str] = None,
                     ax: Optional[plt.Axes] = None,
                     figure: Optional[plt.Figure] = None,
                     qhull_options: Optional[str] = None,
                     plot_points: bool = True,
                     point_style: str = 'ko',
                     point_size: float = 2,
                     clip_boundary_m: Optional[float] = None) -> Optional[plt.Axes]:
        """
        Plots the Voronoi diagram for 2D trajectories.

        For 3D trajectories, it currently prints a message and shows a 3D scatter plot.
        For 1D trajectories, it plots the points on a line.

        Args:
            title (Optional[str]): Plot title.
            ax (Optional[plt.Axes]): Matplotlib Axes to plot on.
            figure (Optional[plt.Figure]): Matplotlib Figure to use if `ax` is None.
            qhull_options (Optional[str]): Options for Qhull (Voronoi calculation).
            plot_points (bool): If True, overlay k-space points on the Voronoi diagram.
            point_style (str): Style for plotted points.
            point_size (float): Size for plotted points.
            clip_boundary_m (Optional[float]): Radius for clipping infinite Voronoi regions.
                                            If None, infinite regions might not be drawn or
                                            might extend to plot edges.

        Returns:
            Optional[plt.Axes]: The Matplotlib Axes object, or None if plotting fails early.
                                Returns Axes for all valid plot scenarios.
        """

        final_plot_title = title if title else f"Voronoi Diagram: {self.name}"

        if self._N == 0:
            print(f"Trajectory '{self.name}' has no k-space points to plot for Voronoi.")
            if ax is None:
                fig = figure if figure else plt.figure()
                ax = fig.add_subplot(111)
            ax.set_title(final_plot_title + " (No points)")
            ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)")
            return ax

        if self._D == 1:
            print(f"Voronoi plotting for 1D trajectory '{self.name}' is not standard. Plotting points instead.")
            if ax is None:
                fig = figure if figure else plt.figure()
                ax = fig.add_subplot(111)
            k_data = self.kspace_points_rad_per_m
            ax.plot(k_data[0,:], np.zeros_like(k_data[0,:]), point_style, markersize=point_size)
            ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("")
            ax.set_title(final_plot_title + " (1D points)")
            return ax

        if self._D != 2 : # For now, only implement 2D Voronoi
            print(f"Voronoi plot for {self._D}D trajectory '{self.name}' is not implemented. Plotting a scatter of points.")
            if self._D == 3:
                return self.plot_3d(title=final_plot_title + " (3D Scatter Fallback)", ax=ax, figure=figure, plot_style=point_style)
            else: # Fallback for other dimensions if any
                if ax is None:
                    fig = figure if figure else plt.figure()
                    ax = fig.add_subplot(111)
                # Generic scatter for D > 3 or if plot_3d is not suitable
                points_for_plot = self.kspace_points_rad_per_m.T # (N,D)
                ax.scatter(*points_for_plot[:,:2].T, s=point_size, c='k' if point_style.startswith('k') else None)
                ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)")
                ax.set_title(final_plot_title + f" ({self._D}D Scatter Fallback)")
                return ax

        # Proceed with 2D Voronoi Plotting
        if ax is None:
            fig = figure if figure else plt.figure()
            ax = fig.add_subplot(111)

        k_points_nd = self.kspace_points_rad_per_m.T # Expected (N, D) by Voronoi

        if k_points_nd.shape[0] < self._D + 1: # Need at least D+1 points for Voronoi in D dimensions
             print(f"Not enough unique points ({k_points_nd.shape[0]}) for a {self._D}D Voronoi diagram. Plotting points.")
             ax.plot(k_points_nd[:,0], k_points_nd[:,1], point_style, markersize=point_size)
             ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)")
             ax.set_title(final_plot_title + " (Too few points for Voronoi)")
             ax.axis('equal')
             return ax

        default_qhull_options = 'Qbb Qc Qz' # Basic options for robustness
        final_qhull_options = qhull_options if qhull_options is not None else default_qhull_options

        try:
            vor = Voronoi(k_points_nd, qhull_options=final_qhull_options)
        except QhullError as e:
            print(f"QhullError during Voronoi computation for '{self.name}': {e}. Plotting points only.")
            ax.plot(k_points_nd[:,0], k_points_nd[:,1], point_style, markersize=point_size)
            ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)")
            ax.set_title(final_plot_title + " (QhullError, points shown)")
            ax.axis('equal')
            return ax
        except Exception as e: # Catch any other error during Voronoi creation
            print(f"Error during Voronoi computation for '{self.name}': {e}. Plotting points only.")
            ax.plot(k_points_nd[:,0], k_points_nd[:,1], point_style, markersize=point_size)
            ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)")
            ax.set_title(final_plot_title + " (Voronoi Error, points shown)")
            ax.axis('equal')
            return ax

        # Plot Voronoi regions
        patches = []
        min_coord = vor.min_bound if hasattr(vor, 'min_bound') else np.min(vor.vertices, axis=0)
        max_coord = vor.max_bound if hasattr(vor, 'max_bound') else np.max(vor.vertices, axis=0)
        plot_radius_factor = 1.5

        # Determine plot limits for clipping infinite regions
        # If clip_boundary_m is given, use that. Otherwise, estimate from points.
        if clip_boundary_m is not None and clip_boundary_m > 0:
            visible_min = -clip_boundary_m
            visible_max = clip_boundary_m
            plot_center = np.array([0.0, 0.0])
        else:
            # Fallback if clip_boundary_m is not provided: estimate from point cloud
            ptp_bound = np.max(vor.points, axis=0) - np.min(vor.points, axis=0)
            plot_center = np.mean(vor.points, axis=0)
            # Ensure plot_radius is not zero, e.g. for single point or colinear points
            plot_radius = max(np.max(ptp_bound) * plot_radius_factor, 1e-3)
            visible_min = plot_center - plot_radius
            visible_max = plot_center + plot_radius


        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not -1 in region: # Finite region
                polygon_verts = vor.vertices[region]
                patches.append(Polygon(polygon_verts, closed=True))
            else: # Infinite region, try to clip (basic clipping)
                # Get ridges for this point
                point_idx = np.where(vor.point_region == region_idx)[0][0]

                # Filter ridges connected to this point
                ridges_for_point = []
                for ridge_points_indices, ridge_vertices_indices in zip(vor.ridge_points, vor.ridge_vertices):
                    if point_idx in ridge_points_indices:
                        # Check if this ridge involves an infinite vertex
                        if -1 in ridge_vertices_indices:
                             ridges_for_point.append((ridge_points_indices, ridge_vertices_indices))

                if not ridges_for_point: continue # Should not happen for infinite regions typically

                # Sort vertices of the region, handling -1s by finding intersections with boundary
                # This part is complex. For now, we'll skip drawing complex infinite regions or use a simpler method.
                # A very simple approach: don't draw infinite regions if they are too complex or extend too far.
                # Or, create a large bounding box and clip polygon there.
                # For now, let's just skip drawing infinite regions if clipping boundary not specified.
                if clip_boundary_m is None:
                    continue

                # A more robust way for infinite regions (simplified):
                # Use the finite vertices and create new vertices at the intersection of ridges with the bounding box.
                # This is non-trivial. For this implementation, we will rely on matplotlib's default behavior for patches
                # that might go out of bounds, or we can choose not to plot them if clip_boundary_m is not set.
                # For now, we are skipping them if clip_boundary_m is None
                # If clip_boundary_m is set, we could attempt to draw them, but they might look odd without proper clipping.
                # A simple fill will often fail for non-convex or open polygons.
                # A better approach for infinite regions involves computing intersections with a bounding box.
                # This is non-trivial. For now, let's use a simpler representation or skip.
                # Fallback: simply don't draw infinite regions for now, unless a clipping boundary is provided.
                # This part needs a more robust geometric library or algorithm for proper clipping.
                pass # Skip infinite regions if no clip_boundary_m


        if patches: # Only add collection if there are finite polygons to draw
             p = PatchCollection(patches, alpha=0.4, edgecolor='gray', facecolor='lightblue') # Example styling
             ax.add_collection(p)

        if plot_points:
            ax.plot(vor.points[:,0], vor.points[:,1], point_style, markersize=point_size)

        ax.set_xlabel("Kx (rad/m)"); ax.set_ylabel("Ky (rad/m)")
        ax.set_title(final_plot_title)

        # Set plot limits
        if clip_boundary_m is not None and clip_boundary_m > 0:
            ax.set_xlim([visible_min, visible_max])
            ax.set_ylim([visible_min, visible_max])
        else: # Auto-scale based on Voronoi vertices or points
            # Use min/max of Voronoi vertices if available and finite, else use points
            finite_vertices = vor.vertices[np.all(np.isfinite(vor.vertices), axis=1)]
            if finite_vertices.size > 0:
                 ax.set_xlim([np.min(finite_vertices[:,0]), np.max(finite_vertices[:,0])])
                 ax.set_ylim([np.min(finite_vertices[:,1]), np.max(finite_vertices[:,1])])
            elif vor.points.size >0 : # Fallback to points if no finite vertices
                 ax.set_xlim([np.min(vor.points[:,0]), np.max(vor.points[:,0])])
                 ax.set_ylim([np.min(vor.points[:,1]), np.max(vor.points[:,1])])
            # ax.axis('equal') # 'equal' axis can make clipped regions look very large if plot_radius is small

        return ax
        
    def summary(self) -> None:
        """Prints a summary of the trajectory's properties and metadata to stdout."""
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
