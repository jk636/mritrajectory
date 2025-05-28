import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial.qhull import QhullError


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


import numpy as np

import numpy as np

class Trajectory:
    """
    Container for a k-space trajectory and associated data.
    Includes additional trajectory metrics like max PNS, max slew, FOV, and resolution.
    """
    def __init__(self, name, kspace_points_rad_per_m, 
                 gradient_waveforms_Tm=None, dt_seconds=None, 
                 metadata=None, gamma_Hz_per_T=42.576e6,
                 dead_time_start_seconds=0.0, dead_time_end_seconds=0.0):
        """
        Args:
            name (str): Trajectory name/description.
            kspace_points_rad_per_m (np.ndarray): [D, N] or [N, D] k-space coordinates in rad/m.
            gradient_waveforms_Tm (np.ndarray, optional): [D, N] or [N, D] gradient waveforms in T/m.
                                                         If None, will be computed on demand.
            dt_seconds (float, optional): Dwell/sample time in seconds. Required if gradients are to be computed.
            metadata (dict, optional): Additional information.
            gamma_Hz_per_T (float, optional): Gyromagnetic ratio. Defaults to 42.576e6 Hz/T (for protons).
            dead_time_start_seconds (float, optional): Dead time at the beginning of the trajectory. Defaults to 0.0.
            dead_time_end_seconds (float, optional): Dead time at the end of the trajectory. Defaults to 0.0.
        """
        self.name = name
        self.kspace_points_rad_per_m = np.array(kspace_points_rad_per_m)
        # gradient_waveforms_Tm now acts as a cache: initially stores provided gradients,
        # or None if not provided, to be computed later by get_gradient_waveforms_Tm()
        # Use self.gradient_waveforms_Tm as the cache. Initialize with provided gradients or None.
        self.gradient_waveforms_Tm = np.array(gradient_waveforms_Tm) if gradient_waveforms_Tm is not None else None
        self.dt_seconds = dt_seconds
        self.metadata = metadata or {}
        
        # Store dead times as attributes
        self.dead_time_start_seconds = dead_time_start_seconds
        self.dead_time_end_seconds = dead_time_end_seconds

        # Prioritize gamma from metadata if it exists (e.g. from file import), else use provided or default.
        if 'gamma_Hz_per_T' not in self.metadata:
             self.metadata['gamma_Hz_per_T'] = gamma_Hz_per_T
        
        self._update_dead_time_metadata() # Populate metadata with deadtime info

        # Automatically populate additional metrics if possible
        self._compute_metrics() # This will call slew, pns, fov, resolution which might use dt.

    def _update_dead_time_metadata(self):
        """Updates metadata with dead time information in seconds and samples."""
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
        """
        Returns the gradient waveforms in T/m.
        If not provided at initialization, they are computed from k-space points and dt, then cached.
        The cached gradients are consistently stored in [D, N] orientation.
        """
        if self.gradient_waveforms_Tm is not None: # Check cache first
            return self.gradient_waveforms_Tm

        if self.dt_seconds is None or self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            return None

        k_data = np.array(self.kspace_points_rad_per_m)
        D = self.get_num_dimensions()
        N = self.get_num_points()

        k_for_gradient = k_data
        # Ensure k_for_gradient is [D, N] for gradient calculation along axis=-1
        if k_data.ndim == 2 and k_data.shape[0] == N and k_data.shape[1] == D: # Actual [N, D]
            if N != 0 and D != 0 : k_for_gradient = k_data.T
        elif k_data.ndim == 1 and N == k_data.shape[0] and D == 1: # Actual 1D array [N], treat as [1,N]
            k_for_gradient = k_data.reshape(1, N)
        elif k_data.ndim == 2 and k_data.shape[0] == D and k_data.shape[1] == N: # Actual [D, N]
            pass # Already in correct [D,N] orientation
        elif k_data.ndim == 1 and D == k_data.shape[0] and N == 1: # Actual 1D array [D] representing a single point, treat as [D,1]
             k_for_gradient = k_data.reshape(D, 1)
        else:
            # Fallback or ambiguous shape. If N or D is 0, it might also lead here.
            # print(f"Warning: Ambiguous k-space data shape {k_data.shape} for gradient computation (D={D}, N={N}).")
            if N==0 or D==0: # Cannot compute gradient if one dimension is zero.
                self.gradient_waveforms_Tm = np.array([]).reshape(D,N) # Store empty array of correct shape
                return self.gradient_waveforms_Tm
            # If still ambiguous, return None as we cannot safely proceed.
            return None
        
        gamma = self.metadata.get('gamma_Hz_per_T', 42.576e6)
        if gamma == 0: gamma = 42.576e6 # Avoid division by zero

        computed_gradients = None
        if k_for_gradient.shape[-1] < 2: # N < 2 (i.e., 0 or 1 point)
            # For single or zero point trajectories, gradient is zero.
            computed_gradients = np.zeros_like(k_for_gradient)
        else:
            try:
                computed_gradients = np.gradient(k_for_gradient, self.dt_seconds, axis=-1) / gamma
            except ValueError as e: # Should be caught by N < 2, but as a safeguard
                # print(f"Warning: np.gradient failed for shape {k_for_gradient.shape}: {e}")
                # Attempt to return zeros if k_for_gradient is valid, otherwise re-raise or return None
                if hasattr(k_for_gradient, 'shape'):
                    computed_gradients = np.zeros_like(k_for_gradient)
                else:
                    return None # Cannot even form zeros.
            
        self.gradient_waveforms_Tm = computed_gradients # Cache the result (now always [D,N] or empty)
        return self.gradient_waveforms_Tm

    def _calculate_slew_rate(self):
        """Calculates slew rate and stores it in metadata. Assumes gradients are [D,N]."""
        gradients = self.get_gradient_waveforms_Tm() # Should be [D,N]
        if gradients is not None and gradients.size > 0 and self.dt_seconds is not None and gradients.shape[-1] > 1: # N > 1
            slew = np.diff(gradients, axis=-1) / self.dt_seconds # diff along N axis
            max_slew_rate_Tm_per_s = np.max(np.linalg.norm(slew, axis=0)) # norm over D axis
            self.metadata['max_slew_rate_Tm_per_s'] = max_slew_rate_Tm_per_s
        else:
            self.metadata['max_slew_rate_Tm_per_s'] = None if gradients is None else 0.0


    def _calculate_pns(self):
        """Calculates PNS metrics and stores them in metadata. Assumes gradients are [D,N]."""
        gradients = self.get_gradient_waveforms_Tm() # Should be [D,N]
        if gradients is not None and gradients.size > 0 and self.dt_seconds is not None:
            abs_grad_sum = np.sum(np.abs(gradients), axis=0) # Sum over D axis
            self.metadata['pns_max_abs_gradient_sum_xyz'] = np.max(abs_grad_sum)

            if gradients.shape[-1] > 1: # N > 1 for slew calculation
                slew = np.diff(gradients, axis=-1) / self.dt_seconds # diff along N axis
                abs_slew_sum = np.sum(np.abs(slew), axis=0) # Sum over D axis
                self.metadata['pns_max_abs_slew_sum_xyz'] = np.max(abs_slew_sum)
            else: # Single point, slew is zero.
                self.metadata['pns_max_abs_slew_sum_xyz'] = 0.0
        else:
            self.metadata['pns_max_abs_gradient_sum_xyz'] = None
            self.metadata['pns_max_abs_slew_sum_xyz'] = None

    def _calculate_fov(self):
        """Calculates FOV and stores it in metadata."""
        if self.kspace_points_rad_per_m is not None:
            # Ensure points are [D, N]
            points = self.kspace_points_rad_per_m
            if points.shape[0] > points.shape[1] and points.shape[0] > 3 : # Heuristic for N, D vs D, N
                 points = points.T

            k_extent_rad_per_m = np.max(np.abs(points), axis=-1)
            # Add a small epsilon to prevent division by zero if k_extent is 0 for a dimension
            fov_m = 1 / (2 * k_extent_rad_per_m + 1e-9)
            self.metadata['fov_estimate_m'] = fov_m.tolist()
            self.metadata['fov_estimate_mm'] = (fov_m * 1e3).tolist()
        else:
            self.metadata['fov_estimate_m'] = None
            self.metadata['fov_estimate_mm'] = None

    def _calculate_resolution(self):
        """Calculates resolution and stores it in metadata."""
        if self.kspace_points_rad_per_m is not None:
            # Ensure points are [D, N]
            points = self.kspace_points_rad_per_m
            if points.shape[0] > points.shape[1] and points.shape[0] > 3 : # Heuristic for N, D vs D, N
                 points = points.T

            # For anisotropic resolution, we might consider resolution along each k-space axis.
            # A common definition is related to the max k-space extent along each axis.
            # However, for a general trajectory, "resolution" is often simplified to 1 / (2 * max_k_radius)
            # Here we'll provide both: an estimate per dimension and an overall estimate.

            # Overall resolution based on max k-space radius
            max_k_radius_rad_per_m = np.max(np.linalg.norm(points, axis=0))
            resolution_m_overall = 1 / (2 * max_k_radius_rad_per_m + 1e-9) # meters
            self.metadata['resolution_overall_estimate_m'] = resolution_m_overall
            self.metadata['resolution_overall_estimate_mm'] = resolution_m_overall * 1e3

            # Per-dimension resolution estimate (can be debated, this is one way)
            # This is similar to FOV calculation but for resolution it's 1 / (2 * max_k_coord_on_axis)
            # This might not be the most standard definition for arbitrary trajectories but can be useful.
            # For now, let's stick to the overall resolution as it's more common for non-Cartesian.
            # If anisotropic definition is strictly needed, it would be more like:
            # delta_k = np.max(points, axis=1) - np.min(points, axis=1) # k-space coverage along each axis
            # resolution_anisotropic_m = 1 / (delta_k + 1e-9)
            # self.metadata['resolution_anisotropic_estimate_m'] = resolution_anisotropic_m.tolist()
            # self.metadata['resolution_anisotropic_estimate_mm'] = (resolution_anisotropic_m * 1e3).tolist()
            # For now, only overall resolution is stored.
        else:
            self.metadata['resolution_overall_estimate_m'] = None
            self.metadata['resolution_overall_estimate_mm'] = None


    def _compute_metrics(self):
        """Computes all trajectory metrics."""
        # self._update_dead_time_metadata() # Already called in __init__ after crucial attrs are set
        self._calculate_slew_rate()
        self._calculate_pns()
        self._calculate_fov()
        self._calculate_resolution()

    def get_duration_seconds(self) -> Optional[float]:
        """
        Returns total trajectory duration in seconds, including dead times.
        Returns None if dt_seconds is not defined.
        """
        if self.dt_seconds is None: # Cannot calculate sampling duration
            # If there's deadtime but no sampling time, duration is ambiguous.
            # Return None, or sum of deadtimes if that's meaningful.
            # For now, if sampling duration is undefined, total duration is undefined.
            return None

        num_points = self.get_num_points()
        # If kspace_points_rad_per_m is None or empty, get_num_points() should handle it (e.g. return 0).
        # If num_points is 0 (e.g. empty kspace), sampling_duration is 0.
        sampling_duration = num_points * self.dt_seconds
        
        total_duration = self.dead_time_start_seconds + sampling_duration + self.dead_time_end_seconds
        return total_duration

    def get_max_grad_Tm(self) -> Optional[float]:
        """Returns the maximum absolute gradient amplitude (T/m)."""
        gradients = self.get_gradient_waveforms_Tm()
        if gradients is not None:
            return np.max(np.linalg.norm(gradients, axis=0))
        return None

    def get_max_slew_Tm_per_s(self) -> Optional[float]:
        """Returns the maximum absolute slew rate (T/m/s)."""
        # This value is calculated and stored in metadata by _calculate_slew_rate
        if 'max_slew_rate_Tm_per_s' in self.metadata and self.metadata['max_slew_rate_Tm_per_s'] is not None:
            return self.metadata['max_slew_rate_Tm_per_s']
        # Fallback if not in metadata or called before _compute_metrics, try to compute if possible
        gradients = self.get_gradient_waveforms_Tm()
        if gradients is not None and self.dt_seconds is not None and gradients.shape[-1] > 1:
            slew = np.diff(gradients, axis=-1) / self.dt_seconds
            return np.max(np.linalg.norm(slew, axis=0))
        return None

    def get_num_points(self) -> int:
        """Returns the number of k-space points."""
        # Assuming kspace_points_rad_per_m is [D, N] or [N,D]
        # If [D,N], N is shape[1]. If [N,D], N is shape[0], assuming D < N.
        # A common convention is D <= 3.
        if self.kspace_points_rad_per_m.shape[0] <= 3 or self.kspace_points_rad_per_m.shape[0] < self.kspace_points_rad_per_m.shape[1]:
            return self.kspace_points_rad_per_m.shape[1]
        return self.kspace_points_rad_per_m.shape[0]


    def get_num_dimensions(self) -> int:
        """Returns the number of spatial dimensions."""
        # Assuming kspace_points_rad_per_m is [D, N] or [N,D]
        # If [D,N], D is shape[0]. If [N,D], D is shape[1].
        if self.kspace_points_rad_per_m.shape[0] <= 3 or self.kspace_points_rad_per_m.shape[0] < self.kspace_points_rad_per_m.shape[1]:
            return self.kspace_points_rad_per_m.shape[0]
        return self.kspace_points_rad_per_m.shape[1]


    def export(self, filename, filetype=None):
        """
        Export trajectory to file (CSV, .npy, .npz, .txt).
        Args:
            filename (str): Output file name.
            filetype (str, optional): 'csv', 'npy', 'npz', or 'txt'. Inferred from extension if not given.
        """
        if filetype is None:
            if filename.endswith('.csv'):
                filetype = 'csv'
            elif filename.endswith('.npy'):
                filetype = 'npy'
            elif filename.endswith('.npz'):
                filetype = 'npz'
            else: # Default to text if extension is unknown or not provided and not one of the above
                filetype = 'txt'
        
        # Ensure points are [N, D] for export (unless 1D)
        points_original_ ήταν_DN = False
        points_to_export = np.array(self.kspace_points_rad_per_m)

        if points_to_export.ndim == 2:
            # Heuristic from get_num_dimensions/get_num_points:
            # If shape[0] is D (<=3 or less than shape[1]), it's [D,N]
            # This is the check for [D,N] orientation.
            if (points_to_export.shape[0] <= 3 or points_to_export.shape[0] < points_to_export.shape[1]) and \
               points_to_export.shape[0] == self.get_num_dimensions() and \
               points_to_export.shape[1] == self.get_num_points():
                points_original_ήταν_DN = True
                points_to_export = points_to_export.T # Export k-space as [N,D]

        # Gradients from getter are always [D,N] (or empty with D,N shape, or None)
        gradients_from_getter = self.get_gradient_waveforms_Tm()
        gradients_to_export = gradients_from_getter

        if gradients_from_getter is not None and gradients_from_getter.ndim == 2:
            if points_original_ήταν_DN:
                # K-space was [D,N] and was transposed to [N,D] for export.
                # Gradients (which are [D,N]) must also be transposed to [N,D].
                gradients_to_export = gradients_from_getter.T
            else:
                # K-space was originally [N,D] (or 1D). Gradients from getter are [D,N].
                # To make gradients [N,D] for export, they need to be transposed.
                # (This assumes original k-space was [N,D] and gradients are [D,N])
                if points_to_export.ndim == 2 and \
                   points_to_export.shape[0] == self.get_num_points() and \
                   points_to_export.shape[1] == self.get_num_dimensions(): # k-space is N,D
                     if gradients_from_getter.shape[0] == self.get_num_dimensions() and \
                        gradients_from_getter.shape[1] == self.get_num_points(): # Gradients are D,N
                        gradients_to_export = gradients_from_getter.T
        
        # For 1D k-space (e.g. [N_points]), points_to_export is [N_points].
        # gradients_from_getter would be [1, N_points].
        # gradients_to_export should ideally be [N_points] or [N_points, 1] for consistency if saved.
        if points_to_export.ndim == 1 and gradients_to_export is not None and gradients_to_export.ndim == 2:
            if gradients_to_export.shape[0] == 1 : # Gradients are [1, N]
                gradients_to_export = gradients_to_export.reshape(gradients_to_export.shape[1]) # Make it [N]
            elif gradients_to_export.shape[1] == 1: # Gradients are [N, 1] (unlikely from getter)
                gradients_to_export = gradients_to_export.reshape(gradients_to_export.shape[0]) # Make it [N]


        if filetype == 'csv':
            np.savetxt(filename, points_to_export, delimiter=',')
            # Consider saving gradients to a separate file if needed for CSV:
            # if gradients_to_export is not None:
            #    np.savetxt(filename.replace('.csv', '_grad.csv'), gradients_to_export, delimiter=',')
        elif filetype == 'npy':
            # Standard practice for .npy is to save the array as is.
            # For multiple arrays, .npz is better.
            # If saving both, use npz. If only k-space, this is fine.
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
        """
        Import a trajectory from a file.
        """
        if filename.endswith('.csv') or filename.endswith('.txt'):
            points = np.loadtxt(filename, delimiter=',' if filename.endswith('.csv') else None)
            # Assuming imported points are N, D.
            # A more robust check could be added here if D > N and D > 3 is a possible case for CSV/TXT.
            return cls(name=filename, kspace_points_rad_per_m=points)
        elif filename.endswith('.npy'):
            points = np.load(filename)
            # .npy could be D,N or N,D. The constructor will handle it.
            return cls(name=filename, kspace_points_rad_per_m=points)
        elif filename.endswith('.npz'):
            data = np.load(filename, allow_pickle=True)
            # Convert legacy keys if present, prioritizing new names
            points_key = 'kspace_points_rad_per_m' if 'kspace_points_rad_per_m' in data else 'points' if 'points' in data else 'kspace'
            gradients_key = 'gradient_waveforms_Tm' if 'gradient_waveforms_Tm' in data else 'gradients'
            dt_key = 'dt_seconds' if 'dt_seconds' in data else 'dt'
            
            points = data[points_key]
            # NPZ may store gradients as None, handle this
            gradients_data = data.get(gradients_key)
            gradients = np.array(gradients_data) if gradients_data is not None else None
            
            dt_data = data.get(dt_key)
            dt = dt_data.item() if dt_data is not None and hasattr(dt_data, 'item') else dt_data

            # Ensure metadata is a dictionary, even if stored as an array by older versions
            metadata_raw = data.get('metadata')
            metadata_dict = {}
            if metadata_raw is not None:
                try:
                    # Attempt to convert to dict if it's a structured array or similar
                    # Check if item() is callable before calling, for some np array types
                    metadata_dict = metadata_raw.item() if hasattr(metadata_raw, 'item') and callable(metadata_raw.item) else dict(metadata_raw)
                except (TypeError, AttributeError, ValueError): # Added ValueError for cases like `dict(np.array([1]))`
                    if isinstance(metadata_raw, dict):
                        metadata_dict = metadata_raw
                    else: 
                        # print(f"Warning: Could not parse metadata from npz, type: {type(metadata_raw)}")
                        metadata_dict = {'raw_metadata': metadata_raw} if metadata_raw is not None else {}
            
            gamma_from_file = metadata_dict.get('gamma_Hz_per_T')
            dts_s_from_file = metadata_dict.get('dead_time_start_seconds', 0.0)
            dte_s_from_file = metadata_dict.get('dead_time_end_seconds', 0.0)
            # Note: dead_time_..._samples will be recalculated in __init__ via _update_dead_time_metadata

            traj_instance = cls(name=filename, kspace_points_rad_per_m=points, 
                               gradient_waveforms_Tm=gradients, dt_seconds=dt, 
                               metadata=metadata_dict, 
                               gamma_Hz_per_T=gamma_from_file if gamma_from_file is not None else 42.576e6,
                               dead_time_start_seconds=dts_s_from_file,
                               dead_time_end_seconds=dte_s_from_file)
            # Instance attributes (dead_time_start_samples etc.) are set by _update_dead_time_metadata in __init__
            return traj_instance
        else:
            raise ValueError(f"Unsupported filetype or extension for: {filename}")

    def calculate_voronoi_density(self, force_recompute=False, qhull_options=None):
        """
        Calculates the Voronoi cell size for each k-space point.

        Args:
            force_recompute (bool): If True, recalculate even if results are cached.
            qhull_options (str, optional): Additional Qhull options for Voronoi calculation.
                                           Defaults to 'Qbb Qc Qz' for 2D/3D.

        Returns:
            np.ndarray: Array of Voronoi cell sizes (area for 2D, volume for 3D)
                        corresponding to each k-space point. Values can be np.inf
                        for unbounded cells or np.nan for calculation errors/degenerate cells.
                        Returns None if calculation is not possible or fails globally.
        """
        if not force_recompute and 'voronoi_cell_sizes' in self.metadata:
            return self.metadata['voronoi_cell_sizes']

        self.metadata.pop('voronoi_cell_sizes', None) # Clear previous results
        self.metadata['voronoi_calculation_status'] = "Not Attempted"

        if self.kspace_points_rad_per_m is None or self.kspace_points_rad_per_m.size == 0:
            self.metadata['voronoi_calculation_status'] = "Error: K-space data is empty or None."
            return None

        k_data_original = np.array(self.kspace_points_rad_per_m)
        D = self.get_num_dimensions()
        N = self.get_num_points()

        # Voronoi expects points as (N, D)
        points_ND = k_data_original
        if k_data_original.shape[0] == D and k_data_original.shape[1] == N and D != N : # Check if it's likely D, N and not ambiguous square
            points_ND = k_data_original.T
        elif k_data_original.ndim == 1 and D == 1: # Single dimension data [N]
             points_ND = k_data_original.reshape(-1,1)
        elif points_ND.shape[0] != N or points_ND.shape[1] != D :
             # This case might occur if D==N, and the heuristic was ambiguous.
             # Or if the input shape is truly strange.
             # Assuming if D==N, the input is already N,D or D,N.
             # If input is D,N and D==N, transposing is not harmful if already N,D,
             # but might be wrong if it was truly D,N.
             # A common convention is N > D. If N==D, Voronoi will run, but interpretation of D,N vs N,D matters.
             # Let's assume if N==D, points are already (N,D).
             # If after potential transpose, shape is still not (N,D), then error.
            if points_ND.shape[0] != N or points_ND.shape[1] != D:
                 self.metadata['voronoi_calculation_status'] = f"Error: Ambiguous k-space data shape {k_data_original.shape} for (N,D) format. N={N}, D={D}."
                 return None


        if N < D + 1:
            self.metadata['voronoi_calculation_status'] = f"Error: Not enough points ({N}) for Voronoi in {D}D (need at least {D+1})."
            return None

        if D not in [2, 3]:
            self.metadata['voronoi_calculation_status'] = f"Error: Voronoi calculation only supported for 2D/3D (D={D})."
            return None
        
        # Deduplicate points for Voronoi, as it can fail with duplicates
        # Keep track of original indices to map results back
        unique_points, unique_indices = np.unique(points_ND, axis=0, return_inverse=True)
        
        if unique_points.shape[0] < D + 1:
            self.metadata['voronoi_calculation_status'] = f"Error: Not enough unique points ({unique_points.shape[0]}) for Voronoi in {D}D."
            return None

        cell_sizes_unique = np.full(unique_points.shape[0], np.nan)

        try:
            default_qhull_options = 'Qbb Qc Qz' # Qbb prevents issues with very large coords, Qc keeps coplanar points, Qz handles co-spherical
            vor = Voronoi(unique_points, qhull_options=qhull_options if qhull_options is not None else default_qhull_options)
            
            for i in range(unique_points.shape[0]):
                region_idx = vor.point_region[i]
                vertex_indices = vor.regions[region_idx]

                if -1 in vertex_indices or not vertex_indices: # Unbounded or empty region
                    cell_sizes_unique[i] = np.inf
                    continue

                region_vertices = vor.vertices[vertex_indices]

                if region_vertices.shape[0] < D + 1: # Not enough vertices for a hull
                    cell_sizes_unique[i] = 0.0 # Degenerate cell, effectively zero area/volume
                    continue
                
                try:
                    # QJ can help with precision issues for ConvexHull, especially in 3D
                    hull_qhull_options = 'QJ' if D > 1 else None # QJ only for 2D+
                    current_hull = ConvexHull(region_vertices, qhull_options=hull_qhull_options)
                    cell_sizes_unique[i] = current_hull.volume # .volume is area in 2D, volume in 3D
                except QhullError: # If ConvexHull fails for a specific region
                    cell_sizes_unique[i] = np.nan # Mark as error for this cell
                except Exception: # Other errors during hull calculation
                    cell_sizes_unique[i] = np.nan


            # Map unique cell sizes back to original points
            cell_sizes = cell_sizes_unique[unique_indices]
            self.metadata['voronoi_cell_sizes'] = cell_sizes
            self.metadata['voronoi_calculation_status'] = "Success"
            return cell_sizes

        except QhullError as e:
            self.metadata['voronoi_calculation_status'] = f"Error: QhullError during Voronoi: {e}"
            return None
        except Exception as e: # Catch any other unexpected errors
            self.metadata['voronoi_calculation_status'] = f"Error: Unexpected error during Voronoi: {e}"
            return None

    def summary(self):
        """
        Print a detailed summary of the trajectory, including its properties and calculated metrics.
        """
        num_dims = self.get_num_dimensions()
        num_points = self.get_num_points()
        duration_ms = self.get_duration_seconds() * 1e3 if self.get_duration_seconds() is not None else "N/A"
        
        print(f"\n--- Trajectory Summary: '{self.name}' ---")
        print(f"  Dimensions: {num_dims}D")
        print(f"  Number of Points: {num_points}")
        print(f"  Duration: {duration_ms:.2f} ms" if isinstance(duration_ms, float) else f"Duration: {duration_ms}")
        print(f"  Dwell Time (dt): {self.dt_seconds * 1e6:.2f} µs" if self.dt_seconds is not None else "Dwell Time (dt): N/A")
        print(f"  Gamma: {self.metadata.get('gamma_Hz_per_T', 'N/A'):.2e} Hz/T" if isinstance(self.metadata.get('gamma_Hz_per_T'), (float, int)) else f"  Gamma: {self.metadata.get('gamma_Hz_per_T', 'N/A')}")

        # Use the getter for gradients to ensure they are computed if needed
        current_gradients = self.get_gradient_waveforms_Tm()
        if current_gradients is not None:
            print("\n  Gradients:")
            # get_max_grad_Tm and get_max_slew_Tm_per_s now use the getter internally
            max_grad_mT_m = self.get_max_grad_Tm() * 1e3 if self.get_max_grad_Tm() is not None else "N/A"
            max_slew_Tm_s = self.get_max_slew_Tm_per_s() # Already computed by _compute_metrics
            max_slew_Tm_s_val = "N/A"
            if isinstance(max_slew_Tm_s, (float, np.floating)):
                 max_slew_Tm_s_val = f"{max_slew_Tm_s:.2f} T/m/s"
            elif max_slew_Tm_s is not None : # if it's a string or other non-float from metadata
                 max_slew_Tm_s_val = str(max_slew_Tm_s)


            print(f"    Max Gradient Amplitude: {max_grad_mT_m:.2f} mT/m" if isinstance(max_grad_mT_m, float) else f"    Max Gradient Amplitude: {max_grad_mT_m}")
            print(f"    Max Slew Rate: {max_slew_Tm_s_val}")
        else:
            print("\n  Gradients: Not available or not computed.")

        print("\n  Calculated Metrics (from metadata):")
        if not self.metadata:
            print("    No metadata computed or stored.")
        else:
            for key, value in self.metadata.items():
                if value is None:
                    print(f"    {key}: N/A")
                elif isinstance(value, list) and all(isinstance(item, (float, np.floating, int))) :
                    unit = ""
                    if "_mm" in key: unit = "mm"
                    elif "_m" in key: unit = "m"
                    elif "_Tm_per_s" in key: unit = "T/m/s"
                    elif "_xyz" in key: unit = "a.u." # Assuming arbitrary units for PNS sums for now
                    
                    # Format numbers: scientific for small/large, fixed for others
                    formatted_values = []
                    for v_item in value:
                        if abs(v_item) < 1e-3 or abs(v_item) > 1e4 and v_item != 0:
                             formatted_values.append(f"{v_item:.2e}")
                        else:
                             formatted_values.append(f"{v_item:.3f}")
                    value_str = ", ".join(formatted_values)
                    print(f"    {key}: [{value_str}] {unit}")
                elif isinstance(value, (float, np.floating, int)):
                    unit = ""
                    if "_mm" in key: unit = "mm"
                    elif "_m" in key: unit = "m"
                    elif "_Tm_per_s" in key: unit = "T/m/s"
                    elif "_xyz" in key: unit = "a.u."

                    if abs(value) < 1e-3 or abs(value) > 1e4 and value != 0:
                        print(f"    {key}: {value:.2e} {unit}")
                    else:
                        print(f"    {key}: {value:.3f} {unit}")
                else:
                    print(f"    {key}: {value}")
        
        if 'voronoi_calculation_status' in self.metadata:
            print(f"\n  Voronoi Density Calculation Status: {self.metadata['voronoi_calculation_status']}")
            if self.metadata['voronoi_calculation_status'] == "Success" and 'voronoi_cell_sizes' in self.metadata:
                sizes = self.metadata['voronoi_cell_sizes']
                finite_sizes = sizes[np.isfinite(sizes) & ~np.isnan(sizes)]
                num_inf = np.sum(np.isinf(sizes))
                num_nan = np.sum(np.isnan(sizes))
                print(f"    Voronoi Cell Sizes ({len(sizes)} total):")
                if finite_sizes.size > 0:
                    print(f"      Finite Cells ({len(finite_sizes)}): Mean={np.mean(finite_sizes):.2e}, Median={np.median(finite_sizes):.2e}")
                    print(f"                     Min={np.min(finite_sizes):.2e}, Max={np.max(finite_sizes):.2e}, Std={np.std(finite_sizes):.2e}")
                if num_inf > 0:
                    print(f"      Unbounded Cells (np.inf): {num_inf}")
                if num_nan > 0:
                    print(f"      Error/Degenerate Cells (np.nan): {num_nan}")
        
        print("--- End of Summary ---")
        Args:
            filename (str): Output file name.
            filetype (str, optional): 'csv', 'npy', 'npz', or 'txt'. Inferred from extension if not given.
        
        if filetype is None:
            if filename.endswith('.csv'):
                filetype = 'csv'
            elif filename.endswith('.npy'):
                filetype = 'npy'
            elif filename.endswith('.npz'):
                filetype = 'npz'
            else:
                filetype = 'txt'
        arr = self.points.T if self.points.shape[0] < self.points.shape[1] else self.points
        if filetype == 'csv':
            np.savetxt(filename, arr, delimiter=',')
        elif filetype == 'npy':
            np.save(filename, arr)
        elif filetype == 'npz':
            np.savez(filename, kspace=arr, gradients=self.get_gradient_waveforms_Tm(), dt=self.dt, metadata=self.metadata)
        elif filetype == 'txt':
            np.savetxt(filename, arr)
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")

    @classmethod
    def import_from(cls, filename):
        """
        Import a trajectory from a file.
        """
        if filename.endswith('.csv') or filename.endswith('.txt'):
            points = np.loadtxt(filename, delimiter=',' if filename.endswith('.csv') else None)
            return cls(name=filename, points=points)
        elif filename.endswith('.npy'):
            points = np.load(filename)
            return cls(name=filename, points=points)
        elif filename.endswith('.npz'):
            data = np.load(filename, allow_pickle=True)
            points = data['kspace']
            gradients = data['gradients'] if 'gradients' in data else None
            dt = data['dt'].item() if 'dt' in data else None
            metadata = data['metadata'].item() if 'metadata' in data else {}
            return cls(name=filename, points=points, gradients=gradients, dt=dt, metadata=metadata)
        else:
            raise ValueError(f"Unsupported filetype: {filename}")

    def summary(self):
        """
        Print a summary of the trajectory.
        """
        d, n = self.points.shape if self.points.shape[0] < self.points.shape[1] else self.points.T.shape
        print(f"Trajectory '{self.name}': {n} points, {d} dimensions")
        if self.get_gradient_waveforms_Tm() is not None:
            print("Gradients available.")
        if self.dt is not None:
            print(f"Sample time: {self.dt * 1e6:.2f} us")
        if self.metadata:
            for key, value in self.metadata.items():
                print(f"{key}: {value}")





class KSpaceTrajectoryGenerator:
    def __init__(
        self,
        fov=0.24,
        resolution=0.001,
        dt=4e-6,
        g_max=40e-3,
        s_max=150.0,
        n_interleaves=8,
        gamma=42.576e6, # Gyromagnetic ratio in Hz/T. Defaults to 42.576e6 (for 1H). Common values in COMMON_NUCLEI_GAMMA_HZ_PER_T
        traj_type='spiral',
        turns=1,
        ramp_fraction=0.1,
        add_rewinder=True,
        add_spoiler=False,
        add_slew_limited_ramps=True,
        dim=2,
        n_stacks: Optional[int] = None,
        zmax: Optional[float] = None,
        custom_traj_func: Optional[Callable[..., Any]] = None,
        per_interleaf_params: Optional[Dict[int, Dict[str, Any]]] = None,
        time_varying_params: Optional[Callable[[float], Dict[str, float]]] = None,
        use_golden_angle: bool = False,
        vd_method: str = "power",
        vd_alpha: Optional[float] = None,
        vd_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        vd_flat: Optional[float] = None,
        vd_sigma: Optional[float] = None,
        vd_rho: Optional[float] = None,
        spiral_out_out: bool = False,          # <--- New option
        spiral_out_out_split: float = 0.5,     # <--- Fraction of samples for first spiral out (0.5=even split)
        # Parameters for 3D EPI
        epi_3d_fov_y: Optional[float] = None,
        epi_3d_resolution_y: Optional[float] = None,
        epi_3d_fov_z: Optional[float] = None,
        epi_3d_resolution_z: Optional[float] = None,
        # UTE ramp sampling
        ute_ramp_sampling: bool = False,
        ):
        """
        Initializes the KSpaceTrajectoryGenerator.

        Args:
            fov (float): Field of View in meters for the primary readout dimension (typically x).
            resolution (float): Resolution in meters for the primary readout dimension.
            dt (float): Dwell time / sampling interval in seconds.
            g_max (float): Maximum gradient amplitude in T/m.
            s_max (float): Maximum slew rate in T/m/s.
            n_interleaves (int): Number of interleaves or shots. For 'stackofspirals', this is spirals per stack.
                               For 'epi_3d', this is total Ky-Kz phase encode lines.
            gamma (float): Gyromagnetic ratio in Hz/T. Defaults to 42.576e6 (for 1H).
                           Common values for other nuclei can be found in `trajgen.COMMON_NUCLEI_GAMMA_HZ_PER_T`.
            traj_type (str): Type of trajectory to generate. Examples: 'spiral', 'radial', 'epi', 'cones',
                             'stackofspirals', 'radial3d', 'zte', 'epi_3d', 'rosette', 'phyllotaxis'.
            turns (int): Number of turns for spiral trajectories.
            ramp_fraction (float): Fraction of points for ramp-up/down for non-UTE profiles.
            add_rewinder (bool): If True, adds a rewinder gradient to return to k-space center.
            add_spoiler (bool): If True, adds a spoiler gradient.
            add_slew_limited_ramps (bool): If True, uses cosine ramps, else linear ramps.
            dim (int): Number of dimensions (2 or 3).
            n_stacks (Optional[int]): Number of stacks for 'stackofspirals'.
            zmax (Optional[float]): Maximum k-space extent in z for 'stackofspirals' (if None, derived from k_max_xy).
            custom_traj_func (Optional[Callable]): User-defined function for custom trajectories.
            per_interleaf_params (Optional[Dict]): Dictionary to override parameters for specific interleaves.
            time_varying_params (Optional[Callable]): Function to vary parameters over time.
            use_golden_angle (bool): If True, uses golden angle scheme for interleaves (radial, spiral).
            vd_method (str): Variable density method for spirals ('power', 'hybrid', 'gaussian', 'exponential', 'flat', 'custom').
            vd_alpha (Optional[float]): Alpha for 'power' or 'hybrid' variable density.
            vd_func (Optional[Callable]): Custom variable density function.
            vd_flat (Optional[float]): Flat region proportion for 'hybrid' variable density.
            vd_sigma (Optional[float]): Sigma for 'gaussian' variable density.
            vd_rho (Optional[float]): Rho for 'exponential' variable density.
            spiral_out_out (bool): If True, generates spiral-out-out trajectories.
            spiral_out_out_split (float): Fraction of samples for the first spiral-out segment.
            epi_3d_fov_y (Optional[float]): FOV in Y for 3D EPI. Defaults to main `fov`.
            epi_3d_resolution_y (Optional[float]): Resolution in Y for 3D EPI. Defaults to main `resolution`.
            epi_3d_fov_z (Optional[float]): FOV in Z for 3D EPI. Defaults to main `fov`.
            epi_3d_resolution_z (Optional[float]): Resolution in Z for 3D EPI. Defaults to main `resolution`.
            ute_ramp_sampling (bool): If True, enables center-out ramp sampling (half-spokes) for radial/cone/ZTE.
        """
        self.fov = fov
        self.resolution = resolution
        self.dt = dt
        self.g_max = g_max
        self.s_max = s_max
        self.n_interleaves = n_interleaves
        self.gamma = gamma  # Gyromagnetic ratio in Hz/T. Defaults to 42.576e6 (for 1H). Common values in COMMON_NUCLEI_GAMMA_HZ_PER_T
        self.traj_type = traj_type
        self.turns = turns
        self.ramp_fraction = ramp_fraction
        self.add_rewinder = add_rewinder
        self.add_spoiler = add_spoiler
        self.add_slew_limited_ramps = add_slew_limited_ramps
        self.dim = dim
        self.n_stacks = n_stacks
        self.zmax = zmax
        self.custom_traj_func = custom_traj_func
        self.per_interleaf_params = per_interleaf_params or {}
        self.time_varying_params = time_varying_params
        self.use_golden_angle = use_golden_angle
        self.vd_method = vd_method
        self.vd_alpha = vd_alpha
        self.vd_func = vd_func
        self.vd_flat = vd_flat
        self.vd_sigma = vd_sigma
        self.vd_rho = vd_rho
        self.spiral_out_out = spiral_out_out
        self.spiral_out_out_split = spiral_out_out_split
        
        # Store 3D EPI specific parameters
        self.epi_3d_fov_y = epi_3d_fov_y
        self.epi_3d_resolution_y = epi_3d_resolution_y
        self.epi_3d_fov_z = epi_3d_fov_z
        self.epi_3d_resolution_z = epi_3d_resolution_z
        self.ute_ramp_sampling = ute_ramp_sampling

        self.k_max = 1 / (2 * self.resolution)
        self.g_required = min(self.k_max / (self.gamma * self.dt), self.g_max)
        self.n_samples = int(np.ceil((self.k_max * 2 * np.pi * self.fov) / (self.gamma * self.g_required * self.dt)))
        self.n_samples = max(self.n_samples, 1)
        self.ramp_samples = int(np.ceil(self.ramp_fraction * self.n_samples))
        self.flat_samples = self.n_samples - 2 * self.ramp_samples

    def _slew_limited_ramp(self, N, sign=1):
        t_ramp = np.linspace(0, 1, N)
        ramp = 0.5 * (1 - np.cos(np.pi * t_ramp))
        return sign * ramp

    def _make_radius_profile(self, n_samples=None):
        n_samples = n_samples or self.n_samples
        
        if self.ute_ramp_sampling:
            if self.add_slew_limited_ramps:
                # Slew-limited ramp from 0 to 1
                r_profile = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n_samples)))
            else:
                # Linear ramp from 0 to 1
                r_profile = np.linspace(0, 1, n_samples)
        else:
            # Standard profile with ramp-up, flat-top, ramp-down
            ramp_samples = int(self.ramp_fraction * n_samples)
            if ramp_samples * 2 > n_samples : # Ensure ramp_samples are not too large for n_samples
                ramp_samples = n_samples // 2 
            flat_samples = n_samples - 2 * ramp_samples
            
            if self.add_slew_limited_ramps:
                ramp_up = self._slew_limited_ramp(ramp_samples)
                flat = np.ones(flat_samples)
                ramp_down = 1 - self._slew_limited_ramp(ramp_samples) # This ramps from 1 down to 0
                r_profile = np.concatenate([ramp_up, flat, ramp_down])
            else:
                r_profile = np.ones(n_samples)
                if ramp_samples > 0: # Avoid issues if n_samples is very small
                    r_profile[:ramp_samples] = np.linspace(0, 1, ramp_samples)
                    r_profile[-ramp_samples:] = np.linspace(1, 0, ramp_samples)
        return r_profile

    def _variable_density_spiral(self, t):
        if self.vd_func is not None or self.vd_method == "custom":
            return self.vd_func(t)
        if self.vd_method == "power":
            alpha = self.vd_alpha if self.vd_alpha is not None else 1
            return t ** alpha
        elif self.vd_method == "hybrid":
            flat = self.vd_flat if self.vd_flat is not None else 0.2
            alpha = self.vd_alpha if self.vd_alpha is not None else 2
            r = np.zeros_like(t)
            mask = t < flat
            r[mask] = t[mask] / flat
            r[~mask] = ((t[~mask] - flat) / (1 - flat)) ** alpha
            return r
        elif self.vd_method == "gaussian":
            sigma = self.vd_sigma if self.vd_sigma is not None else 0.25
            from scipy.special import erf
            return erf(t / sigma)
        elif self.vd_method == "exponential":
            rho = self.vd_rho if self.vd_rho is not None else 3
            return (np.exp(rho * t) - 1) / (np.exp(rho) - 1)
        elif self.vd_method == "flat":
            return t
        else:
            return t

    def _enforce_gradient_limits(self, gx, gy, gz=None):
        g_norm = np.sqrt(gx ** 2 + gy ** 2 + (gz**2 if gz is not None else 0))
        over_gmax = g_norm > self.g_max
        if np.any(over_gmax):
            scale = self.g_max / np.max(g_norm)
            gx[over_gmax] *= scale
            gy[over_gmax] *= scale
            if gz is not None:
                gz[over_gmax] *= scale

        slew = np.sqrt(np.gradient(gx, self.dt) ** 2 +
                       np.gradient(gy, self.dt) ** 2 +
                       (np.gradient(gz, self.dt) ** 2 if gz is not None else 0))
        over_smax = slew > self.s_max
        if np.any(over_smax):
            scale = self.s_max / np.max(slew)
            gx[over_smax] *= scale
            gy[over_smax] *= scale
            if gz is not None:
                gz[over_smax] *= scale
        return gx, gy, gz

    def _golden_angle(self, idx):
        if self.dim == 2:
            golden = np.pi * (3 - np.sqrt(5))
            return idx * golden
        elif self.dim == 3:
            indices = idx + 0.5
            phi = np.arccos(1 - 2*indices/self.n_interleaves)
            theta = np.pi * (1 + 5**0.5) * indices
            return theta, phi

    def _generate_spiral_out_out(self, t, n_samples, k_max, turns, phi, r_profile):
        """
        Generate a spiral-out-spiral-out trajectory (see e.g. https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.24476)
        """
        split = self.spiral_out_out_split
        n1 = int(np.floor(split * n_samples))
        n2 = n_samples - n1

        # First spiral out (0 -> k_max)
        t1 = np.linspace(0, 1, n1, endpoint=False)
        vd1 = self._variable_density_spiral(t1)
        r1 = vd1 * k_max * r_profile[:n1]
        theta1 = turns * 2 * np.pi * t1 + phi

        # Second spiral out (0 -> k_max), starting at origin but shifted in angle
        t2 = np.linspace(0, 1, n2)
        vd2 = self._variable_density_spiral(t2)
        # angle offset for the second spiral
        phi2 = phi + np.pi  # 180 deg offset (can be parameter)
        r2 = vd2 * k_max * r_profile[n1:]
        theta2 = turns * 2 * np.pi * t2 + phi2

        # Both start at (0,0), end at (k_max, angle)
        kx1 = r1 * np.cos(theta1)
        ky1 = r1 * np.sin(theta1)
        kx2 = r2 * np.cos(theta2)
        ky2 = r2 * np.sin(theta2)

        kx = np.concatenate([kx1, kx2])
        ky = np.concatenate([ky1, ky2])
        return kx, ky

    def _generate_standard(self, interleaf_idx, t, n_samples, **params):
        local_params = {**self.__dict__, **params}
        fov = local_params.get("fov", self.fov)
        resolution = local_params.get("resolution", self.resolution)
        turns = local_params.get("turns", self.turns)
        k_max = 1 / (2 * resolution)
        r_profile = self._make_radius_profile(n_samples)
        if self.time_varying_params is not None:
            for i, ti in enumerate(t):
                for key, val in self.time_varying_params(ti).items():
                    if key == "fov":
                        fov = val
                    if key == "resolution":
                        resolution = val
                k_max = 1 / (2 * resolution)
                r_profile[i] = min(r_profile[i], k_max / self.k_max)

        if self.dim == 2:
            if self.traj_type == "spiral":
                if self.spiral_out_out:
                    if self.use_golden_angle:
                        phi = self._golden_angle(interleaf_idx)
                    else:
                        phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                    kx, ky = self._generate_spiral_out_out(t, n_samples, k_max, turns, phi, r_profile)
                else:
                    t_norm = np.linspace(0, 1, n_samples)
                    vd = self._variable_density_spiral(t_norm)
                    r = vd * k_max * r_profile
                    if self.use_golden_angle:
                        phi = self._golden_angle(interleaf_idx)
                    else:
                        phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                    theta = turns * 2 * np.pi * t / t[-1] + phi
                    kx = r * np.cos(theta)
                    ky = r * np.sin(theta)
                kz = None
            elif self.traj_type == "radial":
                angle = (self._golden_angle(interleaf_idx) if self.use_golden_angle
                         else np.pi * interleaf_idx / self.n_interleaves)
                if self.ute_ramp_sampling:
                    # r_profile is 0 to 1, k_max is the max radius
                    k_line = r_profile * k_max # Half-spoke from 0 to k_max
                else:
                    # r_profile is symmetric [ramp, flat, ramp_down]
                    k_line = np.linspace(-k_max, k_max, n_samples) * r_profile # Full spoke, scaled by r_profile
                kx = k_line * np.cos(angle)
                ky = k_line * np.sin(angle)
                kz = None
            elif self.traj_type == "epi":
                kx = np.linspace(-k_max, k_max, n_samples)
                ky = np.zeros(n_samples)
                kz = None
            elif self.traj_type == "rosette":
                f1 = params.get("f1", 5)
                f2 = params.get("f2", 7)
                a = params.get("a", 0.5)
                phase = 2*np.pi*interleaf_idx/self.n_interleaves
                tt = np.linspace(0, 2*np.pi, n_samples)
                kx = k_max * (a * np.sin(f1*tt+phase) + (1-a) * np.sin(f2*tt+phase))
                ky = k_max * (a * np.cos(f1*tt+phase) + (1-a) * np.cos(f2*tt+phase))
                kz = None
            else:
                raise ValueError(f"Unknown 2D traj_type {self.traj_type}")
            gx = np.gradient(kx, self.dt) / self.gamma
            gy = np.gradient(ky, self.dt) / self.gamma
            gz = None
        elif self.dim == 3:
            if self.traj_type == "stackofspirals":
                n_stacks = self.n_stacks or 8
                zmax = self.zmax or k_max
                stack_idx = interleaf_idx // self.n_interleaves
                slice_idx = interleaf_idx % self.n_interleaves
                z_locations = np.linspace(-zmax, zmax, n_stacks)
                z = z_locations[stack_idx]
                phi = 2 * np.pi * slice_idx / self.n_interleaves
                t_norm = np.linspace(0, 1, n_samples)
                vd = self._variable_density_spiral(t_norm)
                r = vd * k_max * r_profile
                theta = turns * 2 * np.pi * t / t[-1] + phi
                kx = r * np.cos(theta)
                ky = r * np.sin(theta)
                kz = np.ones(n_samples) * z
            elif self.traj_type == "phyllotaxis":
                golden_angle = np.pi * (3 - np.sqrt(5))
                theta = golden_angle * interleaf_idx
                z = np.linspace(1 - 1/n_samples, -1 + 1/n_samples, n_samples)
                radius = np.sqrt(1 - z**2)
                kx = k_max * radius * np.cos(theta)
                ky = k_max * radius * np.sin(theta)
                kz = k_max * z
            elif self.traj_type == "cones":
                phi = 2 * np.pi * interleaf_idx / self.n_interleaves
                tt = np.linspace(0, 1, n_samples)
                theta = np.arccos(1 - 2*tt)
                vd = self._variable_density_spiral(tt)
                kx = k_max * vd * np.sin(theta) * np.cos(phi)
                ky = k_max * vd * np.sin(theta) * np.sin(phi)
                kz = k_max * vd * r_profile * np.cos(theta) # r_profile incorporated here
            elif self.traj_type == "radial3d":
                theta, phi = self._golden_angle(interleaf_idx)
                if self.ute_ramp_sampling:
                    k_line = r_profile * k_max # Half-spoke
                else:
                    # For standard radial3d, r_profile is likely intended to make it ramped if not all ones
                    k_line = np.linspace(-k_max, k_max, n_samples) * r_profile
                kx = k_line * np.sin(theta) * np.cos(phi)
                ky = k_line * np.sin(theta) * np.sin(phi)
                kz = k_line * np.cos(theta)
            elif self.traj_type == "epi_3d":
                if self.dim != 3:
                    raise ValueError("epi_3d trajectory requires dim=3")
                # n_samples here is n_samples_x for the readout
                return self._generate_epi_3d(interleaf_idx, t, n_samples, **local_params)
            elif self.traj_type == "zte": # Added ZTE
                if self.dim != 3:
                    raise ValueError("ZTE trajectory (type 'zte') requires dim=3")
                # ZTE behaves like radial3d; ute_ramp_sampling flag controls center-out ramp
                theta, phi = self._golden_angle(interleaf_idx)
                # k_max and r_profile are already resolved at the start of _generate_standard
                if self.ute_ramp_sampling:
                    k_line = r_profile * k_max # Half-spoke (center-out ramp)
                else:
                    # Full spoke, scaled by symmetric r_profile (not typical ZTE if this branch is hit)
                    k_line = np.linspace(-k_max, k_max, n_samples) * r_profile
                kx = k_line * np.sin(theta) * np.cos(phi)
                ky = k_line * np.sin(theta) * np.sin(phi)
                kz = k_line * np.cos(theta)
                # Gradients will be calculated below, common for most 3D trajectories
            else:
                raise ValueError(f"Unknown 3D traj_type {self.traj_type}")
            gx = np.gradient(kx, self.dt) / self.gamma
            gy = np.gradient(ky, self.dt) / self.gamma
            gz = np.gradient(kz, self.dt) / self.gamma
        else:
            raise ValueError("dim must be 2 or 3")
        gx, gy, gz = self._enforce_gradient_limits(gx, gy, gz)
        return kx, ky, kz, gx, gy, gz

    def _generate_epi_3d(self, interleaf_idx, time_vector_x, n_samples_x, **params):
        """Generates a single Kx readout line for a 3D EPI trajectory at a specific (Ky, Kz) encode."""
        
        # Resolve parameters (use instance attributes, override with params if provided)
        fov_x = params.get("fov", self.fov) # Readout FOV
        res_x = params.get("resolution", self.resolution) # Readout resolution

        fov_y = params.get("epi_3d_fov_y", self.epi_3d_fov_y)
        if fov_y is None: fov_y = fov_x # Default to readout FOV
        res_y = params.get("epi_3d_resolution_y", self.epi_3d_resolution_y)
        if res_y is None: res_y = res_x # Default to readout resolution
        
        fov_z = params.get("epi_3d_fov_z", self.epi_3d_fov_z)
        if fov_z is None: fov_z = fov_x # Default to readout FOV
        res_z = params.get("epi_3d_resolution_z", self.epi_3d_resolution_z)
        if res_z is None: res_z = res_x # Default to readout resolution

        k_max_x = 1 / (2 * res_x + 1e-9)
        k_max_y = 1 / (2 * res_y + 1e-9)
        k_max_z = 1 / (2 * res_z + 1e-9)

        num_phase_encodes_y = int(round(fov_y / res_y)) if res_y > 0 else 1
        num_phase_encodes_y = max(1, num_phase_encodes_y)
        num_phase_encodes_z = int(round(fov_z / res_z)) if res_z > 0 else 1
        num_phase_encodes_z = max(1, num_phase_encodes_z)

        delta_ky = (2 * k_max_y) / num_phase_encodes_y if num_phase_encodes_y > 1 else 0.0
        delta_kz = (2 * k_max_z) / num_phase_encodes_z if num_phase_encodes_z > 1 else 0.0
        
        # Map interleaf_idx to (ky_idx, kz_idx)
        # This assumes n_interleaves is set by the user to cover the desired YZ plane
        # (e.g., n_interleaves = num_phase_encodes_y * num_phase_encodes_z for full coverage)
        kz_idx = interleaf_idx // num_phase_encodes_y
        ky_idx = interleaf_idx % num_phase_encodes_y

        # Check if current interleaf is beyond the planned Kz coverage based on num_phase_encodes_z
        # This might happen if n_interleaves > num_phase_encodes_y * num_phase_encodes_z
        if kz_idx >= num_phase_encodes_z:
            # This shot is outside the primary Kz encoding range.
            # Depending on desired behavior, could error, clamp, or let kz_target go beyond k_max_z.
            # For now, let it proceed; kz_target will simply be beyond the nominal k_max_z.
            # Or, more safely, one might want to cap it or warn.
            # print(f"Warning: interleaf_idx {interleaf_idx} results in kz_idx {kz_idx} >= num_phase_encodes_z {num_phase_encodes_z}")
            pass


        current_ky_target = -k_max_y + ky_idx * delta_ky + (delta_ky / 2 if num_phase_encodes_y > 1 else 0.0)
        current_kz_target = -k_max_z + kz_idx * delta_kz + (delta_kz / 2 if num_phase_encodes_z > 1 else 0.0)
        if num_phase_encodes_y == 1: current_ky_target = 0.0 # Single PE line in Y is at center
        if num_phase_encodes_z == 1: current_kz_target = 0.0 # Single PE line in Z is at center


        # Kx readout (frequency encoding)
        kx_readout = np.linspace(-k_max_x, k_max_x, n_samples_x)
        if ky_idx % 2 == 1: # Flip Kx readout for odd Ky lines (traditional EPI raster)
            kx_readout = kx_readout[::-1]

        kx_interleaf = kx_readout
        ky_interleaf = np.full(n_samples_x, current_ky_target)
        kz_interleaf = np.full(n_samples_x, current_kz_target)

        # Calculate gradients (np.gradient will handle mostly zero grads for ky, kz)
        # Note: self.dt and self.gamma are available from the instance
        gx = np.gradient(kx_interleaf, self.dt) / self.gamma
        gy = np.gradient(ky_interleaf, self.dt) / self.gamma 
        gz = np.gradient(kz_interleaf, self.dt) / self.gamma
        
        return kx_interleaf, ky_interleaf, kz_interleaf, gx, gy, gz

    def _add_spoiler(self, kx, ky, kz, gx, gy, gz):
        n_spoil = self.ramp_samples
        spoil_area = 2 * self.k_max
        kx_out, ky_out, kz_out, gx_out, gy_out, gz_out = [], [], [], [], [], []
        for idx in range(self.n_interleaves):
            if self.dim == 2:
                end_g = np.array([gx[idx, -1], gy[idx, -1]])
                if np.linalg.norm(end_g) == 0:
                    end_g = np.array([1, 0])
                else:
                    end_g /= np.linalg.norm(end_g)
                g_spoil = end_g * (spoil_area / (self.gamma * self.dt * n_spoil))
                kx_s = np.full(n_spoil, kx[idx, -1])
                ky_s = np.full(n_spoil, ky[idx, -1])
                gx_s = np.full(n_spoil, g_spoil[0])
                gy_s = np.full(n_spoil, g_spoil[1])
                kx_out.append(np.concatenate([kx[idx], kx_s]))
                ky_out.append(np.concatenate([ky[idx], ky_s]))
                gx_out.append(np.concatenate([gx[idx], gx_s]))
                gy_out.append(np.concatenate([gy[idx], gy_s]))
            else:
                end_g = np.array([gx[idx, -1], gy[idx, -1], gz[idx, -1]])
                if np.linalg.norm(end_g) == 0:
                    spoil_dir = np.array([0, 0, 1])
                else:
                    spoil_dir = end_g / np.linalg.norm(end_g)
                g_spoil = spoil_dir * (spoil_area / (self.gamma * self.dt * n_spoil))
                kx_s = np.full(n_spoil, kx[idx, -1])
                ky_s = np.full(n_spoil, ky[idx, -1])
                kz_s = np.full(n_spoil, kz[idx, -1])
                gx_s = np.full(n_spoil, g_spoil[0])
                gy_s = np.full(n_spoil, g_spoil[1])
                gz_s = np.full(n_spoil, g_spoil[2])
                kx_out.append(np.concatenate([kx[idx], kx_s]))
                ky_out.append(np.concatenate([ky[idx], ky_s]))
                kz_out.append(np.concatenate([kz[idx], kz_s]))
                gx_out.append(np.concatenate([gx[idx], gx_s]))
                gy_out.append(np.concatenate([gy[idx], gy_s]))
                gz_out.append(np.concatenate([gz[idx], gz_s]))
        if self.dim == 2:
            return (np.array(kx_out), np.array(ky_out), None,
                    np.array(gx_out), np.array(gy_out), None)
        else:
            return (np.array(kx_out), np.array(ky_out), np.array(kz_out),
                    np.array(gx_out), np.array(gy_out), np.array(gz_out))

    def _add_rewinder(self, kx, ky, kz, gx, gy, gz):
        n_rw = self.ramp_samples
        kx_out, ky_out, kz_out, gx_out, gy_out, gz_out = [], [], [], [], [], []
        for idx in range(self.n_interleaves):
            if self.dim == 2:
                net_kx = kx[idx, -1]
                net_ky = ky[idx, -1]
                k_rewind = np.linspace([net_kx, net_ky], [0, 0], n_rw)
                gx_rewind = np.gradient(k_rewind[:, 0], self.dt) / self.gamma
                gy_rewind = np.gradient(k_rewind[:, 1], self.dt) / self.gamma
                kx_out.append(np.concatenate([kx[idx], k_rewind[:, 0]]))
                ky_out.append(np.concatenate([ky[idx], k_rewind[:, 1]]))
                gx_out.append(np.concatenate([gx[idx], gx_rewind]))
                gy_out.append(np.concatenate([gy[idx], gy_rewind]))
            else:
                net_kx = kx[idx, -1]
                net_ky = ky[idx, -1]
                net_kz = kz[idx, -1]
                k_rewind = np.linspace([net_kx, net_ky, net_kz], [0, 0, 0], n_rw)
                gx_rewind = np.gradient(k_rewind[:, 0], self.dt) / self.gamma
                gy_rewind = np.gradient(k_rewind[:, 1], self.dt) / self.gamma
                gz_rewind = np.gradient(k_rewind[:, 2], self.dt) / self.gamma
                kx_out.append(np.concatenate([kx[idx], k_rewind[:, 0]]))
                ky_out.append(np.concatenate([ky[idx], k_rewind[:, 1]]))
                kz_out.append(np.concatenate([kz[idx], k_rewind[:, 2]]))
                gx_out.append(np.concatenate([gx[idx], gx_rewind]))
                gy_out.append(np.concatenate([gy[idx], gy_rewind]))
                gz_out.append(np.concatenate([gz[idx], gz_rewind]))
        if self.dim == 2:
            return (np.array(kx_out), np.array(ky_out), None,
                    np.array(gx_out), np.array(gy_out), None)
        else:
            return (np.array(kx_out), np.array(ky_out), np.array(kz_out),
                    np.array(gx_out), np.array(gy_out), np.array(gz_out))

    def generate(self):
        n_interleaves = self.n_interleaves
        n_samples = self.n_samples
        t = np.arange(n_samples) * self.dt

        if self.dim == 2:
            kx = np.zeros((n_interleaves, n_samples))
            ky = np.zeros((n_interleaves, n_samples))
            gx = np.zeros((n_interleaves, n_samples))
            gy = np.zeros((n_interleaves, n_samples))
            kz = gz = None
        else:
            kx = np.zeros((n_interleaves, n_samples))
            ky = np.zeros((n_interleaves, n_samples))
            kz = np.zeros((n_interleaves, n_samples))
            gx = np.zeros((n_interleaves, n_samples))
            gy = np.zeros((n_interleaves, n_samples))
            gz = np.zeros((n_interleaves, n_samples))

        for idx in range(n_interleaves):
            params = self.per_interleaf_params.get(idx, {})
            if self.custom_traj_func is not None:
                k_vals, g_vals = self.custom_traj_func(idx, t, n_samples, **params)
                kx[idx], ky[idx] = k_vals[:2]
                gx[idx], gy[idx] = g_vals[:2]
                if self.dim == 3 and len(k_vals) > 2:
                    kz[idx] = k_vals[2]
                    gz[idx] = g_vals[2]
                continue
            kx_i, ky_i, kz_i, gx_i, gy_i, gz_i = self._generate_standard(idx, t, n_samples, **params)
            kx[idx], ky[idx] = kx_i, ky_i
            gx[idx], gy[idx] = gx_i, gy_i
            if self.dim == 3:
                kz[idx] = kz_i
                gz[idx] = gz_i

        if self.add_spoiler:
            kx, ky, kz, gx, gy, gz = self._add_spoiler(kx, ky, kz, gx, gy, gz)
        if self.add_rewinder:
            kx, ky, kz, gx, gy, gz = self._add_rewinder(kx, ky, kz, gx, gy, gz)

        t = np.arange(kx.shape[1]) * self.dt
        if self.dim == 2:
            return kx, ky, gx, gy, t
        else:
            return kx, ky, kz, gx, gy, gz, t
    def generate_3d_from_2d(
        self,
        n_3d_shots: int,
        fov_3d: float = None,
        resolution_3d: float = None,
        phi_theta_func: Optional[Callable[[int], tuple]] = None,
        traj2d_type: str = "spiral",
        **kwargs
    ):
        """
        Generate a 3D k-space trajectory by rotating a 2D trajectory (e.g., spiral, radial) 
        according to phi and theta angles, with logic for full 3D k-space coverage.
        
        Args:
            n_3d_shots: Number of shots (orientations) covering 3D k-space
            fov_3d: 3D field of view (if None, uses self.fov)
            resolution_3d: 3D resolution (if None, uses self.resolution)
            phi_theta_func: User callback for phi, theta for each shot (idx) (optional)
            traj2d_type: Type of 2D trajectory to generate ("spiral", "radial", etc.)
            kwargs: Passed to 2D trajectory generator
        
        Returns:
            kx, ky, kz: [n_3d_shots, n_samples] arrays
            gx, gy, gz: [n_3d_shots, n_samples] arrays
            t: time vector
            
            # Example usage:
            # gen = KSpaceTrajectoryGenerator(fov=0.22, resolution=0.0015, traj_type='spiral', vd_method='power', vd_alpha=1.5)
            # kx, ky, kz, gx, gy, gz, t = gen.generate_3d_from_2d(n_3d_shots=64)
        """
        fov_3d = fov_3d if fov_3d is not None else self.fov
        resolution_3d = resolution_3d if resolution_3d is not None else self.resolution
        k_max_3d = 1/(2 * resolution_3d)
        # Estimate number of samples for 2D trajectory in the 3D context
        n_samples = int(np.ceil((k_max_3d * 2 * np.pi * fov_3d) / (self.gamma * self.g_max * self.dt)))
        n_samples = max(n_samples, 1)
        t = np.arange(n_samples) * self.dt

        # Uniform sphere coverage: use spherical Fibonacci or golden angle
        def default_phi_theta(idx):
            # Spherical Fibonacci lattice for uniform 3D coverage
            ga = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * (idx + 0.5) / n_3d_shots
            phi = ga * idx
            theta = np.arccos(z)
            return phi, theta

        phi_theta = phi_theta_func if phi_theta_func is not None else default_phi_theta

        kx = np.zeros((n_3d_shots, n_samples))
        ky = np.zeros((n_3d_shots, n_samples))
        kz = np.zeros((n_3d_shots, n_samples))
        gx = np.zeros((n_3d_shots, n_samples))
        gy = np.zeros((n_3d_shots, n_samples))
        gz = np.zeros((n_3d_shots, n_samples))

        # Prepare a 2D trajectory generator for the base (xy) plane
        base2d = KSpaceTrajectoryGenerator(
            fov=fov_3d, resolution=resolution_3d, dt=self.dt, g_max=self.g_max, s_max=self.s_max,
            n_interleaves=1, gamma=self.gamma, traj_type=traj2d_type, turns=self.turns,
            ramp_fraction=self.ramp_fraction, add_slew_limited_ramps=self.add_slew_limited_ramps,
            vd_method=self.vd_method, vd_alpha=self.vd_alpha, vd_func=self.vd_func,
            vd_flat=self.vd_flat, vd_sigma=self.vd_sigma, vd_rho=self.vd_rho
        )

        # Generate a single "base" 2D trajectory
        kx2d, ky2d, gx2d, gy2d, t2d = base2d.generate()
        kx2d = kx2d[0]
        ky2d = ky2d[0]
        gx2d = gx2d[0]
        gy2d = gy2d[0]

        # Pad or cut to n_samples for consistency
        if len(kx2d) > n_samples:
            kx2d = kx2d[:n_samples]; ky2d = ky2d[:n_samples]
            gx2d = gx2d[:n_samples]; gy2d = gy2d[:n_samples]
        elif len(kx2d) < n_samples:
            pad = n_samples - len(kx2d)
            kx2d = np.pad(kx2d, (0, pad))
            ky2d = np.pad(ky2d, (0, pad))
            gx2d = np.pad(gx2d, (0, pad))
            gy2d = np.pad(gy2d, (0, pad))

        # For each 3D shot, rotate the 2D trajectory by phi and theta
        for idx in range(n_3d_shots):
            phi, theta = phi_theta(idx)
            # 3D rotation matrix (first rotate around y by theta, then around z by phi)
            # Ry(theta) * Rz(phi)
            # [cos(phi)*cos(theta) -sin(phi) cos(phi)*sin(theta)]
            # [sin(phi)*cos(theta)  cos(phi) sin(phi)*sin(theta)]
            # [      -sin(theta)         0         cos(theta) ]
            cphi, sphi = np.cos(phi), np.sin(phi)
            ctheta, stheta = np.cos(theta), np.sin(theta)
            R = np.array([
                [cphi*ctheta, -sphi, cphi*stheta],
                [sphi*ctheta,  cphi, sphi*stheta],
                [      -stheta,     0,      ctheta]
            ])
            # Stack 2D trajectory as [x, y, 0]
            traj2d = np.stack([kx2d, ky2d, np.zeros_like(kx2d)], axis=0)
            grad2d = np.stack([gx2d, gy2d, np.zeros_like(gx2d)], axis=0)
            traj3d = R @ traj2d
            grad3d = R @ grad2d
            kx[idx] = traj3d[0]
            ky[idx] = traj3d[1]
            kz[idx] = traj3d[2]
            gx[idx] = grad3d[0]
            gy[idx] = grad3d[1]
            gz[idx] = grad3d[2]

        return kx, ky, kz, gx, gy, gz, t

    @staticmethod
    def plugin_example(idx, t, n_samples, **kwargs):
        kx = np.cos(2 * np.pi * t / t[-1])
        ky = np.sin(2 * np.pi * t / t[-1])
        gx = np.gradient(kx, t)
        gy = np.gradient(ky, t)
        return (kx, ky), (gx, gy)

    def check_gradient_and_slew_limits(self, k_traj):
        gamma = self.gamma
        G = np.diff(k_traj, axis=0) / self.dt / gamma
        slew = np.diff(G, axis=0) / self.dt
        grad_ok = np.all(np.abs(G) <= self.g_max)
        slew_ok = np.all(np.abs(slew) <= self.s_max)
        return grad_ok, slew_ok, G, slew

# Example usage for spiral out-out:
# gen = KSpaceTrajectoryGenerator(
#     fov=0.24, resolution=0.001, dt=4e-6, g_max=40e-3, s_max=150.0,
#     n_interleaves=8, traj_type='spiral', spiral_out_out=True,
#     spiral_out_out_split=0.5, use_golden_angle=True, vd_method="power", vd_alpha=1.2
# )
# kx, ky, gx, gy, t = gen.generate()
