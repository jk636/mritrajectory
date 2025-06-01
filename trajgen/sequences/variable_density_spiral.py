"""
Defines the VariableDensitySpiralSequence class.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from trajgen.generators import generate_spiral_trajectory
from trajgen.sequence_base import MRISequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

__all__ = ['VariableDensitySpiralSequence']


class VariableDensitySpiralSequence(MRISequence):
    """
    Represents a variable density spiral MRI sequence.

    This class uses `trajgen.generators.generate_spiral_trajectory` to create
    the k-space points based on the specified spiral parameters.
    """

    def __init__(self,
                 # Common MRISequence parameters
                 name: str,
                 fov_mm: Union[float, Tuple[float, ...]],
                 resolution_mm: Union[float, Tuple[float, ...]],
                 num_dimensions: int,
                 dt_seconds: float,
                 # Spiral-specific parameters
                 num_interleaves: int,
                 points_per_interleaf: int,
                 spiral_type: str = 'archimedean',
                 density_transition_radius_factor: Optional[float] = None,
                 density_factor_at_center: Optional[float] = None,
                 undersampling_factor: float = 1.0,
                 # Other Trajectory/MRISequence parameters
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds: float = 0.0,
                 dead_time_end_seconds: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a VariableDensitySpiralSequence object.

        Args:
            name (str): Name of the sequence.
            fov_mm (Union[float, Tuple[float, ...]]): Field of View in millimeters.
            resolution_mm (Union[float, Tuple[float, ...]]): Desired resolution in millimeters.
            num_dimensions (int): Number of spatial dimensions (typically 2 for spirals).
            dt_seconds (float): Dwell time (time between k-space samples) in seconds.
            num_interleaves (int): Number of spiral interleaves.
            points_per_interleaf (int): Number of k-space points per interleaf.
            spiral_type (str): Type of spiral (e.g., 'archimedean', 'logarithmic').
            density_transition_radius_factor (Optional[float]): Factor determining the radius
                at which k-space density transitions.
            density_factor_at_center (Optional[float]): Factor determining k-space density
                at the center relative to the periphery.
            undersampling_factor (float): Factor by which to undersample the spiral.
            gamma_Hz_per_T (float): Gyromagnetic ratio in Hz/T.
            dead_time_start_seconds (float): Dead time at the beginning of the sequence.
            dead_time_end_seconds (float): Dead time at the end of the sequence.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
        """
        self.num_interleaves = num_interleaves
        self.points_per_interleaf = points_per_interleaf
        self.spiral_type = spiral_type
        self.density_transition_radius_factor = density_transition_radius_factor
        self.density_factor_at_center = density_factor_at_center
        self.undersampling_factor = undersampling_factor

        sequence_specific_params = {
            'num_interleaves': num_interleaves,
            'points_per_interleaf': points_per_interleaf,
            'spiral_type': spiral_type,
            'density_transition_radius_factor': density_transition_radius_factor,
            'density_factor_at_center': density_factor_at_center,
            'undersampling_factor': undersampling_factor,
        }

        # Combine with other important params for Trajectory's sequence_params if needed
        # For now, MRISequence.__init__ handles basic fov, res, etc. in its metadata creation
        # and passes sequence_specific_params to Trajectory's sequence_params.

        super().__init__(
            name=name,
            fov_mm=fov_mm,
            resolution_mm=resolution_mm,
            num_dimensions=num_dimensions,
            dt_seconds=dt_seconds,
            gamma_Hz_per_T=gamma_Hz_per_T,
            sequence_specific_params=sequence_specific_params,
            dead_time_start_seconds=dead_time_start_seconds,
            dead_time_end_seconds=dead_time_end_seconds,
            metadata=metadata
        )

    def _generate_kspace_points(self) -> np.ndarray:
        """
        Generates k-space points for the variable density spiral sequence.
        """
        # The generate_spiral_trajectory function expects FOV and resolution in meters.
        # self.fov_mm and self.resolution_mm can be float or tuple.

        fov_m: Union[float, Tuple[float, ...]]
        if isinstance(self.fov_mm, tuple):
            fov_m = tuple(f / 1000.0 for f in self.fov_mm)
        else:
            fov_m = self.fov_mm / 1000.0

        resolution_m: Union[float, Tuple[float, ...]]
        if isinstance(self.resolution_mm, tuple):
            resolution_m = tuple(r / 1000.0 for r in self.resolution_mm)
        else:
            resolution_m = self.resolution_mm / 1000.0

        # Ensure num_dimensions is handled correctly, typically 2 for spirals
        if self.num_dimensions != 2:
            # Or raise error, or adapt. For now, assume generator handles/expects 2D based on typical spiral usage.
            print(f"Warning: VariableDensitySpiral is typically 2D, but num_dimensions is {self.num_dimensions}.")


        kspace_points, _, _ = generate_spiral_trajectory(
            fov_m=fov_m, # type: ignore
            resolution_m=resolution_m, # type: ignore
            num_interleaves=self.num_interleaves,
            points_per_interleaf=self.points_per_interleaf,
            spiral_type=self.spiral_type,
            density_transition_radius_factor=self.density_transition_radius_factor,
            density_factor_at_center=self.density_factor_at_center,
            undersampling_factor=self.undersampling_factor,
            gamma_Hz_per_T=self.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
        )
        # generate_spiral_trajectory returns k-space of shape (N, D)
        # Trajectory class expects (D, N)
        return kspace_points.T if kspace_points.ndim == 2 and kspace_points.shape[0] > 0 else np.empty((self.num_dimensions,0))


    def check_gradient_limits(self, system_limits: Dict[str, Any]) -> bool:
        """
        Checks if the trajectory's gradient waveforms and slew rates are within
        the specified system limits.

        Args:
            system_limits (Dict[str, Any]): A dictionary containing system limits,
                e.g., {'max_grad_Tm_per_m': 0.04 (T/m),
                       'max_slew_Tm_per_s_per_m': 150 (T/m/s)}.
                       Note: `generate_spiral_trajectory` might use different units internally
                       (e.g. G/cm, G/cm/ms), so ensure consistency or conversion if limits
                       are directly passed to generators. Here, we check post-generation.

        Returns:
            bool: True if within limits, False otherwise.
        """
        max_grad_limit_Tm_per_m = system_limits.get('max_grad_Tm_per_m')
        max_slew_limit_Tm_per_s = system_limits.get('max_slew_Tm_per_s_per_m') # Key used in prompt

        # Get calculated values from Trajectory base class (which computes them if needed)
        # These are properties that might trigger computation if not already done.
        # self.get_gradient_waveforms_Tm() # ensure gradients are computed
        # self._calculate_slew_rate() # ensure slew is computed

        actual_max_grad_Tm = self.get_max_grad_Tm() # This is norm, check if limit is per-axis or norm
        actual_max_slew_Tm_per_s = self.get_max_slew_Tm_per_s() # This is norm

        grad_ok = True
        slew_ok = True

        print_prefix = f"Gradient Limit Check for '{self.name}':"

        if actual_max_grad_Tm is None:
            print(f"{print_prefix} Could not determine actual max gradient.")
            grad_ok = False # Cannot confirm
        elif max_grad_limit_Tm_per_m is not None:
            grad_ok = actual_max_grad_Tm <= max_grad_limit_Tm_per_m
            print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. Limit: {max_grad_limit_Tm_per_m:.4f} T/m. Status: {'OK' if grad_ok else 'EXCEEDED'}")
        else:
            print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. No limit provided.")
            # grad_ok remains True if no limit is specified

        if actual_max_slew_Tm_per_s is None:
            print(f"{print_prefix} Could not determine actual max slew rate.")
            slew_ok = False # Cannot confirm
        elif max_slew_limit_Tm_per_s is not None:
            slew_ok = actual_max_slew_Tm_per_s <= max_slew_limit_Tm_per_s
            print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. Limit: {max_slew_limit_Tm_per_s:.2f} T/m/s. Status: {'OK' if slew_ok else 'EXCEEDED'}")
        else:
            print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. No limit provided.")
            # slew_ok remains True if no limit is specified

        if not grad_ok or not slew_ok:
            print(f"{print_prefix} One or more limits exceeded or could not be verified.")
            return False

        print(f"{print_prefix} All specified limits met.")
        return True

    def assess_kspace_coverage(self) -> str:
        """
        Provides a qualitative assessment of k-space coverage for this sequence.
        """
        assessment = f"Variable Density Spiral ({self.spiral_type}) trajectory. "
        if self.density_factor_at_center is not None and self.density_factor_at_center > 1.0:
            assessment += f"Typically oversamples the k-space center (density factor: {self.density_factor_at_center}). "
        elif self.density_factor_at_center is not None and self.density_factor_at_center == 1.0:
            assessment += "Designed for relatively uniform density. "
        else:
            assessment += "Density profile varies based on parameters. "

        if self.undersampling_factor != 1.0:
            assessment += f"Global undersampling factor of {self.undersampling_factor} is applied. "
        return assessment

    def estimate_off_resonance_sensitivity(self) -> str:
        """
        Estimates the sequence's sensitivity to off-resonance effects.
        """
        return (f"Spiral trajectories like '{self.name}' are generally sensitive to "
                "off-resonance effects, which can lead to image blurring or geometric distortions "
                "if not corrected during reconstruction.")

    def assess_motion_robustness(self) -> str:
        """
        Assesses the sequence's robustness to subject motion.
        """
        return (f"The fast imaging capabilities of spiral sequences ({self.num_interleaves} interleaves, "
                f"{self.points_per_interleaf * self.dt_seconds * 1e3:.2f} ms per interleaf readout) "
                "can contribute to motion robustness by reducing scan time per frame/encode. "
                "However, interleaf motion can still be an issue.")

    def suggest_reconstruction_method(self) -> str:
        """
        Suggests a suitable reconstruction method based on sequence properties.
        """
        suggestion = "NUFFT (Non-Uniform Fast Fourier Transform) is the primary reconstruction method for spiral data. "
        if self.estimate_off_resonance_sensitivity().lower().count("sensitive") > 0: # Basic check
            suggestion += "Consider off-resonance correction algorithms (e.g., time-segmented reconstruction, field map correction) "
            suggestion += "to mitigate blurring and distortion. "
        if self.undersampling_factor > 1.0:
             suggestion += "If significantly undersampled, compressed sensing or parallel imaging techniques may be beneficial if applicable. "
        return suggestion

```
