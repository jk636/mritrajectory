"""
Defines the PropellerBladeSequence class.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from trajgen.generators import generate_propeller_blade_trajectory
from trajgen.sequence_base import MRISequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

__all__ = ['PropellerBladeSequence']


class PropellerBladeSequence(MRISequence):
    """
    Represents a PROPELLER/BLADE MRI sequence.

    This sequence consists of multiple "blades" or "strips", where each blade
    is a Cartesian acquisition (a set of parallel lines) that is rotated
    around the k-space center. This design is known for its motion correction
    capabilities.
    """

    def __init__(self,
                 # Common MRISequence parameters
                 name: str,
                 fov_mm: Union[float, Tuple[float, float]], # FOV of a single blade
                 resolution_mm: Union[float, Tuple[float, float]], # Resolution within a blade
                 dt_seconds: float,
                 # PROPELLER/BLADE-specific parameters
                 num_blades: int,
                 lines_per_blade: int,
                 points_per_line: int,
                 blade_rotation_angle_increment_deg: float,
                 # Other Trajectory/MRISequence parameters
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds: float = 0.0,
                 dead_time_end_seconds: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a PropellerBladeSequence object.

        Args:
            name (str): Name of the sequence.
            fov_mm (Union[float, Tuple[float, float]]): Field of View for a single blade/strip.
                The first component is along readout, second along phase-encode within blade.
            resolution_mm (Union[float, Tuple[float, float]]): Resolution within a single blade.
            dt_seconds (float): Dwell time (time between k-space samples) in seconds.
            num_blades (int): Number of blades.
            lines_per_blade (int): Number of phase-encode lines within each blade.
            points_per_line (int): Number of readout points along each line within a blade.
            blade_rotation_angle_increment_deg (float): Angle to rotate successive blades.
            gamma_Hz_per_T (float): Gyromagnetic ratio in Hz/T.
            dead_time_start_seconds (float): Dead time at the beginning of the sequence.
            dead_time_end_seconds (float): Dead time at the end of the sequence.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
        """
        self.num_blades = num_blades
        self.lines_per_blade = lines_per_blade
        self.points_per_line = points_per_line
        self.blade_rotation_angle_increment_deg = blade_rotation_angle_increment_deg

        sequence_specific_params = {
            'num_blades': num_blades,
            'lines_per_blade': lines_per_blade,
            'points_per_line': points_per_line,
            'blade_rotation_angle_increment_deg': blade_rotation_angle_increment_deg,
        }

        # PROPELLER/BLADE is typically 2D
        num_dimensions = 2

        super().__init__(
            name=name,
            fov_mm=fov_mm, # This FOV/res is for the blade, not overall image
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
        Generates k-space points for the PROPELLER/BLADE sequence.
        The generator returns k-space points in rad/m with shape (2, N).
        """
        kspace_points_D_N = generate_propeller_blade_trajectory(
            fov_mm=self.fov_mm, # Passed as defined for the blade
            resolution_mm=self.resolution_mm, # Passed as defined for the blade
            num_blades=self.num_blades,
            lines_per_blade=self.lines_per_blade,
            points_per_line=self.points_per_line,
            blade_rotation_angle_increment_deg=self.blade_rotation_angle_increment_deg,
            gamma_Hz_per_T=self.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
        )
        return kspace_points_D_N

    def check_gradient_limits(self, system_limits: Dict[str, Any]) -> bool:
        max_grad_limit_Tm_per_m = system_limits.get('max_grad_Tm_per_m')
        max_slew_limit_Tm_per_s = system_limits.get('max_slew_Tm_per_s_per_m')

        actual_max_grad_Tm = self.get_max_grad_Tm()
        actual_max_slew_Tm_per_s = self.get_max_slew_Tm_per_s()
        grad_ok, slew_ok = True, True
        print_prefix = f"Gradient Limit Check for '{self.name}':"

        if actual_max_grad_Tm is None: grad_ok = False; print(f"{print_prefix} Could not determine actual max gradient.")
        elif max_grad_limit_Tm_per_m is not None:
            grad_ok = actual_max_grad_Tm <= max_grad_limit_Tm_per_m
            print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. Limit: {max_grad_limit_Tm_per_m:.4f} T/m. Status: {'OK' if grad_ok else 'EXCEEDED'}")
        else: print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. No limit provided.")

        if actual_max_slew_Tm_per_s is None: slew_ok = False; print(f"{print_prefix} Could not determine actual max slew rate.")
        elif max_slew_limit_Tm_per_s is not None:
            slew_ok = actual_max_slew_Tm_per_s <= max_slew_limit_Tm_per_s
            print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. Limit: {max_slew_limit_Tm_per_s:.2f} T/m/s. Status: {'OK' if slew_ok else 'EXCEEDED'}")
        else: print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. No limit provided.")

        return grad_ok and slew_ok

    def assess_kspace_coverage(self) -> str:
        return (f"PROPELLER/BLADE sequence '{self.name}' with {self.num_blades} blades. "
                "K-space center is oversampled due to overlapping blades. "
                "Peripheral k-space coverage depends on the number of blades and their width. "
                "Gaps may exist between blades at higher k-space radii if not enough blades are used.")

    def estimate_off_resonance_sensitivity(self) -> str:
        return (f"Each blade in PROPELLER/BLADE ('{self.name}') is Cartesian, making it relatively robust "
                "to off-resonance within the blade. However, phase inconsistencies between blades "
                "due to off-resonance or other effects can cause artifacts if not corrected.")

    def assess_motion_robustness(self) -> str:
        return (f"PROPELLER/BLADE sequences like '{self.name}' are designed for motion robustness. "
                "Each blade provides a consistent set of k-space data that can be used to detect "
                "and correct for motion (translation and rotation) between blade acquisitions. "
                "The oversampling of the k-space center also contributes to this robustness.")

    def suggest_reconstruction_method(self) -> str:
        return (f"Reconstruction for PROPELLER/BLADE ('{self.name}') typically involves: "
                "1. Motion correction using data from individual blades (phase correction, registration). "
                "2. Gridding of the corrected, rotated blade data onto a Cartesian grid. "
                "3. FFT of the gridded k-space. Iterative methods can also be used.")
```
