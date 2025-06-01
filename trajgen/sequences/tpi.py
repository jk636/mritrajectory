"""
Defines the TwistedProjectionImagingSequence class.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from trajgen.generators import generate_tpi_trajectory # Corrected import path
from trajgen.sequence_base import MRISequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

__all__ = ['TwistedProjectionImagingSequence']


class TwistedProjectionImagingSequence(MRISequence):
    """
    Represents a Twisted Projection Imaging (TPI) MRI sequence.

    This class uses `trajgen.generators.generate_tpi_trajectory` to create
    the 3D k-space points based on the specified TPI parameters.
    """

    def __init__(self,
                 # Common MRISequence parameters
                 name: str,
                 fov_mm: Union[float, Tuple[float, float, float]],
                 resolution_mm: Union[float, Tuple[float, float, float]],
                 dt_seconds: float,
                 # TPI-specific parameters
                 num_twists: int,
                 points_per_segment: int,
                 cone_angle_deg: float,
                 spiral_turns_per_twist: float,
                 undersampling_factor: float = 1.0,
                 # Other Trajectory/MRISequence parameters
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds: float = 0.0,
                 dead_time_end_seconds: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a TwistedProjectionImagingSequence object.

        Args:
            name (str): Name of the sequence.
            fov_mm (Union[float, Tuple[float,float,float]]): Field of View in millimeters.
            resolution_mm (Union[float, Tuple[float,float,float]]): Desired resolution in millimeters.
            dt_seconds (float): Dwell time (time between k-space samples) in seconds.
            num_twists (int): Number of twisted projection arms/segments.
            points_per_segment (int): Number of k-space points along each segment.
            cone_angle_deg (float): Angle of the cone (half-angle) w.r.t. kz-axis.
            spiral_turns_per_twist (float): Number of spiral turns along one projection.
            undersampling_factor (float): Overall undersampling factor.
            gamma_Hz_per_T (float): Gyromagnetic ratio in Hz/T.
            dead_time_start_seconds (float): Dead time at the beginning of the sequence.
            dead_time_end_seconds (float): Dead time at the end of the sequence.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
        """
        self.num_twists = num_twists
        self.points_per_segment = points_per_segment
        self.cone_angle_deg = cone_angle_deg
        self.spiral_turns_per_twist = spiral_turns_per_twist
        self.undersampling_factor = undersampling_factor

        sequence_specific_params = {
            'num_twists': num_twists,
            'points_per_segment': points_per_segment,
            'cone_angle_deg': cone_angle_deg,
            'spiral_turns_per_twist': spiral_turns_per_twist,
            'undersampling_factor': undersampling_factor,
        }

        # TPI is always 3D
        num_dimensions = 3

        super().__init__(
            name=name,
            fov_mm=fov_mm,
            resolution_mm=resolution_mm,
            num_dimensions=num_dimensions, # TPI is 3D
            dt_seconds=dt_seconds,
            gamma_Hz_per_T=gamma_Hz_per_T,
            sequence_specific_params=sequence_specific_params,
            dead_time_start_seconds=dead_time_start_seconds,
            dead_time_end_seconds=dead_time_end_seconds,
            metadata=metadata
        )

    def _generate_kspace_points(self) -> np.ndarray:
        """
        Generates k-space points for the TPI sequence.
        The `generate_tpi_trajectory` function returns k-space points in rad/m
        with shape (3, N), which is the format expected by the Trajectory base class.
        """
        # FOV and resolution are handled by the generator, including conversion from mm to m.
        # self.num_dimensions is set to 3 in __init__.

        kspace_points_D_N = generate_tpi_trajectory(
            fov_mm=self.fov_mm,
            resolution_mm=self.resolution_mm,
            num_twists=self.num_twists,
            points_per_segment=self.points_per_segment,
            cone_angle_deg=self.cone_angle_deg,
            spiral_turns_per_twist=self.spiral_turns_per_twist,
            undersampling_factor=self.undersampling_factor, # Passed to generator
            gamma_Hz_per_T=self.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
        )
        # The generator already returns (D, N) as per prompt for generate_tpi_trajectory.
        # If it were (N,D), we would kspace_points_D_N.T
        return kspace_points_D_N


    def check_gradient_limits(self, system_limits: Dict[str, Any]) -> bool:
        """
        Checks if the trajectory's gradient waveforms and slew rates are within
        the specified system limits.
        """
        max_grad_limit_Tm_per_m = system_limits.get('max_grad_Tm_per_m')
        max_slew_limit_Tm_per_s = system_limits.get('max_slew_Tm_per_s_per_m')

        actual_max_grad_Tm = self.get_max_grad_Tm()
        actual_max_slew_Tm_per_s = self.get_max_slew_Tm_per_s()

        grad_ok = True
        slew_ok = True

        print_prefix = f"Gradient Limit Check for '{self.name}':"

        if actual_max_grad_Tm is None:
            print(f"{print_prefix} Could not determine actual max gradient.")
            grad_ok = False
        elif max_grad_limit_Tm_per_m is not None:
            grad_ok = actual_max_grad_Tm <= max_grad_limit_Tm_per_m
            print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. Limit: {max_grad_limit_Tm_per_m:.4f} T/m. Status: {'OK' if grad_ok else 'EXCEEDED'}")
        else:
            print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. No limit provided.")

        if actual_max_slew_Tm_per_s is None:
            print(f"{print_prefix} Could not determine actual max slew rate.")
            slew_ok = False
        elif max_slew_limit_Tm_per_s is not None:
            slew_ok = actual_max_slew_Tm_per_s <= max_slew_limit_Tm_per_s
            print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. Limit: {max_slew_limit_Tm_per_s:.2f} T/m/s. Status: {'OK' if slew_ok else 'EXCEEDED'}")
        else:
            print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. No limit provided.")

        if not grad_ok or not slew_ok:
            print(f"{print_prefix} One or more limits exceeded or could not be verified.")
            return False

        print(f"{print_prefix} All specified limits met.")
        return True

    def assess_kspace_coverage(self) -> str:
        """
        Provides a qualitative assessment of k-space coverage for TPI.
        """
        return (f"Twisted Projection Imaging (TPI) with {self.num_twists} twists "
                f"and a cone angle of {self.cone_angle_deg} degrees. "
                "TPI aims for efficient isotropic 3D k-space coverage. The use of spiral segments "
                "on conical surfaces allows for good sampling density control.")

    def estimate_off_resonance_sensitivity(self) -> str:
        """
        Estimates TPI's sensitivity to off-resonance effects.
        """
        return (f"TPI sequences like '{self.name}' can be sensitive to off-resonance effects "
                "due to the spiral nature of the k-space segments. This may lead to blurring or "
                "distortions. Sensitivity might be somewhat less than pure 3D spiral trajectories "
                "depending on the parameters, but correction is often beneficial.")

    def assess_motion_robustness(self) -> str:
        """
        Assesses TPI's robustness to subject motion.
        """
        return (f"The efficient 3D k-space coverage of TPI ({self.name}) can lead to shorter scan times "
                "for volumetric imaging, which indirectly contributes to motion robustness. "
                "The trajectory itself does not have inherent self-navigation properties like radial spokes, "
                "but faster imaging reduces the window for motion.")

    def suggest_reconstruction_method(self) -> str:
        """
        Suggests a suitable reconstruction method for TPI.
        """
        suggestion = "NUFFT (Non-Uniform Fast Fourier Transform) is essential for reconstructing 3D non-Cartesian TPI data. "
        suggestion += "Off-resonance correction algorithms may be necessary. "
        if self.undersampling_factor > 1.0:
             suggestion += "If undersampled, iterative reconstruction techniques incorporating parallel imaging (if applicable) or compressed sensing could be valuable. "
        return suggestion
```
