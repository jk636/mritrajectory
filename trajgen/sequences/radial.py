"""
Defines the RadialSequence class.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from trajgen.generators import generate_radial_trajectory
from trajgen.sequence_base import MRISequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

__all__ = ['RadialSequence']


class RadialSequence(MRISequence):
    """
    Represents a radial MRI sequence.

    This class uses `trajgen.generators.generate_radial_trajectory` to create
    the k-space points based on the specified radial parameters.
    """

    def __init__(self,
                 # Common MRISequence parameters
                 name: str,
                 fov_mm: Union[float, Tuple[float, ...]],
                 resolution_mm: Union[float, Tuple[float, ...]],
                 num_dimensions: int, # Typically 2 or 3 for radial
                 dt_seconds: float,
                 # Radial-specific parameters
                 num_spokes: int,
                 points_per_spoke: int,
                 projection_angle_increment: Union[str, float] = 'golden_angle',
                 # Other Trajectory/MRISequence parameters
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds: float = 0.0,
                 dead_time_end_seconds: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a RadialSequence object.

        Args:
            name (str): Name of the sequence.
            fov_mm (Union[float, Tuple[float, ...]]): Field of View in millimeters.
            resolution_mm (Union[float, Tuple[float, ...]]): Desired resolution in millimeters.
            num_dimensions (int): Number of spatial dimensions (2 for 2D radial, 3 for 3D radial).
            dt_seconds (float): Dwell time (time between k-space samples) in seconds.
            num_spokes (int): Number of radial spokes.
            points_per_spoke (int): Number of k-space points per spoke.
            projection_angle_increment (Union[str, float]): Angle increment between spokes.
                Can be 'golden_angle' or a float value in radians.
            gamma_Hz_per_T (float): Gyromagnetic ratio in Hz/T.
            dead_time_start_seconds (float): Dead time at the beginning of the sequence.
            dead_time_end_seconds (float): Dead time at the end of the sequence.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
        """
        self.num_spokes = num_spokes
        self.points_per_spoke = points_per_spoke
        self.projection_angle_increment = projection_angle_increment

        sequence_specific_params = {
            'num_spokes': num_spokes,
            'points_per_spoke': points_per_spoke,
            'projection_angle_increment': projection_angle_increment,
        }

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
        Generates k-space points for the radial sequence.
        The `generate_radial_trajectory` function is expected to handle FOV/resolution
        conversion if necessary, or accept them in mm. Based on its previous usage
        (though not explicitly stated for radial), it's safer to assume it might
        need meters, or that it handles units internally based on provided values.
        The prompt states "fov_mm" and "resolution_mm" are passed explicitly.
        The generator `generate_radial_trajectory` returns k-space points in rad/m.
        """

        # generate_radial_trajectory expects fov in m and resolution in m.
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

        kspace_points, _, _ = generate_radial_trajectory(
            num_dimensions=self.num_dimensions,
            num_spokes=self.num_spokes,
            points_per_spoke=self.points_per_spoke,
            fov_m=fov_m, # type: ignore
            resolution_m=resolution_m, # type: ignore
            projection_angle_increment=self.projection_angle_increment,
            gamma_Hz_per_T=self.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
        )
        # generate_radial_trajectory returns k-space of shape (N, D)
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

        Returns:
            bool: True if within limits, False otherwise.
        """
        max_grad_limit_Tm_per_m = system_limits.get('max_grad_Tm_per_m')
        max_slew_limit_Tm_per_s = system_limits.get('max_slew_Tm_per_s_per_m') # Key used in prompt

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
        Provides a qualitative assessment of k-space coverage for this sequence.
        """
        angle_info = self.projection_angle_increment
        if isinstance(self.projection_angle_increment, str):
            angle_info = f"'{self.projection_angle_increment}' scheme"
        else:
            angle_info = f"{self.projection_angle_increment:.4f} rad increment"

        return (f"Radial trajectory ({self.num_dimensions}D) with {self.num_spokes} spokes "
                f"using {angle_info}. "
                "K-space center is inherently oversampled, promoting robustness. "
                "Peripheral coverage depends on the number of spokes and desired resolution.")

    def estimate_off_resonance_sensitivity(self) -> str:
        """
        Estimates the sequence's sensitivity to off-resonance effects.
        """
        return (f"Radial trajectories like '{self.name}' are generally considered less sensitive "
                "to off-resonance effects than spiral trajectories due to the repeated sampling "
                "of the k-space center. However, significant off-resonance can still cause "
                "artifacts, such as blurring or streaking, particularly at the periphery.")

    def assess_motion_robustness(self) -> str:
        """
        Assesses the sequence's robustness to subject motion.
        """
        return (f"Radial trajectories ({self.name}) exhibit good motion robustness. This is due to "
                "the frequent oversampling of the k-space center, which makes them less sensitive "
                "to motion artifacts and allows for potential motion correction/self-navigation "
                "during reconstruction by using data consistency across spokes.")

    def suggest_reconstruction_method(self) -> str:
        """
        Suggests a suitable reconstruction method based on sequence properties.
        """
        methods = ["NUFFT (Non-Uniform Fast Fourier Transform)", "Filtered Back-Projection (FBP)"]
        suggestion = f"Common reconstruction methods for radial data include: {', '.join(methods)}. "
        suggestion += "Radial data is well-suited for motion correction techniques. "
        if self.num_spokes * self.points_per_spoke < (np.pi / 4 * (self.fov_mm / self.resolution_mm)**2 if isinstance(self.fov_mm, float) and isinstance(self.resolution_mm, float) else 0) : # Simplified Nyquist check for 2D
             suggestion += "If undersampled, iterative reconstruction or compressed sensing can be beneficial. "
        return suggestion
```
