"""
Defines the base class for MRI sequences.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T, Trajectory

__all__ = ['MRISequence']


class MRISequence(Trajectory, ABC):
    """
    Abstract base class for specific MRI pulse sequences.

    This class provides a common interface for generating k-space trajectories
    based on sequence-specific parameters and inherits k-space handling
    and analysis capabilities from the Trajectory class.

    Attributes:
        fov_mm (Union[float, Tuple[float, ...]]): Field of View in millimeters.
        resolution_mm (Union[float, Tuple[float, ...]]): Desired resolution in millimeters.
        num_dimensions (int): Number of spatial dimensions (1, 2, or 3).
    """

    def __init__(self,
                 name: str,
                 fov_mm: Union[float, Tuple[float, ...]],
                 resolution_mm: Union[float, Tuple[float, ...]],
                 num_dimensions: int,
                 dt_seconds: float,
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 sequence_specific_params: Optional[Dict[str, Any]] = None,
                 dead_time_start_seconds: float = 0.0,
                 dead_time_end_seconds: float = 0.0,
                 # Allow passing other Trajectory metadata if needed
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes an MRISequence object.

        Args:
            name (str): Name of the sequence.
            fov_mm (Union[float, Tuple[float, ...]]): Field of View in millimeters.
                Can be a single float for isotropic FOV or a tuple for anisotropic.
            resolution_mm (Union[float, Tuple[float, ...]]): Desired resolution in millimeters.
                Can be a single float for isotropic resolution or a tuple.
            num_dimensions (int): Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
            dt_seconds (float): Dwell time (time between k-space samples) in seconds.
            gamma_Hz_per_T (float): Gyromagnetic ratio in Hz/T.
            sequence_specific_params (Optional[Dict[str, Any]]): Dictionary of parameters
                specific to the derived pulse sequence.
            dead_time_start_seconds (float): Dead time at the beginning of the sequence.
            dead_time_end_seconds (float): Dead time at the end of the sequence.
            metadata (Optional[Dict[str, Any]]): Additional metadata to be passed to Trajectory.
        """
        self.fov_mm = fov_mm
        self.resolution_mm = resolution_mm
        self.num_dimensions = num_dimensions

        # _generate_kspace_points is responsible for using fov, resolution, etc.
        kspace_points_rad_per_m = self._generate_kspace_points()

        # Combine sequence_specific_params with other important params for Trajectory's sequence_params
        # The Trajectory class expects 'sequence_params' for its own new attribute.
        # We also store sequence_specific_params directly if needed by derived classes.
        self.sequence_specific_params = sequence_specific_params or {}

        # Initialize metadata if None, then update with sequence-specific identifying info
        _metadata = metadata or {}
        _metadata.update({
            'sequence_name': name,
            'fov_mm': fov_mm,
            'resolution_mm': resolution_mm,
            'num_dimensions': num_dimensions,
        })


        super().__init__(
            name=name,
            kspace_points_rad_per_m=kspace_points_rad_per_m,
            dt_seconds=dt_seconds,
            gamma_Hz_per_T=gamma_Hz_per_T,
            metadata=_metadata, # Pass combined metadata
            sequence_params=self.sequence_specific_params, # Pass specific params to Trajectory's sequence_params
            dead_time_start_seconds=dead_time_start_seconds,
            dead_time_end_seconds=dead_time_end_seconds
        )

    @abstractmethod
    def _generate_kspace_points(self) -> np.ndarray:
        """
        Abstract method to generate k-space points for the specific sequence.

        This method must be implemented by derived classes. It should use
        parameters like fov_mm, resolution_mm, and num_dimensions stored in the
        instance to calculate the k-space trajectory.

        Returns:
            np.ndarray: Array of k-space points in rad/m, shape (D, N).
        """
        pass

    @abstractmethod
    def check_gradient_limits(self, system_limits: Dict[str, Any]) -> bool:
        """
        Checks if the trajectory's gradient waveforms and slew rates are within
        the specified system limits. Overrides Trajectory.check_gradient_limits.

        Args:
            system_limits (Dict[str, Any]): A dictionary containing system limits,
                                  e.g., {'max_grad_mT_per_m': 40, 'max_slew_T_per_m_per_s': 150}.
                                  Units should be consistent (e.g., T/m, T/m/s).

        Returns:
            bool: True if within limits, False otherwise.
        """
        pass

    @abstractmethod
    def assess_kspace_coverage(self) -> str:
        """
        Provides a qualitative assessment of k-space coverage for this sequence.
        Overrides Trajectory.assess_kspace_coverage.

        Returns:
            str: A string describing the k-space coverage.
        """
        pass

    @abstractmethod
    def estimate_off_resonance_sensitivity(self) -> str:
        """
        Estimates the sequence's sensitivity to off-resonance effects.
        Overrides Trajectory.estimate_off_resonance_sensitivity.

        Returns:
            str: A string describing the sensitivity.
        """
        pass

    @abstractmethod
    def assess_motion_robustness(self) -> str:
        """
        Assesses the sequence's robustness to subject motion.
        Overrides Trajectory.assess_motion_robustness.

        Returns:
            str: A string describing the motion robustness.
        """
        pass

    @abstractmethod
    def suggest_reconstruction_method(self) -> str:
        """
        Suggests a suitable reconstruction method based on sequence properties.
        Overrides Trajectory.suggest_reconstruction_method.

        Returns:
            str: A string suggesting a reconstruction method.
        """
        pass
