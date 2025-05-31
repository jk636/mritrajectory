"""
High-Level K-Space Trajectory Generator Class
---------------------------------------------

This module defines the `KSpaceTrajectoryGenerator` class, which acts as a
factory and manager for creating various k-space trajectories. It uses the
underlying functions from `trajgen.generators` and wraps the results into
`trajgen.trajectory.Trajectory` objects, optionally applying hardware
constraints via `trajgen.utils`.
"""
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any

from .trajectory import Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T
from . import generators
from . import utils

__all__ = ['KSpaceTrajectoryGenerator']

class KSpaceTrajectoryGenerator:
    """
    A high-level generator for creating various k-space trajectories.
    """
    def __init__(self,
                 fov_mm: Union[float, Tuple[float, ...]],
                 resolution_mm: Union[float, Tuple[float, ...]],
                 num_dimensions: int,
                 dt_s: float,
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 max_grad_mT_per_m: Optional[float] = None,
                 max_slew_Tm_per_s_ms: Optional[float] = None):
        """
        Initializes the KSpaceTrajectoryGenerator.

        Parameters:
        - fov_mm: Field of view in millimeters (e.g., (200,200) or 200 for isotropic).
        - resolution_mm: Desired resolution in millimeters (e.g., (1,1) or 1 for isotropic).
        - num_dimensions: Number of spatial dimensions (2 or 3).
        - dt_s: Dwell time in seconds.
        - gamma_Hz_per_T: Gyromagnetic ratio.
        - max_grad_mT_per_m (Optional): Maximum gradient strength (mT/m).
        - max_slew_Tm_per_s_ms (Optional): Maximum slew rate (T/m/s/ms).
        """
        if num_dimensions not in [2, 3]:
            raise ValueError("num_dimensions must be 2 or 3.")
        self.num_dimensions = num_dimensions

        # Validate and store FOV
        if isinstance(fov_mm, (int, float)):
            self.fov_mm: Tuple[float, ...] = tuple([float(fov_mm)] * self.num_dimensions)
        elif isinstance(fov_mm, (tuple, list)) and len(fov_mm) == self.num_dimensions:
            self.fov_mm = tuple(map(float, fov_mm))
        else:
            raise ValueError(f"fov_mm must be a number or a tuple/list of length {self.num_dimensions}.")

        # Validate and store resolution
        if isinstance(resolution_mm, (int, float)):
            self.resolution_mm: Tuple[float, ...] = tuple([float(resolution_mm)] * self.num_dimensions)
        elif isinstance(resolution_mm, (tuple, list)) and len(resolution_mm) == self.num_dimensions:
            self.resolution_mm = tuple(map(float, resolution_mm))
        else:
            raise ValueError(f"resolution_mm must be a number or a tuple/list of length {self.num_dimensions}.")

        if not (dt_s > 0):
            raise ValueError("dt_s must be positive.")
        self.dt_s = float(dt_s)
        self.gamma_Hz_per_T = float(gamma_Hz_per_T)

        self.max_grad_mT_per_m = float(max_grad_mT_per_m) if max_grad_mT_per_m is not None else None
        self.max_slew_Tm_per_s_ms = float(max_slew_Tm_per_s_ms) if max_slew_Tm_per_s_ms is not None else None

        self.params_initialized = True # Flag to indicate successful init

    def get_params(self) -> Dict[str, Any]:
        """Returns a dictionary of the generator's current settings."""
        return {
            "fov_mm": self.fov_mm,
            "resolution_mm": self.resolution_mm,
            "num_dimensions": self.num_dimensions,
            "dt_s": self.dt_s,
            "gamma_Hz_per_T": self.gamma_Hz_per_T,
            "max_grad_mT_per_m": self.max_grad_mT_per_m,
            "max_slew_Tm_per_s_ms": self.max_slew_Tm_per_s_ms
        }

    def set_params(self, **kwargs: Any) -> None:
        """Allows updating generator settings after initialization."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Re-run validation if specific setters are implemented, for now direct set
                if key == "fov_mm" or key == "resolution_mm" or key == "num_dimensions":
                     raise ValueError(f"Parameter {key} should be set at init or via a dedicated method with validation.")
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        # Could re-validate related params here if needed

    def create_spiral(self,
                      num_interleaves: int,
                      points_per_interleaf: int,
                      undersampling_factor: float = 1.0,
                      spiral_type: str = 'archimedean',
                      density_transition_radius_factor: Optional[float] = None,
                      density_factor_at_center: Optional[float] = None,
                      apply_constraints: bool = True,
                      name_prefix: str = "spiral",
                      **kwargs) -> Trajectory:
        """
        Creates a spiral trajectory.

        Parameters:
        - num_interleaves (int): Number of spiral interleaves.
        - points_per_interleaf (int): Number of k-space points per interleaf.
        - undersampling_factor (float): Factor for undersampling.
        - spiral_type (str): Type of spiral (e.g., 'archimedean').
        - density_transition_radius_factor (Optional[float]): Factor of k_max (0-1) where density transition occurs for variable density spirals.
        - density_factor_at_center (Optional[float]): Factor by which center is denser than periphery for variable density spirals.
        - apply_constraints (bool): Whether to apply configured hardware constraints.
        - name_prefix (str): Prefix for the trajectory name.
        - **kwargs: Additional metadata to store in the trajectory.

        Returns:
        - Trajectory: The generated spiral trajectory object.
        """
        k_space_points = generators.generate_spiral_trajectory(
            fov_mm=self.fov_mm,
            resolution_mm=self.resolution_mm,
            num_dimensions=self.num_dimensions,
            num_interleaves=num_interleaves,
            points_per_interleaf=points_per_interleaf,
            undersampling_factor=undersampling_factor,
            spiral_type=spiral_type,
            density_transition_radius_factor=density_transition_radius_factor, # Added
            density_factor_at_center=density_factor_at_center,       # Added
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        density_suffix = ""
        if density_factor_at_center is not None and density_factor_at_center != 1.0 and density_transition_radius_factor is not None:
            density_suffix = f"_vd_t{density_transition_radius_factor}_c{density_factor_at_center}"

        traj_name = f"{name_prefix}_{self.num_dimensions}D_{num_interleaves}il_{points_per_interleaf}pts{density_suffix}"
        metadata = self.get_params()
        metadata.update({
            "trajectory_type": "spiral",
            "num_interleaves": num_interleaves,
            "points_per_interleaf": points_per_interleaf,
            "undersampling_factor": undersampling_factor,
            "spiral_type_requested": spiral_type,
            "density_transition_radius_factor": density_transition_radius_factor, # Added
            "density_factor_at_center": density_factor_at_center,           # Added
            "constraints_applied_request": apply_constraints
        })
        metadata.update(kwargs) # Add any extra kwargs to metadata

        trajectory_obj = Trajectory(
            name=traj_name,
            kspace_points_rad_per_m=k_space_points,
            dt_seconds=self.dt_s,
            metadata=metadata,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        if apply_constraints and self.max_grad_mT_per_m is not None and self.max_slew_Tm_per_s_ms is not None:
            trajectory_obj = utils.constrain_trajectory(
                trajectory_obj=trajectory_obj,
                max_grad_mT_per_m=self.max_grad_mT_per_m,
                max_slew_Tm_per_s_ms=self.max_slew_Tm_per_s_ms
                # dt_s and gamma_Hz_per_T will be taken from trajectory_obj by constrain_trajectory
            )
            trajectory_obj.metadata['constraints_actually_applied'] = True
        else:
            trajectory_obj.metadata['constraints_actually_applied'] = False

        return trajectory_obj

    def create_radial(self,
                      num_spokes: int,
                      points_per_spoke: int,
                      projection_angle_increment: Union[str, float] = 'golden_angle',
                      apply_constraints: bool = True,
                      name_prefix: str = "radial",
                      **kwargs) -> Trajectory:
        """
        Creates a radial trajectory.

        Args:
            num_spokes (int): Number of radial spokes.
            points_per_spoke (int): Number of k-space points per spoke.
            projection_angle_increment (Union[str, float]): Angle increment strategy.
                Defaults to 'golden_angle'. Can be a float for fixed angle in degrees.
            apply_constraints (bool): Whether to apply hardware constraints.
            name_prefix (str): Prefix for the trajectory name.
            **kwargs: Additional metadata.

        Returns:
            Trajectory: The generated radial trajectory object.
        """
        k_space_points = generators.generate_radial_trajectory(
            num_spokes=num_spokes,
            points_per_spoke=points_per_spoke,
            num_dimensions=self.num_dimensions,
            fov_mm=self.fov_mm,
            resolution_mm=self.resolution_mm,
            projection_angle_increment=projection_angle_increment,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        traj_name = f"{name_prefix}_{self.num_dimensions}D_{num_spokes}spk_{points_per_spoke}pts"
        metadata = self.get_params()
        metadata.update({
            "trajectory_type": "radial",
            "num_spokes": num_spokes,
            "points_per_spoke": points_per_spoke,
            "projection_angle_increment": projection_angle_increment,
            "constraints_applied_request": apply_constraints
        })
        metadata.update(kwargs)


        trajectory_obj = Trajectory(
            name=traj_name,
            kspace_points_rad_per_m=k_space_points,
            dt_seconds=self.dt_s,
            metadata=metadata,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        if apply_constraints and self.max_grad_mT_per_m is not None and self.max_slew_Tm_per_s_ms is not None:
            trajectory_obj = utils.constrain_trajectory(
                trajectory_obj=trajectory_obj,
                max_grad_mT_per_m=self.max_grad_mT_per_m,
                max_slew_Tm_per_s_ms=self.max_slew_Tm_per_s_ms
            )
            trajectory_obj.metadata['constraints_actually_applied'] = True
        else:
            trajectory_obj.metadata['constraints_actually_applied'] = False

        return trajectory_obj

    def create_cones_trajectory(self,
                                num_cones: int,
                                points_per_cone: int,
                                cone_angle_deg: float,
                                undersampling_factor: float = 1.0,
                                rotation_angle_increment_deg: Optional[float] = None,
                                apply_constraints: bool = True,
                                name_prefix: str = "cones",
                                **kwargs) -> Trajectory:
        """
        Creates a 3D Cones trajectory.

        Args:
            num_cones (int): Number of cones.
            points_per_cone (int): Number of k-space points per cone.
            cone_angle_deg (float): Half-angle of the cone in degrees.
            undersampling_factor (float): Undersampling factor for spiral on cone.
            rotation_angle_increment_deg (Optional[float]): Rotation between cones.
            apply_constraints (bool): Whether to apply hardware constraints.
            name_prefix (str): Prefix for the trajectory name.
            **kwargs: Additional metadata.

        Returns:
            Trajectory: The generated cones trajectory object.

        Raises:
            ValueError: If the generator `num_dimensions` is not 3.
        """
        if self.num_dimensions != 3:
            raise ValueError("Cones trajectory is inherently 3D. Initialize KSpaceTrajectoryGenerator with num_dimensions=3.")

        k_space_points = generators.generate_cones_trajectory(
            fov_mm=self.fov_mm, # Should be 3D FOV from constructor
            resolution_mm=self.resolution_mm, # Should be 3D res from constructor
            num_cones=num_cones,
            points_per_cone=points_per_cone,
            cone_angle_deg=cone_angle_deg,
            undersampling_factor=undersampling_factor,
            rotation_angle_increment_deg=rotation_angle_increment_deg,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        traj_name = f"{name_prefix}_{num_cones}cones_{points_per_cone}pts_angle{cone_angle_deg}"
        metadata = self.get_params()
        metadata.update({
            "trajectory_type": "cones",
            "num_cones": num_cones,
            "points_per_cone": points_per_cone,
            "cone_angle_deg": cone_angle_deg,
            "undersampling_factor": undersampling_factor,
            "rotation_angle_increment_deg": rotation_angle_increment_deg,
            "constraints_applied_request": apply_constraints
        })
        metadata.update(kwargs)

        trajectory_obj = Trajectory(
            name=traj_name,
            kspace_points_rad_per_m=k_space_points,
            dt_seconds=self.dt_s,
            metadata=metadata,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        if apply_constraints and self.max_grad_mT_per_m is not None and self.max_slew_Tm_per_s_ms is not None:
            trajectory_obj = utils.constrain_trajectory(
                trajectory_obj=trajectory_obj,
                max_grad_mT_per_m=self.max_grad_mT_per_m,
                max_slew_Tm_per_s_ms=self.max_slew_Tm_per_s_ms
            )
            trajectory_obj.metadata['constraints_actually_applied'] = True
        else:
            trajectory_obj.metadata['constraints_actually_applied'] = False

        return trajectory_obj

    def create_epi_trajectory(self,
                              num_echoes: int,
                              points_per_echo: int,
                              ramp_sample_percentage: float = 0.1,
                              epi_type: str = 'flyback',
                              phase_encode_direction: str = 'y',
                              acquire_every_other_line: bool = False,
                              apply_constraints: bool = True,
                              name_prefix: str = "epi",
                              **kwargs) -> Trajectory:
        """
        Creates a 2D EPI trajectory.

        Args:
            num_echoes (int): Number of echoes (k-space lines).
            points_per_echo (int): Number of points per echo.
            ramp_sample_percentage (float): Percentage of points for ramps.
            epi_type (str): 'flyback' or 'gradient_recalled'.
            phase_encode_direction (str): 'x' or 'y'.
            acquire_every_other_line (bool): If True, skip lines.
            apply_constraints (bool): Whether to apply hardware constraints.
            name_prefix (str): Prefix for the trajectory name.
            **kwargs: Additional metadata.

        Returns:
            Trajectory: The generated EPI trajectory object.

        Raises:
            ValueError: If the generator `num_dimensions` is not 2.
        """
        if self.num_dimensions != 2:
            raise ValueError("EPI trajectory is 2D. Initialize KSpaceTrajectoryGenerator with num_dimensions=2.")

        k_space_points = generators.generate_epi_trajectory(
            fov_mm=self.fov_mm, # Should be 2D FOV
            resolution_mm=self.resolution_mm, # Should be 2D res
            num_echoes=num_echoes,
            points_per_echo=points_per_echo,
            ramp_sample_percentage=ramp_sample_percentage,
            epi_type=epi_type,
            phase_encode_direction=phase_encode_direction,
            acquire_every_other_line=acquire_every_other_line,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        actual_num_acquired_echoes = k_space_points.shape[1] // points_per_echo if points_per_echo > 0 else 0


        traj_name = f"{name_prefix}_{epi_type}_{phase_encode_direction}pe_{actual_num_acquired_echoes}x{points_per_echo}"
        metadata = self.get_params()
        metadata.update({
            "trajectory_type": "epi",
            "num_echoes_requested": num_echoes,
            "num_echoes_acquired": actual_num_acquired_echoes,
            "points_per_echo": points_per_echo,
            "ramp_sample_percentage": ramp_sample_percentage,
            "epi_type": epi_type,
            "phase_encode_direction": phase_encode_direction,
            "acquire_every_other_line": acquire_every_other_line,
            "constraints_applied_request": apply_constraints
        })
        metadata.update(kwargs)

        trajectory_obj = Trajectory(
            name=traj_name,
            kspace_points_rad_per_m=k_space_points,
            dt_seconds=self.dt_s,
            metadata=metadata,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        if apply_constraints and self.max_grad_mT_per_m is not None and self.max_slew_Tm_per_s_ms is not None:
            trajectory_obj = utils.constrain_trajectory(
                trajectory_obj=trajectory_obj,
                max_grad_mT_per_m=self.max_grad_mT_per_m,
                max_slew_Tm_per_s_ms=self.max_slew_Tm_per_s_ms
            )
            trajectory_obj.metadata['constraints_actually_applied'] = True
        else:
            trajectory_obj.metadata['constraints_actually_applied'] = False

        return trajectory_obj

    def create_rosette_trajectory(self,
                                  num_petals: int,
                                  total_points: int, # Renamed from points_per_petal_cycle for clarity
                                  num_radial_cycles: int, # Renamed from num_cycles
                                  k_max_rosette_factor: float = 1.0,
                                  apply_constraints: bool = True,
                                  name_prefix: str = "rosette",
                                  **kwargs) -> Trajectory:
        """
        Creates a 2D Rosette trajectory.

        Args:
            num_petals (int): Number of major petals.
            total_points (int): Total k-space points for the trajectory.
            num_radial_cycles (int): Number of radial oscillations (lobes).
            k_max_rosette_factor (float): Scaling factor for k_max (0 to 1.0).
            apply_constraints (bool): Whether to apply hardware constraints.
            name_prefix (str): Prefix for the trajectory name.
            **kwargs: Additional metadata.

        Returns:
            Trajectory: The generated rosette trajectory object.

        Raises:
            ValueError: If the generator `num_dimensions` is not 2.
        """
        if self.num_dimensions != 2:
            raise ValueError("Rosette trajectory is 2D. Initialize KSpaceTrajectoryGenerator with num_dimensions=2.")

        k_space_points = generators.generate_rosette_trajectory(
            fov_mm=self.fov_mm,
            resolution_mm=self.resolution_mm,
            num_petals=num_petals,
            total_points=total_points,
            num_radial_cycles=num_radial_cycles,
            k_max_rosette_factor=k_max_rosette_factor,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        traj_name = f"{name_prefix}_{num_petals}petals_{num_radial_cycles}cycles_{total_points}pts"
        metadata = self.get_params()
        metadata.update({
            "trajectory_type": "rosette",
            "num_petals": num_petals,
            "total_points": total_points,
            "num_radial_cycles": num_radial_cycles,
            "k_max_rosette_factor": k_max_rosette_factor,
            "constraints_applied_request": apply_constraints
        })
        metadata.update(kwargs)

        trajectory_obj = Trajectory(
            name=traj_name,
            kspace_points_rad_per_m=k_space_points,
            dt_seconds=self.dt_s,
            metadata=metadata,
            gamma_Hz_per_T=self.gamma_Hz_per_T
        )

        if apply_constraints and self.max_grad_mT_per_m is not None and self.max_slew_Tm_per_s_ms is not None:
            trajectory_obj = utils.constrain_trajectory(
                trajectory_obj=trajectory_obj,
                max_grad_mT_per_m=self.max_grad_mT_per_m,
                max_slew_Tm_per_s_ms=self.max_slew_Tm_per_s_ms
            )
            trajectory_obj.metadata['constraints_actually_applied'] = True
        else:
            trajectory_obj.metadata['constraints_actually_applied'] = False

        return trajectory_obj


# Removed __main__ block, examples to be covered by tests.
