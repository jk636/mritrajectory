"""
Trajgen: MRI K-Space Trajectory Generation Package
=================================================

`trajgen` is a Python package designed for the generation, manipulation,
and analysis of k-space trajectories used in Magnetic Resonance Imaging (MRI).

It provides tools to:
- Generate various types of k-space trajectories (Spiral, Radial, EPI, Cones, Rosette).
- Handle trajectory objects, including calculation of associated parameters like
  gradient waveforms, slew rates, and performance metrics.
- Apply hardware constraints (maximum gradient, maximum slew rate) to trajectories.
- Visualize trajectories and their properties.
- Perform basic image reconstruction from k-space data acquired on these trajectories.

The package is structured into modules for trajectory definition, generation algorithms,
utility functions, and a high-level trajectory generator class.
"""

__version__ = "0.1.0"

from .trajectory import Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T
from . import generators
from . import utils
from .kspace_generator import KSpaceTrajectoryGenerator

# For easier access to specific generator functions if desired by users
from .generators import generate_spiral_trajectory, generate_radial_trajectory, generate_cones_trajectory, generate_epi_trajectory, generate_rosette_trajectory

# For easier access to specific utility functions if desired
from .utils import constrain_trajectory, reconstruct_image, display_trajectory


__all__ = [
    'Trajectory',
    'COMMON_NUCLEI_GAMMA_HZ_PER_T',
    'KSpaceTrajectoryGenerator',
    'generate_spiral_trajectory',
    'generate_radial_trajectory',
    'generate_cones_trajectory',
    'generate_epi_trajectory',
    'generate_rosette_trajectory', # Added
    'constrain_trajectory',
    'reconstruct_image',
    'display_trajectory',
    'generators', # Expose the modules themselves too
    'utils'
]
