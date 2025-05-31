# Trajgen: MRI K-Space Trajectory Generation Package

## Introduction

`trajgen` is a Python package for generating, manipulating, and analyzing k-space trajectories used in Magnetic Resonance Imaging (MRI). It provides a flexible framework for researchers and developers working on pulse sequence design and image reconstruction.

The package allows users to:
- Generate various 2D and 3D k-space trajectories.
- Manage trajectory data, including k-space coordinates, gradient waveforms, and timing information.
- Calculate trajectory-specific properties and performance metrics.
- Apply simplified hardware constraints (maximum gradient, maximum slew rate).
- Visualize trajectories and associated waveforms.
- Perform basic image reconstruction using gridding techniques.

## Key Features

*   **Trajectory Types Implemented:**
    *   **2D:**
        *   Spiral (uniform and variable density)
        *   Radial
        *   EPI (Echo Planar Imaging - flyback and gradient-recalled)
        *   Rosette
    *   **3D:**
        *   Stack-of-Spirals (based on 2D spiral)
        *   3D Radial (Kooshball-like, using golden angle distribution)
        *   Cones (spirals on cone surfaces)
*   **Customizable Parameters:** Control Field of View (FOV), resolution, dwell time (`dt_s`), gyromagnetic ratio (`gamma_Hz_per_T`), and trajectory-specific parameters (e.g., number of interleaves, spokes, petals, echoes).
*   **Hardware Constraints:** Option to apply simplified maximum gradient and slew rate constraints.
*   **Trajectory Analysis:**
    *   Automatic calculation of FOV, resolution, max gradient, and max slew rate estimates.
    *   Voronoi-based density compensation weight calculation (for 2D/3D).
*   **Data Handling:**
    *   `Trajectory` class to encapsulate k-space data, gradients, and metadata.
    *   Export/import capabilities for common formats (NPZ, CSV, NPY, TXT).
*   **Visualization:** Plotting utilities for k-space trajectories, gradient waveforms, and slew rates.
*   **Image Reconstruction:** Basic gridding-based image reconstruction.

## Installation

To use this package, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your_username/trajgen.git # Replace with actual URL later
cd trajgen
pip install numpy scipy matplotlib
```

## Basic Usage Example

```python
from trajgen import KSpaceTrajectoryGenerator, COMMON_NUCLEI_GAMMA_HZ_PER_T
from trajgen.utils import display_trajectory
import matplotlib.pyplot as plt # For showing plots

# 1. Initialize the trajectory generator for 2D trajectories
gen_params_2d = {
    'fov_mm': (220, 220),        # Field of View in mm
    'resolution_mm': (2.0, 2.0), # Target resolution in mm
    'num_dimensions': 2,
    'dt_s': 4e-6,                # Dwell time: 4 microseconds
    'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'], # Gyromagnetic ratio for 1H
    'max_grad_mT_per_m': 40.0,   # Optional: Max gradient strength
    'max_slew_Tm_per_s_ms': 150.0 # Optional: Max slew rate
}
generator_2d = KSpaceTrajectoryGenerator(**gen_params_2d)

# 2. Create a spiral trajectory
spiral_traj = generator_2d.create_spiral(
    num_interleaves=16,
    points_per_interleaf=1024,
    density_factor_at_center=2.0, # Example of variable density
    density_transition_radius_factor=0.2,
    apply_constraints=True
)
print("\\n--- Spiral Trajectory ---")
spiral_traj.summary()

# 3. Create an EPI trajectory
epi_traj = generator_2d.create_epi_trajectory(
    num_echoes=128,
    points_per_echo=128,
    epi_type='gradient_recalled',
    phase_encode_direction='y',
    apply_constraints=True
)
print("\\n--- EPI Trajectory ---")
epi_traj.summary()

# 4. Initialize a generator for 3D trajectories
gen_params_3d = {
    'fov_mm': (200, 200, 100),
    'resolution_mm': (2.0, 2.0, 2.0),
    'num_dimensions': 3,
    'dt_s': 4e-6,
    'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']
}
generator_3d = KSpaceTrajectoryGenerator(**gen_params_3d)

# 5. Create a 3D Radial trajectory
radial_3d_traj = generator_3d.create_radial(
    num_spokes=2000,
    points_per_spoke=128,
    apply_constraints=False # Constraints not applied for this example
)
print("\\n--- 3D Radial Trajectory ---")
radial_3d_traj.summary()

# 6. Display a trajectory (e.g., the spiral)
# Note: In a script, plt.show() would be needed.
# In Jupyter, %matplotlib inline or %matplotlib widget handles display.
# fig_spiral = display_trajectory(spiral_traj, show_gradients=True, show_slew=True)
# plt.show()
# fig_epi = display_trajectory(epi_traj)
# plt.show()
# fig_radial_3d = display_trajectory(radial_3d_traj)
# plt.show()

print("\\nBasic usage example finished.")
```

## Running Tests

To run the unit tests for the `trajgen` package, navigate to the root directory of the repository and execute:

```bash
python -m unittest discover -s tests -v
```

## Core Components

*   **`trajgen.KSpaceTrajectoryGenerator`**: A high-level class to configure and generate various trajectories.
*   **`trajgen.Trajectory`**: Represents a k-space trajectory, holding its data and metadata, and offering methods for analysis and I/O.
*   **`trajgen.generators`**: Module containing the underlying mathematical functions for generating specific trajectory shapes (e.g., `generate_spiral_trajectory`, `generate_radial_trajectory`).
*   **`trajgen.utils`**: Module with utility functions for tasks like constraint application (`constrain_trajectory`), image reconstruction (`reconstruct_image`), and trajectory visualization (`display_trajectory`).
*   **`trajgen.COMMON_NUCLEI_GAMMA_HZ_PER_T`**: A dictionary of gyromagnetic ratios for common nuclei.

This structure allows for both high-level trajectory creation and access to lower-level generation and utility functions if needed.
