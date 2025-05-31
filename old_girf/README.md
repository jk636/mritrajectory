# MRI K-Space Trajectory Generator (mritrajectory)

## Description

`mritrajectory` is a Python library for generating and managing k-space trajectories for Magnetic Resonance Imaging (MRI). It provides tools to create various 2D and 3D trajectories, calculate their properties, and export them in different formats. This library is designed to be flexible for research and educational purposes in MRI pulse sequence development.

## Features

*   **Versatile Trajectory Generation:**
    *   **2D Trajectories:** Spiral, Radial, EPI (Echo Planar Imaging), Rosette.
    *   **3D Trajectories:** Stack-of-Spirals, 3D Radial (Kooshball/Phyllotaxis-like), Cones, 3D EPI, ZTE (Zero Echo Time).
    *   Generation of 3D trajectories by rotating 2D trajectories (e.g., spiral stack with golden angle).
*   **Customizable Parameters:** Control FOV, resolution, gradient limits (Gmax, Smax), dwell time, number of interleaves, etc.
*   **Advanced Spiral Features:**
    *   Variable density spirals (power law, Gaussian, exponential, hybrid, custom).
    *   Spiral-out-out trajectories.
    *   Golden angle spirals.
*   **UTE (Ultrashort Echo Time) Support:**
    *   Ramp sampling for radial, cone, and ZTE trajectories (center-out half-spokes).
*   **Trajectory Analysis and Metrics:**
    *   Calculation of FOV, resolution.
    *   Maximum gradient amplitude and slew rate.
    *   Peripheral Nerve Stimulation (PNS) estimates (max abs gradient sum, max abs slew sum).
    *   Voronoi cell size calculation for density compensation (2D and 3D).
*   **Deadtime Handling:** Ability to specify and account for dead times at the beginning and end of the trajectory.
*   **Flexible Gyromagnetic Ratio (Gamma):**
    *   Easily specify gamma for different nuclei (e.g., 1H, 13C, 31P).
    *   Includes a predefined dictionary `COMMON_NUCLEI_GAMMA_HZ_PER_T` for common nuclei.
*   **Export/Import:** Save and load trajectories in `.npz`, `.csv`, `.npy`, or `.txt` formats.
*   **Object-Oriented Design:**
    *   `KSpaceTrajectoryGenerator`: Class for generating trajectory coordinates and gradient waveforms.
    *   `Trajectory`: Class for holding trajectory data, metadata, and performing calculations.

## Installation / Setup

Currently, the library consists of the main file `trajgen.py`. To use it, ensure this file is in your Python path or in the same directory as your script/notebook.

Required Python packages:
*   NumPy
*   SciPy (for Voronoi calculations and some signal processing tools)

You can typically install these using pip:
```bash
pip install numpy scipy matplotlib
```
(Note: `matplotlib` is useful for plotting examples from notebooks).

## Quick Start

Here's a basic example of generating a 2D spiral trajectory:

```python
from trajgen import KSpaceTrajectoryGenerator, Trajectory, COMMON_NUCLEI_GAMMA_HZ_PER_T
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize the trajectory generator
gen_params = {
    'fov': 0.22,  # meters
    'resolution': 0.002,  # meters
    'dt': 4e-6,  # seconds
    'gamma': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'], # Use proton gamma
    'traj_type': 'spiral',
    'dim': 2,
    'n_interleaves': 16,
    'turns': 2
}
generator = KSpaceTrajectoryGenerator(**gen_params)

# 2. Generate k-space coordinates and gradients
# Output shapes: (n_interleaves, n_samples_per_interleaf) for kx, ky, etc.
# t is 1D array of time points for the full duration of one interleaf (after rewinders etc.)
kx_per_interleaf, ky_per_interleaf, gx_per_interleaf, gy_per_interleaf, t_interleaf = generator.generate()

# For a simple trajectory object, let's take the first interleaf
k_points_single_interleaf = np.stack([kx_per_interleaf[0], ky_per_interleaf[0]]) # Shape (2, N_samples)
# Or, to combine all interleaves into one Trajectory object:
# k_points_all_interleaves = np.stack([kx_per_interleaf.ravel(), ky_per_interleaf.ravel()])
# gradients_all_interleaves = np.stack([gx_per_interleaf.ravel(), gy_per_interleaf.ravel()])


# 3. Create a Trajectory object
# Gradients can be passed or computed automatically if dt and gamma are provided
trajectory = Trajectory(
    name="2D Spiral Example (Interleaf 0)",
    kspace_points_rad_per_m=k_points_single_interleaf,
    dt_seconds=generator.dt,
    gamma_Hz_per_T=generator.gamma,
    metadata={'generator_params': gen_params}
)

# 4. Display trajectory summary
trajectory.summary()

# 5. Plot the first interleaf (optional)
if k_points_single_interleaf.shape[1] > 0: # Check if there are points to plot
    plt.figure()
    plt.plot(k_points_single_interleaf[0, :], k_points_single_interleaf[1, :])
    plt.title(f"K-space for {trajectory.name}")
    plt.xlabel("Kx (rad/m)")
    plt.ylabel("Ky (rad/m)")
    plt.axis('equal')
    plt.show()
```

## Detailed Examples

For more detailed examples and advanced usage, please refer to the Jupyter Notebooks in the `examples/` directory:

*   `01_basic_2d_trajectories.ipynb`: Demonstrates spirals, radials, and 2D EPI.
*   `02_basic_3d_trajectories.ipynb`: Covers stack-of-spirals, 3D radial, and 3D EPI.
*   `03_ute_zte_trajectories.ipynb`: Shows UTE ramp sampling with radial/cone trajectories and ZTE.
*   `04_trajectory_features.ipynb`: Illustrates deadtime handling, export/import, Voronoi density, and using predefined gyromagnetic ratios.
*   `05_advanced_features.ipynb`: Contains examples of using `generate_3d_from_2d`, custom trajectory functions, and other advanced topics. (Note: Please verify the name and content of the fifth notebook as reported by the worker who created them).

## Key Classes

*   **`KSpaceTrajectoryGenerator`**:
    *   The primary class for designing and generating k-space trajectories.
    *   Initialised with parameters like FOV, resolution, trajectory type, etc.
    *   Its `generate()` method produces the k-space coordinates and gradient waveforms.
    *   The `generate_3d_from_2d()` method allows creating complex 3D trajectories by rotating 2D bases.
*   **`Trajectory`**:
    *   A container class that holds the k-space points, time step (`dt_seconds`), and other relevant data for a generated trajectory.
    *   Automatically calculates various metrics (FOV, resolution, max gradient, max slew, PNS estimates).
    *   Can compute gradient waveforms if not provided.
    *   Supports Voronoi density calculation via `calculate_voronoi_density()`.
    *   Handles export to various file formats and import from `.npz`.
    *   Provides a `summary()` method for quick inspection.
*   **`COMMON_NUCLEI_GAMMA_HZ_PER_T`**:
    *   A module-level dictionary providing gyromagnetic ratios (in Hz/T) for common MRI nuclei (e.g., '1H', '13C', '31P'). Useful for setting the `gamma` parameter in `KSpaceTrajectoryGenerator`.