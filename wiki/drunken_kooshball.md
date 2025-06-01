# Drunken Kooshball Trajectory (3D)

The "Drunken Kooshball" is a 3D stochastic k-space trajectory, extending the concept of the 2D "Drunken Spiral." It aims for variable density sampling in three dimensions with inherent incoherent aliasing, making it suitable for 3D Compressed Sensing (CS) reconstruction.

## Concept

-   **Base Trajectory**: A 3D spherical spiral forms the underlying structure, ensuring progressive coverage of k-space from the center outwards.
-   **Stochastic Perturbation**: Random noise is added to the kx, ky, and kz coordinates of the base spherical spiral.
-   **Variable Density**: The amplitude of the stochastic perturbation is weighted to be stronger near the k-space center and weaker towards the periphery. This results in denser effective sampling at the center.
-   **Hardware Constraints**: The generation process includes iterative smoothing to help the trajectory respect specified maximum gradient and slew rate limits. Warnings are issued if limits might still be exceeded.

## Example Usage

Here's how to generate and plot a 3D Drunken Kooshball trajectory:

```python
import numpy as np
import matplotlib.pyplot as plt
from trajgen.sequences.drunken_kooshball import DrunkenKooshballSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

# Define sequence parameters
params = {
    'name': "drunken_kooshball_example",
    'fov_mm': (200.0, 200.0, 150.0), # 3D FOV
    'resolution_mm': (2.5, 2.5, 2.5), # 3D Resolution
    'num_points': 2048,       # Fewer points for a quicker example
    'dt_seconds': 4e-6,       # Dwell time
    'base_spherical_spiral_turns': 12,
    'perturbation_amplitude_factor': 0.1,
    'density_sigma_factor': 0.3,
    'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
    # Optional: target hardware limits for the generator
    'max_grad_Tm_per_m': 0.035,
    'max_slew_Tm_per_s_per_m': 120.0,
    'num_smoothing_iterations': 3,
    'smoothing_kernel_size': 5
}

# Create the sequence
drunken_kooshball_seq = DrunkenKooshballSequence(**params)

# Access k-space points
k_points = drunken_kooshball_seq.kspace_points_rad_per_m
print(f"Generated 3D k-space points with shape: {k_points.shape}")
print(f"Number of dimensions: {drunken_kooshball_seq.get_num_dimensions()}")

# Plot the 3D trajectory (plotting a subset for clarity if many points)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
# Plot every Nth point to avoid overplotting, e.g., every 10th point for 2048 points
num_plot_points = k_points.shape[1]
point_stride_3d = max(1, num_plot_points // 500) # Aim for around 500 points in the plot

drunken_kooshball_seq.plot_3d(ax=ax, title="Drunken Kooshball Trajectory (Subsampled Plot)", point_stride=point_stride_3d, plot_style='-')
# For individual points:
# drunken_kooshball_seq.plot_3d(ax=ax, title="Drunken Kooshball Trajectory", point_stride=point_stride_3d, plot_style='.')
plt.tight_layout()
plt.show()

# Assess k-space coverage
coverage_assessment = drunken_kooshball_seq.assess_kspace_coverage()
print(f"K-space Coverage Assessment:\n{coverage_assessment}")

# Suggest reconstruction method
recon_suggestion = drunken_kooshball_seq.suggest_reconstruction_method()
print(f"Suggested Reconstruction:\n{recon_suggestion}")

# Check against system limits
system_limits_check = {'max_grad_Tm_per_m': 0.040, 'max_slew_Tm_per_s_per_m': 150.0}
is_compliant = drunken_kooshball_seq.check_gradient_limits(system_limits_check)
print(f"Compliant with runtime system limits: {is_compliant}")
print(f"Actual max gradient: {drunken_kooshball_seq.get_max_grad_Tm():.3f} T/m")
print(f"Actual max slew rate: {drunken_kooshball_seq.get_max_slew_Tm_per_s():.1f} T/m/s")
```

## Design Principle Considerations

-   **K-Space Coverage**: Offers pseudo-random, variable-density coverage in 3D. Denser sampling at the k-space center is achieved via weighted perturbations.
-   **Off-Resonance Sensitivity**: As with other long readout trajectories, it can be susceptible to off-resonance effects, potentially leading to diffuse, noise-like artifacts.
-   **Motion Robustness**: Dense central k-space sampling is advantageous. The overall speed of acquiring the 3D volume (dependent on `num_points` and `dt_seconds`) is critical for mitigating motion effects.
-   **Reconstruction**: Best suited for **3D Compressed Sensing (CS)** reconstruction techniques, utilizing a **3D NUFFT** for the non-Cartesian data.

## Parameters for `generate_drunken_kooshball_trajectory`

-   `fov_mm`: Field of view in millimeters (isotropic or per dimension for 3D).
-   `resolution_mm`: Desired resolution in millimeters (isotropic or per dimension for 3D).
-   `num_points`: Total number of k-space points for the trajectory.
-   `dt_seconds`: Dwell time (time between k-space samples) in seconds.
-   `base_spherical_spiral_turns`: Number of turns for the phi component of the base 3D spherical spiral.
-   `perturbation_amplitude_factor`: Factor scaling the random noise amplitude. The actual perturbation scale is relative to `k_max / cbrt(num_points)`.
-   `density_sigma_factor`: Sigma for the Gaussian weighting of noise, relative to `k_max`. Noise is stronger at k-space center.
-   `max_grad_Tm_per_m` (Optional): Target maximum gradient amplitude constraint (T/m) for the generator's iterative smoothing.
-   `max_slew_Tm_per_s_per_m` (Optional): Target maximum slew rate constraint (T/m/s) for the generator's iterative smoothing.
-   `num_smoothing_iterations`: Number of iterations to apply smoothing if constraints (passed to generator) are violated.
-   `smoothing_kernel_size`: Size of the moving average filter kernel for smoothing (should be odd).
```
