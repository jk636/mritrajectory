# Drunken Spiral Trajectory

The "Drunken Spiral" is a type of stochastic k-space trajectory designed to provide variable density sampling with inherent incoherent aliasing, making it suitable for Compressed Sensing (CS) reconstruction. It's based on a standard spiral trajectory whose path is perturbed by a random, weighted noise component.

## Concept

-   **Base Spiral**: An Archimedean spiral forms the underlying structure, ensuring progressive coverage of k-space.
-   **Stochastic Perturbation**: Random noise is added to the kx and ky coordinates of the base spiral.
-   **Variable Density**: The amplitude of the stochastic perturbation is typically weighted to be stronger near the k-space center and weaker towards the periphery. This results in denser effective sampling (due to more erratic movement) at the center, capturing more low-frequency information (important for SNR and contrast), while still covering the outer k-space for resolution.
-   **Hardware Constraints**: The generation process includes an iterative smoothing mechanism to help ensure that the trajectory respects specified maximum gradient and slew rate limits. Warnings are issued if the trajectory, even after smoothing, might exceed these limits.

## Example Usage

Here's how to generate and plot a 2D Drunken Spiral trajectory:

```python
import numpy as np
import matplotlib.pyplot as plt
from trajgen.sequences.drunken_spiral import DrunkenSpiralSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

# Define sequence parameters
params = {
    'name': "drunken_spiral_example",
    'fov_mm': 256.0,
    'resolution_mm': 2.0, # Lower resolution for fewer points in example
    'num_points': 2048,   # Number of k-space points
    'dt_seconds': 4e-6,   # Dwell time
    'base_spiral_turns': 8,
    'perturbation_amplitude_factor': 0.15, # Controls the "drunkenness"
    'density_sigma_factor': 0.3, # Controls how localized the strong perturbation is
    'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
    # Optional: target hardware limits for the generator
    'max_grad_Tm_per_m': 0.035,  # T/m
    'max_slew_Tm_per_s_per_m': 120.0, # T/m/s
    'num_smoothing_iterations': 5,
    'smoothing_kernel_size': 5
}

# Create the sequence
drunken_spiral_seq = DrunkenSpiralSequence(**params)

# Access k-space points
k_points = drunken_spiral_seq.kspace_points_rad_per_m
print(f"Generated k-space points with shape: {k_points.shape}")
print(f"Number of dimensions: {drunken_spiral_seq.get_num_dimensions()}")

# Plot the trajectory
plt.figure(figsize=(7, 7))
drunken_spiral_seq.plot_2d(ax=plt.gca(), title="Drunken Spiral Trajectory", point_stride=1, plot_style='-') # Show lines
# To see individual points better for sparse trajectories:
# drunken_spiral_seq.plot_2d(ax=plt.gca(), title="Drunken Spiral Trajectory", plot_style='.', point_size=1)
plt.xlabel("Kx (rad/m)")
plt.ylabel("Ky (rad/m)")
plt.tight_layout()
plt.show()

# Assess k-space coverage
coverage_assessment = drunken_spiral_seq.assess_kspace_coverage()
print(f"K-space Coverage Assessment:\n{coverage_assessment}")

# Suggest reconstruction method
recon_suggestion = drunken_spiral_seq.suggest_reconstruction_method()
print(f"Suggested Reconstruction:\n{recon_suggestion}")

# Check against (potentially different) system limits
system_limits_check = {'max_grad_Tm_per_m': 0.040, 'max_slew_Tm_per_s_per_m': 150.0}
is_compliant = drunken_spiral_seq.check_gradient_limits(system_limits_check)
print(f"Compliant with runtime system limits: {is_compliant}")
print(f"Actual max gradient: {drunken_spiral_seq.get_max_grad_Tm():.3f} T/m")
print(f"Actual max slew rate: {drunken_spiral_seq.get_max_slew_Tm_per_s():.1f} T/m/s")

```

## Design Principle Considerations

-   **K-Space Coverage**: Provides pseudo-random, variable-density coverage. The center of k-space is sampled more erratically (leading to denser coverage over time or averaging) due to higher perturbation amplitudes, while the periphery is sampled more sparsely but still reached by the underlying spiral.
-   **Off-Resonance Sensitivity**: Like other trajectories with extended readout segments, it can be sensitive to off-resonance effects. However, the stochastic nature might lead to more noise-like artifacts rather than structured blurring typical of pure spirals.
-   **Motion Robustness**: The inherent oversampling of the k-space center can be beneficial. Fast acquisition (if `num_points` and `dt_seconds` allow) is the primary strategy for motion robustness.
-   **Reconstruction**: Due to its non-Cartesian and stochastic nature, reconstruction typically requires **NUFFT** (Non-Uniform Fast Fourier Transform) and is highly suited for **Compressed Sensing (CS)** algorithms. Iterative reconstruction methods are common.

## Parameters for `generate_drunken_spiral_trajectory`

-   `fov_mm`: Field of view in millimeters (for the underlying spiral coverage).
-   `resolution_mm`: Desired resolution in millimeters (determines `k_max`).
-   `num_points`: Total number of k-space points for the trajectory.
-   `dt_seconds`: Dwell time (time between k-space samples) in seconds.
-   `base_spiral_turns`: Number of turns for the underlying Archimedean spiral.
-   `perturbation_amplitude_factor`: Factor scaling the random noise amplitude. The actual perturbation scale is relative to `k_max / sqrt(num_points)`.
-   `density_sigma_factor`: Sigma for the Gaussian weighting of noise, relative to `k_max`. Noise is stronger at k-space center where `r_base/k_max` is small.
-   `max_grad_Tm_per_m` (Optional): Target maximum gradient amplitude constraint (T/m) for the generator's iterative smoothing.
-   `max_slew_Tm_per_s_per_m` (Optional): Target maximum slew rate constraint (T/m/s) for the generator's iterative smoothing.
-   `num_smoothing_iterations`: Number of iterations to apply smoothing if constraints (passed to generator) are violated.
-   `smoothing_kernel_size`: Size of the moving average filter kernel for smoothing (should be odd).

```
