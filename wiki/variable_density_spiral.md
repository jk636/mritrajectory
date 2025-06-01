# Variable Density Spiral Sequence Example

This example demonstrates how to create, visualize, and assess a Variable Density Spiral sequence using the `trajgen` library.

## 1. Import necessary libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from trajgen.sequences import VariableDensitySpiralSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T
```

## 2. Define Sequence Parameters

```python
# Common parameters for the sequence
fov_mm = 220.0  # Field of View in mm
resolution_mm = 1.5  # Desired resolution in mm
num_dimensions = 2 # For a 2D spiral
dt_seconds = 4e-6  # Dwell time (time between k-space samples) in seconds
gamma_Hz_per_T = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'] # Gyromagnetic ratio for 1H

# Spiral-specific parameters
num_interleaves = 24
points_per_interleaf = 1536
spiral_type = 'archimedean' # Type of spiral
density_transition_radius_factor = 0.6 # Factor determining k-space density transition
density_factor_at_center = 2.5 # How much denser the k-space center is

# Optional: dead times
dead_time_start_seconds = 0.001 # 1 ms
dead_time_end_seconds = 0.001   # 1 ms
```

## 3. Create the Variable Density Spiral Sequence

```python
spiral_sequence = VariableDensitySpiralSequence(
    name="ExampleVDS",
    fov_mm=fov_mm,
    resolution_mm=resolution_mm,
    num_dimensions=num_dimensions,
    dt_seconds=dt_seconds,
    gamma_Hz_per_T=gamma_Hz_per_T,
    num_interleaves=num_interleaves,
    points_per_interleaf=points_per_interleaf,
    spiral_type=spiral_type,
    density_transition_radius_factor=density_transition_radius_factor,
    density_factor_at_center=density_factor_at_center,
    dead_time_start_seconds=dead_time_start_seconds,
    dead_time_end_seconds=dead_time_end_seconds
)

# Print the shape of the generated k-space points array
print(f"K-space points shape (Dimensions, Num Points): {spiral_sequence.kspace_points_rad_per_m.shape}")
print(f"Total duration: {spiral_sequence.get_duration_seconds() * 1000:.2f} ms")
```

## 4. Plot the 2D Trajectory

```python
plt.figure(figsize=(7, 7))
ax = spiral_sequence.plot_2d(
    max_total_points=5000, # Limit points for clarity if trajectory is very dense
    plot_style='.-',
    point_stride=10 # Plot every 10th point for less clutter
)
if ax: # plot_2d returns None if not 2D or no points
    plt.title("Variable Density Spiral Trajectory (2D)")
    plt.show()
else:
    print("Could not generate 2D plot for the spiral sequence.")
```
*(Note: In a Jupyter Notebook, `plt.show()` might not be necessary as plots often display automatically.)*

## 5. Assess Trajectory Properties

### K-space Coverage Assessment

```python
coverage_assessment = spiral_sequence.assess_kspace_coverage()
print(f"K-space Coverage Assessment:\n{coverage_assessment}")
```

### Gradient Limit Check

```python
# Define example system limits (T/m for max_grad, T/m/s for max_slew)
system_limits = {
    'max_grad_Tm_per_m': 0.040,      # 40 mT/m
    'max_slew_Tm_per_s_per_m': 150.0 # 150 T/m/s
}

# Check if the trajectory adheres to these limits
# The check_gradient_limits method will print details of the check.
within_limits = spiral_sequence.check_gradient_limits(system_limits)
print(f"Trajectory within specified gradient limits: {within_limits}")

# Example with very strict limits that should fail
strict_limits = {
    'max_grad_Tm_per_m': 0.010,      # 10 mT/m
    'max_slew_Tm_per_s_per_m': 50.0  # 50 T/m/s
}
within_strict_limits = spiral_sequence.check_gradient_limits(strict_limits)
print(f"Trajectory within strict gradient limits: {within_strict_limits}")

```

This example provides a basic workflow for using the `VariableDensitySpiralSequence` class. You can adjust the parameters to explore different trajectory characteristics.
