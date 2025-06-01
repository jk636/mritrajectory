# Radial Sequence Example

This example demonstrates how to create, visualize, and assess a Radial sequence using the `trajgen` library, for both 2D and 3D cases.

## 1. Import necessary libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from trajgen.sequences import RadialSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T
```

## 2. Define Parameters for a 2D Radial Sequence

```python
# Parameters for a 2D radial sequence
fov_mm_2d = 200.0
resolution_mm_2d = 2.0
num_dimensions_2d = 2
dt_seconds_2d = 4e-6
gamma_Hz_per_T = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H']

num_spokes_2d = 128
points_per_spoke_2d = 256
projection_angle_increment_2d = 'golden_angle' # Use golden angle for even distribution
```

## 3. Create and Plot the 2D Radial Sequence

```python
radial_seq_2d = RadialSequence(
    name="ExampleRadial2D",
    fov_mm=fov_mm_2d,
    resolution_mm=resolution_mm_2d,
    num_dimensions=num_dimensions_2d,
    dt_seconds=dt_seconds_2d,
    gamma_Hz_per_T=gamma_Hz_per_T,
    num_spokes=num_spokes_2d,
    points_per_spoke=points_per_spoke_2d,
    projection_angle_increment=projection_angle_increment_2d
)

print(f"2D Radial K-space points shape: {radial_seq_2d.kspace_points_rad_per_m.shape}")

# Plot the 2D trajectory
plt.figure(figsize=(7, 7))
ax_2d = radial_seq_2d.plot_2d(max_total_points=num_spokes_2d * 5) # Plot only a few points per spoke for clarity
if ax_2d:
    plt.title("2D Radial Trajectory")
    plt.show()
else:
    print("Could not generate 2D plot for the 2D radial sequence.")
```

## 4. Define Parameters for a 3D Radial Sequence

```python
# Parameters for a 3D radial sequence (Stack-of-Stars like if angle is fixed, or true 3D)
fov_mm_3d = (180.0, 180.0, 180.0) # Can be isotropic or anisotropic
resolution_mm_3d = (3.0, 3.0, 3.0)
num_dimensions_3d = 3
dt_seconds_3d = 4e-6

num_spokes_3d = 2000 # More spokes needed for 3D coverage
points_per_spoke_3d = 128
projection_angle_increment_3d = 'golden_angle' # Uses 3D golden angle (spherical)
```

## 5. Create and Plot the 3D Radial Sequence

```python
radial_seq_3d = RadialSequence(
    name="ExampleRadial3D",
    fov_mm=fov_mm_3d,
    resolution_mm=resolution_mm_3d,
    num_dimensions=num_dimensions_3d,
    dt_seconds=dt_seconds_3d,
    gamma_Hz_per_T=gamma_Hz_per_T,
    num_spokes=num_spokes_3d,
    points_per_spoke=points_per_spoke_3d,
    projection_angle_increment=projection_angle_increment_3d
)

print(f"3D Radial K-space points shape: {radial_seq_3d.kspace_points_rad_per_m.shape}")

# Plot the 3D trajectory
fig_3d = plt.figure(figsize=(8, 8))
ax_3d = radial_seq_3d.plot_3d(
    max_total_points=num_spokes_3d * 5, # Limit points for visual clarity
    point_stride=points_per_spoke_3d // 5 if points_per_spoke_3d > 5 else 1 # Show a few points per spoke
)
if ax_3d:
    plt.title("3D Radial Trajectory")
    plt.show()
else:
    print("Could not generate 3D plot for the 3D radial sequence.")
```

## 6. Assess Trajectory Properties (using 3D sequence as example)

### Motion Robustness Assessment

```python
motion_robustness_info = radial_seq_3d.assess_motion_robustness()
print(f"Motion Robustness Assessment (3D Radial):\n{motion_robustness_info}")
```

### Gradient Limit Check

```python
# Define example system limits
system_limits = {
    'max_grad_Tm_per_m': 0.035,      # 35 mT/m
    'max_slew_Tm_per_s_per_m': 120.0 # 120 T/m/s
}

within_limits_3d = radial_seq_3d.check_gradient_limits(system_limits)
print(f"3D Radial trajectory within specified gradient limits: {within_limits_3d}")
```

This example illustrates how to generate both 2D and 3D radial trajectories and utilize some of their assessment methods. The `'golden_angle'` increment is particularly useful for achieving uniform coverage in both 2D and 3D.
