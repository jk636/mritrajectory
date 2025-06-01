# Twisted Projection Imaging (TPI) Sequence Example

This example demonstrates how to create, visualize, and assess a Twisted Projection Imaging (TPI) sequence using the `trajgen` library. TPI is an inherently 3D trajectory.

## 1. Import necessary libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from trajgen.sequences import TwistedProjectionImagingSequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T
```

## 2. Define Sequence Parameters

```python
# Common parameters for the sequence
fov_mm = 200.0  # Isotropic Field of View in mm for 3D
resolution_mm = 2.0  # Isotropic desired resolution in mm for 3D
# num_dimensions is implicitly 3 for TPI, set by the sequence class
dt_seconds = 4e-6  # Dwell time (time between k-space samples) in seconds
gamma_Hz_per_T = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'] # Gyromagnetic ratio

# TPI-specific parameters
num_twists = 100 # Number of twisted projection arms/segments (reduced for quick example)
points_per_segment = 512 # Points along each twisted segment
cone_angle_deg = 25.0 # Angle of the cone on which projections are twisted (degrees)
spiral_turns_per_twist = 4.0 # How many turns the spiral makes along one twist

# Optional: dead times
dead_time_start_seconds = 0.002 # 2 ms
dead_time_end_seconds = 0.002   # 2 ms
```

## 3. Create the Twisted Projection Imaging Sequence

```python
tpi_sequence = TwistedProjectionImagingSequence(
    name="ExampleTPI",
    fov_mm=fov_mm,
    resolution_mm=resolution_mm,
    # num_dimensions is handled by the TPISequence class (set to 3)
    dt_seconds=dt_seconds,
    gamma_Hz_per_T=gamma_Hz_per_T,
    num_twists=num_twists,
    points_per_segment=points_per_segment,
    cone_angle_deg=cone_angle_deg,
    spiral_turns_per_twist=spiral_turns_per_twist,
    dead_time_start_seconds=dead_time_start_seconds,
    dead_time_end_seconds=dead_time_end_seconds
)

# Print the shape of the generated k-space points array
# The TPI generator returns shape (3, Total Points)
print(f"K-space points shape (Dimensions, Num Points): {tpi_sequence.kspace_points_rad_per_m.shape}")
print(f"Total duration: {tpi_sequence.get_duration_seconds() * 1000:.2f} ms")
```

## 4. Plot the 3D Trajectory

```python
fig = plt.figure(figsize=(8, 8))
ax = tpi_sequence.plot_3d(
    max_total_points=num_twists * 10, # Limit points for clarity, e.g., 10 points per twist
    point_stride=points_per_segment // 10 if points_per_segment > 10 else 1
)
if ax:
    plt.title("Twisted Projection Imaging (TPI) Trajectory (3D)")
    plt.show()
else:
    print("Could not generate 3D plot for the TPI sequence.")

```
*(Note: In a Jupyter Notebook, `plt.show()` might not be necessary.)*

## 5. Assess Trajectory Properties

### Suggested Reconstruction Method

```python
reconstruction_suggestion = tpi_sequence.suggest_reconstruction_method()
print(f"Suggested Reconstruction Method:\n{reconstruction_suggestion}")
```

### K-space Coverage Assessment

```python
coverage_assessment = tpi_sequence.assess_kspace_coverage()
print(f"K-space Coverage Assessment:\n{coverage_assessment}")
```

### Gradient Limit Check

```python
# Define example system limits
system_limits = {
    'max_grad_Tm_per_m': 0.040,      # 40 mT/m
    'max_slew_Tm_per_s_per_m': 160.0 # 160 T/m/s
}

within_limits = tpi_sequence.check_gradient_limits(system_limits)
print(f"Trajectory within specified gradient limits: {within_limits}")
```

This example provides a basic workflow for using the `TwistedProjectionImagingSequence`. TPI is a versatile 3D trajectory, and its characteristics can be tuned by adjusting parameters like `cone_angle_deg`, `num_twists`, and `spiral_turns_per_twist`.
