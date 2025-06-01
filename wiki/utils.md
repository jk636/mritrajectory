# Trajectory Utility Functions

This page describes various utility functions available in `trajgen.utils` for manipulating and generating k-space trajectories.

## Generating a Series of Rotated 3D Trajectories

The `generate_rotated_trajectory_series` function can be used to create a list of 3D trajectory objects, where each trajectory in the list is a rotated version of an input `base_trajectory_3d`. This is useful for simulating dynamic imaging with different k-space views over time (e.g., using golden angle increments) or for generating data for rotational studies.

**Function Signature:**
```python
from trajgen.utils import generate_rotated_trajectory_series
from trajgen.trajectory import Trajectory # For type hint
from typing import List, Tuple # For type hints
import numpy as np # For np.sqrt if used in default args

def generate_rotated_trajectory_series(
    base_trajectory_3d: Trajectory,
    num_frames: int,
    rotation_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    rotation_angle_increment_deg: float = 180.0 * (3.0 - np.sqrt(5.0)),
    output_name_template: str = "frame_{i:03d}_"
) -> List[Trajectory]:
    # ... implementation ...
    pass
```

**Example Usage:**

This example demonstrates generating a series of rotated "Drunken Kooshball" trajectories.

```python
import numpy as np
import matplotlib.pyplot as plt
from trajgen.sequences.drunken_kooshball import DrunkenKooshballSequence # Assuming this exists
from trajgen.utils import generate_rotated_trajectory_series
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

# 1. Create a base 3D trajectory
base_params = {
    'name': "base_kooshball",
    'fov_mm': (180.0, 180.0, 180.0),
    'resolution_mm': (3.0, 3.0, 3.0),
    'num_points': 512, # Keep low for quick example
    'dt_seconds': 4e-6,
    'base_spherical_spiral_turns': 6,
    'perturbation_amplitude_factor': 0.1,
    'density_sigma_factor': 0.3,
    'gamma_Hz_per_T': COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
    'num_smoothing_iterations': 1,
    'smoothing_kernel_size': 3
}
base_kooshball_seq = DrunkenKooshballSequence(**base_params)

# 2. Generate the series of rotated trajectories
num_dynamic_frames = 5
trajectory_series = generate_rotated_trajectory_series(
    base_trajectory_3d=base_kooshball_seq,
    num_frames=num_dynamic_frames,
    rotation_axis=(0, 0, 1), # Rotate around Z-axis
    rotation_angle_increment_deg=36.0 # 5 frames * 36 deg = 180 deg total spread for example
)

print(f"Generated {len(trajectory_series)} trajectory frames.")

# Plot the first and last frame's trajectory for comparison (optional)
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
if len(trajectory_series) > 0:
    trajectory_series[0].plot_3d(ax=ax1, title=trajectory_series[0].name, point_stride=1, plot_style='-')

ax2 = fig.add_subplot(122, projection='3d')
if len(trajectory_series) > 0:
    trajectory_series[-1].plot_3d(ax=ax2, title=trajectory_series[-1].name, point_stride=1, plot_style='-')

plt.tight_layout()
plt.show()

# Each element in trajectory_series is a full Trajectory object
# Example of accessing metadata for each frame:
# for trajectory_frame in trajectory_series:
#     print(f"Trajectory: {trajectory_frame.name}, "
#           f"Rotation Angle: {trajectory_frame.metadata.get('series_cumulative_rotation_angle_deg')} deg, "
#           f"Frame: {trajectory_frame.metadata.get('frame_number')}")
```
This utility internally uses `rotate_3d_trajectory` for each frame.

---
*More utilities will be documented here as they are implemented.*
