# Basic Trajectory Generation

This document provides an overview and examples of how to generate common k-space trajectories using the functions available in `trajgen.py`.

## 2D Trajectories

### 1. Radial Trajectories

Radial trajectories sample k-space along lines (spokes) that pass through the k-space origin at various angles. They are relatively simple to design and are robust to motion.

**Function**: `generate_radial_trajectory()`

**Essential Parameters for a Minimal Example**:
*   `num_spokes (int)`: Number of radial lines.
*   `num_samples_per_spoke (int)`: Number of k-space samples along each line.
*   `fov_m (float)`: Field of View in meters (determines k-space extent if `max_k_rad_per_m` is not given).

**Minimal Python Code Example**:
```python
from trajgen import generate_radial_trajectory, Trajectory

# Parameters for a simple radial trajectory
num_spokes = 64
num_samples_per_spoke = 128
fov = 0.256  # meters

# Generate the trajectory
radial_trajectory = generate_radial_trajectory(
    num_spokes=num_spokes,
    num_samples_per_spoke=num_samples_per_spoke,
    fov_m=fov
)

# The 'radial_trajectory' object is an instance of the Trajectory class
# print(f"Generated radial trajectory: {radial_trajectory.name}")
# print(f"K-space points shape: {radial_trajectory.kspace_points_rad_per_m.shape}")
```

### 2. Spiral Trajectories

Spiral trajectories sample k-space along spiral paths, typically starting from the center and spiraling outwards. They can be very efficient in terms of k-space coverage over time.

**Function**: `generate_spiral_trajectory()`

**Essential Parameters for a Minimal Example**:
*   `num_arms (int)`: Number of spiral interleaves/arms.
*   `num_samples_per_arm (int)`: Number of k-space samples along each arm.
*   `fov_m (float)`: Field of View in meters (determines k-space extent if `max_k_rad_per_m` is not given).

**Minimal Python Code Example**:
```python
from trajgen import generate_spiral_trajectory, Trajectory

# Parameters for a simple spiral trajectory
num_arms = 8
num_samples_per_arm = 1024
fov = 0.224  # meters

# Generate the trajectory
spiral_trajectory = generate_spiral_trajectory(
    num_arms=num_arms,
    num_samples_per_arm=num_samples_per_arm,
    fov_m=fov
)

# The 'spiral_trajectory' object is an instance of the Trajectory class
# print(f"Generated spiral trajectory: {spiral_trajectory.name}")
# print(f"K-space points shape: {spiral_trajectory.kspace_points_rad_per_m.shape}")
```
**Note**: The `generate_spiral_trajectory` function also supports applying gradient and slew rate constraints during generation via `max_gradient_Tm_per_m` and `max_slew_rate_Tm_per_s` parameters.

### 3. Echo-Planar Imaging (EPI) Trajectories

Echo-Planar Imaging (EPI) trajectories are widely used for fast MRI, covering k-space rapidly using a series of gradient echoes, typically in a raster-scan pattern (e.g., Cartesian lines) or spiral readouts.

**Standalone Generator Function**:
A direct standalone function for simple, high-level EPI generation (e.g., by specifying only FOV, matrix size, and echo train length) is not currently available in `trajgen.py`. EPI trajectories typically involve precise sequencing of gradient blips for phase encoding and fast readouts along the frequency encoding direction, often requiring more detailed sequence parameters or construction via a lower-level trajectory generation class like `KSpaceTrajectoryGenerator`.

## 3D Trajectories

### 1. 3D Radial / Phyllotaxis (Golden Angle) Trajectories

These trajectories extend radial sampling concepts to 3D. The "Golden Angle" or "Phyllotaxis" method distributes points (or spoke directions) approximately uniformly on the surface of spheres, providing efficient 3D k-space coverage. The implementation generates points to fill a spherical or ellipsoidal k-space volume.

**Function**: `generate_golden_angle_3d_trajectory()`

**Essential Parameters for a Minimal Example**:
*   `num_points (int)`: Total number of k-space points to generate in 3D.
*   `fov_m (float or tuple/list of 3 floats)`: Field of View in meters.
    *   A single float assumes isotropic FOV.
    *   A tuple/list `(fov_x, fov_y, fov_z)` defines anisotropic FOV.

**Minimal Python Code Example**:
```python
from trajgen import generate_golden_angle_3d_trajectory, Trajectory

# Parameters for a simple 3D golden angle trajectory
num_3d_points = 5000
fov_isotropic = 0.2  # meters for isotropic FOV

# Generate the trajectory (isotropic FOV)
golden_angle_3d_iso_trajectory = generate_golden_angle_3d_trajectory(
    num_points=num_3d_points,
    fov_m=fov_isotropic
)
# print(f"Generated 3D Golden Angle (iso FOV) trajectory: {golden_angle_3d_iso_trajectory.name}")
# print(f"K-space points shape: {golden_angle_3d_iso_trajectory.kspace_points_rad_per_m.shape}")

# Example with anisotropic FOV
fov_anisotropic = (0.2, 0.25, 0.18) # meters for fov_x, fov_y, fov_z
golden_angle_3d_aniso_trajectory = generate_golden_angle_3d_trajectory(
    num_points=num_3d_points,
    fov_m=fov_anisotropic,
    name="golden_angle_3d_anisotropic"
)
# print(f"Generated 3D Golden Angle (aniso FOV) trajectory: {golden_angle_3d_aniso_trajectory.name}")
# print(f"K-space points shape: {golden_angle_3d_aniso_trajectory.kspace_points_rad_per_m.shape}")
# print(f"Calculated k_max: {golden_angle_3d_aniso_trajectory.metadata['k_max_calculated_rad_m_xyz']}")
```

### 2. Stack-of-Spirals Trajectories

Stack-of-Spirals trajectories acquire 3D k-space data by collecting multiple 2D spiral datasets at different slice-encoding positions (kz steps). This is a common method for achieving fast 3D volumetric coverage with spiral characteristics.

**Standalone Generator Function**:
Stack-of-Spirals trajectories are typically constructed by generating a base 2D spiral trajectory and then applying rotations and kz-offsets to form the 3D stack. A dedicated helper function for the simple, direct generation of a full Stack-of-Spirals trajectory (e.g., by only specifying overall 3D FOV, resolution, and spiral parameters) is not currently available as a single standalone function in `trajgen.py`. Generation is often handled by more comprehensive classes like `KSpaceTrajectoryGenerator` or by manually combining 2D spirals.

### 3. Cones Trajectories

Cones trajectories sample 3D k-space along paths that resemble cones or twisted spirals on conical surfaces, originating from the k-space center. They offer good isotropic resolution properties and SNR efficiency.

**Standalone Generator Function**:
A direct standalone function for simple Cones trajectory generation (e.g., by specifying basic parameters like FOV and resolution) is not currently available in `trajgen.py`. Cones trajectories often involve complex calculations for the gradient waveforms along three axes and are typically generated using more specialized sequence design tools or lower-level classes like `KSpaceTrajectoryGenerator`.
