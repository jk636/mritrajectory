# Gradient Impulse Response Function (GIRF) Features in `mritrajectory`

## 1. Introduction to GIRF in `mritrajectory`

The Gradient Impulse Response Function (GIRF) is a crucial concept in Magnetic Resonance Imaging (MRI) that describes how the actual gradient waveforms produced by an MRI scanner deviate from the nominally commanded (or programmed) waveforms. These deviations arise from various system imperfections, including eddy currents, amplifier limitations, and other hardware characteristics. Understanding and accounting for the GIRF is important for:

*   **Accurate Image Reconstruction**: Deviations between commanded and actual k-space trajectories can lead to image artifacts like blurring, geometric distortions, and ghosting.
*   **Sequence Design and Optimization**: Knowing the system's response allows for designing gradient waveforms that will produce the desired k-space trajectory more faithfully.
*   **Quantitative MRI**: Precise k-space encoding is vital for many quantitative imaging techniques.

The `mritrajectory` library now includes a set_of tools to represent GIRFs and utilize them for analyzing the impact on k-space trajectories and for compensating gradient waveforms.

## 2. The `GIRF` Class

### Purpose
The `GIRF` class serves as a container to store the measured or estimated Gradient Impulse Response Functions for each of the three physical gradient axes (X, Y, Z), along with the time resolution (`dt_girf`) at which these impulse responses were sampled.

### Creation
A `GIRF` object can be created in two ways:

*   **Direct Instantiation**:
    If you have the GIRF impulse response data as NumPy arrays, you can directly instantiate the class:
    ```python
    import numpy as np
    from trajgen import GIRF

    ht_x = np.array([...]) # 1D array for X-axis GIRF
    ht_y = np.array([...]) # 1D array for Y-axis GIRF
    ht_z = np.array([...]) # 1D array for Z-axis GIRF
    dt_girf_sampling = 4e-6 # Time resolution of GIRF data in seconds

    my_girf = GIRF(h_t_x=ht_x, h_t_y=ht_y, h_t_z=ht_z, 
                   dt_girf=dt_girf_sampling, name="ScannerGIRF_ModelA")
    ```

*   **Loading from Files**:
    Alternatively, GIRF data can be loaded from `.npy` files (assuming each file contains a 1D NumPy array for one axis):
    ```python
    from trajgen import GIRF

    dt_girf_sampling = 4e-6
    girf_from_files = GIRF.from_files(
        filepath_x="path/to/girf_x.npy",
        filepath_y="path/to/girf_y.npy",
        filepath_z="path/to/girf_z.npy",
        dt_girf=dt_girf_sampling,
        name="SystemGIRF_Site1"
    )
    ```
    If the `name` parameter is omitted in `from_files`, a default name will be generated based on the input filepaths.

### Key Attributes
*   `h_t_x`: NumPy array, impulse response for the X-axis.
*   `h_t_y`: NumPy array, impulse response for the Y-axis.
*   `h_t_z`: NumPy array, impulse response for the Z-axis.
*   `dt_girf`: Float, time resolution of the GIRF data in seconds.
*   `name`: Optional string, an identifier for the GIRF profile (e.g., "ScannerGIRF_System1").

## 3. Core Utility: `apply_girf_convolution`

### Purpose
The `apply_girf_convolution` function is a fundamental utility that performs the convolution of a single 1D gradient waveform (for one axis) with its corresponding 1D GIRF impulse response.

### Key Feature
A key feature of this function is its ability to handle situations where the time resolution (`dt_gradient`) of the input gradient waveform differs from the time resolution (`dt_girf`) of the GIRF data. In such cases, the GIRF is automatically resampled to match `dt_gradient` before convolution. This resampling process includes a normalization step to ensure that the integral (sum) of the GIRF is preserved, maintaining its physical meaning.

### Usage Hint
```python
from trajgen import apply_girf_convolution
import numpy as np

# Example:
# commanded_gradient_x: 1D NumPy array for X-axis
# girf_ht_x: 1D NumPy array for X-axis GIRF
# dt_gradient_waveform: Time step of commanded_gradient_x
# dt_girf_data: Time step of girf_ht_x

# convolved_gradient_x = apply_girf_convolution(
#     commanded_gradient_x, 
#     girf_ht_x, 
#     dt_gradient_waveform, 
#     dt_girf_data
# )
```

## 4. Predicting Actual Gradients: `predict_actual_gradients`

### Purpose
The `predict_actual_gradients` function simulates the actual gradient waveforms that would be produced by the MRI scanner. It takes a `Trajectory` object (which contains the *commanded* k-space trajectory and thus implies commanded gradients) and a `GIRF` object. It then convolves the commanded gradient for each axis with the respective GIRF component.

### Usage Hint
```python
from trajgen import Trajectory, GIRF, predict_actual_gradients
# Assume 'trajectory_obj' is an existing Trajectory object
# Assume 'girf_obj' is an existing GIRF object

# predicted_gradients_Tm = predict_actual_gradients(trajectory_obj, girf_obj)
```

### Output
The function returns a 2D NumPy array of shape `(num_dimensions, N_points)` representing the predicted actual gradient waveforms in T/m for each axis.

## 5. Correcting K-space (Post-hoc): `correct_kspace_with_girf`

### Purpose
The `correct_kspace_with_girf` function provides a way to retrospectively estimate the actual k-space trajectory traversed, given an original (commanded) `Trajectory` and a system `GIRF`. This is useful for understanding deviations and for potential artifact correction if raw data was acquired with the commanded trajectory.

### Process
1.  It first calls `predict_actual_gradients` internally to get the simulated actual gradient waveforms.
2.  Then, it re-integrates these actual gradient waveforms (using the appropriate gyromagnetic ratio and time step) to calculate the corrected k-space path. The initial k-space point of the original trajectory is preserved.

### Usage Hint
```python
from trajgen import Trajectory, GIRF, correct_kspace_with_girf
# Assume 'original_trajectory' and 'girf_obj' exist

# corrected_trajectory_obj = correct_kspace_with_girf(original_trajectory, girf_obj)
```

### Output
This function returns a *new* `Trajectory` object. This new object contains:
*   The GIRF-corrected k-space points.
*   The predicted actual gradient waveforms that would produce this corrected k-space.
*   Updated metadata, including a `girf_correction` dictionary detailing whether the correction was applied, the name of the GIRF used, and the gamma value used for k-space recalculation.

## 6. Pre-compensating Gradients: `precompensate_gradients_with_girf`

### Purpose
The `precompensate_gradients_with_girf` function is designed to calculate a set of *pre-distorted* commanded gradient waveforms. The idea is that when these pre-compensated gradients are played on the scanner (which has a response characterized by the GIRF), the *actual* gradients produced by the scanner will be much closer to the originally desired target gradient waveforms. This is a feed-forward compensation technique.

### Method
The function employs an iterative deconvolution algorithm, specifically the Van Cittert method. It iteratively refines the commanded gradient waveform for each axis:
`G_command_new = G_command_old + alpha * (G_target - G_actual_simulated)`
where `G_actual_simulated` is the result of convolving the current `G_command_old` with the GIRF.

### Usage Hint
```python
from trajgen import Trajectory, GIRF, precompensate_gradients_with_girf
# Assume 'target_trajectory_obj' (whose gradients are the desired output) 
# and 'girf_obj' exist

# precomp_trajectory_obj = precompensate_gradients_with_girf(
#     target_trajectory=target_trajectory_obj,
#     girf=girf_obj,
#     num_iterations=15, 
#     alpha=0.5,
#     tolerance=0.01 # Optional relative error for early stopping
# )
```

### Output
This function returns a *new* `Trajectory` object. This object contains:
*   The pre-compensated commanded gradient waveforms (these are the waveforms one would send to the scanner).
*   The k-space trajectory that these pre-compensated gradients would ideally trace (i.e., recalculated by integrating the pre-compensated gradients). This should be very close to the k-space of the `target_trajectory`.
*   Updated metadata, including a `girf_precompensation` dictionary detailing parameters like `num_iterations`, `alpha`, `tolerance`, final relative errors per axis, and the GIRF used.
