# GIRF/sGIRF Correction and Precompensation Features

This document outlines the functionalities related to Gradient Impulse Response Functions (GIRF) and Switched Gradient Impulse Response Functions (sGIRF) for trajectory correction and precompensation.

## 1. Introduction to GIRF and sGIRF

Gradient systems in MRI are not perfect. The actual gradient waveform produced can deviate from the commanded (ideal) waveform due to factors like eddy currents, amplifier imperfections, and mechanical vibrations.

*   **GIRF (Gradient Impulse Response Function)**: Traditionally, GIRF models the system's linear time-invariant response for each gradient axis independently. It helps predict the actual gradient output based on a commanded input for that same axis.
*   **sGIRF (Switched Gradient Impulse Response Function)**: sGIRF extends this concept by characterizing the system's response to rapid switching, including cross-axis effects. This means a command on one axis (e.g., X) can induce responses on other axes (e.g., Y and Z). This is crucial for sequences like EPI, diffusion, and spiral imaging.

Accounting for these responses allows for more accurate k-space trajectory information, leading to improved image reconstruction.

## 2. Representing System Impulse Responses

### `GIRF` Class
*   **Purpose**: Stores independent impulse responses for each of the X, Y, and Z gradient axes (`h_t_x`, `h_t_y`, `h_t_z`) and their common time resolution `dt_girf`.
*   **Assumes**: Each axis responds independently to its own command (no cross-axis effects are modeled by this class itself).
*   **Creation**:
    *   Directly: `GIRF(h_t_x, h_t_y, h_t_z, dt_girf, name=None)`
    *   From `.npy` files: `GIRF.from_files(filepath_x, filepath_y, filepath_z, dt_girf, name=None)`

### `sGIRF` Class
*   **Purpose**: Stores a 3x3 matrix of time-domain impulse responses `h_ij(t)` (where `i` is output axis, `j` is input axis) and their common time resolution `dt_sgirf`. This explicitly models on-axis (`h_xx`, `h_yy`, `h_zz`) and cross-axis (`h_xy`, `h_xz`, `h_yx`, etc.) responses.
*   **Creation**:
    *   Directly: `sGIRF(h_xx_t, h_xy_t, ..., h_zz_t, dt_sgirf, name=None)` (all 9 components required).
    *   From `.npy` files: `sGIRF.from_numpy_files(filepaths_dict, dt_sgirf, name=None)` where `filepaths_dict` maps keys like 'xx', 'xy' to file paths.

## 3. Predicting Actual Gradient Waveforms

### Using `GIRF` (On-Axis Prediction)
*   **Function**: `predict_actual_gradients(trajectory: Trajectory, girf: GIRF) -> np.ndarray`
*   **Action**: Convolves each commanded gradient axis from the `trajectory` with its corresponding GIRF component (e.g., Gx_cmd \* hx).
*   **Output**: A `(D, N)` array, where `D` is the number of dimensions in the input `trajectory`.

### Using `sGIRF` (On-Axis and Cross-Axis Prediction)
*   **Function**: `predict_actual_gradients_from_sgirf(trajectory: Trajectory, sgirf: sGIRF) -> np.ndarray`
*   **Action**: Calculates the full 3-axis actual gradient response. For example, for the X-axis:
    `Gx_act = Gx_cmd*h_xx + Gy_cmd*h_xy + Gz_cmd*h_xz` (and similarly for Gy_act, Gz_act). If the input `trajectory` has fewer than 3 commanded axes, the missing commanded axes are treated as zero.
*   **Output**: Always a `(3, N)` array representing the predicted actual gradients for X, Y, and Z physical axes.

## 4. Applying GIRF/sGIRF Corrections and Precompensation

These functions modify a `Trajectory` object based on the system response model.

### `correct_kspace_with_girf`
*   **Signature**: `correct_kspace_with_girf(trajectory: Trajectory, girf_system: Union[GIRF, sGIRF], gamma_Hz_per_T: Optional[float] = None) -> Trajectory`
*   **Purpose**: Recalculates the k-space trajectory based on the *predicted actual* gradient waveforms.
*   **Behavior with `GIRF`**:
    *   Calls `predict_actual_gradients` internally.
    *   The output `Trajectory` will have the same number of dimensions as the input `trajectory`.
*   **Behavior with `sGIRF`**:
    *   Calls `predict_actual_gradients_from_sgirf` internally.
    *   The predicted actual gradients are always 3-dimensional.
    *   Consequently, the k-space of the output `Trajectory` will be **3-dimensional**, even if the input `trajectory` was 1D or 2D. The initial k-space point for any new dimension is assumed to be zero.
*   **Metadata**: The output `Trajectory`'s metadata is updated under the `'girf_correction'` key to include `type` ('GIRF' or 'sGIRF') and `girf_name`.

### `precompensate_gradients_with_girf`
*   **Signature**: `precompensate_gradients_with_girf(target_trajectory: Trajectory, girf_system: Union[GIRF, sGIRF], num_iterations: int = 10, alpha: float = 0.5, tolerance: Optional[float] = None, gamma_Hz_per_T: Optional[float] = None) -> Trajectory`
*   **Purpose**: Iteratively modifies the *commanded* gradient waveforms in the `target_trajectory` such that when these pre-compensated commands are passed through the `girf_system`, the resulting *actual* gradients match the original commanded gradients of the `target_trajectory`.
*   **Behavior with `GIRF`**:
    *   Pre-compensation is performed independently for each gradient axis present in the `target_trajectory`.
    *   The `final_errors_per_axis` are stored in the output metadata.
*   **Behavior with `sGIRF`**:
    *   Pre-compensation is performed globally, considering all on-axis and cross-axis terms.
    *   The iterative update internally uses 3-component commanded gradients (padded with zeros if the target is 1D/2D).
    *   The `target_gradients_Tm` are also padded to 3D if necessary for error calculation against the 3D simulated actual gradients.
    *   The final `commanded_gradients_precompensated` stored in the output `Trajectory` object are **trimmed back to match the number of dimensions of the original `target_trajectory`'s commanded gradients**.
    *   An `final_relative_error_overall` (scalar) is stored in the output metadata.
*   **Output**: A new `Trajectory` object containing the pre-compensated commanded gradients and the k-space recalculated from these pre-compensated gradients.
*   **Metadata**: Updated under `'girf_precompensation'` key, including `type` ('GIRF' or 'sGIRF'), `girf_name`, and relevant error metrics.

## 5. Utility for Convolution
*   **Function**: `apply_girf_convolution(gradient_waveform_1d: np.ndarray, girf_h_t_1d: np.ndarray, dt_gradient: float, dt_girf: float) -> np.ndarray`
*   **Purpose**: A core utility that convolves a single 1D gradient waveform with a single 1D impulse response. It handles resampling of the GIRF to match the gradient's time resolution (`dt_gradient`) if `dt_girf` is different, while preserving the integral of the GIRF.
*   **Used by**: All prediction and correction functions.

This framework allows for flexible and accurate modeling and correction of gradient system imperfections.
[end of wiki/girf_features.md]
[start of wiki/sgirf_and_pulses.md]
# Switched Gradient Impulse Response Function (sGIRF) and Pulse Generation

This document provides an overview of the Switched Gradient Impulse Response Function (sGIRF) concept and details the tools available for representing sGIRF data and generating suitable stimulus pulses.

## 1. Introduction to Switched Gradient Impulse Response Function (sGIRF)

The **Switched Gradient Impulse Response Function (sGIRF)** characterizes the behavior of an MRI gradient system in response to rapidly switched gradient waveforms. Unlike the traditional Gradient Impulse Response Function (GIRF) which primarily focuses on the linear time-invariant response to a single gradient axis input, the sGIRF aims to capture more complex dynamics that become significant during fast gradient switching.

Key aspects of sGIRF include:

*   **Focus on Switching Dynamics**: It's particularly important for sequences that rely on rapid and repeated gradient switching, such as Echo-Planar Imaging (EPI), Diffusion-Weighted Imaging (DWI), spiral imaging, and other advanced pulse sequences.
*   **Characterization of System Imperfections**: sGIRF measurements can help identify and quantify system imperfections like eddy currents, mechanical vibrations, and group delays that are more pronounced during high-speed operation.
*   **Cross-Axis Effects**: A crucial component of sGIRF is its ability to describe **cross-axis coupling**. This means that applying a gradient pulse on one logical axis (e.g., X) can induce responses not only on that axis but also on the other physical axes (Y and Z). These cross-term responses can lead to image artifacts if not accounted for.

Understanding and modeling the sGIRF allows for more accurate prediction of the actual gradient fields produced by the system, which can then be used for prospective or retrospective correction of k-space trajectories and image reconstruction, leading to improved image quality.

## 2. The `sGIRF` Class for Representing sGIRF Data

The `sGIRF` class is designed to store and manage the data associated with a switched GIRF characterization. It encapsulates the 3x3 matrix of time-domain impulse responses.

### Purpose

The primary purpose of the `sGIRF` class is to hold the nine distinct impulse response functions, `h_ij(t)`, where `i` denotes the output (measured) axis and `j` denotes the input (commanded) axis. It also stores the common time resolution, `dt_sgirf`, at which these responses were sampled.

### Key Attributes

*   `h_t_matrix (np.ndarray)`: A 3x3 NumPy array with `dtype=object`. Each element `h_t_matrix[i, j]` contains a 1D NumPy array representing the impulse response `h_ij(t)`. For example, `h_t_matrix[0, 1]` would be `h_xy(t)`, the response measured on the X-axis due to a stimulus on the Y-axis.
*   `dt_sgirf (float)`: The time resolution (sampling interval) in seconds for all impulse response functions stored in `h_t_matrix`.
*   `name (Optional[str])`: An optional user-defined name for the sGIRF profile (e.g., "SystemA_Default_sGIRF").

### Creation

#### Direct Instantiation

You can create an `sGIRF` object by directly providing all nine impulse response arrays and the time resolution.

```python
import numpy as np
from trajgen import sGIRF # Assuming sGIRF is in trajgen.py

# Example: Create dummy impulse responses (replace with actual data)
response_len = 128
dt = 4e-6 # 4 us
h_xx = np.sin(np.linspace(0, np.pi, response_len))
h_xy = np.zeros(response_len) 
# ... define all other 7 components (h_xz, h_yx, ..., h_zz) ...
h_yy = h_xx.copy() # Example
h_zz = h_xx.copy() # Example
h_xz = np.zeros(response_len)
h_yx = np.zeros(response_len)
h_yz = np.zeros(response_len)
h_zx = np.zeros(response_len)
h_zy = np.zeros(response_len)


sgirf_profile = sGIRF(
    h_xx_t=h_xx, h_xy_t=h_xy, h_xz_t=h_xz,
    h_yx_t=h_yx, h_yy_t=h_yy, h_yz_t=h_yz,
    h_zx_t=h_zx, h_zy_t=h_zy, h_zz_t=h_zz,
    dt_sgirf=dt,
    name="Example_sGIRF"
)
```

**Important Considerations for Direct Instantiation:**
*   All nine component arrays (`h_xx_t` through `h_zz_t`) must be provided.
*   Each component array must be a 1D NumPy array.
*   All component arrays must have the same, non-zero length.

#### Loading from NumPy Files

Alternatively, sGIRF data can be loaded from a set of `.npy` files, where each file contains one of the nine impulse response components.

```python
# Assume .npy files exist: 'sgirf_data_xx.npy', 'sgirf_data_xy.npy', ..., 'sgirf_data_zz.npy'
filepaths = {
    'xx': 'path/to/sgirf_data_xx.npy', 'xy': 'path/to/sgirf_data_xy.npy', 'xz': 'path/to/sgirf_data_xz.npy',
    'yx': 'path/to/sgirf_data_yx.npy', 'yy': 'path/to/sgirf_data_yy.npy', 'yz': 'path/to/sgirf_data_yz.npy',
    'zx': 'path/to/sgirf_data_zx.npy', 'zy': 'path/to/sgirf_data_zy.npy', 'zz': 'path/to/sgirf_data_zz.npy'
}
dt_measurement = 4e-6 # Time resolution of the saved .npy data

sgirf_from_files = sGIRF.from_numpy_files(filepaths, dt_sgirf=dt_measurement, name="Loaded_sGIRF_Profile")
```

The `from_numpy_files` method expects a dictionary where keys are strings like 'xx', 'xy', etc., and values are the corresponding filepaths.

## 3. Stimulus Pulse Generation: `generate_tw_ssi_pulse`

To measure an sGIRF, a stimulus gradient waveform with broad spectral content is typically required. The `generate_tw_ssi_pulse` function provides a way to generate such a pulse.

### Purpose

This function generates a **Tukey-windowed Shifted Sine-Integral (Tw-SSI) pulse**. The SSI pulse itself has a desirable sharp spectral cutoff, and the Tukey window is applied to smoothly taper the pulse edges, reducing ringing artifacts that might arise from abrupt waveform starts and ends. This type of pulse is often used as a stimulus for sGIRF characterization experiments. The mathematical formulation is based on the work by J. P. M. van der Zwaag et al. (MRM 2010; 64:1748-1757).

### Function Signature

```python
generate_tw_ssi_pulse(
    duration_s: float, 
    bandwidth_hz: float, 
    dt_s: float = 4e-6, 
    tukey_alpha: float = 0.3
) -> np.ndarray
```

### Key Parameters

*   `duration_s (float)`: The total duration of the pulse in seconds.
*   `bandwidth_hz (float)`: The desired bandwidth of the pulse in Hz. This parameter influences the sharpness of the spectral features of the SSI component.
*   `dt_s (float)`: The sampling interval (time step) in seconds for the generated waveform. Defaults to `4e-6` s (4 µs).
*   `tukey_alpha (float)`: The shape parameter for the Tukey window. It represents the ratio of the taper duration to the total window duration.
    *   `tukey_alpha = 0` results in a rectangular window (no tapering).
    *   `tukey_alpha = 1` results in a Hann window (fully tapered).
    *   Defaults to `0.3`.

### Output

The function returns a 1D NumPy array representing the generated gradient waveform. The waveform is normalized such that its peak absolute amplitude is 1.0.

### Example Usage

```python
from trajgen import generate_tw_ssi_pulse
import matplotlib.pyplot as plt # For visualization

pulse_duration = 2e-3  # 2 ms
pulse_bandwidth = 5000 # 5 kHz
time_step = 4e-6       # 4 us

tw_ssi_waveform = generate_tw_ssi_pulse(
    duration_s=pulse_duration,
    bandwidth_hz=pulse_bandwidth,
    dt_s=time_step,
    tukey_alpha=0.25
)

# To visualize (optional)
# time_axis = np.arange(len(tw_ssi_waveform)) * time_step
# plt.plot(time_axis * 1000, tw_ssi_waveform) # Time in ms
# plt.title("Tukey-Windowed SSI Pulse")
# plt.xlabel("Time (ms)")
# plt.ylabel("Normalized Amplitude")
# plt.grid(True)
# plt.show()
```

### Dependencies

This function relies on functions from the `scipy` library:
*   `scipy.special.sici` for calculating the Sine Integral.
*   `scipy.signal.windows.tukey` for generating the Tukey window.

Ensure `scipy` is installed in your Python environment to use this function.

## 4. Using sGIRF in Trajectory Correction and Pre-compensation

The `sGIRF` objects can be used with the trajectory correction and pre-compensation functions available in `trajgen.py` to account for cross-axis gradient imperfections. These functions now accept a `girf_system` parameter which can be either a `GIRF` or an `sGIRF` object.

*   **`correct_kspace_with_girf(trajectory: Trajectory, girf_system: Union[GIRF, sGIRF], ...)`**:
    *   When an `sGIRF` object is passed as `girf_system`, this function internally uses `predict_actual_gradients_from_sgirf` to determine the 3-axis actual gradient waveforms.
    *   **Important Dimensionality Note**: If the input `trajectory` is 1D or 2D, the resulting corrected `Trajectory` object will have a **3-dimensional k-space and 3-dimensional actual gradient waveforms**. This is because the sGIRF model inherently predicts responses on all three physical axes (X, Y, Z). The initial k-space point for any newly introduced dimension (e.g., Z for a 2D input) is assumed to be zero.

*   **`precompensate_gradients_with_girf(target_trajectory: Trajectory, girf_system: Union[GIRF, sGIRF], ...)`**:
    *   When an `sGIRF` object is passed as `girf_system`, the iterative pre-compensation algorithm considers the full 3x3 sGIRF matrix for a global correction.
    *   The internal iterative process updates a 3-component commanded gradient. The target gradients (from `target_trajectory`) are padded to 3D if necessary for error calculation against the 3D simulated actual gradients from the sGIRF.
    *   The final `commanded_gradients_precompensated` stored in the output `Trajectory` object are **trimmed back to match the dimensionality of the original `target_trajectory`'s commanded gradients**. For instance, if a 2D target trajectory is supplied, the output pre-compensated trajectory will also feature 2D commanded gradients, despite the internal use of a 3x3 sGIRF model for calculations.

Using `sGIRF` objects with these functions enables a more comprehensive correction by accounting for cross-axis gradient system imperfections.
[end of wiki/sgirf_and_pulses.md]
