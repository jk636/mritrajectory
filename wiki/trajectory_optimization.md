# Trajectory Optimization Toolbox

The `trajgen` library includes a Trajectory Optimization Toolbox designed to help find optimal parameters for trajectory generator functions based on a user-defined cost function. This allows for tailoring trajectories to specific hardware constraints, imaging goals, or other desired properties.

## Core Components

The toolbox primarily consists of:

1.  **`TrajectoryOptimizer` Class (`trajgen.optimizers.TrajectoryOptimizer`)**:
    *   The main class that manages the optimization process.
    *   It uses `scipy.optimize.minimize` (typically with methods like 'L-BFGS-B' or 'SLSQP') to find the best set of parameters.
    *   **Initialization**:
        *   `generator_func`: The trajectory generator function you want to optimize (e.g., `generate_drunken_kooshball_trajectory`).
        *   `optimizable_params_config`: A dictionary specifying which parameters of the `generator_func` to optimize, along with their (min, max) bounds. Example: `{'param_name': (lower_bound, upper_bound)}`.
        *   `cost_evaluator_func`: A Python function you define that takes a `Trajectory` object and returns a single floating-point number representing the "cost" or "badness" of that trajectory. Lower is better.
        *   `fixed_generator_params`: A dictionary of any other parameters required by your `generator_func` that will remain fixed during this optimization run. This *must* include `dt_seconds` and typically `gamma_Hz_per_T` if not default.

    *   **Key Method**:
        *   `optimize(initial_guess_values: Optional[List[float]] = None, method: str = 'L-BFGS-B', optimizer_options: Optional[Dict] = None) -> Tuple[Optional[Trajectory], float, Any]`:
            *   Runs the optimization.
            *   `initial_guess_values`: Starting point for the optimizable parameters. If `None`, midpoints of bounds are used.
            *   `method`: Algorithm for `scipy.optimize.minimize`.
            *   `optimizer_options`: Options passed to the scipy optimizer (e.g., `{'maxiter': 100, 'disp': True}`).
            *   Returns: The best `Trajectory` found, its cost, and the full result object from `scipy.optimize.minimize`.

2.  **Cost Component Functions (`trajgen.optimizers.cost_components`)**:
    *   A set of pre-defined functions to help you build your `cost_evaluator_func`. These calculate penalties for common trajectory attributes:
        *   `calculate_hardware_penalty(trajectory, grad_limit_Tm_per_m, slew_limit_Tm_per_s_per_m, penalty_factor)`: Penalizes exceeding specified gradient and slew rate limits.
        *   `calculate_gradient_roughness_penalty(trajectory, penalty_factor)`: Penalizes trajectories with "rough" or rapidly changing gradients, often used as a proxy for reducing acoustic noise. (Currently implemented as sum of squared slew magnitudes).
        *   `calculate_pns_proxy_penalty(trajectory, pns_threshold_T_per_s, penalty_factor)`: Penalizes trajectories if their maximum slew rate (used as a simplified proxy for PNS) exceeds a threshold.
        *   `calculate_signal_decay_penalty(trajectory: Trajectory, T1_ms: Optional[float], T2_ms: float, b_value_s_per_mm2: float, k_space_weighting_func: Optional[Callable], is_13c: bool, penalty_factor: float) -> float`:
            *   Calculates a penalty based on MR signal decay due to T1 (especially if `is_13c=True`), T2, and diffusion effects (`b_value_s_per_mm2`).
            *   Penalizes late acquisition of k-space points, especially high k-space radii (weighted by `radii^2` or `radii^3` by default).
            *   Allows an optional `k_space_weighting_func` to further define importance of k-space regions.
            *   Includes an additional penalty if the total acquisition time significantly exceeds limits derived from T1/T2.
        *   `calculate_psf_incoherence_penalty(trajectory: Trajectory, k_max_rad_per_m: float, target_density_func: Optional[Callable], penalty_factor: float, voronoi_qhull_options: str) -> float`:
            *   Estimates k-space coverage quality and incoherence using Voronoi tessellation as a proxy for PSF properties.
            *   Penalizes:
                *   High variance in Voronoi cell sizes (squared coefficient of variation, indicating non-uniform sampling).
                *   Deviation from a `target_density_func` if provided (compares normalized actual vs. target density profiles).
                *   Large empty k-space regions (based on max distance from Voronoi vertices to k-space samples relative to average sample spacing).

## Workflow

1.  **Choose a Generator**: Identify the trajectory generator function whose parameters you want to optimize (e.g., `generate_drunken_kooshball_trajectory`).
2.  **Define Fixed Parameters**: Set values for generator parameters that will not be optimized.
3.  **Define Optimizable Parameters & Bounds**: Specify which parameters of the generator should be tuned by the optimizer and define their valid ranges (min/max bounds).
4.  **Write a Cost Evaluator Function**: This is a crucial step. You create a Python function that:
    *   Accepts a single `Trajectory` object as input.
    *   Uses the provided cost component functions (and/or your own custom logic) to calculate a total cost (a single float). This function embodies your optimization goals (e.g., stay within hardware limits, minimize roughness, achieve certain k-space coverage).
    *   The optimizer will try to find parameters that minimize the output of this function.
5.  **Instantiate `TrajectoryOptimizer`**: Provide the generator, parameter configs, your cost evaluator, and fixed parameters.
6.  **Run Optimization**: Call the `optimize()` method.
7.  **Analyze Results**: Examine the returned best trajectory, its cost, and the optimized parameters.

## Example

For a detailed, runnable example, please see the Jupyter Notebook:
-   [**Optimizing a Drunken Kooshball Trajectory**](../examples/07_trajectory_optimization.ipynb)
    *(Note: If viewing this on GitHub, you might need to adjust the path or run a Jupyter Notebook server from the repository root to open the .ipynb file correctly. The path `../examples/` assumes this `trajectory_optimization.md` file is in a `wiki` directory that is a sibling to the `examples` directory.)*

This notebook demonstrates:
-   Setting up fixed and optimizable parameters for `generate_drunken_kooshball_trajectory`.
-   Defining a custom cost function combining several cost components.
-   Running the `TrajectoryOptimizer`.
-   Visualizing the resulting optimized trajectory.

## Tips for Effective Optimization

-   **Start Simple**: Begin with optimizing only 1-2 parameters and a simple cost function.
-   **Parameter Bounds**: Well-chosen bounds are critical. If bounds are too wide, optimization can be slow. If too narrow, the true optimum might be missed. `L-BFGS-B` requires finite bounds.
-   **Cost Function Weights**: The relative weights (`penalty_factor` values) you assign to different components in your cost function will determine the trade-offs in the optimized trajectory. Experiment with these.
-   **Initial Guess**: Providing a good initial guess can sometimes speed up convergence, though the optimizer can also start from the midpoint of bounds if an explicit guess is not provided.
-   **Optimizer Method & Options**: Different `scipy.optimize.minimize` methods may perform better on different problems. `L-BFGS-B` is a good default for bound-constrained problems. `maxiter` in `optimizer_options` is important; complex problems may need many iterations. `disp: True` can provide more insight during the optimization run.
-   **Normalization**: When combining different cost components (e.g., hardware penalties vs. roughness), ensure they are on a somewhat comparable scale, or that their weights reflect their relative importance accurately. The provided cost components use some internal normalization for penalties (e.g., dividing by the limit before squaring), but overall scaling might still need attention in your combined cost function.
    - **`dt_seconds` for Time-Dependent Costs**: Ensure `dt_seconds` is correctly set in your `fixed_generator_params` when using time-dependent cost functions like `calculate_signal_decay_penalty`, as it's used by the `Trajectory` object (via `get_acquisition_times_ms()`) to calculate acquisition times.

This toolbox provides a flexible way to move towards more automated and objective-driven trajectory design.
