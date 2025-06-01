# trajgen/optimizers/optimizer_core.py
"""
Core class for trajectory optimization.
"""
import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Callable, Dict, Tuple, Any, List, Optional
from trajgen.trajectory import Trajectory # Assuming Trajectory class
# COMMON_NUCLEI_GAMMA_HZ_PER_T could be imported if needed for default gamma,
# but Trajectory class has its own default.
# from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T


__all__ = ['TrajectoryOptimizer']

class TrajectoryOptimizer:
    def __init__(
        self,
        generator_func: Callable[..., np.ndarray], # e.g., generate_drunken_kooshball_trajectory
        optimizable_params_config: Dict[str, Tuple[Optional[float], Optional[float]]], # {'param_name': (min_bound, max_bound)}
        cost_evaluator_func: Callable[[Trajectory], float],
        fixed_generator_params: Dict[str, Any]
    ):
        """
        Initializes the TrajectoryOptimizer.

        Args:
            generator_func: The trajectory generator function
                            (e.g., generate_drunken_kooshball_trajectory).
                            Expected to return k-space points (np.ndarray).
            optimizable_params_config: Dictionary mapping parameter names (strings)
                                       to tuples of (min_bound, max_bound). These are
                                       the parameters of `generator_func` to be optimized.
                                       Use None for unbound parameters (though L-BFGS-B requires bounds).
            cost_evaluator_func: A function that takes a `Trajectory` object
                                 and returns a single float value (the cost).
            fixed_generator_params: A dictionary of parameters for `generator_func`
                                    that are fixed during this optimization. Must include
                                    at least 'dt_seconds' for Trajectory object creation,
                                    and any other params required by the generator not being optimized.
                                    'gamma_Hz_per_T' can also be provided here.
        """
        self.generator_func = generator_func
        self.optimizable_params_config = optimizable_params_config
        self.cost_evaluator_func = cost_evaluator_func
        self.fixed_generator_params = fixed_generator_params

        if 'dt_seconds' not in self.fixed_generator_params:
            raise ValueError("fixed_generator_params must include 'dt_seconds'.")

        self.param_names_to_optimize: List[str] = list(optimizable_params_config.keys())

        bounds_min: List[float] = []
        bounds_max: List[float] = []
        for param_name in self.param_names_to_optimize:
            min_b, max_b = optimizable_params_config[param_name]
            # L-BFGS-B requires finite bounds, replace None with -inf/+inf if truly unbounded,
            # or user should provide reasonable large numbers. For now, assume finite.
            if min_b is None:
                print(f"Warning: Parameter '{param_name}' has no min_bound. Optimizer might behave unexpectedly or require a method supporting unbounded optimization.")
                min_b = -np.inf
            if max_b is None:
                print(f"Warning: Parameter '{param_name}' has no max_bound. Optimizer might behave unexpectedly or require a method supporting unbounded optimization.")
                max_b = np.inf
            bounds_min.append(min_b)
            bounds_max.append(max_b)
        self.bounds_for_optimizer = Bounds(bounds_min, bounds_max)

        self.iteration_count = 0
        self.best_cost_so_far = float('inf')
        self.best_params_so_far: Optional[np.ndarray] = None
        self.all_evaluated_params: List[Dict[str, Any]] = [] # For debugging/logging

    def _objective_function(self, param_values_array: np.ndarray) -> float:
        """
        Objective function for scipy.optimize.minimize.
        Generates a trajectory with the given parameters and evaluates its cost.
        """
        self.iteration_count += 1

        current_generator_params = self.fixed_generator_params.copy()
        current_optimizable_params_dict: Dict[str, Any] = {}
        for i, param_name in enumerate(self.param_names_to_optimize):
            current_generator_params[param_name] = param_values_array[i]
            current_optimizable_params_dict[param_name] = param_values_array[i]

        self.all_evaluated_params.append(current_optimizable_params_dict)

        try:
            k_space_points = self.generator_func(**current_generator_params)
        except Exception as e:
            print(f"Error during trajectory generation in objective function (Iter {self.iteration_count}): {e}")
            # print(f"Parameters: {current_generator_params}") # Can be too verbose
            return float('inf')

        traj_name = f"opt_iter_{self.iteration_count}"
        dt = current_generator_params['dt_seconds']
        gamma = current_generator_params.get('gamma_Hz_per_T',
                                             self.fixed_generator_params.get('gamma_Hz_per_T'))

        try:
            trajectory = Trajectory(
                name=traj_name,
                kspace_points_rad_per_m=k_space_points,
                dt_seconds=dt,
                gamma_Hz_per_T=gamma,
                metadata={'optimized_params_values': param_values_array.tolist(), # Store current values
                          'optimizable_param_names': self.param_names_to_optimize}
            )
            cost = self.cost_evaluator_func(trajectory)
        except Exception as e:
            print(f"Error during Trajectory creation or cost evaluation (Iter {self.iteration_count}): {e}")
            return float('inf')


        print(f"Iter {self.iteration_count}: Params={param_values_array}, Cost={cost:.6e}")

        if cost < self.best_cost_so_far:
            self.best_cost_so_far = cost
            self.best_params_so_far = param_values_array.copy()
            print(f"  New best cost: {self.best_cost_so_far:.6e}")

        return cost

    def optimize(
        self,
        initial_guess_values: Optional[List[float]] = None,
        method: str = 'L-BFGS-B', # Bounds are handled by L-BFGS-B, SLSQP, TNC, etc.
        optimizer_options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Trajectory], float, Any]:
        self.iteration_count = 0
        self.best_cost_so_far = float('inf')
        self.best_params_so_far = None
        self.all_evaluated_params = []

        initial_guess_np: np.ndarray
        if initial_guess_values is None:
            initial_guess_np = np.array([
                (min_b + max_b) / 2.0 if np.isfinite(min_b) and np.isfinite(max_b) \
                else min_b if np.isfinite(min_b) \
                else max_b if np.isfinite(max_b) \
                else 0.0
                for min_b, max_b in zip(self.bounds_for_optimizer.lb, self.bounds_for_optimizer.ub)
            ])
            # Ensure guess is within bounds if bounds are one-sided inf
            for i in range(len(initial_guess_np)):
                if initial_guess_np[i] == -np.inf and self.bounds_for_optimizer.ub[i] != np.inf:
                    initial_guess_np[i] = self.bounds_for_optimizer.ub[i] - 1.0 # Adjust if only upper bound
                elif initial_guess_np[i] == np.inf and self.bounds_for_optimizer.lb[i] != -np.inf:
                    initial_guess_np[i] = self.bounds_for_optimizer.lb[i] + 1.0 # Adjust if only lower bound
                elif initial_guess_np[i] == -np.inf or initial_guess_np[i] == np.inf:
                     initial_guess_np[i] = 0.0 # Default if unbounded on both sides

        else:
            initial_guess_np = np.array(initial_guess_values)
            if len(initial_guess_np) != len(self.param_names_to_optimize):
                raise ValueError("Length of initial_guess_values must match number of optimizable parameters.")

        # Clip initial guess to be within bounds if provided
        initial_guess_np = np.clip(initial_guess_np, self.bounds_for_optimizer.lb, self.bounds_for_optimizer.ub)


        if optimizer_options is None:
            optimizer_options = {'disp': False, 'maxiter': 50, 'ftol': 1e-7, 'gtol': 1e-6}

        print(f"Starting optimization for: {self.param_names_to_optimize}")
        # print(f"With fixed params: {self.fixed_generator_params}") # Can be too verbose
        print(f"Initial guess: {initial_guess_np}")
        print(f"Bounds: LB={self.bounds_for_optimizer.lb}, UB={self.bounds_for_optimizer.ub}")

        result = minimize(
            self._objective_function,
            initial_guess_np,
            method=method,
            bounds=self.bounds_for_optimizer,
            options=optimizer_options
        )

        print(f"Optimization finished. Status: {result.message}")
        print(f"Total iterations by optimizer: {result.nit}, Objective function calls: {self.iteration_count}")

        optimal_params_to_use: Optional[np.ndarray] = None
        final_cost_to_use: float = float('inf')

        if self.best_params_so_far is not None:
            optimal_params_to_use = self.best_params_so_far
            final_cost_to_use = self.best_cost_so_far
            print(f"Best parameters tracked during iterations: {optimal_params_to_use}")
            print(f"Best cost tracked during iterations: {final_cost_to_use:.6e}")
        elif result.success : # Fallback if best_params_so_far wasn't updated (e.g. initial guess was best)
            optimal_params_to_use = result.x
            final_cost_to_use = result.fun # This is the cost at result.x
            print(f"Using parameters from scipy result: {optimal_params_to_use}")
            print(f"Cost from scipy result: {final_cost_to_use:.6e}")

        if optimal_params_to_use is None:
            print("Optimization did not find a successful set of parameters or improve upon initial state.")
            return None, float('inf'), result

        final_generator_params = self.fixed_generator_params.copy()
        optimized_params_dict = {}
        for i, param_name in enumerate(self.param_names_to_optimize):
            final_generator_params[param_name] = optimal_params_to_use[i]
            optimized_params_dict[param_name] = optimal_params_to_use[i]

        try:
            best_k_space_points = self.generator_func(**final_generator_params)
        except Exception as e:
            print(f"Error generating trajectory with final optimal parameters: {e}")
            return None, final_cost_to_use, result

        dt = final_generator_params['dt_seconds']
        gamma = final_generator_params.get('gamma_Hz_per_T',
                                           self.fixed_generator_params.get('gamma_Hz_per_T'))

        best_trajectory = Trajectory(
            name=f"optimized_{self.generator_func.__name__}",
            kspace_points_rad_per_m=best_k_space_points,
            dt_seconds=dt,
            gamma_Hz_per_T=gamma,
            metadata={
                'optimized_params': optimized_params_dict,
                'fixed_params_subset': {k:v for k,v in self.fixed_generator_params.items() if isinstance(v, (int, float, str, bool, tuple, type(None)))},
                'optimization_result_message': result.message,
                'final_cost': final_cost_to_use,
                'optimizer_iterations': result.nit,
                'objective_function_calls': self.iteration_count
            }
        )
        return best_trajectory, final_cost_to_use, result
```
