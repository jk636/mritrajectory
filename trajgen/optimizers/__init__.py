# trajgen/optimizers/__init__.py
"""
Trajectory Optimization Toolbox for trajgen.

This package provides tools to optimize trajectory generation parameters
based on user-defined cost functions.
"""

from .optimizer_core import TrajectoryOptimizer
from .cost_components import (
    calculate_hardware_penalty,
    calculate_gradient_roughness_penalty,
    calculate_pns_proxy_penalty
)

__all__ = [
    'TrajectoryOptimizer',
    'calculate_hardware_penalty',
    'calculate_gradient_roughness_penalty',
    'calculate_pns_proxy_penalty'
]
