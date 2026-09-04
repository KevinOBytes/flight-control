"""
VectorFlight offline validation, analysis, and geometry search tools.
"""

from .geometry import PodGeometry, RotorGeometry, VehicleModel, rodrigues_rotation, ypr_to_rotation_matrix
from .jacobians import (
    analyze_controllability,
    compute_full_effectiveness_matrix,
    compute_motor_tilt_jacobian,
    compute_pod_tilt_jacobian,
    compute_thrust_jacobian,
)
from .optimizer import OptimizationResult, evaluate_fault_tolerance, optimize_baseline_tilts

__version__ = "0.1.0"

__all__ = [
    "VehicleModel",
    "PodGeometry",
    "RotorGeometry",
    "rodrigues_rotation",
    "ypr_to_rotation_matrix",
    "compute_thrust_jacobian",
    "compute_motor_tilt_jacobian",
    "compute_pod_tilt_jacobian",
    "compute_full_effectiveness_matrix",
    "analyze_controllability",
    "optimize_baseline_tilts",
    "evaluate_fault_tolerance",
    "OptimizationResult",
]
