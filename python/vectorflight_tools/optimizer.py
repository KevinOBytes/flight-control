"""
Nonlinear orientation and geometry optimizer using SciPy.
Finds baseline motor and pod tilt configurations that maximize control authority (sigma_min)
while maintaining zero net parasitic moments and horizontal forces in hover.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from .geometry import VehicleModel
from .jacobians import compute_thrust_jacobian, analyze_controllability


@dataclass
class OptimizationResult:
    success: bool
    message: str
    optimal_motor_tilts: np.ndarray
    optimal_pod_tilts: np.ndarray
    optimal_sigma_min: float
    optimal_condition_number: float
    hover_wrench_residual: np.ndarray
    fault_tolerance_summary: Dict[str, Any]


def evaluate_fault_tolerance(
    model: VehicleModel,
    motor_tilts: np.ndarray,
    pod_tilts: np.ndarray,
    nominal_thrust: float,
) -> Dict[str, Any]:
    """
    Evaluates control authority and minimum singular value under every possible single-motor failure.
    """
    single_failures = []
    nominal_thrusts = np.full(16, nominal_thrust, dtype=np.float64)

    for failed_idx in range(16):
        t_failed = nominal_thrusts.copy()
        t_failed[failed_idx] = 0.0

        j_t = compute_thrust_jacobian(model, t_failed, motor_tilts, pod_tilts)
        # Drop the failed motor's column
        j_healthy = np.delete(j_t, failed_idx, axis=1)

        metrics = analyze_controllability(j_healthy)
        single_failures.append({
            "failed_rotor_id": failed_idx + 1,
            "rank": metrics["rank"],
            "sigma_min": metrics["sigma_min"],
            "condition_number": metrics["condition_number"],
        })

    sigma_mins = [f["sigma_min"] for f in single_failures]
    ranks = [f["rank"] for f in single_failures]

    return {
        "all_single_failures": single_failures,
        "worst_case_sigma_min": float(min(sigma_mins)),
        "mean_sigma_min": float(np.mean(sigma_mins)),
        "all_rank_6": all(r == 6 for r in ranks),
    }


def optimize_baseline_tilts(
    model: VehicleModel,
    target_sigma_min: float = 0.20,
    max_tilt_rad: float = 0.25,
    optimize_pods: bool = True,
) -> OptimizationResult:
    """
    Optimizes baseline static tilt angles (motor tilts and optionally pod tilts)
    to maximize minimum singular value of J_thrust while maintaining exact hover moment balance.
    """
    hover_thrust_total = model.mass_kg * 9.80665
    nominal_thrust_per_motor = hover_thrust_total / 16.0
    nominal_thrusts = np.full(16, nominal_thrust_per_motor, dtype=np.float64)

    # Initial guess: alternating small tilts (10 degrees = 0.174 rad)
    x0_motor = np.array([0.15 * ((-1) ** i) for i in range(16)], dtype=np.float64)
    x0_pod = np.array([0.10, -0.10, -0.10, 0.10, 0.10, 0.10, -0.10, -0.10], dtype=np.float64)

    if optimize_pods:
        x0 = np.concatenate([x0_motor, x0_pod])
        lower_bounds = np.concatenate([
            np.full(16, -max_tilt_rad),
            np.full(8, -max_tilt_rad)
        ])
        upper_bounds = np.concatenate([
            np.full(16, max_tilt_rad),
            np.full(8, max_tilt_rad)
        ])
    else:
        x0 = x0_motor
        lower_bounds = np.full(16, -max_tilt_rad)
        upper_bounds = np.full(16, max_tilt_rad)

    def unpack_tilts(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if optimize_pods:
            return x[:16], x[16:]
        else:
            return x[:16], np.zeros(8, dtype=np.float64)

    # Objective function: Maximize sigma_min (i.e. minimize -sigma_min) + smoothness/regularization
    def objective(x: np.ndarray) -> float:
        m_tilts, p_tilts = unpack_tilts(x)
        j_t = compute_thrust_jacobian(model, nominal_thrusts, m_tilts, p_tilts)
        metrics = analyze_controllability(j_t)

        sigma_min = metrics["sigma_min"]
        # Reward sigma_min up to and above target_sigma_min
        cost = -10.0 * sigma_min + 0.1 * np.sum(x ** 2)
        return cost

    # Equality constraints: Zero parasitic horizontal forces (Fx, Fy) and moments (Mx, My, Mz)
    def equilibrium_constraints(x: np.ndarray) -> np.ndarray:
        m_tilts, p_tilts = unpack_tilts(x)
        wrench = model.compute_forward_wrench(nominal_thrusts, m_tilts, p_tilts)
        # We constrain [Fx, Fy, Mx, My, Mz] to be zero
        return np.array([wrench[0], wrench[1], wrench[3], wrench[4], wrench[5]], dtype=np.float64)

    eq_constraint = NonlinearConstraint(
        equilibrium_constraints,
        lb=np.zeros(5),
        ub=np.zeros(5),
    )

    bounds = Bounds(lower_bounds, upper_bounds)

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[eq_constraint],
        options={"maxiter": 300, "ftol": 1e-6, "disp": False},
    )

    opt_m_tilts, opt_p_tilts = unpack_tilts(res.x)
    opt_jt = compute_thrust_jacobian(model, nominal_thrusts, opt_m_tilts, opt_p_tilts)
    opt_metrics = analyze_controllability(opt_jt)
    opt_wrench = model.compute_forward_wrench(nominal_thrusts, opt_m_tilts, opt_p_tilts)

    fault_summary = evaluate_fault_tolerance(
        model, opt_m_tilts, opt_p_tilts, nominal_thrust_per_motor
    )

    return OptimizationResult(
        success=bool(res.success),
        message=str(res.message),
        optimal_motor_tilts=opt_m_tilts,
        optimal_pod_tilts=opt_p_tilts,
        optimal_sigma_min=opt_metrics["sigma_min"],
        optimal_condition_number=opt_metrics["condition_number"],
        hover_wrench_residual=opt_wrench,
        fault_tolerance_summary=fault_summary,
    )
