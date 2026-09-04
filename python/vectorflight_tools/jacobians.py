"""
Numerical finite-difference Jacobian computations and control effectiveness analysis.
"""

from typing import Any, Dict
import numpy as np
from .geometry import VehicleModel


def compute_thrust_jacobian(
    model: VehicleModel,
    thrusts: np.ndarray,
    motor_tilts: np.ndarray,
    pod_tilts: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Computes 6x16 Thrust Jacobian J_T = dW/dT using central finite differences.
    """
    thrusts = np.asarray(thrusts, dtype=np.float64)
    motor_tilts = np.asarray(motor_tilts, dtype=np.float64)
    pod_tilts = np.asarray(pod_tilts, dtype=np.float64)

    j_thrust = np.zeros((6, 16), dtype=np.float64)
    for i in range(16):
        t_plus = thrusts.copy()
        t_minus = thrusts.copy()
        t_plus[i] += eps
        t_minus[i] -= eps

        w_plus = model.compute_forward_wrench(t_plus, motor_tilts, pod_tilts)
        w_minus = model.compute_forward_wrench(t_minus, motor_tilts, pod_tilts)
        j_thrust[:, i] = (w_plus - w_minus) / (2.0 * eps)

    return j_thrust


def compute_motor_tilt_jacobian(
    model: VehicleModel,
    thrusts: np.ndarray,
    motor_tilts: np.ndarray,
    pod_tilts: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Computes 6x16 Motor Tilt Jacobian J_gamma = dW/dgamma using central finite differences.
    """
    thrusts = np.asarray(thrusts, dtype=np.float64)
    motor_tilts = np.asarray(motor_tilts, dtype=np.float64)
    pod_tilts = np.asarray(pod_tilts, dtype=np.float64)

    j_tilt = np.zeros((6, 16), dtype=np.float64)
    for i in range(16):
        g_plus = motor_tilts.copy()
        g_minus = motor_tilts.copy()
        g_plus[i] += eps
        g_minus[i] -= eps

        w_plus = model.compute_forward_wrench(thrusts, g_plus, pod_tilts)
        w_minus = model.compute_forward_wrench(thrusts, g_minus, pod_tilts)
        j_tilt[:, i] = (w_plus - w_minus) / (2.0 * eps)

    return j_tilt


def compute_pod_tilt_jacobian(
    model: VehicleModel,
    thrusts: np.ndarray,
    motor_tilts: np.ndarray,
    pod_tilts: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Computes 6x8 Pod Tilt Jacobian J_pod = dW/dtheta using central finite differences.
    """
    thrusts = np.asarray(thrusts, dtype=np.float64)
    motor_tilts = np.asarray(motor_tilts, dtype=np.float64)
    pod_tilts = np.asarray(pod_tilts, dtype=np.float64)

    j_pod = np.zeros((6, 8), dtype=np.float64)
    for i in range(8):
        p_plus = pod_tilts.copy()
        p_minus = pod_tilts.copy()
        p_plus[i] += eps
        p_minus[i] -= eps

        w_plus = model.compute_forward_wrench(thrusts, motor_tilts, p_plus)
        w_minus = model.compute_forward_wrench(thrusts, motor_tilts, p_minus)
        j_pod[:, i] = (w_plus - w_minus) / (2.0 * eps)

    return j_pod


def compute_full_effectiveness_matrix(
    model: VehicleModel,
    thrusts: np.ndarray,
    motor_tilts: np.ndarray,
    pod_tilts: np.ndarray,
) -> np.ndarray:
    """
    Computes 6x40 total control effectiveness matrix B = [J_T | J_gamma | J_theta].
    """
    j_t = compute_thrust_jacobian(model, thrusts, motor_tilts, pod_tilts)
    j_g = compute_motor_tilt_jacobian(model, thrusts, motor_tilts, pod_tilts)
    j_p = compute_pod_tilt_jacobian(model, thrusts, motor_tilts, pod_tilts)
    return np.hstack([j_t, j_g, j_p])


def analyze_controllability(jacobian: np.ndarray) -> Dict[str, Any]:
    """
    Performs SVD and rank/condition analysis of a control Jacobian (e.g. 6x16 or 6x40).
    """
    u, s, vh = np.linalg.svd(jacobian, full_matrices=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    sigma_max = float(s[0]) if len(s) > 0 else 0.0
    sigma_min = float(s[-1]) if len(s) > 0 else 0.0
    condition_number = float(sigma_max / max(sigma_min, 1e-12))

    return {
        "rank": rank,
        "singular_values": s,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": condition_number,
        "u_matrix": u,
        "v_matrix": vh.T,
    }
