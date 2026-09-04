"""
Unit tests for numerical Jacobians and controllability analysis.
"""

from pathlib import Path
import numpy as np
import pytest

from vectorflight_tools.geometry import VehicleModel
from vectorflight_tools.jacobians import (
    analyze_controllability,
    compute_full_effectiveness_matrix,
    compute_motor_tilt_jacobian,
    compute_pod_tilt_jacobian,
    compute_thrust_jacobian,
)


@pytest.fixture
def config_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "configs" / "vehicle_v1.toml"


@pytest.fixture
def model(config_path: Path) -> VehicleModel:
    return VehicleModel.from_toml(config_path)


def test_thrust_jacobian_dimensions_and_properties(model: VehicleModel):
    thrusts = np.full(16, 15.0)
    # Give small opposing tilts to evaluate full 6-DOF effectiveness
    motor_tilts = np.array([0.15 * ((-1) ** i) for i in range(16)])
    pod_tilts = np.array([0.10, -0.10, -0.10, 0.10, 0.10, 0.10, -0.10, -0.10])

    j_t = compute_thrust_jacobian(model, thrusts, motor_tilts, pod_tilts)
    assert j_t.shape == (6, 16)

    metrics = analyze_controllability(j_t)
    assert metrics["rank"] == 6
    assert metrics["sigma_min"] > 0.02
    assert metrics["condition_number"] < 500.0


def test_motor_tilt_jacobian_dimensions(model: VehicleModel):
    thrusts = np.full(16, 15.0)
    motor_tilts = np.zeros(16)
    pod_tilts = np.zeros(8)

    j_g = compute_motor_tilt_jacobian(model, thrusts, motor_tilts, pod_tilts)
    assert j_g.shape == (6, 16)

    # When motors are vertical and spinning, motor tilt around Y-axis produces Fx force
    # Each column should have non-zero Fx derivative (row 0)
    assert np.all(np.abs(j_g[0, :]) > 1.0)


def test_pod_tilt_jacobian_dimensions(model: VehicleModel):
    thrusts = np.full(16, 15.0)
    motor_tilts = np.zeros(16)
    pod_tilts = np.zeros(8)

    j_p = compute_pod_tilt_jacobian(model, thrusts, motor_tilts, pod_tilts)
    assert j_p.shape == (6, 8)

    # Axis 1 (indices 0, 2, 4, 6) is roll axis ([1, 0, 0]), tilting it produces Fy force (row 1)
    for ax1_idx in [0, 2, 4, 6]:
        assert np.abs(j_p[1, ax1_idx]) > 5.0

    # Axis 2 (indices 1, 3, 5, 7) is pitch axis ([0, 1, 0]), tilting it produces Fx force (row 0)
    for ax2_idx in [1, 3, 5, 7]:
        assert np.abs(j_p[0, ax2_idx]) > 5.0


def test_full_effectiveness_matrix_rank(model: VehicleModel):
    thrusts = np.full(16, 15.0)
    motor_tilts = np.zeros(16)
    pod_tilts = np.zeros(8)

    b_mat = compute_full_effectiveness_matrix(model, thrusts, motor_tilts, pod_tilts)
    assert b_mat.shape == (6, 40)

    # Combined matrix must have full rank 6 even at zero tilt angles
    rank = np.linalg.matrix_rank(b_mat)
    assert rank == 6
