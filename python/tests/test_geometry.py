"""
Unit tests for Python vehicle geometry and forward wrench model.
"""

from pathlib import Path
import numpy as np
import pytest

from vectorflight_tools.geometry import (
    VehicleModel,
    rodrigues_rotation,
    ypr_to_rotation_matrix,
)


@pytest.fixture
def config_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "configs" / "vehicle_v1.toml"


@pytest.fixture
def model(config_path: Path) -> VehicleModel:
    return VehicleModel.from_toml(config_path)


def test_load_vehicle_model(model: VehicleModel):
    assert model.mass_kg == 25.0
    assert len(model.pods) == 4
    assert len(model.rotors) == 16
    assert set(model.pods.keys()) == {"FL", "FR", "RL", "RR"}


def test_rodrigues_formula_properties():
    axis = np.array([1.0, 2.0, 3.0])
    angle = 0.5235987756  # 30 deg
    r = rodrigues_rotation(axis, angle)

    # Orthogonality: R^T * R = I
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    # Determinant = +1 (proper rotation)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)


def test_nominal_hover_symmetry(model: VehicleModel):
    thrusts = np.full(16, 10.0)
    motor_tilts = np.zeros(16)
    pod_tilts = np.zeros(8)

    wrench = model.compute_forward_wrench(thrusts, motor_tilts, pod_tilts)

    # In nominal flat hover, total force is purely vertical (-160 N in body FRD)
    assert np.isclose(wrench[0], 0.0, atol=1e-10)  # Fx = 0
    assert np.isclose(wrench[1], 0.0, atol=1e-10)  # Fy = 0
    assert np.isclose(wrench[2], -160.0, atol=1e-10)  # Fz = -160 N

    # Net parasitic moments should be exactly zero due to pod and spin symmetry
    assert np.isclose(wrench[3], 0.0, atol=1e-10)  # Mx = 0
    assert np.isclose(wrench[4], 0.0, atol=1e-10)  # My = 0
    assert np.isclose(wrench[5], 0.0, atol=1e-10)  # Mz = 0


def test_single_rotor_thrust(model: VehicleModel):
    thrusts = np.zeros(16)
    thrusts[0] = 10.0  # Rotor 1 only
    motor_tilts = np.zeros(16)
    pod_tilts = np.zeros(8)

    wrench = model.compute_forward_wrench(thrusts, motor_tilts, pod_tilts)
    assert np.isclose(wrench[2], -10.0, atol=1e-10)
    # Rotor 1 has reaction torque in body frame matching its spin direction and torque ratio
    r1 = model.rotors[0]
    expected_reaction_torque = r1.spin_direction * r1.torque_per_thrust_m * 10.0 * (-1.0)
    assert np.isclose(wrench[5], expected_reaction_torque + (r1.position_pod_m[0] + model.pods[r1.pod_id].position_body_m[0]) * wrench[1] - (r1.position_pod_m[1] + model.pods[r1.pod_id].position_body_m[1]) * wrench[0], atol=1e-6)
