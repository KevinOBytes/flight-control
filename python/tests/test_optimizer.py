"""
Unit tests for nonlinear orientation optimizer and fault tolerance evaluation.
"""

from pathlib import Path
import numpy as np
import pytest

from vectorflight_tools.geometry import VehicleModel
from vectorflight_tools.optimizer import evaluate_fault_tolerance, optimize_baseline_tilts


@pytest.fixture
def config_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "configs" / "vehicle_v1.toml"


@pytest.fixture
def model(config_path: Path) -> VehicleModel:
    return VehicleModel.from_toml(config_path)


def test_baseline_tilt_optimization(model: VehicleModel):
    opt = optimize_baseline_tilts(model, target_sigma_min=0.15, max_tilt_rad=0.25)

    # Optimization must find a valid trim configuration
    assert opt.optimal_sigma_min > 0.10, f"Expected sigma_min > 0.10, got {opt.optimal_sigma_min}"

    # Residual parasitic forces and moments in hover must be negligible
    assert np.abs(opt.hover_wrench_residual[0]) < 1e-2  # Fx
    assert np.abs(opt.hover_wrench_residual[1]) < 1e-2  # Fy
    assert np.abs(opt.hover_wrench_residual[3]) < 1e-2  # Mx
    assert np.abs(opt.hover_wrench_residual[4]) < 1e-2  # My
    assert np.abs(opt.hover_wrench_residual[5]) < 1e-2  # Mz

    # All single motor failure cases must remain rank-6 controllable
    assert opt.fault_tolerance_summary["all_rank_6"] is True
    assert opt.fault_tolerance_summary["worst_case_sigma_min"] > 0.05


def test_fault_tolerance_evaluator(model: VehicleModel):
    motor_tilts = np.array([0.15 * ((-1) ** i) for i in range(16)])
    pod_tilts = np.array([0.10, -0.10, -0.10, 0.10, 0.10, 0.10, -0.10, -0.10])
    nominal_thrust = 15.32

    summary = evaluate_fault_tolerance(model, motor_tilts, pod_tilts, nominal_thrust)
    assert len(summary["all_single_failures"]) == 16
    assert summary["all_rank_6"] is True
    assert summary["worst_case_sigma_min"] > 0.02
