"""
Independent Python geometry representation and forward kinematics for VectorFlight.
Coordinate Convention: Right-handed FRD (+X Forward, +Y Right, +Z Down), radians, SI units.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Computes 3x3 rotation matrix for rotation around a unit axis by angle_rad
    using Rodrigues' formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = axis / norm
    kx, ky, kz = u
    k_mat = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0]
    ], dtype=np.float64)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.eye(3, dtype=np.float64) + s * k_mat + (1.0 - c) * (k_mat @ k_mat)


def ypr_to_rotation_matrix(ypr_rad: np.ndarray) -> np.ndarray:
    """
    Converts intrinsic Yaw, Pitch, Roll (Z-Y-X sequence) to 3x3 rotation matrix:
    R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    """
    yaw, pitch, roll = ypr_rad
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)

    return rz @ ry @ rx


@dataclass
class PodGeometry:
    name: str
    position_body_m: np.ndarray
    base_orientation_ypr_rad: np.ndarray
    axis_1_local: np.ndarray
    axis_2_local: np.ndarray
    axis_1_min_rad: float
    axis_1_max_rad: float
    axis_1_rate_limit_rad_s: float
    axis_2_min_rad: float
    axis_2_max_rad: float
    axis_2_rate_limit_rad_s: float


@dataclass
class RotorGeometry:
    id: int
    pod_id: str
    position_pod_m: np.ndarray
    base_orientation_ypr_rad: np.ndarray
    motor_tilt_axis_local: np.ndarray
    spin_direction: int  # +1 for CCW, -1 for CW
    thrust_min_n: float
    thrust_max_n: float
    thrust_rate_limit_n_s: float
    motor_tilt_min_rad: float
    motor_tilt_max_rad: float
    motor_tilt_rate_limit_rad_s: float
    torque_per_thrust_m: float


@dataclass
class VehicleModel:
    mass_kg: float
    inertia_diag: np.ndarray
    center_of_mass_body_m: np.ndarray
    pods: Dict[str, PodGeometry]
    rotors: List[RotorGeometry]

    @classmethod
    def from_toml(cls, path: str | Path) -> "VehicleModel":
        with open(path, "rb") as f:
            data = tomllib.load(f)

        v_cfg = data["vehicle"]
        mass_kg = float(v_cfg["mass_kg"])
        inertia_diag = np.array(v_cfg["inertia_diag"], dtype=np.float64)
        center_of_mass_body_m = np.array(v_cfg["center_of_mass_body_m"], dtype=np.float64)

        pods = {}
        for pod_name, p_cfg in data["pods"].items():
            pods[pod_name] = PodGeometry(
                name=pod_name,
                position_body_m=np.array(p_cfg["position_body_m"], dtype=np.float64),
                base_orientation_ypr_rad=np.array(p_cfg["base_orientation_ypr_rad"], dtype=np.float64),
                axis_1_local=np.array(p_cfg["axis_1_local"], dtype=np.float64),
                axis_2_local=np.array(p_cfg["axis_2_local"], dtype=np.float64),
                axis_1_min_rad=float(p_cfg["axis_1_min_rad"]),
                axis_1_max_rad=float(p_cfg["axis_1_max_rad"]),
                axis_1_rate_limit_rad_s=float(p_cfg["axis_1_rate_limit_rad_s"]),
                axis_2_min_rad=float(p_cfg["axis_2_min_rad"]),
                axis_2_max_rad=float(p_cfg["axis_2_max_rad"]),
                axis_2_rate_limit_rad_s=float(p_cfg["axis_2_rate_limit_rad_s"]),
            )

        rotors = []
        for r_cfg in data["rotors"]:
            spin = 1 if r_cfg["spin_direction"] == "CCW" else -1
            rotors.append(
                RotorGeometry(
                    id=int(r_cfg["id"]),
                    pod_id=r_cfg["pod_id"],
                    position_pod_m=np.array(r_cfg["position_pod_m"], dtype=np.float64),
                    base_orientation_ypr_rad=np.array(r_cfg["base_orientation_ypr_rad"], dtype=np.float64),
                    motor_tilt_axis_local=np.array(r_cfg["motor_tilt_axis_local"], dtype=np.float64),
                    spin_direction=spin,
                    thrust_min_n=float(r_cfg["thrust_min_n"]),
                    thrust_max_n=float(r_cfg["thrust_max_n"]),
                    thrust_rate_limit_n_s=float(r_cfg["thrust_rate_limit_n_s"]),
                    motor_tilt_min_rad=float(r_cfg["motor_tilt_min_rad"]),
                    motor_tilt_max_rad=float(r_cfg["motor_tilt_max_rad"]),
                    motor_tilt_rate_limit_rad_s=float(r_cfg["motor_tilt_rate_limit_rad_s"]),
                    torque_per_thrust_m=float(r_cfg["torque_per_thrust_m"]),
                )
            )

        # Sort rotors by ID 1..16
        rotors.sort(key=lambda r: r.id)
        return cls(
            mass_kg=mass_kg,
            inertia_diag=inertia_diag,
            center_of_mass_body_m=center_of_mass_body_m,
            pods=pods,
            rotors=rotors,
        )

    def get_pod_tilt_indices(self) -> Dict[str, Tuple[int, int]]:
        """
        Returns mapping of pod name -> (axis_1_index, axis_2_index) into 8-element pod_tilts array.
        Ordering: FL(0,1), FR(2,3), RL(4,5), RR(6,7)
        """
        return {
            "FL": (0, 1),
            "FR": (2, 3),
            "RL": (4, 5),
            "RR": (6, 7),
        }

    def compute_forward_wrench(
        self,
        thrusts: np.ndarray,
        motor_tilts: np.ndarray,
        pod_tilts: np.ndarray,
    ) -> np.ndarray:
        """
        Computes 6-DOF net body wrench [Fx, Fy, Fz, Mx, My, Mz] in FRD body frame.
        - thrusts: array-like of shape (16,) (Newtons)
        - motor_tilts: array-like of shape (16,) (radians)
        - pod_tilts: array-like of shape (8,) (radians) [FL_ax1, FL_ax2, FR_ax1, FR_ax2, RL_ax1, RL_ax2, RR_ax1, RR_ax2]
        """
        thrusts = np.asarray(thrusts, dtype=np.float64)
        motor_tilts = np.asarray(motor_tilts, dtype=np.float64)
        pod_tilts = np.asarray(pod_tilts, dtype=np.float64)

        pod_indices = self.get_pod_tilt_indices()
        total_force = np.zeros(3, dtype=np.float64)
        total_moment = np.zeros(3, dtype=np.float64)

        for rotor in self.rotors:
            idx = rotor.id - 1
            t_k = thrusts[idx]
            gamma_k = motor_tilts[idx]

            pod = self.pods[rotor.pod_id]
            ax1_idx, ax2_idx = pod_indices[rotor.pod_id]
            alpha_1 = pod_tilts[ax1_idx]
            alpha_2 = pod_tilts[ax2_idx]

            # 1. Pod base rotation
            r_base = ypr_to_rotation_matrix(pod.base_orientation_ypr_rad)
            # 2. Pod gimbals (axis 1 then axis 2)
            r_g1 = rodrigues_rotation(pod.axis_1_local, alpha_1)
            r_g2 = rodrigues_rotation(pod.axis_2_local, alpha_2)
            r_pod = r_base @ r_g1 @ r_g2

            # 3. Rotor hub position in body frame
            p_hub_body = pod.position_body_m + r_pod @ rotor.position_pod_m

            # 4. Rotor orientation relative to pod
            r_rotor_base = ypr_to_rotation_matrix(rotor.base_orientation_ypr_rad)
            r_tilt = rodrigues_rotation(rotor.motor_tilt_axis_local, gamma_k)
            r_rotor_body = r_pod @ r_rotor_base @ r_tilt

            # 5. Thrust vector (default rotor thrust points in local -Z, opposite to normal vector)
            # In FRD frame, rotor spinning upwards pushes vehicle up in -Z direction
            thrust_dir_body = r_rotor_body @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
            f_k = t_k * thrust_dir_body

            # 6. Moment arm from vehicle COM
            r_k = p_hub_body - self.center_of_mass_body_m
            m_thrust_k = np.cross(r_k, f_k)

            # 7. Reaction torque: spin_sign * k_t * T_k * thrust_dir
            m_reaction_k = rotor.spin_direction * rotor.torque_per_thrust_m * t_k * thrust_dir_body

            total_force += f_k
            total_moment += (m_thrust_k + m_reaction_k)

        return np.concatenate([total_force, total_moment])
