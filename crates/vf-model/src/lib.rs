//! Kinematics, geometry, and forward wrench models for VectorFlight.

use nalgebra::{SMatrix, Unit, UnitQuaternion, Vector3, Vector6};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use vf_core::{BodyWrench, PodAxis, PodId, RotorId, SpinDirection, VfError};

/// Geometry and limits for a single rotor, resolved or configured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotorGeometry {
    pub id: RotorId,
    pub pod_id: PodId,
    /// Rotor position relative to the pod center (local pod frame)
    pub position_pod_m: Vector3<f64>,
    /// Base orientation of the motor frame relative to the pod frame
    pub base_orientation_pod: UnitQuaternion<f64>,
    /// Motor tilt axis in local motor frame
    pub motor_tilt_axis_local: Unit<Vector3<f64>>,
    pub spin_direction: SpinDirection,
    pub thrust_min_n: f64,
    pub thrust_max_n: f64,
    pub thrust_rate_limit_n_s: f64,
    pub motor_tilt_min_rad: f64,
    pub motor_tilt_max_rad: f64,
    pub motor_tilt_rate_limit_rad_s: f64,
    pub torque_per_thrust_m: f64,
}

/// Geometry and limits for a single pod.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodGeometry {
    pub id: PodId,
    /// Pod center position relative to vehicle COM (body frame)
    pub position_body_m: Vector3<f64>,
    /// Pod base orientation relative to body frame
    pub base_orientation_body: UnitQuaternion<f64>,
    /// First gimbal tilt axis (local pod frame)
    pub axis_1_local: Unit<Vector3<f64>>,
    /// Second gimbal tilt axis (local pod frame)
    pub axis_2_local: Unit<Vector3<f64>>,
    pub axis_1_min_rad: f64,
    pub axis_1_max_rad: f64,
    pub axis_1_rate_limit_rad_s: f64,
    pub axis_2_min_rad: f64,
    pub axis_2_max_rad: f64,
    pub axis_2_rate_limit_rad_s: f64,
}

/// Raw representation of the configuration file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleConfig {
    pub vehicle: VehicleParams,
    pub loop_rates: LoopRatesParams,
    pub allocator_weights: AllocatorWeightsParams,
    pub fault_thresholds: FaultThresholdsParams,
    pub pods: HashMap<String, PodConfig>,
    pub rotors: Vec<RotorConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleParams {
    pub mass_kg: f64,
    pub inertia_diag: [f64; 3],
    pub center_of_mass_body_m: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopRatesParams {
    pub thrust_allocation_hz: f64,
    pub motor_tilt_planner_hz: f64,
    pub pod_tilt_planner_hz: f64,
    pub telemetry_hz: f64,
    pub health_monitoring_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocatorWeightsParams {
    pub wrench_error: [f64; 6],
    pub smoothness: f64,
    pub power: f64,
    pub center: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultThresholdsParams {
    pub stale_feedback_timeout_s: f64,
    pub min_singular_value_degraded: f64,
    pub min_singular_value_critical: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodConfig {
    pub position_body_m: [f64; 3],
    pub base_orientation_ypr_rad: [f64; 3],
    pub axis_1_local: [f64; 3],
    pub axis_2_local: [f64; 3],
    pub axis_1_min_rad: f64,
    pub axis_1_max_rad: f64,
    pub axis_1_rate_limit_rad_s: f64,
    pub axis_2_min_rad: f64,
    pub axis_2_max_rad: f64,
    pub axis_2_rate_limit_rad_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotorConfig {
    pub id: u32,
    pub pod_id: String,
    pub position_pod_m: [f64; 3],
    pub base_orientation_ypr_rad: [f64; 3],
    pub motor_tilt_axis_local: [f64; 3],
    pub spin_direction: String,
    pub thrust_min_n: f64,
    pub thrust_max_n: f64,
    pub thrust_rate_limit_n_s: f64,
    pub motor_tilt_min_rad: f64,
    pub motor_tilt_max_rad: f64,
    pub motor_tilt_rate_limit_rad_s: f64,
    pub torque_per_thrust_m: f64,
    pub can_node_id: u32,
}

/// The complete resolved vehicle model.
#[derive(Debug, Clone)]
pub struct VehicleModel {
    pub mass_kg: f64,
    pub inertia: Vector3<f64>,
    pub center_of_mass: Vector3<f64>,
    pub pods: HashMap<PodId, PodGeometry>,
    pub rotors: Vec<RotorGeometry>,
    pub min_singular_value_degraded: f64,
    pub min_singular_value_critical: f64,
    pub stale_feedback_timeout_s: f64,
}

impl VehicleModel {
    /// Loads a vehicle model from a TOML configuration file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, VfError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| VfError::ConfigError(format!("Failed to read config file: {}", e)))?;
        let config: VehicleConfig = toml::from_str(&content)
            .map_err(|e| VfError::ConfigError(format!("Failed to parse TOML: {}", e)))?;
        Self::from_config(config)
    }

    /// Resolves raw configuration into the internal representation.
    pub fn from_config(config: VehicleConfig) -> Result<Self, VfError> {
        let mut pods = HashMap::new();

        for (name, pod_cfg) in config.pods {
            let pod_id = match name.as_str() {
                "FL" => PodId::FL,
                "FR" => PodId::FR,
                "RL" => PodId::RL,
                "RR" => PodId::RR,
                _ => return Err(VfError::ConfigError(format!("Invalid pod name: {}", name))),
            };

            let pos = Vector3::from(pod_cfg.position_body_m);
            // Convert Yaw, Pitch, Roll to UnitQuaternion
            let rot = UnitQuaternion::from_euler_angles(
                pod_cfg.base_orientation_ypr_rad[2], // Roll
                pod_cfg.base_orientation_ypr_rad[1], // Pitch
                pod_cfg.base_orientation_ypr_rad[0], // Yaw
            );

            let axis_1 =
                Unit::try_new(Vector3::from(pod_cfg.axis_1_local), 1e-6).ok_or_else(|| {
                    VfError::ConfigError(format!("Pod {} axis_1 is zero-length", name))
                })?;
            let axis_2 =
                Unit::try_new(Vector3::from(pod_cfg.axis_2_local), 1e-6).ok_or_else(|| {
                    VfError::ConfigError(format!("Pod {} axis_2 is zero-length", name))
                })?;

            pods.insert(
                pod_id,
                PodGeometry {
                    id: pod_id,
                    position_body_m: pos,
                    base_orientation_body: rot,
                    axis_1_local: axis_1,
                    axis_2_local: axis_2,
                    axis_1_min_rad: pod_cfg.axis_1_min_rad,
                    axis_1_max_rad: pod_cfg.axis_1_max_rad,
                    axis_1_rate_limit_rad_s: pod_cfg.axis_1_rate_limit_rad_s,
                    axis_2_min_rad: pod_cfg.axis_2_min_rad,
                    axis_2_max_rad: pod_cfg.axis_2_max_rad,
                    axis_2_rate_limit_rad_s: pod_cfg.axis_2_rate_limit_rad_s,
                },
            );
        }

        let mut rotors = Vec::new();
        for r_cfg in config.rotors {
            let pod_id = match r_cfg.pod_id.as_str() {
                "FL" => PodId::FL,
                "FR" => PodId::FR,
                "RL" => PodId::RL,
                "RR" => PodId::RR,
                _ => {
                    return Err(VfError::ConfigError(format!(
                        "Invalid pod_id {} for rotor {}",
                        r_cfg.pod_id, r_cfg.id
                    )))
                }
            };

            let pos = Vector3::from(r_cfg.position_pod_m);
            let rot = UnitQuaternion::from_euler_angles(
                r_cfg.base_orientation_ypr_rad[2],
                r_cfg.base_orientation_ypr_rad[1],
                r_cfg.base_orientation_ypr_rad[0],
            );

            let tilt_axis = Unit::try_new(Vector3::from(r_cfg.motor_tilt_axis_local), 1e-6)
                .ok_or_else(|| {
                    VfError::ConfigError(format!("Rotor {} tilt axis is zero-length", r_cfg.id))
                })?;

            let spin = match r_cfg.spin_direction.as_str() {
                "CW" => SpinDirection::CW,
                "CCW" => SpinDirection::CCW,
                _ => {
                    return Err(VfError::ConfigError(format!(
                        "Invalid spin direction for rotor {}",
                        r_cfg.id
                    )))
                }
            };

            rotors.push(RotorGeometry {
                id: RotorId(r_cfg.id),
                pod_id,
                position_pod_m: pos,
                base_orientation_pod: rot,
                motor_tilt_axis_local: tilt_axis,
                spin_direction: spin,
                thrust_min_n: r_cfg.thrust_min_n,
                thrust_max_n: r_cfg.thrust_max_n,
                thrust_rate_limit_n_s: r_cfg.thrust_rate_limit_n_s,
                motor_tilt_min_rad: r_cfg.motor_tilt_min_rad,
                motor_tilt_max_rad: r_cfg.motor_tilt_max_rad,
                motor_tilt_rate_limit_rad_s: r_cfg.motor_tilt_rate_limit_rad_s,
                torque_per_thrust_m: r_cfg.torque_per_thrust_m,
            });
        }

        // Validate count
        if rotors.len() != 16 {
            return Err(VfError::ConfigError(format!(
                "Expected exactly 16 rotors, found {}",
                rotors.len()
            )));
        }

        Ok(Self {
            mass_kg: config.vehicle.mass_kg,
            inertia: Vector3::from(config.vehicle.inertia_diag),
            center_of_mass: Vector3::from(config.vehicle.center_of_mass_body_m),
            pods,
            rotors,
            min_singular_value_degraded: config.fault_thresholds.min_singular_value_degraded,
            min_singular_value_critical: config.fault_thresholds.min_singular_value_critical,
            stale_feedback_timeout_s: config.fault_thresholds.stale_feedback_timeout_s,
        })
    }
}

/// Representation of the actuator states at any given tick.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActuatorState {
    /// 16 rotor thrust values in Newtons. Indexed by `RotorId - 1`.
    pub motor_thrusts: [f64; 16],
    /// 16 motor tilt values in radians. Indexed by `RotorId - 1`.
    pub motor_tilts: [f64; 16],
    /// 8 pod tilt values in radians. Ordered:
    /// `[FL_axis1, FL_axis2, FR_axis1, FR_axis2, RL_axis1, RL_axis2, RR_axis1, RR_axis2]`
    pub pod_tilts: [f64; 8],
}

impl ActuatorState {
    pub fn zero() -> Self {
        Self {
            motor_thrusts: [0.0; 16],
            motor_tilts: [0.0; 16],
            pod_tilts: [0.0; 8],
        }
    }

    pub fn get_motor_thrust(&self, id: RotorId) -> Result<f64, VfError> {
        let idx = (id.0 as usize)
            .checked_sub(1)
            .ok_or_else(|| VfError::InvalidValue(format!("Invalid RotorId: {:?}", id)))?;
        self.motor_thrusts
            .get(idx)
            .copied()
            .ok_or_else(|| VfError::InvalidValue(format!("RotorId out of bounds: {:?}", id)))
    }

    pub fn get_motor_tilt(&self, id: RotorId) -> Result<f64, VfError> {
        let idx = (id.0 as usize)
            .checked_sub(1)
            .ok_or_else(|| VfError::InvalidValue(format!("Invalid RotorId: {:?}", id)))?;
        self.motor_tilts
            .get(idx)
            .copied()
            .ok_or_else(|| VfError::InvalidValue(format!("RotorId out of bounds: {:?}", id)))
    }

    pub fn get_pod_tilts(&self, pod: PodId) -> (f64, f64) {
        let base_idx = match pod {
            PodId::FL => 0,
            PodId::FR => 2,
            PodId::RL => 4,
            PodId::RR => 6,
        };
        (self.pod_tilts[base_idx], self.pod_tilts[base_idx + 1])
    }
}

/// Result of computing kinematics for a single rotor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotorKinematics {
    pub position_body_m: Vector3<f64>,
    pub thrust_direction_body: Vector3<f64>,
    pub reaction_torque_body: Vector3<f64>,
}

/// Computes the forward kinematics and wrench contribution for a single rotor.
pub fn rotor_kinematics(
    model: &VehicleModel,
    state: &ActuatorState,
    rotor_id: RotorId,
) -> Result<RotorKinematics, VfError> {
    // 1. Find rotor geometry
    let rotor = model
        .rotors
        .iter()
        .find(|r| r.id == rotor_id)
        .ok_or_else(|| VfError::InvalidValue(format!("Rotor {:?} not found in model", rotor_id)))?;

    // 2. Find pod geometry
    let pod = model.pods.get(&rotor.pod_id).ok_or_else(|| {
        VfError::InvalidValue(format!("Pod {:?} not found in model", rotor.pod_id))
    })?;

    // 3. Compute Pod rotation
    // R_body_to_pod = R_body_to_pod_base * R_pod_axis_1(alpha) * R_pod_axis_2(beta)
    let (alpha, beta) = state.get_pod_tilts(rotor.pod_id);
    let r_axis1 = UnitQuaternion::from_axis_angle(&pod.axis_1_local, alpha);
    let r_axis2 = UnitQuaternion::from_axis_angle(&pod.axis_2_local, beta);
    let r_body_to_pod = pod.base_orientation_body * r_axis1 * r_axis2;

    // 4. Compute Motor rotation relative to Pod
    // R_pod_to_rotor = R_motor_base * R_motor_tilt(gamma)
    let gamma = state.get_motor_tilt(rotor_id)?;
    let r_motor_tilt = UnitQuaternion::from_axis_angle(&rotor.motor_tilt_axis_local, gamma);
    let r_pod_to_rotor = rotor.base_orientation_pod * r_motor_tilt;

    // 5. Total rotation composition
    // R_body_to_rotor = R_body_to_pod * R_pod_to_rotor
    let r_body_to_rotor = r_body_to_pod * r_pod_to_rotor;

    // 6. Position in body frame
    // position_body_m = pod.position_body_m + R_body_to_pod * rotor.position_pod_m
    let pos_body = pod.position_body_m + r_body_to_pod * rotor.position_pod_m;

    // 7. Thrust direction in body frame
    // Positive thrust magnitude acts along -Z (lift is upward, Z is down)
    // Thrust vector points in direction of -Z axis in local rotor frame: n_i = R_body_to_rotor * [0, 0, -1]
    let thrust_dir_local = Vector3::new(0.0, 0.0, -1.0);
    let thrust_dir_body = r_body_to_rotor * thrust_dir_local;

    // 8. Reaction torque direction in body frame
    // Reaction torque is along the motor rotation axis (local Z).
    // Direction depends on spin direction.
    // Standard convention: CW rotation creates reaction torque in CCW direction (positive local +Z or similar).
    // Let's specify that reaction torque is:
    // M_reaction = spin_sign * torque_per_thrust * thrust_magnitude * n_i
    // Spin sign: CW = +1.0, CCW = -1.0
    // Wait, let's verify if reaction torque opposes rotation. If rotor spin is CW (looking along thrust vector, which is -Z),
    // then reaction torque points in CCW direction.
    // Let's define the spin sign coefficient:
    let spin_sign = match rotor.spin_direction {
        SpinDirection::CW => 1.0,
        SpinDirection::CCW => -1.0,
    };
    // Reaction torque vector in body frame:
    let reaction_torque_body = spin_sign * rotor.torque_per_thrust_m * thrust_dir_body;

    Ok(RotorKinematics {
        position_body_m: pos_body,
        thrust_direction_body: thrust_dir_body,
        reaction_torque_body,
    })
}

/// Computes the total vehicle wrench produced by the actuator state.
pub fn wrench_from_actuators(
    model: &VehicleModel,
    state: &ActuatorState,
) -> Result<BodyWrench, VfError> {
    let mut total_force = Vector3::zeros();
    let mut total_moment = Vector3::zeros();

    for rotor in &model.rotors {
        let thrust = state.get_motor_thrust(rotor.id)?;
        if thrust < 0.0 {
            return Err(VfError::ConstraintViolation(format!(
                "Rotor {:?} commanded negative thrust: {}",
                rotor.id, thrust
            )));
        }

        let kin = rotor_kinematics(model, state, rotor.id)?;

        let force_i = thrust * kin.thrust_direction_body;
        // Moment = (r_i - COM) × F_i + M_reaction_i
        let arm = kin.position_body_m - model.center_of_mass;
        let moment_i = arm.cross(&force_i) + thrust * kin.reaction_torque_body;

        total_force += force_i;
        total_moment += moment_i;
    }

    Ok(BodyWrench::new(total_force, total_moment))
}

/// Computes the thrust effectiveness matrix B (6x16 Jacobian) under current actuator angles.
pub fn compute_thrust_effectiveness(
    model: &VehicleModel,
    state: &ActuatorState,
) -> Result<SMatrix<f64, 6, 16>, VfError> {
    let mut b = SMatrix::<f64, 6, 16>::zeros();

    for (i, rotor) in model.rotors.iter().enumerate() {
        let kin = rotor_kinematics(model, state, rotor.id)?;

        // Force column component (direction vector)
        b[(0, i)] = kin.thrust_direction_body.x;
        b[(1, i)] = kin.thrust_direction_body.y;
        b[(2, i)] = kin.thrust_direction_body.z;

        // Moment column component: arm x n_i + reaction_torque
        let arm = kin.position_body_m - model.center_of_mass;
        let moment_unit = arm.cross(&kin.thrust_direction_body) + kin.reaction_torque_body;

        b[(3, i)] = moment_unit.x;
        b[(4, i)] = moment_unit.y;
        b[(5, i)] = moment_unit.z;
    }

    Ok(b)
}

/// Computes the derivative of the thrust direction and reaction torque with respect to motor tilt angle gamma.
pub fn compute_motor_tilt_derivative(
    model: &VehicleModel,
    state: &ActuatorState,
    rotor_id: RotorId,
) -> Result<(Vector3<f64>, Vector3<f64>), VfError> {
    let rotor = model
        .rotors
        .iter()
        .find(|r| r.id == rotor_id)
        .ok_or_else(|| VfError::InvalidValue(format!("Rotor {:?} not found", rotor_id)))?;
    let pod = model
        .pods
        .get(&rotor.pod_id)
        .ok_or_else(|| VfError::InvalidValue(format!("Pod {:?} not found", rotor.pod_id)))?;

    // 1. Compute R_body_to_pod
    let (alpha, beta) = state.get_pod_tilts(rotor.pod_id);
    let r_axis1 = UnitQuaternion::from_axis_angle(&pod.axis_1_local, alpha);
    let r_axis2 = UnitQuaternion::from_axis_angle(&pod.axis_2_local, beta);
    let r_body_to_pod = pod.base_orientation_body * r_axis1 * r_axis2;

    // 2. R_fixed = R_body_to_pod * base_orientation_pod
    let r_fixed = r_body_to_pod * rotor.base_orientation_pod;

    // 3. Rotate tilt axis to body frame
    let v_body = r_fixed * rotor.motor_tilt_axis_local.into_inner();

    // 4. Get current kinematics (for current n_i)
    let kin = rotor_kinematics(model, state, rotor_id)?;

    // dn_dgamma = v_body x n
    let dn_dgamma = v_body.cross(&kin.thrust_direction_body);

    // reaction torque derivative is spin_sign * k_t * dn_dgamma
    let spin_sign = match rotor.spin_direction {
        SpinDirection::CW => 1.0,
        SpinDirection::CCW => -1.0,
    };
    let dm_reaction_dgamma = spin_sign * rotor.torque_per_thrust_m * dn_dgamma;

    Ok((dn_dgamma, dm_reaction_dgamma))
}

/// Computes the motor tilt effectiveness matrix J_gamma (6x16 Jacobian) under the current actuator state.
pub fn compute_motor_tilt_effectiveness(
    model: &VehicleModel,
    state: &ActuatorState,
) -> Result<SMatrix<f64, 6, 16>, VfError> {
    let mut j_gamma = SMatrix::<f64, 6, 16>::zeros();

    for (i, rotor) in model.rotors.iter().enumerate() {
        let thrust = state.get_motor_thrust(rotor.id)?;
        let kin = rotor_kinematics(model, state, rotor.id)?;
        let (dn_dgamma, dm_reaction_dgamma) =
            compute_motor_tilt_derivative(model, state, rotor.id)?;

        // dF_dgamma = T * dn_dgamma
        let df_dgamma = thrust * dn_dgamma;

        // dM_dgamma = arm x dF_dgamma + T * dm_reaction_dgamma
        let arm = kin.position_body_m - model.center_of_mass;
        let dm_dgamma = arm.cross(&df_dgamma) + thrust * dm_reaction_dgamma;

        j_gamma[(0, i)] = df_dgamma.x;
        j_gamma[(1, i)] = df_dgamma.y;
        j_gamma[(2, i)] = df_dgamma.z;
        j_gamma[(3, i)] = dm_dgamma.x;
        j_gamma[(4, i)] = dm_dgamma.y;
        j_gamma[(5, i)] = dm_dgamma.z;
    }

    Ok(j_gamma)
}

/// Computes the total wrench derivative with respect to a single pod tilt axis.
pub fn compute_pod_tilt_derivative(
    model: &VehicleModel,
    state: &ActuatorState,
    pod_id: PodId,
    axis: PodAxis,
) -> Result<Vector6<f64>, VfError> {
    let pod = model
        .pods
        .get(&pod_id)
        .ok_or_else(|| VfError::InvalidValue(format!("Pod {:?} not found", pod_id)))?;

    let (alpha, _beta) = state.get_pod_tilts(pod_id);

    // Compute rotation axis in body frame
    let v_body = match axis {
        PodAxis::Axis1 => pod.base_orientation_body * pod.axis_1_local.into_inner(),
        PodAxis::Axis2 => {
            let r_axis1 = UnitQuaternion::from_axis_angle(&pod.axis_1_local, alpha);
            pod.base_orientation_body * (r_axis1 * pod.axis_2_local.into_inner())
        }
    };

    let mut df_total = Vector3::zeros();
    let mut dm_total = Vector3::zeros();

    for rotor in model.rotors.iter().filter(|r| r.pod_id == pod_id) {
        let thrust = state.get_motor_thrust(rotor.id)?;
        let kin = rotor_kinematics(model, state, rotor.id)?;

        // Hub displacement relative to pod center in body frame
        let hub_rel_pod = kin.position_body_m - pod.position_body_m;
        let dr_dtheta = v_body.cross(&hub_rel_pod);

        // Thrust direction derivative
        let dn_dtheta = v_body.cross(&kin.thrust_direction_body);

        // Force derivative
        let df_dtheta = thrust * dn_dtheta;

        // Reaction torque derivative
        let spin_sign = match rotor.spin_direction {
            SpinDirection::CW => 1.0,
            SpinDirection::CCW => -1.0,
        };
        let dm_reaction_dtheta = spin_sign * rotor.torque_per_thrust_m * thrust * dn_dtheta;

        // Moment derivative
        let f_rotor = thrust * kin.thrust_direction_body;
        let arm_com = kin.position_body_m - model.center_of_mass;
        let dm_dtheta = dr_dtheta.cross(&f_rotor) + arm_com.cross(&df_dtheta) + dm_reaction_dtheta;

        df_total += df_dtheta;
        dm_total += dm_dtheta;
    }

    Ok(Vector6::new(
        df_total.x, df_total.y, df_total.z, dm_total.x, dm_total.y, dm_total.z,
    ))
}

/// Computes the pod tilt effectiveness matrix J_pod (6x8 Jacobian) under the current actuator state.
pub fn compute_pod_tilt_effectiveness(
    model: &VehicleModel,
    state: &ActuatorState,
) -> Result<SMatrix<f64, 6, 8>, VfError> {
    let mut j_pod = SMatrix::<f64, 6, 8>::zeros();

    let pod_configs = [
        (PodId::FL, PodAxis::Axis1, 0),
        (PodId::FL, PodAxis::Axis2, 1),
        (PodId::FR, PodAxis::Axis1, 2),
        (PodId::FR, PodAxis::Axis2, 3),
        (PodId::RL, PodAxis::Axis1, 4),
        (PodId::RL, PodAxis::Axis2, 5),
        (PodId::RR, PodAxis::Axis1, 6),
        (PodId::RR, PodAxis::Axis2, 7),
    ];

    for (pod_id, axis, col) in pod_configs {
        let deriv = compute_pod_tilt_derivative(model, state, pod_id, axis)?;
        j_pod.set_column(col, &deriv);
    }

    Ok(j_pod)
}
