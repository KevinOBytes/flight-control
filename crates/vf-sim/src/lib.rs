//! 6-DOF rigid body plant simulator with actuator dynamics, lags, and disturbance models for VectorFlight.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use vf_core::{BodyWrench, VfError};
use vf_model::{ActuatorState, VehicleModel};

pub mod scenario;
pub use scenario::FlightScenarioRunner;

/// Full dynamic state of the simulated aircraft plant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantState {
    /// 3D position in global NED frame (meters, +Z down)
    pub position: Vector3<f64>,
    /// Velocity in body FRD frame (meters/second, +X forward, +Y right, +Z down)
    pub velocity: Vector3<f64>,
    /// Attitude quaternion representing rotation from body frame to world NED frame
    pub attitude: UnitQuaternion<f64>,
    /// Angular velocity in body FRD frame (radians/second)
    pub angular_velocity: Vector3<f64>,
    /// Current internal actual state of all physical actuators (with lags and bounds applied)
    pub actuators: ActuatorState,
}

impl Default for PlantState {
    fn default() -> Self {
        Self::new()
    }
}

impl PlantState {
    pub fn new() -> Self {
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            attitude: UnitQuaternion::identity(),
            angular_velocity: Vector3::zeros(),
            actuators: ActuatorState::zero(),
        }
    }
}

/// Simulator configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub gravity_m_s2: f64,
    /// First-order time constant for motor thrust lag (seconds)
    pub motor_thrust_tau: f64,
    /// First-order time constant for motor tilt angle lag (seconds)
    pub motor_tilt_tau: f64,
    /// First-order time constant for pod tilt angle lag (seconds)
    pub pod_tilt_tau: f64,
    /// Standard deviation of white noise added to forces (Newtons)
    pub force_noise_std: f64,
    /// Standard deviation of white noise added to moments (Newton-meters)
    pub moment_noise_std: f64,
    /// Number of sub-steps per simulator step for numerical integration precision
    pub substeps: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gravity_m_s2: 9.80665,
            motor_thrust_tau: 0.020,
            motor_tilt_tau: 0.030,
            pod_tilt_tau: 0.050,
            force_noise_std: 0.0,
            moment_noise_std: 0.0,
            substeps: 10,
        }
    }
}

/// Simulation environment containing the 6-DOF plant model.
pub struct MultirotorSimulator {
    config: SimConfig,
    vehicle_model: VehicleModel,
    state: PlantState,
    total_time_s: f64,
}

impl MultirotorSimulator {
    pub fn new(config: SimConfig, vehicle_model: VehicleModel, initial_state: PlantState) -> Self {
        Self {
            config,
            vehicle_model,
            state: initial_state,
            total_time_s: 0.0,
        }
    }

    pub fn vehicle_model(&self) -> &VehicleModel {
        &self.vehicle_model
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    pub fn total_time_s(&self) -> f64 {
        self.total_time_s
    }

    /// Retrieve the current simulated plant state.
    pub fn get_state(&self) -> &PlantState {
        &self.state
    }

    /// Mutable access to state (e.g. for setting initial condition).
    pub fn state_mut(&mut self) -> &mut PlantState {
        &mut self.state
    }

    /// Advances the plant model state by dt_s seconds using Newton-Euler equations of motion.
    pub fn step(
        &mut self,
        commands: &ActuatorState,
        dt_s: f64,
        wind_force_body: &Vector3<f64>,
        wind_moment_body: &Vector3<f64>,
    ) -> Result<&PlantState, VfError> {
        if dt_s <= 0.0 {
            return Err(VfError::InvalidValue(format!(
                "Simulation step dt must be positive, got {}",
                dt_s
            )));
        }

        let n_sub = self.config.substeps.max(1);
        let sub_dt = dt_s / (n_sub as f64);

        let mass = self.vehicle_model.mass_kg;
        let inertia = self.vehicle_model.inertia;
        let inertia_mat = Matrix3::from_diagonal(&inertia);
        let inv_inertia_mat = Matrix3::from_diagonal(&Vector3::new(
            1.0 / inertia.x,
            1.0 / inertia.y,
            1.0 / inertia.z,
        ));

        for _ in 0..n_sub {
            // 1. Actuator Dynamics Integration (first-order lag with rate and range limits)
            // Motor thrusts:
            for (i, rotor) in self.vehicle_model.rotors.iter().enumerate() {
                let target = commands.motor_thrusts[i];
                let current = self.state.actuators.motor_thrusts[i];
                let raw_rate = (target - current) / self.config.motor_thrust_tau;
                let rate =
                    raw_rate.clamp(-rotor.thrust_rate_limit_n_s, rotor.thrust_rate_limit_n_s);
                let new_val =
                    (current + rate * sub_dt).clamp(rotor.thrust_min_n, rotor.thrust_max_n);
                self.state.actuators.motor_thrusts[i] = new_val;
            }

            // Motor tilts:
            for (i, rotor) in self.vehicle_model.rotors.iter().enumerate() {
                let target = commands.motor_tilts[i];
                let current = self.state.actuators.motor_tilts[i];
                let raw_rate = (target - current) / self.config.motor_tilt_tau;
                let rate = raw_rate.clamp(
                    -rotor.motor_tilt_rate_limit_rad_s,
                    rotor.motor_tilt_rate_limit_rad_s,
                );
                let new_val = (current + rate * sub_dt)
                    .clamp(rotor.motor_tilt_min_rad, rotor.motor_tilt_max_rad);
                self.state.actuators.motor_tilts[i] = new_val;
            }

            // Pod tilts:
            let pod_axes = [
                (vf_core::PodId::FL, vf_core::PodAxis::Axis1, 0),
                (vf_core::PodId::FL, vf_core::PodAxis::Axis2, 1),
                (vf_core::PodId::FR, vf_core::PodAxis::Axis1, 2),
                (vf_core::PodId::FR, vf_core::PodAxis::Axis2, 3),
                (vf_core::PodId::RL, vf_core::PodAxis::Axis1, 4),
                (vf_core::PodId::RL, vf_core::PodAxis::Axis2, 5),
                (vf_core::PodId::RR, vf_core::PodAxis::Axis1, 6),
                (vf_core::PodId::RR, vf_core::PodAxis::Axis2, 7),
            ];

            for (pod_id, axis, idx) in pod_axes {
                let pod = self.vehicle_model.pods.get(&pod_id).ok_or_else(|| {
                    VfError::InvalidValue(format!("Pod {:?} not found in vehicle model", pod_id))
                })?;
                let (min_rad, max_rad, rate_limit) = match axis {
                    vf_core::PodAxis::Axis1 => (
                        pod.axis_1_min_rad,
                        pod.axis_1_max_rad,
                        pod.axis_1_rate_limit_rad_s,
                    ),
                    vf_core::PodAxis::Axis2 => (
                        pod.axis_2_min_rad,
                        pod.axis_2_max_rad,
                        pod.axis_2_rate_limit_rad_s,
                    ),
                };

                let target = commands.pod_tilts[idx];
                let current = self.state.actuators.pod_tilts[idx];
                let raw_rate = (target - current) / self.config.pod_tilt_tau;
                let rate = raw_rate.clamp(-rate_limit, rate_limit);
                let new_val = (current + rate * sub_dt).clamp(min_rad, max_rad);
                self.state.actuators.pod_tilts[idx] = new_val;
            }

            // 2. Compute Body-Frame Propulsion Wrench
            let prop_wrench: BodyWrench =
                vf_model::wrench_from_actuators(&self.vehicle_model, &self.state.actuators)?;

            // 3. Compute Body-Frame Gravity Force
            // In NED frame, gravity is [0, 0, m * g] (+Z down)
            let gravity_world = Vector3::new(0.0, 0.0, mass * self.config.gravity_m_s2);
            let gravity_body = self.state.attitude.inverse_transform_vector(&gravity_world);

            // 4. Total Forces and Moments in Body Frame
            let total_force_body = prop_wrench.force + gravity_body + wind_force_body;
            let total_moment_body = prop_wrench.moment + wind_moment_body;

            // 5. Translational Dynamics (Newton-Euler in rotating body frame):
            // \dot{v}_b = F_total / m - \omega \times v_b
            let v_body = self.state.velocity;
            let omega_body = self.state.angular_velocity;
            let accel_body = total_force_body / mass - omega_body.cross(&v_body);
            let next_v_body = v_body + accel_body * sub_dt;

            // World velocity & position integration:
            // \dot{p}_w = R_b^w * v_b
            let vel_world = self.state.attitude * v_body;
            let next_pos_world = self.state.position + vel_world * sub_dt;

            // 6. Rotational Dynamics (Euler equations):
            // \dot{\omega}_b = I^{-1} * (M_total - \omega \times (I * \omega))
            let h_body = inertia_mat * omega_body;
            let gyro_torque = omega_body.cross(&h_body);
            let alpha_body = inv_inertia_mat * (total_moment_body - gyro_torque);
            let next_omega_body = omega_body + alpha_body * sub_dt;

            // Attitude quaternion integration:
            let delta_q = UnitQuaternion::from_scaled_axis(omega_body * sub_dt);
            let next_attitude =
                UnitQuaternion::new_normalize((self.state.attitude * delta_q).into_inner());

            // 7. Update plant state
            self.state.position = next_pos_world;
            self.state.velocity = next_v_body;
            self.state.angular_velocity = next_omega_body;
            self.state.attitude = next_attitude;
        }

        self.total_time_s += dt_s;
        Ok(&self.state)
    }
}
