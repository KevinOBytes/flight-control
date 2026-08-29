//! 6-DOF rigid body plant simulator skeleton with actuator dynamics, lags, and noise for VectorFlight.

use nalgebra::{UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use vf_core::VfError;
use vf_model::ActuatorState;

/// Full dynamic state of the simulated aircraft plant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantState {
    /// 3D position in global NED frame (meters)
    pub position: Vector3<f64>,
    /// Velocity in body frame (meters/second)
    pub velocity: Vector3<f64>,
    /// Attitude quaternion of the body relative to global frame
    pub attitude: UnitQuaternion<f64>,
    /// Angular velocity in body frame (radians/second)
    pub angular_velocity: Vector3<f64>,
    /// Current internal actual state of all actuators (with lags, rate limits applied)
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
}

/// Simulation environment containing the plant model.
pub struct MultirotorSimulator {
    config: SimConfig,
    state: PlantState,
    mass_kg: f64,
    inertia: Vector3<f64>,
}

impl MultirotorSimulator {
    pub fn new(
        config: SimConfig,
        initial_state: PlantState,
        mass_kg: f64,
        inertia: Vector3<f64>,
    ) -> Self {
        Self {
            config,
            state: initial_state,
            mass_kg,
            inertia,
        }
    }

    pub fn mass_kg(&self) -> f64 {
        self.mass_kg
    }

    pub fn inertia(&self) -> Vector3<f64> {
        self.inertia
    }

    /// Advances the plant model state by dt seconds.
    pub fn step(
        &mut self,
        commands: &ActuatorState,
        dt_s: f64,
        wind_force_body: &Vector3<f64>,
    ) -> Result<&PlantState, VfError> {
        if dt_s <= 0.0 {
            return Err(VfError::InvalidValue(format!(
                "Simulation step dt must be positive, got {}",
                dt_s
            )));
        }

        // 1. Actuator Dynamics Integration (lags and rate-limits)
        // Simple first-order Euler step for actuator state tracking command:
        // dx/dt = (command - x) / tau
        for i in 0..16 {
            let d_thrust = (commands.motor_thrusts[i] - self.state.actuators.motor_thrusts[i])
                / self.config.motor_thrust_tau;
            self.state.actuators.motor_thrusts[i] += d_thrust * dt_s;

            let d_tilt = (commands.motor_tilts[i] - self.state.actuators.motor_tilts[i])
                / self.config.motor_tilt_tau;
            self.state.actuators.motor_tilts[i] += d_tilt * dt_s;
        }

        for i in 0..8 {
            let d_pod = (commands.pod_tilts[i] - self.state.actuators.pod_tilts[i])
                / self.config.pod_tilt_tau;
            self.state.actuators.pod_tilts[i] += d_pod * dt_s;
        }

        // 2. Rigid Body Dynamics Integration (Placeholder)
        // Dynamics will be integrated in later phases.
        // We update position and velocity as a basic skeleton:
        let accel_body = wind_force_body / self.mass_kg;
        self.state.velocity += accel_body * dt_s;
        let vel_world = self.state.attitude * self.state.velocity;
        self.state.position += vel_world * dt_s;

        Ok(&self.state)
    }

    /// Retrieve the current simulated plant state.
    pub fn get_state(&self) -> &PlantState {
        &self.state
    }
}
