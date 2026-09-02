//! Scriptable scenario execution engine for closed-loop flight profile simulation.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use vf_core::{BodyWrench, VfError};
use vf_faults::{ControlAuthorityMode, FaultSet};
use vf_model::ActuatorState;
use vf_runtime::{AllocationRequest, AllocatorRuntime, ElectricalLimits};

use crate::{MultirotorSimulator, PlantState};

/// Time-series telemetry record at each control loop sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPoint {
    pub timestamp_s: f64,
    pub position_ned: Vector3<f64>,
    pub velocity_body: Vector3<f64>,
    pub attitude_euler_rad: Vector3<f64>, // [roll, pitch, yaw]
    pub angular_velocity_body: Vector3<f64>,
    pub desired_wrench: BodyWrench,
    pub achieved_wrench: BodyWrench,
    pub residual_wrench: BodyWrench,
    pub commanded_actuators: ActuatorState,
    pub actual_actuators: ActuatorState,
    pub authority_mode: ControlAuthorityMode,
}

/// Simulation harness running AllocatorRuntime closed-loop against MultirotorSimulator.
pub struct FlightScenarioRunner {
    pub simulator: MultirotorSimulator,
    pub runtime: AllocatorRuntime,
    pub dt_control_s: f64,
    pub history: Vec<TelemetryPoint>,
}

impl FlightScenarioRunner {
    pub fn new(
        simulator: MultirotorSimulator,
        runtime: AllocatorRuntime,
        dt_control_s: f64,
    ) -> Self {
        Self {
            simulator,
            runtime,
            dt_control_s,
            history: Vec::new(),
        }
    }

    /// Clears recorded telemetry history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Returns recorded telemetry data.
    pub fn get_history(&self) -> &[TelemetryPoint] {
        &self.history
    }

    /// Runs a closed-loop scenario simulation for duration_s.
    ///
    /// - `wrench_fn`: provides desired BodyWrench at time t (seconds)
    /// - `faults_fn`: provides active FaultSet at time t (seconds)
    /// - `wind_fn`: provides (wind_force_body, wind_moment_body) at time t (seconds)
    pub fn run_scenario<FW, FF, FG>(
        &mut self,
        duration_s: f64,
        mut wrench_fn: FW,
        mut faults_fn: FF,
        mut wind_fn: FG,
    ) -> Result<&[TelemetryPoint], VfError>
    where
        FW: FnMut(f64) -> BodyWrench,
        FF: FnMut(f64) -> FaultSet,
        FG: FnMut(f64) -> (Vector3<f64>, Vector3<f64>),
    {
        let steps = (duration_s / self.dt_control_s).round() as usize;

        for _ in 0..steps {
            let t = self.simulator.total_time_s();

            let desired_wrench = wrench_fn(t);
            let active_faults = faults_fn(t);
            let (wind_force, wind_moment) = wind_fn(t);

            // 1. Construct allocation request with actual simulated actuator state
            let request = AllocationRequest {
                timestamp: Instant::now(),
                desired_wrench_body: desired_wrench,
                measured_actuator_state: self.simulator.get_state().actuators,
                electrical_limits: ElectricalLimits {
                    max_current_a: 100.0,
                    battery_voltage_v: 24.0,
                    max_power_w: 2400.0,
                },
                active_faults,
            };

            // 2. Step synchronous control runtime
            let alloc_result = self.runtime.step(&request)?;

            // 3. Step 6-DOF plant simulator with newly allocated actuator commands
            let plant_state = self.simulator.step(
                &alloc_result.commands,
                self.dt_control_s,
                &wind_force,
                &wind_moment,
            )?;

            // 4. Compute Euler angles from attitude quaternion (Roll, Pitch, Yaw)
            let (roll, pitch, yaw) = plant_state.attitude.euler_angles();

            // 5. Record telemetry
            self.history.push(TelemetryPoint {
                timestamp_s: t + self.dt_control_s,
                position_ned: plant_state.position,
                velocity_body: plant_state.velocity,
                attitude_euler_rad: Vector3::new(roll, pitch, yaw),
                angular_velocity_body: plant_state.angular_velocity,
                desired_wrench,
                achieved_wrench: alloc_result.achieved_wrench_estimate,
                residual_wrench: alloc_result.residual_wrench,
                commanded_actuators: alloc_result.commands,
                actual_actuators: plant_state.actuators,
                authority_mode: alloc_result.mode,
            });
        }

        Ok(&self.history)
    }

    /// Access the underlying plant state.
    pub fn plant_state(&self) -> &PlantState {
        self.simulator.get_state()
    }
}
