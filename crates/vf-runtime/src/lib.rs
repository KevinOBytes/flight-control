//! Synchronous scheduler, execution loop, timing metrics, and allocations request/result structs.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use vf_allocator::{
    AllocationDiagnostics, ControlAllocator, OsqpMotorTiltPlanner, OsqpThrustAllocator,
    SolverStatus,
};
use vf_core::BodyWrench;
use vf_faults::{ControlAuthority, ControlAuthorityMode, FaultSet};
use vf_model::{ActuatorState, VehicleModel};

/// Limits regarding electrical power/current boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricalLimits {
    pub max_current_a: f64,
    pub battery_voltage_v: f64,
    pub max_power_w: f64,
}

fn default_instant() -> Instant {
    Instant::now()
}

/// The runtime input containing desired wrench, measurements, limits, and active faults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationRequest {
    #[serde(default = "default_instant", skip)]
    pub timestamp: Instant,
    pub desired_wrench_body: BodyWrench,
    pub measured_actuator_state: ActuatorState,
    pub electrical_limits: ElectricalLimits,
    pub active_faults: FaultSet,
}

/// The control loop output frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationResult {
    pub commands: ActuatorState,
    pub achieved_wrench_estimate: BodyWrench,
    pub residual_wrench: BodyWrench,
    pub solver_status: SolverStatus,
    pub authority: ControlAuthority,
    pub mode: ControlAuthorityMode,
    pub diagnostics: AllocationDiagnostics,
}

/// System execution loop configuration.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub thrust_alloc_hz: f64,
    pub motor_tilt_hz: f64,
    pub pod_tilt_hz: f64,
}

/// Runtime timing diagnostics tracked by the scheduler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingDiagnostics {
    pub elapsed_since_start_s: f64,
    pub loop_execution_time_us: u64,
    pub max_loop_execution_time_us: u64,
    pub missed_deadlines_count: u64,
}

/// Synchronous allocator runtime controller.
pub struct AllocatorRuntime {
    config: RuntimeConfig,
    allocator: OsqpThrustAllocator,
    tilt_planner: OsqpMotorTiltPlanner,
    vehicle_model: VehicleModel,
    last_tick: Option<Instant>,
    timing: TimingDiagnostics,
    pub current_motor_tilts: Option<[f64; 16]>,
    pub target_motor_tilts: Option<[f64; 16]>,
    pub step_counter: u64,
}

impl AllocatorRuntime {
    pub fn new(
        config: RuntimeConfig,
        allocator: OsqpThrustAllocator,
        tilt_planner: OsqpMotorTiltPlanner,
        vehicle_model: VehicleModel,
    ) -> Self {
        Self {
            config,
            allocator,
            tilt_planner,
            vehicle_model,
            last_tick: None,
            timing: TimingDiagnostics::default(),
            current_motor_tilts: None,
            target_motor_tilts: None,
            step_counter: 0,
        }
    }

    /// Executes one step of the synchronous control loop.
    pub fn step(
        &mut self,
        request: &AllocationRequest,
    ) -> Result<AllocationResult, vf_core::VfError> {
        let now = Instant::now();
        let dt_fast = if let Some(last) = self.last_tick {
            now.duration_since(last).as_secs_f64()
        } else {
            1.0 / self.config.thrust_alloc_hz
        };
        self.last_tick = Some(now);

        // 1. Initialize tilts if not yet initialized
        if self.current_motor_tilts.is_none() {
            self.current_motor_tilts = Some(request.measured_actuator_state.motor_tilts);
            self.target_motor_tilts = Some(request.measured_actuator_state.motor_tilts);
        }

        let mut current_tilts = self.current_motor_tilts.unwrap();
        let target_tilts = self.target_motor_tilts.unwrap();

        // 2. Slew current motor tilts toward target motor tilts
        let mut tilt_rates = [0.0; 16];
        for i in 0..16 {
            let rotor_id = vf_core::RotorId(i as u32 + 1);
            let rotor = self
                .vehicle_model
                .rotors
                .iter()
                .find(|r| r.id == rotor_id)
                .ok_or_else(|| {
                    vf_core::VfError::InvalidValue(format!("Rotor {:?} not found", rotor_id))
                })?;

            if let Some(jammed_angle) = request.active_faults.get_jammed_motor_tilt(rotor_id) {
                current_tilts[i] = jammed_angle;
                tilt_rates[i] = 0.0;
            } else {
                let limit = rotor.motor_tilt_rate_limit_rad_s * dt_fast;
                let diff = target_tilts[i] - current_tilts[i];
                let delta = diff.clamp(-limit, limit);
                current_tilts[i] += delta;
                tilt_rates[i] = if dt_fast > 1e-6 { delta / dt_fast } else { 0.0 };
            }
        }
        self.current_motor_tilts = Some(current_tilts);

        // 3. Update the tilt rates in OsqpThrustAllocator to update dynamic bounds
        self.allocator.update_motor_tilt_rates(tilt_rates);

        // 4. Formulate the QP for thrust allocation using the current motor tilts
        let mut nominal_state_with_slewed_tilts = request.measured_actuator_state;
        nominal_state_with_slewed_tilts.motor_tilts = current_tilts;

        let qp = self.allocator.formulate(
            &self.vehicle_model,
            &request.desired_wrench_body,
            &nominal_state_with_slewed_tilts,
            &request.active_faults,
        )?;

        // 5. Solve the thrust QP
        let (mut commands, solver_status, diagnostics) = self
            .allocator
            .solve(&qp, Some(&nominal_state_with_slewed_tilts))?;

        // 6. Post-solver safety check: enforce exactly 0.0 N thrust for failed motors
        for (i, rotor) in self.vehicle_model.rotors.iter().enumerate() {
            if request
                .active_faults
                .is_rotor_failed(rotor.id, rotor.pod_id)
            {
                commands.motor_thrusts[i] = 0.0;
            }
        }

        // Set the active motor tilts in the final commands state
        commands.motor_tilts = current_tilts;

        // 7. Trigger the 50 Hz Individual Motor Tilt Planner
        self.step_counter += 1;
        let planner_step_divider =
            (self.config.thrust_alloc_hz / self.config.motor_tilt_hz).round() as u64;
        let planner_step_divider = if planner_step_divider == 0 {
            1
        } else {
            planner_step_divider
        };

        if self.step_counter.is_multiple_of(planner_step_divider) {
            // Run the tilt planner to update target motor tilts
            let planner_qp = self.tilt_planner.formulate(
                &self.vehicle_model,
                &request.desired_wrench_body,
                &commands,
                &request.active_faults,
            )?;
            let delta_tilts = self
                .tilt_planner
                .solve(&planner_qp, Some(&commands.motor_tilts))?;

            let mut new_targets = current_tilts;
            for i in 0..16 {
                new_targets[i] = current_tilts[i] + delta_tilts[i];
            }
            self.target_motor_tilts = Some(new_targets);
        }

        // 8. Compute achieved wrench estimate and residual
        let achieved_wrench = vf_model::wrench_from_actuators(&self.vehicle_model, &commands)?;
        let residual_wrench = BodyWrench::new(
            request.desired_wrench_body.force - achieved_wrench.force,
            request.desired_wrench_body.moment - achieved_wrench.moment,
        );

        // 9. Update timing diagnostics
        let loop_time = now.elapsed().as_micros() as u64;
        self.timing.loop_execution_time_us = loop_time;
        self.timing.max_loop_execution_time_us =
            self.timing.max_loop_execution_time_us.max(loop_time);

        let deadline_us = (1.0 / self.config.thrust_alloc_hz * 1e6) as u64;
        if loop_time > deadline_us {
            self.timing.missed_deadlines_count += 1;
            tracing::warn!(
                "Control loop deadline missed! Execution time: {} us, deadline: {} us",
                loop_time,
                deadline_us
            );
        }

        // 10. Compute ControlAuthority metrics using SVD and the fault-degraded effectiveness matrix
        let b_matrix = vf_allocator::compute_thrust_effectiveness_under_faults(
            &self.vehicle_model,
            &commands,
            &request.active_faults,
        )?;

        let authority = ControlAuthority::compute(
            &self.vehicle_model,
            &commands,
            &request.active_faults,
            &b_matrix,
            self.vehicle_model.min_singular_value_degraded,
            self.vehicle_model.min_singular_value_critical,
        );

        let authority_mode = authority.mode;

        Ok(AllocationResult {
            commands,
            achieved_wrench_estimate: achieved_wrench,
            residual_wrench,
            solver_status,
            authority,
            mode: authority_mode,
            diagnostics,
        })
    }

    /// Expose runtime timing metrics for logging.
    pub fn get_timing_diagnostics(&self) -> &TimingDiagnostics {
        &self.timing
    }

    /// Returns the current motor tilt rates.
    pub fn get_motor_tilt_rates(&self) -> [f64; 16] {
        self.allocator.motor_tilt_rates
    }
}
