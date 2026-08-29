//! Synchronous scheduler, execution loop, timing metrics, and allocations request/result structs.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use vf_allocator::{AllocationDiagnostics, ControlAllocator, OsqpThrustAllocator, SolverStatus};
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
    vehicle_model: VehicleModel,
    last_tick: Option<Instant>,
    timing: TimingDiagnostics,
}

impl AllocatorRuntime {
    pub fn new(
        config: RuntimeConfig,
        allocator: OsqpThrustAllocator,
        vehicle_model: VehicleModel,
    ) -> Self {
        Self {
            config,
            allocator,
            vehicle_model,
            last_tick: None,
            timing: TimingDiagnostics::default(),
        }
    }

    /// Executes one step of the synchronous control loop.
    pub fn step(
        &mut self,
        request: &AllocationRequest,
    ) -> Result<AllocationResult, vf_core::VfError> {
        let now = Instant::now();
        let _dt = if let Some(last) = self.last_tick {
            now.duration_since(last).as_secs_f64()
        } else {
            1.0 / self.config.thrust_alloc_hz
        };
        self.last_tick = Some(now);

        // 1. Formulate the QP
        let qp = self.allocator.formulate(
            &self.vehicle_model,
            &request.desired_wrench_body,
            &request.measured_actuator_state,
        )?;

        // 2. Solve the QP
        let (commands, solver_status, diagnostics) = self
            .allocator
            .solve(&qp, Some(&request.measured_actuator_state))?;

        // 3. Compute achieved wrench estimate and residual
        let achieved_wrench = vf_model::wrench_from_actuators(&self.vehicle_model, &commands)?;
        let residual_wrench = BodyWrench::new(
            request.desired_wrench_body.force - achieved_wrench.force,
            request.desired_wrench_body.moment - achieved_wrench.moment,
        );

        // 4. Update timing diagnostics
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

        // 5. Build ControlAuthority (mock values for singular values/rank until Phase 3)
        let authority = ControlAuthority {
            mode: ControlAuthorityMode::NORMAL,
            rank: 6,
            singular_values: vec![1.0; 6],
            min_singular_value: 1.0,
            condition_number: 1.0,
            thrust_reserve_n: 800.0,
            saturation_fraction: 0.0,
        };

        Ok(AllocationResult {
            commands,
            achieved_wrench_estimate: achieved_wrench,
            residual_wrench,
            solver_status,
            authority,
            mode: ControlAuthorityMode::NORMAL,
            diagnostics,
        })
    }

    /// Expose runtime timing metrics for logging.
    pub fn get_timing_diagnostics(&self) -> &TimingDiagnostics {
        &self.timing
    }
}
