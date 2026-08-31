use std::path::PathBuf;
use std::time::Instant;
use vf_allocator::{OsqpMotorTiltPlanner, OsqpThrustAllocator, SolverStatus};
use vf_core::BodyWrench;
use vf_faults::FaultSet;
use vf_model::{ActuatorState, VehicleModel};
use vf_runtime::{AllocationRequest, AllocatorRuntime, ElectricalLimits, RuntimeConfig};

#[test]
fn test_allocator_runtime_step() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // to crates
    path.pop(); // to root
    path.push("configs");
    path.push("vehicle_v1.toml");

    let model = VehicleModel::from_file(path).expect("Failed to load vehicle model");

    let config = RuntimeConfig {
        thrust_alloc_hz: 200.0,
        motor_tilt_hz: 50.0,
        pod_tilt_hz: 20.0,
    };

    let wrench_weights = [1e4, 1e4, 1e4, 1e4, 1e4, 1e4];
    let allocator = OsqpThrustAllocator::new(wrench_weights, 1e-3, 1e-1, [10.0; 16], 0.005);
    let tilt_planner = OsqpMotorTiltPlanner::new(wrench_weights, 1.0, 0.1, 0.020);

    let mut runtime = AllocatorRuntime::new(config, allocator, tilt_planner, model);

    // Create a request
    let request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(0.0, 0.0, -160.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        ),
        measured_actuator_state: {
            let mut state = ActuatorState::zero();
            for i in 0..16 {
                state.motor_thrusts[i] = 7.5;
            }
            state
        },
        electrical_limits: ElectricalLimits {
            max_current_a: 100.0,
            battery_voltage_v: 24.0,
            max_power_w: 2400.0,
        },
        active_faults: FaultSet::new(),
    };

    // Execute step
    let result = runtime.step(&request).expect("Step execution failed");

    assert_eq!(result.solver_status, SolverStatus::Success);
    assert!(result.achieved_wrench_estimate.force.z <= -150.0);
}
