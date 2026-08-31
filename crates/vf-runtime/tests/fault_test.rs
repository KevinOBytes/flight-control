use std::path::PathBuf;
use std::time::Instant;
use vf_allocator::{OsqpMotorTiltPlanner, OsqpThrustAllocator};
use vf_core::{BodyWrench, PodId, RotorId};
use vf_faults::{ActuatorFault, ControlAuthorityMode, FaultSet};
use vf_model::{ActuatorState, VehicleModel};
use vf_runtime::{AllocationRequest, AllocatorRuntime, ElectricalLimits, RuntimeConfig};

fn setup_runtime() -> AllocatorRuntime {
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

    AllocatorRuntime::new(config, allocator, tilt_planner, model)
}

/// Generates a measured state with non-zero tilts to ensure full 6-DOF matrix rank.
fn get_nominal_measured_state() -> ActuatorState {
    let mut state = ActuatorState::zero();
    for i in 0..16 {
        state.motor_thrusts[i] = 10.0;
        // set small tilts to ensure 6-DOF control authority is non-singular
        state.motor_tilts[i] = 0.15 * (i as f64).cos();
    }
    // Opposing pod tilts to ensure independent lateral/longitudinal force and moment generation
    state.pod_tilts = [0.15, -0.15, -0.15, 0.15, 0.15, 0.15, -0.15, -0.15];
    state
}

#[test]
fn test_scenario_nominal_hover() {
    let mut runtime = setup_runtime();

    let request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(0.0, 0.0, -160.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        ),
        measured_actuator_state: get_nominal_measured_state(),
        electrical_limits: ElectricalLimits {
            max_current_a: 100.0,
            battery_voltage_v: 24.0,
            max_power_w: 2400.0,
        },
        active_faults: FaultSet::new(),
    };

    let result = runtime.step(&request).expect("Step failed");
    assert_eq!(result.mode, ControlAuthorityMode::NORMAL);
    assert_eq!(result.authority.rank, 6);
    assert!(result.authority.min_singular_value > 0.1);
}

#[test]
fn test_scenario_single_motor_failure() {
    let mut runtime = setup_runtime();

    let mut faults = FaultSet::new();
    faults.insert(ActuatorFault::MotorFailed { rotor: RotorId(1) });

    let request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(0.0, 0.0, -160.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        ),
        measured_actuator_state: {
            let mut state = get_nominal_measured_state();
            for i in 1..16 {
                state.motor_thrusts[i] = 11.0;
            }
            state.motor_thrusts[0] = 0.0; // Rotor 1 is failed
            state
        },
        electrical_limits: ElectricalLimits {
            max_current_a: 100.0,
            battery_voltage_v: 24.0,
            max_power_w: 2400.0,
        },
        active_faults: faults,
    };

    let result = runtime.step(&request).expect("Step failed");

    // Rule check: Failed motor MUST command 0.0 N thrust
    assert_eq!(result.commands.motor_thrusts[0], 0.0);

    // Controllability is degraded but rank is 6
    assert_eq!(result.mode, ControlAuthorityMode::DEGRADED);
    assert_eq!(result.authority.rank, 6);

    // Verify wrench is still achieved (other rotors compensate)
    assert!(result.residual_wrench.force.z.abs() < 1e-1);
}

#[test]
fn test_scenario_pod_bus_failure() {
    let mut runtime = setup_runtime();

    // Lost entire FL pod bus
    let mut faults = FaultSet::new();
    faults.insert(ActuatorFault::PodBusUnavailable { pod: PodId::FL });

    let request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(0.0, 0.0, -160.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        ),
        measured_actuator_state: {
            let mut state = get_nominal_measured_state();
            // FL rotors are 1..4 (indices 0..3)
            for i in 0..4 {
                state.motor_thrusts[i] = 0.0;
            }
            state
        },
        electrical_limits: ElectricalLimits {
            max_current_a: 100.0,
            battery_voltage_v: 24.0,
            max_power_w: 2400.0,
        },
        active_faults: faults,
    };

    let result = runtime.step(&request).expect("Step failed");

    // All FL rotors must be 0
    for i in 0..4 {
        assert_eq!(result.commands.motor_thrusts[i], 0.0);
    }

    // Rank should still be 6 because 12 remaining rotors (on FR, RL, RR) are fully sufficient for 6-DOF control!
    assert_eq!(result.authority.rank, 6);
    assert!(
        result.mode == ControlAuthorityMode::DEGRADED
            || result.mode == ControlAuthorityMode::CRITICAL
    );
}

#[test]
fn test_scenario_uncontrollable_multiple_failures() {
    let mut runtime = setup_runtime();

    // Fail almost all pods (FL, FR, RL) to lose controllability
    let mut faults = FaultSet::new();
    faults.insert(ActuatorFault::PodBusUnavailable { pod: PodId::FL });
    faults.insert(ActuatorFault::PodBusUnavailable { pod: PodId::FR });
    faults.insert(ActuatorFault::PodBusUnavailable { pod: PodId::RL });

    let request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(0.0, 0.0, -160.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        ),
        measured_actuator_state: {
            let mut state = get_nominal_measured_state();
            // All failed except RR (rotors 13..16)
            for i in 0..12 {
                state.motor_thrusts[i] = 0.0;
            }
            state
        },
        electrical_limits: ElectricalLimits {
            max_current_a: 100.0,
            battery_voltage_v: 24.0,
            max_power_w: 2400.0,
        },
        active_faults: faults,
    };

    let result = runtime.step(&request).expect("Step failed");

    // System must classify as UNCONTROLLABLE
    assert_eq!(result.mode, ControlAuthorityMode::UNCONTROLLABLE);
    assert!(result.authority.rank < 6);
}
