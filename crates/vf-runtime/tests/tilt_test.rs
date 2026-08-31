use std::path::PathBuf;
use std::time::Instant;
use vf_allocator::{OsqpMotorTiltPlanner, OsqpThrustAllocator};
use vf_core::{BodyWrench, RotorId};
use vf_faults::{ActuatorFault, FaultSet};
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
    let tilt_planner = OsqpMotorTiltPlanner::new(
        wrench_weights,
        1.0,   // lambda_smooth
        0.1,   // lambda_center
        0.020, // dt
    );

    AllocatorRuntime::new(config, allocator, tilt_planner, model)
}

/// Generates a measured state with non-zero opposing tilts for 6-DOF rank.
fn get_nominal_measured_state() -> ActuatorState {
    let mut state = ActuatorState::zero();
    for i in 0..16 {
        state.motor_thrusts[i] = 10.0;
        state.motor_tilts[i] = 0.15 * (i as f64).cos();
    }
    state.pod_tilts = [0.15, -0.15, -0.15, 0.15, 0.15, 0.15, -0.15, -0.15];
    state
}

#[test]
fn test_tilt_planner_sustained_lateral_force() {
    let mut runtime = setup_runtime();

    // 1. Initial request for hover + longitudinal force in X direction (15.0 N)
    let mut request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(15.0, 0.0, -160.0),
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

    // 2. Run scheduler for multiple steps (e.g. 20 ticks = 0.1 seconds)
    // The tilt planner runs at 50 Hz, so it will run 2 times.
    let mut last_result = None;
    for _step in 0..20 {
        let result = runtime.step(&request).expect("Step failed");

        // Feed command back as next measurement to simulate loop
        request.measured_actuator_state = result.commands;
        last_result = Some(result);
    }

    let final_result = last_result.unwrap();

    // Check that we track the desired longitudinal force and vertical force closely
    assert!(
        final_result.residual_wrench.force.x.abs() < 1.0,
        "Large residual Fx: {}",
        final_result.residual_wrench.force.x
    );
    assert!(
        final_result.residual_wrench.force.z.abs() < 1.0,
        "Large residual Fz: {}",
        final_result.residual_wrench.force.z
    );

    // Check that motor tilts have adjusted from the initial cosine pattern
    // to vector thrust and help produce the lateral force
    let tilts_changed = final_result
        .commands
        .motor_tilts
        .iter()
        .zip(get_nominal_measured_state().motor_tilts.iter())
        .any(|(&c, &m)| (c - m).abs() > 1e-3);
    assert!(tilts_changed, "Motor tilts did not change!");

    // Verify dynamic bounds update: since motor tilts were slewing, at least one motor tilt rate must have been non-zero,
    // which scales down the upper bounds.
    // Let's assert that the active tilt rates were populated
    assert!(
        runtime
            .get_motor_tilt_rates()
            .iter()
            .any(|&r: &f64| r.abs() > 0.0),
        "No active tilt rates!"
    );
}

#[test]
fn test_tilt_planner_jammed_joint() {
    let mut runtime = setup_runtime();

    // Inject jammed motor tilt on rotor 2 at exactly 0.25 rad
    let mut faults = FaultSet::new();
    let jammed_angle = 0.25;
    faults.insert(ActuatorFault::MotorTiltJammed {
        rotor: RotorId(2),
        angle_rad: jammed_angle,
    });

    let mut request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(15.0, 0.0, -160.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        ),
        measured_actuator_state: {
            let mut state = get_nominal_measured_state();
            state.motor_tilts[1] = jammed_angle; // rotor 2
            state
        },
        electrical_limits: ElectricalLimits {
            max_current_a: 100.0,
            battery_voltage_v: 24.0,
            max_power_w: 2400.0,
        },
        active_faults: faults,
    };

    // Run for 10 ticks
    let mut last_result = None;
    for _ in 0..10 {
        let result = runtime.step(&request).expect("Step failed");
        request.measured_actuator_state = result.commands;
        last_result = Some(result);
    }

    let final_result = last_result.unwrap();

    // Verification: Rotor 2 tilt must remain exactly locked at the jammed angle (0.25 rad)
    assert_eq!(
        final_result.commands.motor_tilts[1], jammed_angle,
        "Jammed motor tilt drifted!"
    );
}
