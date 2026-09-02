use std::path::PathBuf;
use std::time::Instant;
use vf_allocator::{OsqpMotorTiltPlanner, OsqpPodTiltPlanner, OsqpThrustAllocator};
use vf_core::{BodyWrench, PodAxis, PodId};
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
    let pod_planner = OsqpPodTiltPlanner::new(
        wrench_weights,
        1.0,   // lambda_smooth
        0.1,   // lambda_center
        0.3,   // alpha_lpf
        0.050, // dt
    );

    AllocatorRuntime::new(config, allocator, tilt_planner, pod_planner, model)
}

/// Generates a nominal measured state with small opposing tilts for 6-DOF rank.
fn get_nominal_measured_state() -> ActuatorState {
    let mut state = ActuatorState::zero();
    for i in 0..16 {
        state.motor_thrusts[i] = 10.0;
        state.motor_tilts[i] = 0.05 * (i as f64).cos();
    }
    state.pod_tilts = [0.05, -0.05, -0.05, 0.05, 0.05, 0.05, -0.05, -0.05];
    state
}

#[test]
fn test_pod_planner_gross_vectoring_lateral_force() {
    let mut runtime = setup_runtime();

    // Request large sustained lateral force (Fy = 30.0 N) + hover lift (Fz = -160.0 N)
    let mut request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(0.0, 30.0, -160.0),
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

    let initial_pod_tilts = get_nominal_measured_state().pod_tilts;
    let mut last_result = None;

    // Run for 150 ticks (0.75s = 15 pod planner cycles at 20 Hz)
    for _ in 0..150 {
        let result = runtime.step(&request).expect("Step failed");
        request.measured_actuator_state = result.commands;
        last_result = Some(result);
    }

    let final_result = last_result.unwrap();

    // 1. Pod gimbals must have tilted to vector thrust sideways
    let pod_tilts = runtime.get_pod_tilts().expect("Pod tilts uninitialized");
    let pod_tilts_changed = pod_tilts
        .iter()
        .zip(initial_pod_tilts.iter())
        .any(|(&c, &m)| (c - m).abs() > 0.02);
    assert!(
        pod_tilts_changed,
        "Pod tilts did not adapt to lateral demand!"
    );

    // 2. Residual lateral and vertical forces should be well-tracked
    assert!(
        final_result.residual_wrench.force.y.abs() < 2.0,
        "Large residual Fy: {}",
        final_result.residual_wrench.force.y
    );
    assert!(
        final_result.residual_wrench.force.z.abs() < 2.0,
        "Large residual Fz: {}",
        final_result.residual_wrench.force.z
    );
}

#[test]
fn test_pod_planner_low_pass_filtering_noise_rejection() {
    let mut runtime = setup_runtime();

    let mut request = AllocationRequest {
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

    let mut max_pod_tilt_delta = 0.0;
    let mut prev_pod_tilts = get_nominal_measured_state().pod_tilts;

    // Simulate 50 ticks of high-frequency oscillating moment noise (+/- 20 Nm at each tick)
    for tick in 0..50 {
        let noise_sign = if tick % 2 == 0 { 1.0 } else { -1.0 };
        request.desired_wrench_body.moment.y = noise_sign * 20.0;

        let result = runtime.step(&request).expect("Step failed");
        request.measured_actuator_state = result.commands;

        let current_pods = runtime.get_pod_tilts().unwrap();
        for i in 0..8 {
            let delta = (current_pods[i] - prev_pod_tilts[i]).abs();
            if delta > max_pod_tilt_delta {
                max_pod_tilt_delta = delta;
            }
        }
        prev_pod_tilts = current_pods;
    }

    // Because of the low-pass filter and slew limiting, the pod gimbals do not jump wildly on single-tick alternating noise
    assert!(
        max_pod_tilt_delta < 0.05,
        "Pod gimbals hunted high-frequency noise excessively: max delta = {}",
        max_pod_tilt_delta
    );
}

#[test]
fn test_pod_planner_jammed_axis() {
    let mut runtime = setup_runtime();

    // Inject jammed axis fault on Pod FL Axis 1 at 0.18 rad
    let mut faults = FaultSet::new();
    let jammed_angle = 0.18;
    faults.insert(ActuatorFault::PodAxisJammed {
        pod: PodId::FL,
        axis: PodAxis::Axis1,
        angle_rad: jammed_angle,
    });

    let mut request = AllocationRequest {
        timestamp: Instant::now(),
        desired_wrench_body: BodyWrench::new(
            nalgebra::Vector3::new(15.0, 15.0, -160.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        ),
        measured_actuator_state: {
            let mut state = get_nominal_measured_state();
            state.pod_tilts[0] = jammed_angle; // FL Axis 1
            state
        },
        electrical_limits: ElectricalLimits {
            max_current_a: 100.0,
            battery_voltage_v: 24.0,
            max_power_w: 2400.0,
        },
        active_faults: faults,
    };

    // Run for 30 ticks
    let mut last_result = None;
    for _ in 0..30 {
        let result = runtime.step(&request).expect("Step failed");
        request.measured_actuator_state = result.commands;
        last_result = Some(result);
    }

    let final_result = last_result.unwrap();

    // Verification: FL Axis 1 must remain exactly locked at jammed_angle (0.18 rad)
    assert_eq!(
        final_result.commands.pod_tilts[0], jammed_angle,
        "Jammed pod axis drifted!"
    );
}
