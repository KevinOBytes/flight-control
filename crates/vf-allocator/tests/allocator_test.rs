use approx::assert_relative_eq;
use std::path::PathBuf;
use vf_allocator::{ControlAllocator, OsqpThrustAllocator, SolverStatus};
use vf_core::BodyWrench;
use vf_model::{ActuatorState, VehicleModel};

#[test]
fn test_osqp_allocator_hover() {
    // Load config
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // to crates
    path.pop(); // to root
    path.push("configs");
    path.push("vehicle_v1.toml");

    let model = VehicleModel::from_file(path).expect("Failed to load vehicle model");

    // Allocate 160 N of lift (-Z direction)
    let desired_wrench = BodyWrench::new(
        nalgebra::Vector3::new(0.0, 0.0, -160.0),
        nalgebra::Vector3::new(0.0, 0.0, 0.0),
    );

    // Initial state: 7.5 N on each motor (under-thrusted, within slew limit of 10.0 N)
    let mut current_state = ActuatorState::zero();
    for i in 0..16 {
        current_state.motor_thrusts[i] = 7.5;
    }

    // Settings
    let wrench_weights = [1e4, 1e4, 1e4, 1e4, 1e4, 1e4];
    let lambda_smooth = 1e-3;
    let lambda_center = 1e-1;
    let f_nominal = [10.0; 16]; // hover nominal
    let dt = 0.005; // 200 Hz

    let mut allocator =
        OsqpThrustAllocator::new(wrench_weights, lambda_smooth, lambda_center, f_nominal, dt);

    let qp = allocator
        .formulate(
            &model,
            &desired_wrench,
            &current_state,
            &vf_faults::FaultSet::new(),
        )
        .expect("Failed to formulate QP");

    // 2. Solve
    let (commands, status, _diag) = allocator
        .solve(&qp, Some(&current_state))
        .expect("Failed to solve QP");

    assert_eq!(status, SolverStatus::Success);

    // Verifications:
    // - Thrust commands should be positive and close to 10.0 N each
    for &thrust in commands.motor_thrusts.iter() {
        assert!(thrust >= 0.0, "Thrust must be non-negative: {}", thrust);
        assert!(thrust <= 50.0, "Thrust must not exceed limit: {}", thrust);
        assert_relative_eq!(thrust, 10.0, epsilon = 0.1);
    }

    // - Verify the achieved wrench matches the desired wrench
    let achieved = vf_model::wrench_from_actuators(&model, &commands)
        .expect("Failed to compute achieved wrench");

    assert_relative_eq!(achieved.force.x, 0.0, epsilon = 1e-2);
    assert_relative_eq!(achieved.force.y, 0.0, epsilon = 1e-2);
    assert_relative_eq!(achieved.force.z, -160.0, epsilon = 1e-2);
    assert_relative_eq!(achieved.moment.x, 0.0, epsilon = 1e-2);
    assert_relative_eq!(achieved.moment.y, 0.0, epsilon = 1e-2);
    assert_relative_eq!(achieved.moment.z, 0.0, epsilon = 1e-2);
}
