use approx::assert_relative_eq;
use std::path::PathBuf;
use vf_model::{wrench_from_actuators, ActuatorState, VehicleModel};

#[test]
fn test_hover_wrench_symmetry() {
    // Load config
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // to crates
    path.pop(); // to root
    path.push("configs");
    path.push("vehicle_v1.toml");

    let model = VehicleModel::from_file(path).expect("Failed to load vehicle model");

    // Command symmetric thrust
    let mut state = ActuatorState::zero();
    for i in 1..=16 {
        state.motor_thrusts[i - 1] = 10.0; // 10 N each
    }

    let wrench = wrench_from_actuators(&model, &state).expect("Failed to compute wrench");

    // Total force should be 160 N in -Z (upwards)
    // Fx, Fy should be near 0
    assert_relative_eq!(wrench.force.x, 0.0, epsilon = 1e-6);
    assert_relative_eq!(wrench.force.y, 0.0, epsilon = 1e-6);
    assert_relative_eq!(wrench.force.z, -160.0, epsilon = 1e-6);

    // Torques should cancel out symmetrically (Mx, My, Mz should be 0)
    assert_relative_eq!(wrench.moment.x, 0.0, epsilon = 1e-6);
    assert_relative_eq!(wrench.moment.y, 0.0, epsilon = 1e-6);
    assert_relative_eq!(wrench.moment.z, 0.0, epsilon = 1e-6);
}
