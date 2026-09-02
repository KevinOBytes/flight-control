use approx::assert_relative_eq;
use std::path::PathBuf;
use vf_model::{compute_thrust_effectiveness, wrench_from_actuators, ActuatorState, VehicleModel};

#[test]
fn test_thrust_jacobian_finite_difference() {
    // Load config
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // to crates
    path.pop(); // to root
    path.push("configs");
    path.push("vehicle_v1.toml");

    let model = VehicleModel::from_file(path).expect("Failed to load vehicle model");

    // We choose a non-zero actuator state with some arbitrary tilts and thrusts
    // to test in a general, non-zero operating point.
    let mut state = ActuatorState::zero();
    for i in 0..16 {
        state.motor_thrusts[i] = 15.0 + 2.0 * (i as f64).sin(); // arbitrary thrusts
        state.motor_tilts[i] = 0.2 * (i as f64).cos(); // arbitrary motor tilts
    }
    // Arbitrary pod tilts
    state.pod_tilts = [0.1, -0.05, -0.1, 0.08, 0.05, -0.07, -0.02, 0.04];

    // Compute analytical Jacobian B (6x16)
    let b_analytical = compute_thrust_effectiveness(&model, &state)
        .expect("Failed to compute analytical Jacobian");

    // Perform finite differences
    let delta = 1e-6;

    for i in 0..16 {
        // Perturb positive
        let mut state_pos = state;
        state_pos.motor_thrusts[i] += delta;
        let w_pos = wrench_from_actuators(&model, &state_pos)
            .expect("Failed to compute positive perturbed wrench")
            .to_vector();

        // Perturb negative
        let mut state_neg = state;
        state_neg.motor_thrusts[i] -= delta;
        let w_neg = wrench_from_actuators(&model, &state_neg)
            .expect("Failed to compute negative perturbed wrench")
            .to_vector();

        // Finite difference derivative: (w_pos - w_neg) / (2 * delta)
        let col_numeric = (w_pos - w_neg) / (2.0 * delta);

        // Compare elements
        for r in 0..6 {
            let val_analytical = b_analytical[(r, i)];
            let val_numeric = col_numeric[r];

            // Use relative or absolute comparison depending on magnitude
            if val_numeric.abs() > 1e-2 {
                let rel_diff = (val_analytical - val_numeric).abs() / val_numeric.abs();
                assert!(
                    rel_diff < 1e-5,
                    "Jacobian mismatch at row {}, col {}: analytical = {}, numeric = {}, rel_diff = {}",
                    r,
                    i,
                    val_analytical,
                    val_numeric,
                    rel_diff
                );
            } else {
                assert_relative_eq!(val_analytical, val_numeric, epsilon = 1e-5);
            }
        }
    }
}

#[test]
fn test_motor_tilt_jacobian_finite_difference() {
    // Load config
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // to crates
    path.pop(); // to root
    path.push("configs");
    path.push("vehicle_v1.toml");

    let model = VehicleModel::from_file(path).expect("Failed to load vehicle model");

    // Choose a non-zero actuator state
    let mut state = ActuatorState::zero();
    for i in 0..16 {
        state.motor_thrusts[i] = 15.0 + 2.0 * (i as f64).sin(); // must be non-zero thrust so derivative is non-zero
        state.motor_tilts[i] = 0.2 * (i as f64).cos();
    }
    state.pod_tilts = [0.1, -0.05, -0.1, 0.08, 0.05, -0.07, -0.02, 0.04];

    // Compute analytical J_gamma (6x16)
    let j_analytical = vf_model::compute_motor_tilt_effectiveness(&model, &state)
        .expect("Failed to compute analytical motor tilt Jacobian");

    let delta = 1e-6;

    for i in 0..16 {
        // Perturb positive
        let mut state_pos = state;
        state_pos.motor_tilts[i] += delta;
        let w_pos = wrench_from_actuators(&model, &state_pos)
            .expect("Failed to compute positive perturbed wrench")
            .to_vector();

        // Perturb negative
        let mut state_neg = state;
        state_neg.motor_tilts[i] -= delta;
        let w_neg = wrench_from_actuators(&model, &state_neg)
            .expect("Failed to compute negative perturbed wrench")
            .to_vector();

        // Finite difference derivative
        let col_numeric = (w_pos - w_neg) / (2.0 * delta);

        for r in 0..6 {
            let val_analytical = j_analytical[(r, i)];
            let val_numeric = col_numeric[r];

            if val_numeric.abs() > 1e-2 {
                let rel_diff = (val_analytical - val_numeric).abs() / val_numeric.abs();
                assert!(
                    rel_diff < 1e-5,
                    "Motor tilt Jacobian mismatch at row {}, col {}: analytical = {}, numeric = {}, rel_diff = {}",
                    r,
                    i,
                    val_analytical,
                    val_numeric,
                    rel_diff
                );
            } else {
                assert_relative_eq!(val_analytical, val_numeric, epsilon = 1e-5);
            }
        }
    }
}

#[test]
fn test_pod_tilt_jacobian_finite_difference() {
    // Load config
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // to crates
    path.pop(); // to root
    path.push("configs");
    path.push("vehicle_v1.toml");

    let model = VehicleModel::from_file(path).expect("Failed to load vehicle model");

    // Choose a non-zero actuator state
    let mut state = ActuatorState::zero();
    for i in 0..16 {
        state.motor_thrusts[i] = 15.0 + 2.0 * (i as f64).sin();
        state.motor_tilts[i] = 0.2 * (i as f64).cos();
    }
    state.pod_tilts = [0.1, -0.05, -0.1, 0.08, 0.05, -0.07, -0.02, 0.04];

    // Compute analytical J_pod (6x8)
    let j_analytical = vf_model::compute_pod_tilt_effectiveness(&model, &state)
        .expect("Failed to compute analytical pod tilt Jacobian");

    let delta = 1e-6;

    for i in 0..8 {
        // Perturb positive
        let mut state_pos = state;
        state_pos.pod_tilts[i] += delta;
        let w_pos = wrench_from_actuators(&model, &state_pos)
            .expect("Failed to compute positive perturbed wrench")
            .to_vector();

        // Perturb negative
        let mut state_neg = state;
        state_neg.pod_tilts[i] -= delta;
        let w_neg = wrench_from_actuators(&model, &state_neg)
            .expect("Failed to compute negative perturbed wrench")
            .to_vector();

        // Finite difference derivative
        let col_numeric = (w_pos - w_neg) / (2.0 * delta);

        for r in 0..6 {
            let val_analytical = j_analytical[(r, i)];
            let val_numeric = col_numeric[r];

            if val_numeric.abs() > 1e-2 {
                let rel_diff = (val_analytical - val_numeric).abs() / val_numeric.abs();
                assert!(
                    rel_diff < 1e-5,
                    "Pod tilt Jacobian mismatch at row {}, col {}: analytical = {}, numeric = {}, rel_diff = {}",
                    r,
                    i,
                    val_analytical,
                    val_numeric,
                    rel_diff
                );
            } else {
                assert_relative_eq!(val_analytical, val_numeric, epsilon = 1e-5);
            }
        }
    }
}
