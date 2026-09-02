use std::path::PathBuf;
use vf_allocator::{OsqpMotorTiltPlanner, OsqpPodTiltPlanner, OsqpThrustAllocator};
use vf_core::{BodyWrench, RotorId};
use vf_faults::{ActuatorFault, ControlAuthorityMode, FaultSet};
use vf_model::{ActuatorState, VehicleModel};
use vf_runtime::{AllocatorRuntime, RuntimeConfig};
use vf_sim::{FlightScenarioRunner, MultirotorSimulator, PlantState, SimConfig};

fn setup_sim_and_runtime() -> (MultirotorSimulator, AllocatorRuntime) {
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
    let allocator = OsqpThrustAllocator::new(wrench_weights, 1e-3, 1e-1, [15.32; 16], 0.005);
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

    let runtime =
        AllocatorRuntime::new(config, allocator, tilt_planner, pod_planner, model.clone());

    let sim_config = SimConfig {
        gravity_m_s2: 9.80665,
        motor_thrust_tau: 0.020,
        motor_tilt_tau: 0.030,
        pod_tilt_tau: 0.050,
        force_noise_std: 0.0,
        moment_noise_std: 0.0,
        substeps: 10,
    };

    let sim = MultirotorSimulator::new(sim_config, model, PlantState::new());

    (sim, runtime)
}

#[test]
fn test_scenario_hover_equilibrium() {
    let (mut sim, runtime) = setup_sim_and_runtime();

    let hover_thrust_total = sim.vehicle_model().mass_kg * sim.config().gravity_m_s2;
    let hover_thrust_per_motor = hover_thrust_total / 16.0;

    // Set initial plant actuator state to steady hover thrust
    for i in 0..16 {
        sim.state_mut().actuators.motor_thrusts[i] = hover_thrust_per_motor;
    }

    let mut runner = FlightScenarioRunner::new(sim, runtime, 0.005);

    // Run hover scenario for 1.0 second
    let history = runner
        .run_scenario(
            1.0,
            |_t| {
                BodyWrench::new(
                    nalgebra::Vector3::new(0.0, 0.0, -hover_thrust_total),
                    nalgebra::Vector3::new(0.0, 0.0, 0.0),
                )
            },
            |_t| FaultSet::new(),
            |_t| (nalgebra::Vector3::zeros(), nalgebra::Vector3::zeros()),
        )
        .expect("Scenario run failed");

    assert_eq!(history.len(), 200);

    let final_point = history.last().unwrap();

    // Verify vertical drift is minimal (< 5 cm over 1 second of closed-loop hover)
    assert!(
        final_point.position_ned.z.abs() < 0.05,
        "Vertical position drifted: z = {}",
        final_point.position_ned.z
    );

    // Verify attitude remains upright
    assert!(
        final_point.attitude_euler_rad.x.abs() < 1e-3,
        "Roll angle drifted: roll = {}",
        final_point.attitude_euler_rad.x
    );
    assert!(
        final_point.attitude_euler_rad.y.abs() < 1e-3,
        "Pitch angle drifted: pitch = {}",
        final_point.attitude_euler_rad.y
    );
}

#[test]
fn test_scenario_actuator_lag_and_rate_limits() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // to crates
    path.pop(); // to root
    path.push("configs");
    path.push("vehicle_v1.toml");

    let model = VehicleModel::from_file(path).expect("Failed to load vehicle model");

    let sim_config = SimConfig {
        gravity_m_s2: 9.80665,
        motor_thrust_tau: 0.020,
        motor_tilt_tau: 0.030,
        pod_tilt_tau: 0.050,
        force_noise_std: 0.0,
        moment_noise_std: 0.0,
        substeps: 20,
    };

    let mut sim = MultirotorSimulator::new(sim_config, model, PlantState::new());

    let mut commanded_actuators = ActuatorState::zero();
    commanded_actuators.motor_thrusts[0] = 10.0; // Step change from 0 to 10 N

    // Step by 1 tau = 0.020 s
    let dt = 0.020;
    sim.step(
        &commanded_actuators,
        dt,
        &nalgebra::Vector3::zeros(),
        &nalgebra::Vector3::zeros(),
    )
    .expect("Step failed");

    let actual_thrust = sim.get_state().actuators.motor_thrusts[0];

    // For 1st-order response x(t) = target * (1 - e^(-t/tau)):
    // at t = tau, x(tau) = 10.0 * (1 - 1/e) ~= 6.321 N
    let expected_thrust = 10.0 * (1.0 - (-1.0_f64).exp());
    let diff = (actual_thrust - expected_thrust).abs();

    assert!(
        diff < 0.5,
        "Thrust lag response mismatch: actual = {}, expected ~= {}",
        actual_thrust,
        expected_thrust
    );
}

#[test]
fn test_scenario_closed_loop_lateral_vectoring() {
    let (mut sim, runtime) = setup_sim_and_runtime();

    let hover_thrust_total = sim.vehicle_model().mass_kg * sim.config().gravity_m_s2;
    let hover_thrust_per_motor = hover_thrust_total / 16.0;

    for i in 0..16 {
        sim.state_mut().actuators.motor_thrusts[i] = hover_thrust_per_motor;
    }

    let mut runner = FlightScenarioRunner::new(sim, runtime, 0.005);

    // Command hover lift + 20 N sustained lateral force
    let history = runner
        .run_scenario(
            1.0,
            |_t| {
                BodyWrench::new(
                    nalgebra::Vector3::new(0.0, 20.0, -hover_thrust_total),
                    nalgebra::Vector3::new(0.0, 0.0, 0.0),
                )
            },
            |_t| FaultSet::new(),
            |_t| (nalgebra::Vector3::zeros(), nalgebra::Vector3::zeros()),
        )
        .expect("Scenario run failed");

    let final_point = history.last().unwrap();

    // Verify vehicle accelerated laterally in +Y direction
    assert!(
        final_point.velocity_body.y > 0.1,
        "Vehicle did not gain lateral velocity: vy = {}",
        final_point.velocity_body.y
    );
    assert!(
        final_point.position_ned.y > 0.05,
        "Vehicle did not translate laterally: py = {}",
        final_point.position_ned.y
    );

    // Pod gimbals rotated to vector the thrust
    let pod_tilts = final_point.actual_actuators.pod_tilts;
    assert!(
        pod_tilts.iter().any(|&p| p.abs() > 0.01),
        "Pod tilts did not vector thrust!"
    );
}

#[test]
fn test_scenario_closed_loop_fault_recovery() {
    let (mut sim, runtime) = setup_sim_and_runtime();

    let hover_thrust_total = sim.vehicle_model().mass_kg * sim.config().gravity_m_s2;
    let hover_thrust_per_motor = hover_thrust_total / 16.0;

    for i in 0..16 {
        sim.state_mut().actuators.motor_thrusts[i] = hover_thrust_per_motor;
        sim.state_mut().actuators.motor_tilts[i] = 0.15 * (i as f64).cos();
    }
    sim.state_mut().actuators.pod_tilts = [0.15, -0.15, -0.15, 0.15, 0.15, 0.15, -0.15, -0.15];

    let mut runner = FlightScenarioRunner::new(sim, runtime, 0.005);

    // Run for 1.0s, with Motor 1 failure injected at t = 0.3s
    let history = runner
        .run_scenario(
            1.0,
            |_t| {
                BodyWrench::new(
                    nalgebra::Vector3::new(0.0, 0.0, -hover_thrust_total),
                    nalgebra::Vector3::new(0.0, 0.0, 0.0),
                )
            },
            |t| {
                let mut faults = FaultSet::new();
                if t >= 0.3 {
                    faults.insert(ActuatorFault::MotorFailed { rotor: RotorId(1) });
                }
                faults
            },
            |_t| (nalgebra::Vector3::zeros(), nalgebra::Vector3::zeros()),
        )
        .expect("Scenario run failed");

    let pre_fault_point = &history[50]; // t = 0.255s
    let post_fault_point = history.last().unwrap(); // t = 1.0s

    assert_eq!(pre_fault_point.authority_mode, ControlAuthorityMode::NORMAL);
    assert_eq!(
        post_fault_point.authority_mode,
        ControlAuthorityMode::DEGRADED
    );

    // Motor 1 thrust must be zeroed out in actual physical actuator state
    assert!(
        post_fault_point.actual_actuators.motor_thrusts[0] < 0.1,
        "Failed motor 1 is still producing thrust: {}",
        post_fault_point.actual_actuators.motor_thrusts[0]
    );

    // The remaining motors took over the thrust load
    let avg_remaining_thrust: f64 = post_fault_point.actual_actuators.motor_thrusts[1..16]
        .iter()
        .sum::<f64>()
        / 15.0;
    assert!(
        avg_remaining_thrust > hover_thrust_per_motor,
        "Remaining motors did not increase thrust: avg = {}, nominal = {}",
        avg_remaining_thrust,
        hover_thrust_per_motor
    );
}
