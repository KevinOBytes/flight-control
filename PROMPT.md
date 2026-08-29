# Coding Prompt: VectorFlight Control Library

Build a research-grade control-allocation and simulation library for a fully actuated, over-actuated multirotor aircraft with nested thrust vectoring.

## Mission

Implement a standalone control library for a Linux SBC that controls a 16-motor aircraft composed of four propulsion pods. Each pod contains four independently driven motors. Every motor has one mechanical tilt axis, and every pod has two mechanical tilt axes.

The library must accept a desired 6-DOF body wrench:

```text
[Fx, Fy, Fz, Mx, My, Mz]
```

and compute actuator commands for:

- 16 motor thrust commands
- 16 individual motor tilt commands
- 8 pod tilt commands, two axes per pod

The real-time allocator must use a QP-based approach. Nonlinear optimization is permitted only in offline Python tooling for geometry optimization, operating-point analysis, validation, and system identification.

The implementation is a research prototype. It must be engineered conservatively, with explicit invariants, fault handling, actuator constraints, deterministic interfaces, and aggressive simulation/testing. Do not represent it as airworthy, certified, or safe for human-carrying use.

## Primary language and libraries

### Rust
Use stable Rust.

Preferred libraries:

- `nalgebra` for matrices, vectors, rotations, and linear algebra.
- `osqp` for quadratic programming.
- `serde` / `serde_json` / `toml` for configuration and telemetry serialization.
- `thiserror` for structured errors.
- `tracing` and `tracing-subscriber` for diagnostics.
- `approx` for floating-point assertions.
- `proptest` for property-based tests.
- `crossbeam-channel` or `tokio` only where asynchronous IO is actually justified. Keep the control loop itself synchronous and deterministic.
- `socketcan` or an equivalent maintained Linux SocketCAN crate for the reference CAN-FD adapter.
- `clap` for CLI tools if a CLI is added.

Do not introduce a large robotics middleware dependency in v1.

### Python
Use Python 3.12+ for offline tooling.

Preferred libraries:

- `numpy`
- `scipy`
- `sympy`
- `matplotlib`
- `pandas`
- `pytest`

Optional only if justified:

- `jax` for automatic differentiation or large geometry searches.
- `casadi` for offline nonlinear optimization.

The Rust control library must not require Python at runtime.

## System architecture

Build the repository as a Cargo workspace. Use crates roughly equivalent to:

```text
vectorflight/
├── Cargo.toml
├── AGENTS.md
├── README.md
├── configs/
│   └── vehicle_v1.toml
├── crates/
│   ├── vf-core/
│   ├── vf-model/
│   ├── vf-allocator/
│   ├── vf-faults/
│   ├── vf-hal/
│   ├── vf-can/
│   ├── vf-runtime/
│   └── vf-sim/
├── python/
│   ├── pyproject.toml
│   ├── vectorflight_tools/
│   └── tests/
├── docs/
│   ├── REQUIREMENTS.md
│   ├── FEATURES.md
│   ├── DESIGN.md
│   └── ARCHITECTURE.md
└── tests/
```

Exact crate names may change if there is a strong reason, but preserve separation between:

1. math/types,
2. vehicle model and kinematics,
3. QP allocation,
4. fault state,
5. hardware abstraction,
6. CAN-FD transport,
7. runtime scheduling,
8. simulation.

## Coordinate conventions

Pick one coordinate convention and document it prominently.

Recommended:

- Body frame: right-handed FRD
  - +X forward
  - +Y right
  - +Z down
- Force and torque vectors use body-frame coordinates.
- Rotor thrust sign conventions must be explicit.
- All angles are radians internally.
- SI units only internally:
  - meters
  - seconds
  - kilograms
  - Newtons
  - Newton-meters
  - radians

Never mix conventions silently.

Create strong types or wrapper types for quantities where it reduces mistakes.

## Vehicle model

The reference v1 vehicle contains:

- 4 pods: FL, FR, RL, RR.
- 4 motors per pod.
- 16 motors total.
- 1 tilt DOF per motor.
- 2 tilt DOF per pod.
- 40 commanded actuator variables total:
  - 16 thrust variables
  - 16 motor tilt variables
  - 8 pod tilt variables

Represent geometry from configuration, not hard-coded constants.

Each rotor must have at minimum:

```rust
struct RotorGeometry {
    id: RotorId,
    pod_id: PodId,
    position_body_m: Vector3<f64>,
    base_orientation_body: UnitQuaternion<f64>,
    motor_tilt_axis_local: Unit<Vector3<f64>>,
    spin_direction: SpinDirection,
    thrust_min_n: f64,
    thrust_max_n: f64,
    thrust_rate_limit_n_s: f64,
    motor_tilt_min_rad: f64,
    motor_tilt_max_rad: f64,
    motor_tilt_rate_limit_rad_s: f64,
    torque_per_thrust_m: f64,
}
```

Each pod must have equivalent geometry, limits, rates, and two orthogonal or explicitly configured tilt axes.

Do not assume pod axes are Euler X/Y unless the configuration says so. Model rotations using rotation matrices or quaternions.

For rotor `i`, compute thrust direction from the composed transformation:

```text
R_body_to_rotor =
    R_body_to_pod_base
  * R_pod_axis_1(alpha)
  * R_pod_axis_2(beta)
  * R_motor_base
  * R_motor_tilt(gamma)
```

Then:

```text
F_i = thrust_i * n_i
M_i = r_i × F_i + reaction_torque_i
```

The total vehicle wrench is the sum of all rotor contributions.

Expose:

```rust
fn wrench_from_actuators(
    model: &VehicleModel,
    state: &ActuatorState
) -> BodyWrench;
```

## Differential effectiveness

The real-time allocator should operate around the current actuator state.

Implement the local actuator-effectiveness Jacobian:

```text
J = dW / dU
```

where actuator increments include whichever variables are enabled for the current allocation stage.

Do not derive a giant symbolic expression manually.

Use analytic derivatives where simple and well-tested. Otherwise provide numerically stable finite-difference derivatives with configurable perturbations.

Offline Python tooling should independently verify the Rust Jacobian.

Track:

- matrix rank,
- singular values,
- condition number,
- minimum singular value,
- controllability degradation after faults.

## Hierarchical allocation

Do not solve an unconstrained 40-variable nonlinear optimization in the real-time loop.

Use a hierarchical approach.

### Fast allocation

At every control tick, solve for thrust deltas using a QP with current tilt angles fixed.

Objective should approximately be:

```text
minimize
    ||W * (B * Δf - wrench_error)||²
  + λ_smooth ||Δf||²
  + λ_power  P_approx(f + Δf)
  + λ_center ||f - f_nominal||²
```

Subject to:

- thrust minimum/maximum
- thrust slew rate
- failed motor constraints
- reserved thrust margin
- power/current constraints where available

### Tilt allocation

Run tilt optimization at a slower rate.

Use the local Jacobian to choose motor tilt and pod tilt setpoints that:

- improve desired wrench tracking,
- improve control authority,
- reduce sustained motor thrust demand,
- reduce power,
- avoid singular or poorly conditioned configurations,
- preserve actuator margin,
- avoid unnecessary gimbal motion.

Generate QPs for local tilt increments where possible.

Pod motion must be penalized more heavily than individual motor tilt motion.

Individual motor tilt must be penalized more heavily than simple thrust changes unless a sustained condition makes vectoring beneficial.

Recommended hierarchy:

```text
thrust adjustment     highest bandwidth / lowest movement penalty
motor tilt            medium bandwidth / medium penalty
pod tilt              lowest bandwidth / highest penalty
```

Make frequencies configurable.

## QP formulation

Create a solver-neutral internal QP representation even though OSQP is the v1 backend.

Expose a model like:

```rust
pub struct QuadraticProgram {
    pub p: DMatrix<f64>,
    pub q: DVector<f64>,
    pub a: DMatrix<f64>,
    pub lower: DVector<f64>,
    pub upper: DVector<f64>,
}
```

Optimize allocation matrices to avoid heap churn in steady-state operation. Preallocate where practical.

Warm-start OSQP using the prior actuator solution.

Handle explicitly:

- solver success,
- solved inaccurate,
- max iteration,
- infeasible,
- numerical failure,
- NaN/Inf input,
- impossible requested wrench.

Never silently convert a failed solve into actuator output.

Provide fallback behavior.

## Fault model

Support at minimum:

```rust
enum ActuatorFault {
    MotorFailed { rotor: RotorId },
    MotorDegraded { rotor: RotorId, max_thrust_fraction: f64 },
    MotorTiltJammed { rotor: RotorId, angle_rad: f64 },
    MotorTiltDegraded { rotor: RotorId, rate_fraction: f64 },
    PodAxisJammed { pod: PodId, axis: PodAxis, angle_rad: f64 },
    PodAxisDegraded { pod: PodId, axis: PodAxis, rate_fraction: f64 },
    EscUnavailable { rotor: RotorId },
    PodBusUnavailable { pod: PodId },
}
```

Faults modify constraints and effectiveness; they must not be handled by ad-hoc special-case mixer code.

The allocator must recompute available control authority after a fault.

Define degraded modes:

- NORMAL
- DEGRADED
- CRITICAL
- UNCONTROLLABLE

The system must distinguish "can mathematically produce some wrench" from "has sufficient reserve to be considered controllable."

## HAL

Define traits independent of the CAN implementation.

Example:

```rust
trait PropulsionHal {
    fn read_feedback(&mut self) -> Result<ActuatorFeedbackFrame, HalError>;
    fn write_commands(&mut self, commands: &ActuatorCommandFrame)
        -> Result<(), HalError>;
    fn emergency_zero_thrust(&mut self) -> Result<(), HalError>;
}
```

Do not let SocketCAN types leak into core crates.

## CAN-FD reference adapter

Provide a Linux SocketCAN CAN-FD reference implementation.

Requirements:

- configurable interface name,
- message IDs defined in one place,
- monotonically increasing sequence number,
- timestamps,
- CRC or equivalent application-level integrity field where useful,
- actuator ID,
- command type,
- feedback type,
- explicit byte order,
- versioned protocol,
- stale-frame detection,
- range validation,
- duplicate detection where practical,
- bus health telemetry.

Do not implement security-by-obscurity. CAN is not an authenticated transport by default. Document this.

Design the HAL so a future authenticated transport can be substituted.

## Timing

The runtime must use Linux monotonic time.

Initial defaults:

- thrust allocation loop: 200 Hz
- motor tilt planner: 50 Hz
- pod tilt planner: 20 Hz
- telemetry: 20-50 Hz
- health monitoring: 50-100 Hz

These are defaults, not guaranteed final values.

Measure and expose:

- loop execution time,
- QP solve time,
- missed deadlines,
- worst-case solve time,
- solver iteration count.

Do not claim hard real-time behavior on ordinary Linux.

Avoid async scheduling inside the control algorithm.

## Safety invariants

At every runtime output boundary enforce:

1. no NaN or Inf values,
2. actuator commands within hard bounds,
3. actuator slew-rate bounds,
4. no thrust command to failed actuators,
5. jammed joints remain at their known angle,
6. timestamps are monotonic,
7. stale state estimates cause safe degradation,
8. stale actuator feedback raises a fault,
9. solver failure cannot propagate arbitrary output,
10. emergency stop always overrides allocator output.

Implement invariant checks both before and after allocation.

## Runtime input

The library does not own the navigation or attitude controller in v1.

Its primary runtime input is:

```rust
pub struct AllocationRequest {
    pub timestamp: Instant,
    pub desired_wrench_body: BodyWrench,
    pub measured_actuator_state: ActuatorState,
    pub electrical_limits: ElectricalLimits,
    pub active_faults: FaultSet,
}
```

Its output is:

```rust
pub struct AllocationResult {
    pub commands: ActuatorCommands,
    pub achieved_wrench_estimate: BodyWrench,
    pub residual_wrench: BodyWrench,
    pub solver_status: SolverStatus,
    pub authority: ControlAuthority,
    pub mode: AllocationMode,
    pub diagnostics: AllocationDiagnostics,
}
```

## Simulation

Implement a simulation environment sufficient to validate the allocator without claiming full aerodynamic fidelity.

Minimum model:

- rigid-body 6-DOF dynamics,
- mass and inertia tensor,
- gravity,
- rotor thrust lag,
- motor tilt actuator dynamics,
- pod tilt actuator dynamics,
- configurable actuator noise,
- actuator rate limits,
- simplified reaction torque,
- optional wind force.

Separate "allocator model" from "plant model" so tests can expose model mismatch.

Do not make the simulator use the exact same helper for every physical calculation as the controller; independent calculations are needed to catch errors.

## Python tooling

Implement tools to:

1. load vehicle geometry,
2. calculate exact wrench from actuator state,
3. numerically calculate Jacobians,
4. compare Python and Rust effectiveness matrices,
5. search motor-axis orientations,
6. search pod geometry,
7. calculate rank and singular values,
8. plot minimum singular value across tilt configurations,
9. run fault sweeps,
10. run Monte Carlo parameter perturbations,
11. plot actuator utilization,
12. plot wrench residual,
13. fit thrust-vs-RPM and reaction-torque coefficients from experimental data.

Provide a geometry optimizer capable of evaluating candidate motor tilt-axis orientations using metrics including:

```text
minimum singular value
condition number
available wrench volume approximation
vertical thrust efficiency
actuator saturation margin
faulted controllability
```

## Configuration

Use TOML or YAML; prefer TOML unless a reason exists.

A complete vehicle configuration must define:

- mass,
- inertia,
- center of mass,
- rotor positions,
- pod positions,
- rotor base orientations,
- pod base orientations,
- motor tilt axes,
- pod axes,
- thrust bounds,
- angular bounds,
- slew limits,
- reaction torque coefficients,
- allocator weights,
- loop rates,
- fault thresholds,
- CAN mappings.

Validate configuration on load.

Reject invalid configurations.

## Tests

Unit tests:

- rotation composition,
- cross-product moment calculation,
- reaction torque sign,
- CW/CCW conventions,
- Jacobian accuracy,
- bound enforcement,
- rate limiting,
- solver status handling,
- fault application,
- CAN codec,
- config validation.

Property tests:

- no valid command exceeds actuator bounds,
- failed actuator thrust remains zero,
- jammed joint remains fixed,
- finite valid inputs never produce NaN/Inf outputs,
- zero wrench around a valid equilibrium does not create arbitrary large actuator movement,
- symmetric vehicle geometry yields expected symmetric behavior.

Scenario tests:

- hover,
- pure roll,
- pure pitch,
- pure yaw,
- lateral force while holding attitude,
- vertical + lateral force,
- simultaneous 6-axis wrench,
- single motor loss,
- two nonadjacent motor losses,
- one motor tilt jam,
- one pod axis jam,
- one full pod unavailable,
- thrust saturation,
- impossible requested wrench,
- stale feedback,
- solver timeout/failure.

## Verification gates

Do not mark a feature complete merely because it compiles.

For allocator features require:

1. unit tests,
2. simulation scenario,
3. residual-wrench metric,
4. actuator-bound validation,
5. diagnostic output,
6. documentation update.

Add benchmark tests for allocation latency.

Record expected machine and OS details in benchmark output.

## Numerical acceptance targets

Treat these as initial research targets, configurable where appropriate:

- finite input -> finite output: 100%
- actuator hard-bound violations: 0
- failed-actuator command violations: 0
- Jacobian relative error vs independent finite difference: < 1e-4 in nonsingular nominal configurations
- hover wrench residual: < 1% of commanded vertical force
- nominal axis wrench residual: < 2% when command lies within authority envelope
- allocator p99 execution time target on development Linux SBC: < 2 ms for thrust QP
- control loop must explicitly detect any deadline overrun

If a target cannot be reached, document the measured result rather than weakening the test silently.

## Research instrumentation

Every allocation cycle should optionally expose:

- requested wrench,
- achieved wrench,
- residual,
- actuator state,
- command state,
- active constraints,
- saturation state,
- singular values,
- authority metric,
- solver iterations,
- solver time,
- mode,
- active faults.

Allow binary or compact logging later, but JSON/CSV export is sufficient for v1 tooling.

## Repository quality

Run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
pytest
```

No TODO placeholders in core allocation logic.

No `unwrap()` or `expect()` in runtime control paths unless an invariant makes failure literally impossible and the reason is documented.

Unsafe Rust is prohibited in v1 unless unavoidable for a hardware boundary; any unsafe block requires a written safety argument.

## Implementation phases

### Phase 1
Repository skeleton, units, geometry, configuration, wrench forward model.

### Phase 2
Thrust-only effectiveness matrix and QP allocator with fixed tilt.

### Phase 3
Fault constraints, authority metrics, degradation logic.

### Phase 4
Individual motor tilt local planner.

### Phase 5
Pod tilt local planner.

### Phase 6
6-DOF simulator, scenario runner, logging.

### Phase 7
Python geometry optimizer and independent validation.

### Phase 8
HAL and Linux CAN-FD adapter.

### Phase 9
Timing instrumentation, fault injection, benchmarks, integration tests.

Do not skip directly to hardware IO before simulation verification exists.

## Deliverables

The finished repository must contain:

- compiling Rust workspace,
- Python tooling,
- vehicle v1 example configuration,
- QP allocator,
- fault handling,
- simulator,
- CAN-FD reference HAL,
- CLI scenario runner,
- automated tests,
- benchmarks,
- README,
- REQUIREMENTS,
- FEATURES,
- DESIGN,
- ARCHITECTURE,
- reproducible example showing level-body lateral-force allocation.

The final report must state:

- what was implemented,
- what was tested,
- benchmark results,
- known limitations,
- assumptions,
- unsafe or unverified behavior,
- next engineering steps.

Do not claim flight readiness.
