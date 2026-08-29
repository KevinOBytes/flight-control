# VectorFlight Features and Development Roadmap

This document outlines the current features and planned development roadmap for the VectorFlight control-allocation stack.

## Roadmap & Implementation Phases

The project progresses through 9 distinct phases:

### Phase 1: Skeleton & Forward Model (Completed)
- [x] Multi-crate Cargo workspace structure.
- [x] Symmetric 16-motor multirotor geometry configuration parser.
- [x] Composed pod-to-rotor rotation matrices/quaternions.
- [x] Forward wrench mapping (`wrench_from_actuators`) using rigid body kinematics.
- [x] Python packaging and offline workspace initialization.

### Phase 2: Thrust-Only QP Allocation (Completed)
- [x] Local Jacobian matrix solver ($J = \frac{\partial W}{\partial f}$ with fixed tilts).
- [x] OSQP backend interface wrapper.
- [x] Thrust QP allocation: solving for $\Delta f$ to track desired body wrench.
- [x] Slew-rate and physical bounds constraints.

### Phase 3: Fault constraints & Authority Tracking (Current Phase)
- [ ] Injection and propagation of actuator faults (failed motors, jammed tilts).
- [ ] Recomputation of local effectiveness Jacobian under faults.
- [ ] Computation of singular values, condition numbers, and rank.
- [ ] Control authority status transitions (`NORMAL`, `DEGRADED`, `CRITICAL`, `UNCONTROLLABLE`).

### Phase 4: Individual Motor Tilt Planner
- [ ] Motor tilt axis rate-limited optimization loop.
- [ ] Allocation of tilt angles for sustained vectoring.
- [ ] Dynamic bounds update for thrust QP based on tilt rates.

### Phase 5: Propulsion Pod Tilt Planner
- [ ] Pod-tilt optimization loop (gross direction changes).
- [ ] Low-pass gimbal filtering to prevent gimbal hunting on high-frequency noise.
- [ ] Pod tilt rate and range bounds enforcement.

### Phase 6: Rigid-Body 6-DOF Simulator
- [ ] 6-DOF plant dynamics numerical integrator.
- [ ] First-order lag model for ESC thrust and servo tilt response.
- [ ] Sim-side actuator noise and wind disturbance models.
- [ ] Scriptable scenario runner for automated flight profile tests (hover, yaw/pitch/roll, lateral translation).

### Phase 7: Python Validation & Geometry Search
- [ ] Independent Python geometry representation and forward model.
- [ ] Numerical finite difference Jacobian calculation.
- [ ] Automated cross-check of Python vs Rust Jacobians.
- [ ] Pod/motor tilt orientation optimizer to maximize minimum singular values.

### Phase 8: HAL & CAN-FD Adapter
- [ ] Hardware-independent Propulsion HAL trait.
- [ ] Linux SocketCAN CAN-FD adapter implementation.
- [ ] Timestamps, monotonically increasing sequence numbers, and byte-order serialization.
- [ ] CRC/integrity fields and stale-frame detection.

### Phase 9: Latency Benchmarks & Hardening
- [ ] Real-time control loop jitter measurements.
- [ ] $p_{50}, p_{95}, p_{99}$ latency metrics log export.
- [ ] Stress-testing via random fault injection.
- [ ] Property-based testing with `proptest`.
