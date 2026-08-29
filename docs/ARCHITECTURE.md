# VectorFlight System Architecture

This document describes the modular crate layout, component boundaries, and data flow patterns of the VectorFlight control stack.

## 1. Crate Layout and Component Boundaries

The project is structured as a Cargo workspace to enforce strict dependency boundaries:

```text
                           ┌────────────────────────┐
                           │       vf-sim           │
                           │   (6-DOF Simulator)    │
                           └───────────┬────────────┘
                                       │
                                       ▼
 ┌───────────────────┐     ┌────────────────────────┐     ┌────────────────────┐
 │    vf-runtime     ├────►│      vf-allocator      │◄────┤     vf-faults      │
 │ (Scheduler Loop)  │     │      (QP Solver)       │     │ (Health/Authority) │
 └─────────┬─────────┘     └───────────▲────────────┘     └──────────┬─────────┘
           │                           │                             │
           ▼                           ▼                             ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                 vf-model                                    │
 │                    (Kinematics & Forward Wrench Model)                      │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                  vf-core                                    │
 │                       (Shared Units & Domain Types)                         │
 └──────────────────────────────────────▲──────────────────────────────────────┘
                                        │
                                        │ (Implements)
 ┌───────────────────┐     ┌────────────┴───────────┐
 │      vf-can       ├────►│         vf-hal         │
 │ (SocketCAN FD)    │     │   (Propulsion HAL)     │
 └───────────────────┘     └────────────────────────┘
```

### Component Details
1. **`vf-core` (Shared Types):** Low-level structures (`RotorId`, `PodId`, `BodyWrench`), spin directions, and common error definitions. Has zero dependencies.
2. **`vf-model` (Kinematics & Parameters):** Config file deserializer and kinematics/geometry solver. Computes the composed rotation and positions of rotors, and aggregates forces to body wrenches.
3. **`vf-allocator` (QP Core):** Formulates the OSQP representation matrices and translates optimization solutions.
4. **`vf-faults` (Degradation & Health):** Accumulates active actuator fault sets and recalculates rank, singular values, and authority status modes.
5. **`vf-runtime` (Loop Execution):** Synchronizes timing, checks loop deadlines, and routes inputs into `AllocationResult` structures.
6. **`vf-hal` (Hardware Interface):** Abstract interfaces and structs mapping logical command and feedback frames.
7. **`vf-can` (CAN FD Transport):** Serializes/deserializes CAN-FD frames. Targets SocketCAN (Linux-only), providing safe stub fallbacks on non-Linux targets.
8. **`vf-sim` (Plant Model):** Complete 6-DOF dynamic simulator. Used only for software-in-the-loop (SIL) validation.

---

## 2. Allocation Runtime Data Flow

Each synchronous cycle at 200 Hz follows this exact pipeline:

```text
 1. Receive desired BodyWrench request and actual measured ActuatorState
                             │
                             ▼
 2. Apply active faults from vf-faults to update lower/upper bounds
                             │
                             ▼
 3. Compute local effectiveness Jacobian around current actuator angles
                             │
                             ▼
 4. Formulate QP matrices (P, q, A, l, u) and feed to OSQP solver
                             │
                             ▼
 5. Check solver outputs (Success, Infeasible, Stale, etc.)
                             │
                             ▼
 6. Apply pre-output safety checks (NaN/Inf, Hard bounds check)
                             │
                             ▼
 7. Dispatch command frame to PropulsionHal interface (vf-hal/vf-can)
```

---

## 3. Real-Time Execution Philosophy

1. **No Async Rust in Critical Path:** The core allocation and kinematics path is synchronous, single-threaded, and deterministic to minimize jitter.
2. **No Heap Allocation:** Matrices are preallocated or stack-allocated. Dynamic sizing is bounded.
3. **Hardware Independence:** Crates like `vf-allocator` and `vf-model` are unaware of SocketCAN, networks, or SBC interfaces. They operate purely on mathematical vectors and configurations.
