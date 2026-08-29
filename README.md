# VectorFlight

VectorFlight is a research control-allocation stack for a fully actuated multirotor using nested thrust vectoring.

The v1 reference vehicle has four propulsion pods. Each pod carries four motors. Every motor has one independent tilt axis, and every pod has two additional tilt axes.

That produces:

```text
16 motor thrust commands
16 individual motor-tilt commands
 8 pod-tilt commands
--------------------------
40 actuator command variables
```

The allocator maps a desired body-frame wrench:

```text
[Fx, Fy, Fz, Mx, My, Mz]
```

into physically constrained actuator commands.

## Status

Research prototype.

This project is intended for simulation, controls research, geometry optimization, benchtop validation, and eventual experimental vehicle integration.

It is not an airworthy flight-control system and has not been certified for safety-critical operation.

## Why Rust + Python

Rust owns the runtime control path:

- vehicle geometry,
- kinematics,
- wrench modeling,
- QP allocation,
- fault constraints,
- HAL,
- CAN-FD adapter,
- runtime scheduling,
- diagnostics.

Python owns offline work:

- nonlinear optimization,
- geometry search,
- Jacobian cross-checks,
- Monte Carlo analysis,
- system identification,
- visualization.

The runtime does not depend on Python.

## Reference architecture

```text
desired 6-DOF wrench
        │
        ▼
┌─────────────────────┐
│ allocation runtime  │
└─────────────────────┘
        │
        ├───────────────┐
        ▼               ▼
 thrust QP         tilt planners
  ~200 Hz         motor ~50 Hz
                  pod   ~20 Hz
        │               │
        └───────┬───────┘
                ▼
        invariant checks
                │
                ▼
              HAL
                │
                ▼
        CAN-FD reference
                │
                ▼
     ESCs / tilt controllers
```

## Control hierarchy

The design intentionally avoids a single nonlinear 40-variable optimization in the real-time loop.

The preferred hierarchy is:

1. motor thrust,
2. individual motor tilt,
3. pod tilt.

Motor thrust handles fast corrections.

Individual motor tilt provides finer sustained vectoring and additional moment authority.

Pod tilt changes the gross direction of a complete four-motor propulsion module and is therefore treated as slower and more expensive motion.

The real-time allocator uses quadratic programming.

Offline Python tools may use nonlinear optimization to search vehicle geometry and validate operating envelopes.

## Reference vehicle

The initial model contains:

```text
Pod FL        Pod FR
M01 M02       M05 M06
M03 M04       M07 M08

       vehicle core

Pod RL        Pod RR
M09 M10       M13 M14
M11 M12       M15 M16
```

Geometry is configuration-driven.

The library does not assume a particular arm length, propeller size, rotor axis, pod axis, or mass property.

## Coordinate system

Default body frame:

- +X forward
- +Y right
- +Z down

The system is right-handed.

All internal units are SI.

All internal angles are radians.

## Core model

For rotor `i`:

```text
n_i = composed pod + motor thrust direction
F_i = T_i n_i
M_i = r_i × F_i + M_reaction_i
```

Total wrench:

```text
W = Σ [F_i, M_i]
```

Local control effectiveness is computed around the current actuator state and used by the allocator.

## Faults

The model supports faults as changes to actuator constraints:

- motor failed,
- motor degraded,
- ESC unavailable,
- individual tilt jammed,
- individual tilt degraded,
- pod axis jammed,
- pod axis degraded,
- pod communications unavailable.

The allocator recomputes control authority after faults.

## Control authority

A vehicle is not considered healthy simply because the effectiveness matrix has rank six.

VectorFlight tracks:

- rank,
- singular values,
- minimum singular value,
- condition number,
- actuator saturation,
- thrust reserve,
- fault state.

Runtime modes are:

```text
NORMAL
DEGRADED
CRITICAL
UNCONTROLLABLE
```

## HAL

The control core depends on an abstract propulsion HAL.

The initial reference implementation targets Linux SocketCAN CAN-FD.

The core library remains independent of CAN so later integrations can use another transport.

Potential future adapters include:

- PX4
- ArduPilot
- EtherCAT
- custom authenticated transport
- microcontroller bridge

## Repository layout

```text
.
├── AGENTS.md
├── README.md
├── Cargo.toml
├── configs/
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
└── docs/
    ├── REQUIREMENTS.md
    ├── FEATURES.md
    ├── DESIGN.md
    └── ARCHITECTURE.md
```

## Development checks

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
pytest
```

## Initial performance target

For the thrust-allocation QP on the development Linux SBC:

```text
p99 solve time < 2 ms
```

This is a research target, not a hard-real-time guarantee.

All deadline overruns must be recorded.

## Hardware integration policy

Do not start by flying the full aircraft.

The intended progression is:

```text
mathematical model
→ allocator tests
→ simulation
→ single-pod bench fixture
→ 6-axis load-cell characterization
→ hardware-in-loop
→ restrained vehicle test
→ experimental free flight
```

The single-pod fixture should empirically map:

```text
(thrusts, motor tilts, pod tilts)
        ↓
(Fx, Fy, Fz, Mx, My, Mz)
```

That empirical model can then be compared against the ideal rotor model.

## Documentation

See:

- `docs/REQUIREMENTS.md`
- `docs/FEATURES.md`
- `docs/DESIGN.md`
- `docs/ARCHITECTURE.md`

## License

Choose a license before external distribution.
