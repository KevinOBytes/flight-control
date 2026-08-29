# AGENTS.md

## Purpose

This repository implements research-grade control allocation for a 16-motor, four-pod, nested thrust-vectoring multirotor.

The system is intentionally split into a deterministic Rust control core and Python offline analysis tooling.

The repository is a research prototype, not an airworthy or certified flight-control product.

## Non-negotiable engineering rules

1. Never bypass actuator bounds.
2. Never silently ignore a solver failure.
3. Never emit NaN or Inf actuator commands.
4. Never command nonzero thrust to an actuator marked failed.
5. Never move a joint marked jammed.
6. Never mix coordinate conventions.
7. Never use degrees internally.
8. Never make hardware transport types part of the control-core API.
9. Never introduce `unsafe` without a documented necessity and safety argument.
10. Never claim flight safety based only on simulation.

## Coordinate convention

Unless explicitly changed by an approved design update:

- body frame is right-handed FRD,
- +X forward,
- +Y right,
- +Z down,
- moments use the right-hand rule,
- SI units internally,
- radians internally.

Any code that crosses a boundary with a different convention must convert explicitly and test the conversion.

## Architecture boundaries

Maintain these conceptual layers:

```text
vehicle geometry / math
        ↓
forward wrench model
        ↓
effectiveness / Jacobian
        ↓
QP control allocation
        ↓
fault and authority logic
        ↓
HAL
        ↓
CAN-FD / hardware transport
```

The simulator and offline Python model must be able to exercise the allocator without hardware.

Do not couple the allocator to PX4, ArduPilot, ROS, CAN, or a specific SBC.

## Real-time philosophy

The control loop is synchronous.

Do not introduce async Rust into the control mathematics.

Async IO may exist outside the control loop, but data handed to the allocator must be a coherent snapshot.

Prefer:

- preallocation,
- bounded work,
- warm-started QPs,
- monotonic clocks,
- explicit deadlines,
- deterministic state transitions.

Ordinary Linux is not hard real time. Documentation and logs must say so.

## Allocator design

The primary real-time algorithm is QP-based.

Do not replace it with a generic nonlinear optimizer.

Use nonlinear optimization only in offline Python tooling.

The hierarchy is:

1. thrust allocation,
2. individual motor tilt,
3. pod tilt.

Do not make the pod gimbals chase high-frequency wrench noise.

Every allocator change must report:

- residual wrench,
- active constraints,
- solver status,
- authority metric,
- execution time.

## Fault handling

Faults are constraints, not mixer hacks.

Examples:

```text
failed motor -> thrust = 0
degraded motor -> reduced upper thrust bound
jammed tilt -> angle fixed
degraded tilt -> reduced rate bound
lost pod bus -> all affected actuators unavailable/frozen according to policy
```

Fault behavior must be testable without hardware.

## Control authority

Do not equate rank 6 with healthy controllability.

Track:

- rank,
- singular values,
- minimum singular value,
- condition number,
- actuator saturation,
- reserve thrust,
- fault state.

Use these to classify:

```text
NORMAL
DEGRADED
CRITICAL
UNCONTROLLABLE
```

Thresholds belong in configuration and must be documented.

## Rust guidelines

Use stable Rust.

Preferred crates:

- nalgebra
- osqp
- serde
- thiserror
- tracing
- approx
- proptest

Runtime paths should avoid:

- `unwrap()`,
- `expect()`,
- unbounded allocations,
- blocking IO,
- hidden global mutable state.

Errors must carry enough context for diagnosis.

Prefer explicit domain types over raw tuples.

## Python guidelines

Python is for:

- geometry search,
- nonlinear optimization,
- model validation,
- system identification,
- plots,
- Monte Carlo experiments.

Python must independently validate important Rust calculations.

Do not simply port Rust helper code line-for-line and call that independent verification.

## Testing requirements

Before marking allocator work complete:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
pytest
```

At minimum add:

- a unit test,
- a scenario test,
- bound validation,
- residual-wrench validation.

For geometry or Jacobian changes, compare against independent Python finite differences.

For fault handling, inject the fault in simulation.

## Benchmarks

Measure allocator execution time.

Report:

- CPU,
- OS/kernel,
- Rust version,
- build mode,
- scenario,
- p50,
- p95,
- p99,
- maximum observed time.

Never report a generic "real-time capable" result without measurements.

## Hardware work

Hardware integration may begin only after:

- forward model tests pass,
- QP allocation tests pass,
- simulator scenarios pass,
- emergency-stop behavior exists,
- stale-feedback behavior exists.

Default reference hardware boundary is Linux SocketCAN CAN-FD behind `vf-hal`.

CAN transport is not authenticated by default. Do not imply otherwise.

## Documentation discipline

Update docs when behavior changes.

`REQUIREMENTS.md` defines what must be true.

`FEATURES.md` defines intended capabilities and roadmap state.

`DESIGN.md` explains design decisions.

`ARCHITECTURE.md` explains component boundaries and data flow.

README is the user entrypoint, not the full technical specification.

## Safety language

Allowed:

- research prototype,
- simulator verified,
- tested under defined scenarios,
- control allocation prototype.

Do not say:

- flight safe,
- airworthy,
- certified,
- production safe,
- fail-safe,

unless a future verification program actually establishes those claims.
