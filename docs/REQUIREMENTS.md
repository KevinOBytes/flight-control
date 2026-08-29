# VectorFlight Requirements Specification

This document details the functional, non-functional, safety, and performance requirements for the VectorFlight control-allocation system.

## 1. System Scope & Actuation

The system controls a nested thrust-vectoring multirotor with 40 command variables:
- **16 Motor Thrust Commands:** Individual motors $i \in [1, 16]$.
- **16 Motor Tilt Commands:** Individual motor mechanical tilt angles $\gamma_i$ (1 DOF per motor).
- **8 Pod Tilt Commands:** Two gimbal axes ($\alpha_j, \beta_j$) for each of the 4 propulsion pods $j \in \{\text{FL}, \text{FR}, \text{RL}, \text{RR}\}$.

## 2. Coordinate System and Units

- **Coordinate Frame:** Vehicle-fixed Body frame must be right-handed **FRD** (+X forward, +Y right, +Z down).
- **Angular Convention:** Moments and angular positions follow the right-hand rule.
- **Internal Units:** System must use SI units exclusively:
  - Position: meters ($m$)
  - Force: Newtons ($N$)
  - Moment/Torque: Newton-meters ($N\cdot m$)
  - Angle: radians ($rad$)
  - Mass: kilograms ($kg$)
  - Time: seconds ($s$)

## 3. Mathematical & Kinematic Model

- **Rotation Composition:** The rotation from body frame to rotor frame $i$ must be:
  $$R_{\text{body}\to\text{rotor}} = R_{\text{body}\to\text{pod\_base}} \cdot R_{\text{pod\_axis\_1}}(\alpha) \cdot R_{\text{pod\_axis\_2}}(\beta) \cdot R_{\text{motor\_base}} \cdot R_{\text{motor\_tilt}}(\gamma)$$
- **Force and Moment:**
  - $F_i = T_i \cdot n_i$ where $n_i = R_{\text{body}\to\text{rotor}} \cdot [0, 0, -1]^T$ (lift acts in local $-Z$ direction).
  - $M_i = (r_i - r_{\text{COM}}) \times F_i + M_{\text{reaction}, i}$
  - $M_{\text{reaction}, i} = \text{spin\_sign} \cdot k_{\text{torque}} \cdot T_i \cdot n_i$ where $\text{spin\_sign} = 1.0$ for CW and $-1.0$ for CCW.
- **Wrench Forward Model:**
  $$W = \sum_{i=1}^{16} \begin{bmatrix} F_i \\ M_i \end{bmatrix}$$
- **Jacobian:** The local effectiveness Jacobian $J = \frac{\partial W}{\partial U}$ must be computed around the current actuator state.

## 4. Control Allocation & QP

- **Hierarchical Loop Rates:**
  - **Fast Thrust Allocation:** Solve for thrust deltas $\Delta f$ at **200 Hz** using a QP with tilt angles fixed.
  - **Motor Tilt Planner:** Update motor tilt setpoints at **50 Hz**.
  - **Pod Tilt Planner:** Update pod gimbal angle setpoints at **20 Hz**.
- **Optimization Priority:**
  - Simple thrust adjustments have the lowest movement penalty (highest bandwidth).
  - Motor tilt changes have medium penalty (medium bandwidth).
  - Pod tilt changes have the highest penalty (lowest bandwidth, gross direction shifts).
- **Execution Invariants:**
  - Preallocate matrices to prevent runtime heap churn.
  - Warm-start the OSQP solver using the prior cycle's solution.
  - Detect and safely handle: Success, Solved Inaccurate, Max Iterations, Infeasible, Numerical Failure, NaN/Inf inputs, and out-of-envelope requested wrenches.

## 5. Fault Management and Control Authority

- **Actuator Fault Matrix:** Support constraints for:
  - Motor Failed: thrust = 0.
  - Motor Degraded: upper thrust bound scaled down.
  - Motor Tilt Jammed: tilt angle fixed at known jammed position.
  - Motor Tilt Degraded: slew rate bound reduced.
  - Pod Axis Jammed: gimbal angle fixed.
  - Pod Axis Degraded: gimbal slew rate bound reduced.
  - ESC/Pod Bus Unavailable: all affected actuators disabled or frozen.
- **Authority Tracking:** Continuously calculate and report:
  - Matrix rank (rank $\ge 6$ is necessary but not sufficient for controllability).
  - Singular values, minimum singular value ($\sigma_{\min}$), and condition number ($\kappa$).
  - Actuator saturation state.
  - Thrust reserve.
- **Authority Modes:** Classify and transition between:
  - `NORMAL`
  - `DEGRADED`
  - `CRITICAL`
  - `UNCONTROLLABLE`

## 6. Safety Invariants (Non-negotiable)

1. **No NaN/Inf:** Ensure no NaN or Inf values are emitted from the allocator or model.
2. **Hard Actuator Bounds:** Actuator commands must never exceed configuration physical bounds.
3. **Actuator Slew Limits:** Actuator commands must respect dynamic rate limits.
4. **No Faulted Actuation:** Command zero thrust to any motor marked failed.
5. **No Jammed Actuation:** Never command motion to a joint marked jammed.
6. **Monotonic Clocks:** Use monotonic Linux clocks (`Instant::now()`) for control scheduling.
7. **Stale State Protection:** Trigger safe degradation or fallback if state estimate is stale.
8. **Stale Feedback Fault:** Trigger a fault if physical actuator feedback is lost.
9. **Solver Failure Fallback:** Provide defined fallback commands (e.g. hold last state or decay thrust) if the QP fails to solve.
10. **Emergency Stop (E-Stop):** Immediate zero-thrust override must bypass allocator logic.

## 7. Performance & Latency Targets

- **QP Solve Time:** $p_{99}$ execution time of the thrust QP must be $< 2.0\text{ ms}$ on the target SBC.
- **Deadline Detection:** The scheduler must explicitly track and log any missed deadlines.
