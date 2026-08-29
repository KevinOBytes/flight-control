pub mod csc;

use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use vf_core::{BodyWrench, VfError};
use vf_model::{compute_thrust_effectiveness, wrench_from_actuators, ActuatorState, VehicleModel};

/// Solver-neutral internal Quadratic Program representation.
/// Minimizes: 1/2 * x^T * P * x + q^T * x
/// Subject to: lower <= A * x <= upper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticProgram {
    pub p: DMatrix<f64>,
    pub q: DVector<f64>,
    pub a: DMatrix<f64>,
    pub lower: DVector<f64>,
    pub upper: DVector<f64>,
}

/// Status of the QP solver output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverStatus {
    Success,
    SolvedInaccurate,
    MaxIterationsReached,
    Infeasible,
    NumericalFailure,
    InvalidInput,
    UnknownFailure,
}

/// Diagnostic information about the allocation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationDiagnostics {
    pub solve_time_us: u64,
    pub iterations: u32,
    pub active_constraints_count: u32,
}

/// Control allocator trait defining the real-time allocator interface.
pub trait ControlAllocator {
    /// Formulates the QP for the current control cycle.
    fn formulate(
        &mut self,
        model: &VehicleModel,
        desired_wrench: &BodyWrench,
        current_state: &ActuatorState,
    ) -> Result<QuadraticProgram, VfError>;

    /// Solves the QP and returns the actuator state update.
    fn solve(
        &mut self,
        qp: &QuadraticProgram,
        warm_start: Option<&ActuatorState>,
    ) -> Result<(ActuatorState, SolverStatus, AllocationDiagnostics), VfError>;
}

/// OSQP-based real-time thrust-only allocator.
pub struct OsqpThrustAllocator {
    settings: osqp::Settings,
    problem: Option<osqp::Problem>,
    last_p: Option<DMatrix<f64>>,
    wrench_weights: [f64; 6],
    lambda_smooth: f64,
    lambda_center: f64,
    f_nominal: [f64; 16],
    dt: f64,
    last_solution: [f64; 16],
}

impl OsqpThrustAllocator {
    /// Creates a new OsqpThrustAllocator.
    pub fn new(
        wrench_weights: [f64; 6],
        lambda_smooth: f64,
        lambda_center: f64,
        f_nominal: [f64; 16],
        dt: f64,
    ) -> Self {
        let mut settings = osqp::Settings::default();
        settings = settings.verbose(false); // Disable solver stdout spam
        settings = settings.eps_abs(1e-4).eps_rel(1e-4);

        Self {
            settings,
            problem: None,
            last_p: None,
            wrench_weights,
            lambda_smooth,
            lambda_center,
            f_nominal,
            dt,
            last_solution: [0.0; 16],
        }
    }
}

impl ControlAllocator for OsqpThrustAllocator {
    fn formulate(
        &mut self,
        model: &VehicleModel,
        desired_wrench: &BodyWrench,
        current_state: &ActuatorState,
    ) -> Result<QuadraticProgram, VfError> {
        // 1. Compute effectiveness matrix B (6x16)
        let b = compute_thrust_effectiveness(model, current_state)?;

        // 2. Setup weighting matrix W_w^2
        let mut w_diag = SMatrix::<f64, 6, 6>::zeros();
        for i in 0..6 {
            w_diag[(i, i)] = self.wrench_weights[i] * self.wrench_weights[i];
        }

        // 3. Compute P = 2 * (B^T * W_w^2 * B + (lambda_smooth + lambda_center) * I)
        let p_wrench = b.transpose() * w_diag * b;
        let mut p_mat = 2.0 * p_wrench;
        let diag_add = 2.0 * (self.lambda_smooth + self.lambda_center);
        for i in 0..16 {
            p_mat[(i, i)] += diag_add;
        }

        // 4. Compute e_wrench = W_des - W_k
        let wk = wrench_from_actuators(model, current_state)?;
        let e_wrench = desired_wrench.to_vector() - wk.to_vector();

        // 5. Compute linear cost q = -2 * B^T * W_w^2 * e_wrench + 2 * lambda_center * (f_k - f_nom)
        let q_wrench = -2.0 * b.transpose() * w_diag * e_wrench;
        let mut q_center = SVector::<f64, 16>::zeros();
        for i in 0..16 {
            q_center[i] =
                2.0 * self.lambda_center * (current_state.motor_thrusts[i] - self.f_nominal[i]);
        }
        let q_vec = q_wrench + q_center;

        // 6. Setup constraints box A = I (16x16)
        let a_mat = SMatrix::<f64, 16, 16>::identity();

        // 7. Calculate bounds lower/upper box limits
        let mut lower = SVector::<f64, 16>::zeros();
        let mut upper = SVector::<f64, 16>::zeros();
        for (i, rotor) in model.rotors.iter().enumerate() {
            let fk_i = current_state.motor_thrusts[i];
            let max_df = rotor.thrust_rate_limit_n_s * self.dt;

            lower[i] = (rotor.thrust_min_n - fk_i).max(-max_df);
            upper[i] = (rotor.thrust_max_n - fk_i).min(max_df);
        }

        Ok(QuadraticProgram {
            p: DMatrix::from_column_slice(16, 16, p_mat.as_slice()),
            q: DVector::from_column_slice(q_vec.as_slice()),
            a: DMatrix::from_column_slice(16, 16, a_mat.as_slice()),
            lower: DVector::from_column_slice(lower.as_slice()),
            upper: DVector::from_column_slice(upper.as_slice()),
        })
    }

    fn solve(
        &mut self,
        qp: &QuadraticProgram,
        warm_start: Option<&ActuatorState>,
    ) -> Result<(ActuatorState, SolverStatus, AllocationDiagnostics), VfError> {
        // Enforce no NaN/Inf input
        if !qp.p.iter().all(|&x| x.is_finite())
            || !qp.q.iter().all(|&x| x.is_finite())
            || !qp.lower.iter().all(|&x| x.is_finite())
            || !qp.upper.iter().all(|&x| x.is_finite())
        {
            return Err(VfError::InvalidValue(
                "QP input contains non-finite values (NaN or Inf)".to_string(),
            ));
        }

        let start_time = Instant::now();

        // 1. Determine if we need to rebuild the OSQP Problem workspace
        let is_p_same = match &self.last_p {
            Some(last_p) => last_p == &qp.p,
            None => false,
        };

        if !is_p_same || self.problem.is_none() {
            let csc_p = csc::convert_to_csc_upper_tri(&qp.p);
            let csc_a = csc::convert_to_csc(&qp.a);
            let prob = osqp::Problem::new(
                csc_p,
                qp.q.as_slice(),
                csc_a,
                qp.lower.as_slice(),
                qp.upper.as_slice(),
                &self.settings,
            )
            .map_err(|e| {
                VfError::SolverFailure(format!("Failed to build OSQP problem: {:?}", e))
            })?;
            self.problem = Some(prob);
            self.last_p = Some(qp.p.clone());
        } else if let Some(ref mut prob) = self.problem {
            // Warm-start: update linear cost and bounds
            prob.update_lin_cost(qp.q.as_slice());
            prob.update_bounds(qp.lower.as_slice(), qp.upper.as_slice());
        }

        // 2. Warm start solution values
        let prob = self
            .problem
            .as_mut()
            .ok_or_else(|| VfError::SolverFailure("OSQP problem uninitialized".to_string()))?;
        prob.warm_start(
            &self.last_solution,
            // Pass zeros for dual variables
            &[0.0; 16],
        );

        // 3. Solve the QP
        let result = prob.solve();
        let solve_time_us = start_time.elapsed().as_micros() as u64;

        // 4. Handle solve status
        let status = match result {
            osqp::Status::Solved(_) => SolverStatus::Success,
            osqp::Status::SolvedInaccurate(_) => SolverStatus::SolvedInaccurate,
            osqp::Status::MaxIterationsReached(_) => SolverStatus::MaxIterationsReached,
            osqp::Status::PrimalInfeasible(_) | osqp::Status::PrimalInfeasibleInaccurate(_) => {
                SolverStatus::Infeasible
            }
            osqp::Status::DualInfeasible(_) | osqp::Status::DualInfeasibleInaccurate(_) => {
                SolverStatus::Infeasible
            }
            _ => SolverStatus::UnknownFailure,
        };

        let iter_count = result.iter();

        let diagnostics = AllocationDiagnostics {
            solve_time_us,
            iterations: iter_count,
            active_constraints_count: 0, // OSQP doesn't expose active constraints count easily, default to 0
        };

        // 5. Extract output and apply safety invariants
        if status == SolverStatus::Success
            || status == SolverStatus::SolvedInaccurate
            || status == SolverStatus::MaxIterationsReached
        {
            let sol_x = match &result {
                osqp::Status::Solved(sol) => Some(sol.x()),
                osqp::Status::SolvedInaccurate(sol) => Some(sol.x()),
                osqp::Status::MaxIterationsReached(sol) => Some(sol.x()),
                _ => None,
            }
            .ok_or_else(|| {
                VfError::SolverFailure("OSQP succeeded but solution vector is empty".to_string())
            })?;

            // Copy to last solution cache
            for (i, &val) in sol_x.iter().enumerate().take(16) {
                self.last_solution[i] = val;
            }

            // The decision variable is the delta thrust \Delta f.
            // We construct the new ActuatorState commands by adding the delta.
            // In Phase 2: tilts are held fixed, so they are copied from warm start.
            let mut commands = ActuatorState::zero();

            if let Some(ws) = warm_start {
                commands.motor_tilts = ws.motor_tilts;
                commands.pod_tilts = ws.pod_tilts;
                for i in 0..16 {
                    let cmd_val = ws.motor_thrusts[i] + self.last_solution[i];
                    // Apply absolute non-negativity constraint clamping just in case
                    commands.motor_thrusts[i] = cmd_val.max(0.0);
                }
            } else {
                for i in 0..16 {
                    commands.motor_thrusts[i] = self.last_solution[i].max(0.0);
                }
            }

            // Safety invariant check: assert no NaN or Inf values
            if !commands.motor_thrusts.iter().all(|&x| x.is_finite())
                || !commands.motor_tilts.iter().all(|&x| x.is_finite())
                || !commands.pod_tilts.iter().all(|&x| x.is_finite())
            {
                return Err(VfError::InvalidValue(
                    "Actuator commands contain non-finite values (NaN or Inf)".to_string(),
                ));
            }

            Ok((commands, status, diagnostics))
        } else {
            // Solver failed. Return current state (delta = 0) to avoid sending erratic commands.
            Err(VfError::SolverFailure(format!(
                "OSQP solver failed to find solution: status = {:?}",
                status
            )))
        }
    }
}
