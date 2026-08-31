use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use vf_core::{BodyWrench, VfError};
use vf_faults::FaultSet;
use vf_model::{compute_motor_tilt_effectiveness, ActuatorState, VehicleModel};

pub struct OsqpMotorTiltPlanner {
    settings: osqp::Settings,
    problem: Option<osqp::Problem>,
    last_p: Option<DMatrix<f64>>,
    wrench_weights: [f64; 6],
    lambda_smooth: f64,
    lambda_center: f64,
    dt: f64,
    last_solution: [f64; 16],
}

impl OsqpMotorTiltPlanner {
    pub fn new(wrench_weights: [f64; 6], lambda_smooth: f64, lambda_center: f64, dt: f64) -> Self {
        let mut settings = osqp::Settings::default();
        settings = settings
            .verbose(false)
            .warm_start(true)
            .eps_abs(1e-5)
            .eps_rel(1e-5)
            .max_iter(500);

        Self {
            settings,
            problem: None,
            last_p: None,
            wrench_weights,
            lambda_smooth,
            lambda_center,
            dt,
            last_solution: [0.0; 16],
        }
    }

    /// Formulates the QP for the motor tilt planner.
    /// Decision variables are the delta tilt angles: \Delta \gamma \in \mathbb{R}^{16}.
    pub fn formulate(
        &self,
        model: &VehicleModel,
        desired_wrench: &BodyWrench,
        current_state: &ActuatorState,
        faults: &FaultSet,
    ) -> Result<crate::QuadraticProgram, VfError> {
        // 1. Compute the motor tilt effectiveness matrix J_gamma (6x16)
        let j_gamma = compute_motor_tilt_effectiveness(model, current_state)?;

        // 2. Setup weighting matrix W_w^2
        let mut w_diag = SMatrix::<f64, 6, 6>::zeros();
        for i in 0..6 {
            w_diag[(i, i)] = self.wrench_weights[i] * self.wrench_weights[i];
        }

        // 3. Compute P = 2 * (J_gamma^T * W_w^2 * J_gamma + (lambda_smooth + lambda_center) * I)
        let p_wrench = j_gamma.transpose() * w_diag * j_gamma;
        let mut p_mat = 2.0 * p_wrench;
        let diag_add = 2.0 * (self.lambda_smooth + self.lambda_center);
        for i in 0..16 {
            p_mat[(i, i)] += diag_add;
        }

        // 4. Compute wrench error: W_e = W_des - W_k
        let wk = vf_model::wrench_from_actuators(model, current_state)?;
        let e_wrench = desired_wrench.to_vector() - wk.to_vector();

        // 5. Compute linear cost q = -2 * J_gamma^T * W_w^2 * e_wrench + 2 * lambda_center * (\gamma_k - \gamma_nominal)
        let q_wrench = -2.0 * j_gamma.transpose() * w_diag * e_wrench;
        let mut q_center = SVector::<f64, 16>::zeros();
        for i in 0..16 {
            // \gamma_nominal = 0.0 radians
            q_center[i] = 2.0 * self.lambda_center * current_state.motor_tilts[i];
        }
        let q_vec = q_wrench + q_center;

        // 6. Constraints box A = I (16x16)
        let a_mat = SMatrix::<f64, 16, 16>::identity();

        // 7. Calculate bounds lower/upper box limits
        let mut lower = SVector::<f64, 16>::zeros();
        let mut upper = SVector::<f64, 16>::zeros();
        for (i, rotor) in model.rotors.iter().enumerate() {
            let gamma_k = current_state.motor_tilts[i];
            let max_dgamma = rotor.motor_tilt_rate_limit_rad_s * self.dt;

            if let Some(jammed_angle) = faults.get_jammed_motor_tilt(rotor.id) {
                // Joint is jammed. Delta must lead exactly to the jammed angle
                let dgamma_jammed = jammed_angle - gamma_k;
                lower[i] = dgamma_jammed;
                upper[i] = dgamma_jammed;
            } else {
                lower[i] = (rotor.motor_tilt_min_rad - gamma_k).max(-max_dgamma);
                upper[i] = (rotor.motor_tilt_max_rad - gamma_k).min(max_dgamma);
            }
        }

        Ok(crate::QuadraticProgram {
            p: DMatrix::from_column_slice(16, 16, p_mat.as_slice()),
            q: DVector::from_column_slice(q_vec.as_slice()),
            a: DMatrix::from_column_slice(16, 16, a_mat.as_slice()),
            lower: DVector::from_column_slice(lower.as_slice()),
            upper: DVector::from_column_slice(upper.as_slice()),
        })
    }

    /// Solves the planner QP and returns the planned tilt delta commands.
    pub fn solve(
        &mut self,
        qp: &crate::QuadraticProgram,
        warm_start: Option<&[f64; 16]>,
    ) -> Result<[f64; 16], VfError> {
        let csc_p = crate::csc::convert_to_csc_upper_tri(&qp.p);
        let csc_a = crate::csc::convert_to_csc(&qp.a);

        let must_init = self.problem.is_none() || self.last_p.as_ref() != Some(&qp.p);

        if must_init {
            self.last_p = Some(qp.p.clone());
            let problem = osqp::Problem::new(
                csc_p,
                qp.q.as_slice(),
                csc_a,
                qp.lower.as_slice(),
                qp.upper.as_slice(),
                &self.settings,
            )
            .map_err(|e| VfError::SolverFailure(format!("Failed to initialize OSQP: {:?}", e)))?;
            self.problem = Some(problem);
        } else {
            let problem = self.problem.as_mut().unwrap();
            problem.update_lin_cost(qp.q.as_slice());
            problem.update_bounds(qp.lower.as_slice(), qp.upper.as_slice());
        }

        let problem = self.problem.as_mut().unwrap();

        if let Some(ws) = warm_start {
            problem.warm_start(ws, &self.last_solution);
        } else {
            problem.warm_start(&self.last_solution, &self.last_solution);
        }

        let result = problem.solve();

        if let osqp::Status::Solved(ref sol)
        | osqp::Status::SolvedInaccurate(ref sol)
        | osqp::Status::MaxIterationsReached(ref sol) = result
        {
            let sol_x = sol.x();
            for (i, &val) in sol_x.iter().enumerate().take(16) {
                self.last_solution[i] = val;
            }
            Ok(self.last_solution)
        } else {
            Err(VfError::SolverFailure(format!(
                "OSQP tilt planner failed to solve: status = {:?}",
                result
            )))
        }
    }
}
