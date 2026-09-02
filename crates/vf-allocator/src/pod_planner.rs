use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use vf_core::{BodyWrench, PodAxis, PodId, VfError};
use vf_faults::FaultSet;
use vf_model::{compute_pod_tilt_effectiveness, ActuatorState, VehicleModel};

pub struct OsqpPodTiltPlanner {
    settings: osqp::Settings,
    problem: Option<osqp::Problem>,
    last_p: Option<DMatrix<f64>>,
    wrench_weights: [f64; 6],
    lambda_smooth: f64,
    lambda_center: f64,
    alpha_lpf: f64,
    dt: f64,
    last_solution: [f64; 8],
    filtered_targets: [f64; 8],
    lpf_initialized: bool,
}

impl OsqpPodTiltPlanner {
    pub fn new(
        wrench_weights: [f64; 6],
        lambda_smooth: f64,
        lambda_center: f64,
        alpha_lpf: f64,
        dt: f64,
    ) -> Self {
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
            alpha_lpf: alpha_lpf.clamp(0.01, 1.0),
            dt,
            last_solution: [0.0; 8],
            filtered_targets: [0.0; 8],
            lpf_initialized: false,
        }
    }

    /// Formulates the QP for the pod tilt planner.
    /// Decision variables are the delta pod tilt angles: \Delta \theta \in \mathbb{R}^8.
    pub fn formulate(
        &self,
        model: &VehicleModel,
        desired_wrench: &BodyWrench,
        current_state: &ActuatorState,
        faults: &FaultSet,
    ) -> Result<crate::QuadraticProgram, VfError> {
        // 1. Compute the pod tilt effectiveness matrix J_pod (6x8)
        let j_pod = compute_pod_tilt_effectiveness(model, current_state)?;

        // 2. Setup weighting matrix W_w^2
        let mut w_diag = SMatrix::<f64, 6, 6>::zeros();
        for i in 0..6 {
            w_diag[(i, i)] = self.wrench_weights[i] * self.wrench_weights[i];
        }

        // 3. Compute P = 2 * (J_pod^T * W_w^2 * J_pod + (lambda_smooth + lambda_center) * I)
        let p_wrench = j_pod.transpose() * w_diag * j_pod;
        let mut p_mat = 2.0 * p_wrench;
        let diag_add = 2.0 * (self.lambda_smooth + self.lambda_center);
        for i in 0..8 {
            p_mat[(i, i)] += diag_add;
        }

        // 4. Compute wrench error: W_e = W_des - W_k
        let wk = vf_model::wrench_from_actuators(model, current_state)?;
        let e_wrench = desired_wrench.to_vector() - wk.to_vector();

        // 5. Compute linear cost q = -2 * J_pod^T * W_w^2 * e_wrench + 2 * lambda_center * (\theta_k - \theta_nominal)
        let q_wrench = -2.0 * j_pod.transpose() * w_diag * e_wrench;
        let mut q_center = SVector::<f64, 8>::zeros();
        for i in 0..8 {
            // \theta_nominal = 0.0 radians
            q_center[i] = 2.0 * self.lambda_center * current_state.pod_tilts[i];
        }
        let q_vec = q_wrench + q_center;

        // 6. Constraints box A = I (8x8)
        let a_mat = SMatrix::<f64, 8, 8>::identity();

        // 7. Calculate bounds lower/upper box limits
        let mut lower = SVector::<f64, 8>::zeros();
        let mut upper = SVector::<f64, 8>::zeros();

        let pod_axes = [
            (PodId::FL, PodAxis::Axis1, 0),
            (PodId::FL, PodAxis::Axis2, 1),
            (PodId::FR, PodAxis::Axis1, 2),
            (PodId::FR, PodAxis::Axis2, 3),
            (PodId::RL, PodAxis::Axis1, 4),
            (PodId::RL, PodAxis::Axis2, 5),
            (PodId::RR, PodAxis::Axis1, 6),
            (PodId::RR, PodAxis::Axis2, 7),
        ];

        for (pod_id, axis, idx) in pod_axes {
            let pod = model
                .pods
                .get(&pod_id)
                .ok_or_else(|| VfError::InvalidValue(format!("Pod {:?} not found", pod_id)))?;

            let theta_k = current_state.pod_tilts[idx];

            let (min_rad, max_rad, rate_limit) = match axis {
                PodAxis::Axis1 => (
                    pod.axis_1_min_rad,
                    pod.axis_1_max_rad,
                    pod.axis_1_rate_limit_rad_s,
                ),
                PodAxis::Axis2 => (
                    pod.axis_2_min_rad,
                    pod.axis_2_max_rad,
                    pod.axis_2_rate_limit_rad_s,
                ),
            };

            let max_dtheta = rate_limit * self.dt;

            if let Some(jammed_angle) = faults.get_jammed_pod_tilt(pod_id, axis) {
                let dtheta_jammed = jammed_angle - theta_k;
                lower[idx] = dtheta_jammed;
                upper[idx] = dtheta_jammed;
            } else if faults.is_pod_bus_failed(pod_id) {
                // Bus failed: joint is frozen/unavailable
                lower[idx] = 0.0;
                upper[idx] = 0.0;
            } else {
                let l = (min_rad - theta_k).max(-max_dtheta);
                let u = (max_rad - theta_k).min(max_dtheta);
                if l > u {
                    if min_rad - theta_k > 0.0 {
                        let val = max_dtheta.min(min_rad - theta_k);
                        lower[idx] = val;
                        upper[idx] = val;
                    } else {
                        let val = (-max_dtheta).max(max_rad - theta_k);
                        lower[idx] = val;
                        upper[idx] = val;
                    }
                } else {
                    lower[idx] = l;
                    upper[idx] = u;
                }
            }
        }

        Ok(crate::QuadraticProgram {
            p: DMatrix::from_column_slice(8, 8, p_mat.as_slice()),
            q: DVector::from_column_slice(q_vec.as_slice()),
            a: DMatrix::from_column_slice(8, 8, a_mat.as_slice()),
            lower: DVector::from_column_slice(lower.as_slice()),
            upper: DVector::from_column_slice(upper.as_slice()),
        })
    }

    /// Solves the planner QP and returns the planned delta pod tilt commands.
    pub fn solve(
        &mut self,
        qp: &crate::QuadraticProgram,
        warm_start: Option<&[f64; 8]>,
    ) -> Result<[f64; 8], VfError> {
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
            for (i, &val) in sol_x.iter().enumerate().take(8) {
                self.last_solution[i] = val;
            }
            Ok(self.last_solution)
        } else {
            Err(VfError::SolverFailure(format!(
                "OSQP pod planner failed to solve: status = {:?}",
                result
            )))
        }
    }

    /// Applies 1st-order low-pass filtering to raw target pod tilt angles:
    /// \theta_{filtered} = (1 - \alpha) \theta_{filtered} + \alpha \theta_{raw}
    pub fn filter_targets(&mut self, raw_targets: &[f64; 8]) -> [f64; 8] {
        if !self.lpf_initialized {
            self.filtered_targets = *raw_targets;
            self.lpf_initialized = true;
            return self.filtered_targets;
        }

        for (i, &raw) in raw_targets.iter().enumerate() {
            self.filtered_targets[i] =
                (1.0 - self.alpha_lpf) * self.filtered_targets[i] + self.alpha_lpf * raw;
        }

        self.filtered_targets
    }

    /// Resets the low pass filter state with given initial values.
    pub fn reset_lpf(&mut self, initial_targets: &[f64; 8]) {
        self.filtered_targets = *initial_targets;
        self.lpf_initialized = true;
    }
}
