//! Fault models and control authority tracking for VectorFlight.

use serde::{Deserialize, Serialize};
use vf_core::{PodAxis, PodId, RotorId};

/// Set of active faults representing degraded state of actuators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActuatorFault {
    MotorFailed {
        rotor: RotorId,
    },
    MotorDegraded {
        rotor: RotorId,
        max_thrust_fraction: f64,
    },
    MotorTiltJammed {
        rotor: RotorId,
        angle_rad: f64,
    },
    MotorTiltDegraded {
        rotor: RotorId,
        rate_fraction: f64,
    },
    PodAxisJammed {
        pod: PodId,
        axis: PodAxis,
        angle_rad: f64,
    },
    PodAxisDegraded {
        pod: PodId,
        axis: PodAxis,
        rate_fraction: f64,
    },
    EscUnavailable {
        rotor: RotorId,
    },
    PodBusUnavailable {
        pod: PodId,
    },
}

/// System-level classification of control authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ControlAuthorityMode {
    NORMAL,
    DEGRADED,
    CRITICAL,
    UNCONTROLLABLE,
}

/// Metrics used to determine overall control authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlAuthority {
    pub mode: ControlAuthorityMode,
    /// Rank of the actuator effectiveness matrix.
    pub rank: usize,
    /// Singular values of the effectiveness matrix.
    pub singular_values: Vec<f64>,
    /// Minimum singular value (measure of control margin).
    pub min_singular_value: f64,
    /// Ratio of max to min singular value.
    pub condition_number: f64,
    /// Remaining thrust margin available (Newtons).
    pub thrust_reserve_n: f64,
    /// Current fraction of actuators saturated.
    pub saturation_fraction: f64,
}

impl ControlAuthority {
    pub fn compute(
        model: &vf_model::VehicleModel,
        commands: &vf_model::ActuatorState,
        faults: &FaultSet,
        b_matrix: &nalgebra::SMatrix<f64, 6, 16>,
        min_sv_degraded: f64,
        min_sv_critical: f64,
    ) -> Self {
        // 1. Perform SVD on the 6x16 Jacobian matrix
        // We do not need U and V, so pass false, false
        let svd = b_matrix.svd(false, false);
        let s = svd.singular_values; // Vector6<f64>

        let mut singular_values = vec![0.0; 6];
        for i in 0..6 {
            singular_values[i] = s[i];
        }

        // 2. Compute rank based on tolerance
        let tol = 1e-5;
        let rank = s.iter().filter(|&&val| val > tol).count();

        // 3. Compute minimum singular value and condition number
        let min_singular_value = s[5];
        let max_singular_value = s[0];
        let condition_number = if min_singular_value > 1e-12 {
            max_singular_value / min_singular_value
        } else {
            f64::INFINITY
        };

        // 4. Compute remaining thrust reserve (sum of remaining capacity of healthy motors)
        let mut thrust_reserve_n = 0.0;
        let mut healthy_count = 0;
        let mut saturated_count = 0;

        for rotor in &model.rotors {
            let is_failed = faults.is_rotor_failed(rotor.id, rotor.pod_id);
            if !is_failed {
                healthy_count += 1;
                let limit_frac = faults.get_motor_thrust_limit_fraction(rotor.id);
                let max_thrust_effective = rotor.thrust_max_n * limit_frac;
                let thrust = commands.motor_thrusts[(rotor.id.0 as usize) - 1];

                let reserve = (max_thrust_effective - thrust).max(0.0);
                thrust_reserve_n += reserve;

                // Saturation check: is it close to bounds?
                let is_sat = (thrust - rotor.thrust_min_n).abs() < 1e-3
                    || (max_thrust_effective - thrust).abs() < 1e-3;
                if is_sat {
                    saturated_count += 1;
                }
            }
        }

        let saturation_fraction = if healthy_count > 0 {
            saturated_count as f64 / healthy_count as f64
        } else {
            0.0
        };

        // 5. Classify ControlAuthorityMode
        // Use threshold configurations:
        let mode = if rank < 6 || min_singular_value < min_sv_critical {
            ControlAuthorityMode::UNCONTROLLABLE
        } else if min_singular_value < min_sv_degraded {
            ControlAuthorityMode::CRITICAL
        } else if !faults.faults.is_empty() {
            ControlAuthorityMode::DEGRADED
        } else {
            ControlAuthorityMode::NORMAL
        };

        Self {
            mode,
            rank,
            singular_values,
            min_singular_value,
            condition_number,
            thrust_reserve_n,
            saturation_fraction,
        }
    }
}

/// Keeps track of the active fault set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FaultSet {
    pub faults: Vec<ActuatorFault>,
}

impl FaultSet {
    pub fn new() -> Self {
        Self { faults: Vec::new() }
    }

    pub fn insert(&mut self, fault: ActuatorFault) {
        self.faults.push(fault);
    }

    pub fn clear(&mut self) {
        self.faults.clear();
    }

    /// Check if a rotor has a complete failure.
    pub fn is_rotor_failed(&self, rotor: RotorId, pod: PodId) -> bool {
        self.faults.iter().any(|f| match f {
            ActuatorFault::MotorFailed { rotor: r } => *r == rotor,
            ActuatorFault::EscUnavailable { rotor: r } => *r == rotor,
            ActuatorFault::PodBusUnavailable { pod: p } => *p == pod,
            _ => false,
        })
    }

    /// Returns the fraction of maximum thrust a motor is degraded to (1.0 = healthy).
    pub fn get_motor_thrust_limit_fraction(&self, rotor: RotorId) -> f64 {
        let mut min_fraction: f64 = 1.0;
        for f in &self.faults {
            if let ActuatorFault::MotorDegraded {
                rotor: r,
                max_thrust_fraction,
            } = f
            {
                if *r == rotor {
                    min_fraction = min_fraction.min(*max_thrust_fraction);
                }
            }
        }
        min_fraction
    }

    /// Gets the jammed angle in radians for a motor tilt joint, if applicable.
    pub fn get_jammed_motor_tilt(&self, rotor: RotorId) -> Option<f64> {
        self.faults.iter().find_map(|f| match f {
            ActuatorFault::MotorTiltJammed {
                rotor: r,
                angle_rad,
            } if *r == rotor => Some(*angle_rad),
            _ => None,
        })
    }

    /// Gets the jammed angle in radians for a pod axis joint, if applicable.
    pub fn get_jammed_pod_tilt(&self, pod: PodId, axis: PodAxis) -> Option<f64> {
        self.faults.iter().find_map(|f| match f {
            ActuatorFault::PodAxisJammed {
                pod: p,
                axis: a,
                angle_rad,
            } if *p == pod && *a == axis => Some(*angle_rad),
            _ => None,
        })
    }
}
