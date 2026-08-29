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
    pub fn is_rotor_failed(&self, rotor: RotorId) -> bool {
        self.faults.iter().any(|f| match f {
            ActuatorFault::MotorFailed { rotor: r } => *r == rotor,
            ActuatorFault::EscUnavailable { rotor: r } => *r == rotor,
            ActuatorFault::PodBusUnavailable { pod: _ } => {
                // If we knew the pod map, we could check here. Let's make sure the caller handles it.
                false
            }
            _ => false,
        })
    }
}
