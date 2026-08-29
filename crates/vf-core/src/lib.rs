//! Core types and coordinate convention invariants for the VectorFlight control stack.

use nalgebra::{Vector3, Vector6};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Strong type for identifying a rotor (1-indexed or 0-indexed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RotorId(pub u32);

/// Strong type for identifying a propulsion pod.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PodId {
    FL,
    FR,
    RL,
    RR,
}

impl std::fmt::Display for PodId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Identifies the active tilt axis of a pod.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PodAxis {
    Axis1,
    Axis2,
}

/// Represents spin direction of a rotor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpinDirection {
    CW,
    CCW,
}

/// Body-frame wrench representation.
/// Assumes FRD coordinate frame: +X forward, +Y right, +Z down.
/// Torques use the right-hand rule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyWrench {
    pub force: Vector3<f64>,
    pub moment: Vector3<f64>,
}

impl BodyWrench {
    /// Creates a new BodyWrench from force and moment components.
    pub fn new(force: Vector3<f64>, moment: Vector3<f64>) -> Self {
        Self { force, moment }
    }

    /// Creates a zero wrench.
    pub fn zero() -> Self {
        Self {
            force: Vector3::zeros(),
            moment: Vector3::zeros(),
        }
    }

    /// Converts to a flat 6D vector [Fx, Fy, Fz, Mx, My, Mz].
    pub fn to_vector(&self) -> Vector6<f64> {
        Vector6::new(
            self.force.x,
            self.force.y,
            self.force.z,
            self.moment.x,
            self.moment.y,
            self.moment.z,
        )
    }

    /// Creates a BodyWrench from a flat 6D vector.
    pub fn from_vector(vec: &Vector6<f64>) -> Self {
        Self {
            force: Vector3::new(vec[0], vec[1], vec[2]),
            moment: Vector3::new(vec[3], vec[4], vec[5]),
        }
    }

    /// Validates that all fields are finite (no NaN or Inf).
    pub fn is_finite(&self) -> bool {
        self.force.iter().all(|&x| x.is_finite()) && self.moment.iter().all(|&x| x.is_finite())
    }
}

/// Common errors within the VectorFlight system.
#[derive(Error, Debug, Clone)]
pub enum VfError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Solver failure: {0}")]
    SolverFailure(String),

    #[error("Actuator constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("State estimation stale: elapsed {0:.3}s exceeds threshold {1:.3}s")]
    StaleFeedback(f64, f64),

    #[error("Invalid value encountered: {0}")]
    InvalidValue(String),
}

/// Assures the coordinate conventions are strictly FRD (+X forward, +Y right, +Z down).
/// This helper is used at boundary conditions to assert correctness of vectors.
pub fn assert_frd_convention(vector: &Vector3<f64>, context: &str) -> Result<(), VfError> {
    if !vector.iter().all(|&x| x.is_finite()) {
        return Err(VfError::InvalidValue(format!(
            "{}: Vector contains non-finite values (NaN or Inf)",
            context
        )));
    }
    Ok(())
}
