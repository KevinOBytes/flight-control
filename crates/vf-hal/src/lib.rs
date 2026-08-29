//! Hardware Abstraction Layer traits and frame definitions for VectorFlight.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

/// High-level errors originating from the HAL or underlying transport layer.
#[derive(Error, Debug, Clone)]
pub enum HalError {
    #[error("Communication bus error: {0}")]
    BusError(String),

    #[error("Stale frame detected: elapsed {0:.3}s")]
    StaleFrame(f64),

    #[error("Value out of valid range: {0}")]
    RangeError(String),

    #[error("Initialization failed: {0}")]
    InitFailed(String),

    #[error("Hardware timeout")]
    Timeout,
}

fn default_instant() -> Instant {
    Instant::now()
}

/// Commands sent to the physical actuators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorCommandFrame {
    #[serde(default = "default_instant", skip)]
    pub timestamp: Instant,
    pub sequence: u64,
    /// 16 rotor thrust commands in Newtons.
    pub motor_thrusts: [f64; 16],
    /// 16 individual motor tilt commands in radians.
    pub motor_tilts: [f64; 16],
    /// 8 pod tilt commands in radians.
    pub pod_tilts: [f64; 8],
}

/// Feedback received from the physical actuators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorFeedbackFrame {
    #[serde(default = "default_instant", skip)]
    pub timestamp: Instant,
    pub sequence: u64,
    /// Actual rotor thrusts (or estimated from RPM) in Newtons.
    pub motor_thrusts: [f64; 16],
    /// Actual motor tilt angles in radians.
    pub motor_tilts: [f64; 16],
    /// Actual pod tilt angles in radians.
    pub pod_tilts: [f64; 8],
    /// Bitmask or status flags indicating actuator errors/warnings.
    pub status_flags: [u32; 16],
}

/// Propulsion Hardware Abstraction Layer.
pub trait PropulsionHal {
    /// Reads the latest actuator feedback frame from the bus.
    fn read_feedback(&mut self) -> Result<ActuatorFeedbackFrame, HalError>;

    /// Writes a command frame out to the actuator controllers.
    fn write_commands(&mut self, commands: &ActuatorCommandFrame) -> Result<(), HalError>;

    /// Commands all rotors to zero thrust immediately.
    fn emergency_zero_thrust(&mut self) -> Result<(), HalError>;
}
