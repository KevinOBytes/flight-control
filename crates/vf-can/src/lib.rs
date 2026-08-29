//! Reference CAN-FD transport adapter implementation for SocketCAN.
//!
//! # Safety and Authentication WARNING
//!
//! CAN-FD is NOT an authenticated transport by default. Standard frames do not have
//! cryptographic signatures, encryption, or built-in sender verification.
//! This implementation is a research prototype. If deployed in real hardware,
//! application-level CRC/authentication must be added or an alternative transport substituted.

use std::time::Instant;
use tracing::warn;
use vf_hal::{ActuatorCommandFrame, ActuatorFeedbackFrame, HalError, PropulsionHal};

/// Configuration for the CAN-FD interface.
#[derive(Debug, Clone)]
pub struct CanConfig {
    pub interface: String,
    pub bitrate: u32,
    pub dbitrate: u32,
}

/// Linux-specific implementation placeholder.
#[cfg(target_os = "linux")]
pub struct SocketCanAdapter {
    config: CanConfig,
    sequence: u64,
    // socketcan::FdSocket would go here in actual implementation
}

#[cfg(target_os = "linux")]
impl SocketCanAdapter {
    pub fn new(config: CanConfig) -> Result<Self, HalError> {
        tracing::info!("Initializing SocketCAN FD adapter on {}", config.interface);
        Ok(Self {
            config,
            sequence: 0,
        })
    }

    pub fn config(&self) -> &CanConfig {
        &self.config
    }
}

#[cfg(target_os = "linux")]
impl PropulsionHal for SocketCanAdapter {
    fn read_feedback(&mut self) -> Result<ActuatorFeedbackFrame, HalError> {
        // In actual implementation, read from socketcan FD frame buffer
        Ok(ActuatorFeedbackFrame {
            timestamp: Instant::now(),
            sequence: self.sequence,
            motor_thrusts: [0.0; 16],
            motor_tilts: [0.0; 16],
            pod_tilts: [0.0; 8],
            status_flags: [0; 16],
        })
    }

    fn write_commands(&mut self, commands: &ActuatorCommandFrame) -> Result<(), HalError> {
        self.sequence = commands.sequence;
        // In actual implementation, format and write CAN-FD frame
        Ok(())
    }

    fn emergency_zero_thrust(&mut self) -> Result<(), HalError> {
        warn!("Sending emergency zero thrust broadcast over CAN-FD");
        Ok(())
    }
}

/// Non-Linux fallback implementation (e.g. macOS developer machine).
#[cfg(not(target_os = "linux"))]
pub struct SocketCanAdapter {
    config: CanConfig,
    sequence: u64,
}

#[cfg(not(target_os = "linux"))]
impl SocketCanAdapter {
    pub fn new(config: CanConfig) -> Result<Self, HalError> {
        warn!(
            "SocketCAN is only supported on Linux. Initializing MOCK SocketCAN adapter on {}",
            config.interface
        );
        Ok(Self {
            config,
            sequence: 0,
        })
    }

    pub fn config(&self) -> &CanConfig {
        &self.config
    }
}

#[cfg(not(target_os = "linux"))]
impl PropulsionHal for SocketCanAdapter {
    fn read_feedback(&mut self) -> Result<ActuatorFeedbackFrame, HalError> {
        Ok(ActuatorFeedbackFrame {
            timestamp: Instant::now(),
            sequence: self.sequence,
            motor_thrusts: [0.0; 16],
            motor_tilts: [0.0; 16],
            pod_tilts: [0.0; 8],
            status_flags: [0; 16],
        })
    }

    fn write_commands(&mut self, commands: &ActuatorCommandFrame) -> Result<(), HalError> {
        self.sequence = commands.sequence;
        Ok(())
    }

    fn emergency_zero_thrust(&mut self) -> Result<(), HalError> {
        warn!("Sending MOCK emergency zero thrust broadcast");
        Ok(())
    }
}
