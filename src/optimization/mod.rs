//! Mathematical optimization framework.
//!
//! An optimizer configuration should remain static during operation
//! and may optionally depend on a given problem.
//! A problem and configuration
//! can be used to start a running optimizer.
//! A running optimizer has state
//! that depends on its problem
//! and configuration.

pub mod config;
pub mod optimizer;
