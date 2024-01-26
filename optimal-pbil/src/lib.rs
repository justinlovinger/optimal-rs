#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use optimal_pbil::PbilBuilder;
//!
//! println!(
//!     "{:?}",
//!     PbilBuilder::default()
//!         .for_(2, |point| point.iter().filter(|x| **x).count())
//!         .argmin()
//! )
//! ```
//!
//! For greater flexibility,
//! introspection,
//! and customization,
//! see [`low_level`].

mod high_level;
pub mod low_level;
pub mod types;

pub use high_level::*;
