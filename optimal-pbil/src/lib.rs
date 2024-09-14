#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use optimal_compute_core::{argvals, run::Value, Run};
//! use optimal_pbil::PbilBuilder;
//!
//! let pbil = PbilBuilder::default()
//!     .for_(2, |point| Value(point.iter().filter(|x| **x).count()))
//!     .computation();
//! println!("{}", pbil);
//! println!("{:?}", pbil.run(argvals![]));
//! ```

mod high_level;
pub mod low_level;
pub mod types;

pub use high_level::*;
