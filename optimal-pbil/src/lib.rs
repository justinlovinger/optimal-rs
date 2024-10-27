#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use optimal_compute_core::{arg1, argvals, peano::Zero, run::Value, Computation, Run};
//! use optimal_pbil::PbilBuilder;
//!
//! let pbil = PbilBuilder::default()
//!     .for_(
//!         2,
//!         arg1!("sample").black_box(|sample: Vec<bool>| {
//!             Value(sample.iter().filter(|x| **x).count())
//!         }),
//!     )
//!     .computation();
//! println!("{}", pbil);
//! println!("{:?}", pbil.run(argvals![]));
//! ```

mod high_level;
pub mod low_level;
pub mod types;

pub use high_level::*;
