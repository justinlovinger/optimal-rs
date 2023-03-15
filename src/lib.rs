//! Mathematical optimization and machine learning framework
//! and algorithms.
//!
//! Optimal provides a composable framework
//! for mathematical optimization
//! and machine learning
//! from the optimization perspective,
//! in addition to algorithm implementations.
//!
//! # Examples
//!
//! Minimize the objective function `f`
//! using a PBIL optimizer:
//!
//! ```
//! use ndarray::{Data, RemoveAxis, prelude::*};
//! use optimal::{
//!     optimizer::derivative_free::pbil::{DoneWhenConvergedConfig, PbilDoneWhenConverged, NumBits},
//!     prelude::*,
//! };
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let config = DoneWhenConvergedConfig::default(NumBits(16));
//!     let mut iter =
//!         PbilDoneWhenConverged::new(&config, &f, config.initial_state()).into_streaming_iter();
//!     let xs = iter
//!         .find(|o| o.is_done())
//!         .expect("should converge")
//!         .best_point();
//!     println!("f({}) = {}", xs, f(xs.view()));
//! }
//!
//! fn f(bs: ArrayView1<bool>) -> u64 {
//!     bs.fold(0, |acc, b| acc + *b as u64)
//! }
//! ```

#![deny(missing_docs)]

mod derive;
mod objective;
pub mod optimizer;
pub mod prelude;
