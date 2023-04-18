#![allow(clippy::needless_doctest_main)]

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
//! use optimal::{optimizer::derivative_free::pbil, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter =
//!         pbil::RunningDoneWhenConverged::new(pbil::DoneWhenConvergedConfig::default(Count))
//!             .into_streaming_iter();
//!     let o = iter.find(|o| o.is_done()).expect("should converge");
//!     println!("f({}) = {}", o.best_point(), o.best_point_value());
//! }
//!
//! struct Count;
//!
//! impl Problem for Count {
//!     type PointElem = bool;
//!     type PointValue = u64;
//!
//!     fn evaluate<S>(&self, point: ArrayBase<S, Ix1>) -> u64
//!     where
//!         S: ndarray::RawData<Elem = bool> + Data,
//!     {
//!         point.fold(0, |acc, b| acc + *b as u64)
//!     }
//! }
//!
//! impl FixedLength for Count {
//!     fn len(&self) -> usize {
//!         16
//!     }
//! }
//! ```

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

mod derive;
pub mod optimizer;
pub mod prelude;
mod problem;
