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
//! use ndarray::prelude::*;
//! use optimal::{optimizer::derivative_free::pbil, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter = pbil::DoneWhenConvergedConfig::default(Count)
//!         .start()
//!         .into_streaming_iter();
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
//!     fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
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
