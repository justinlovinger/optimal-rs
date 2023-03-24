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
//! use optimal::{
//!     optimizer::derivative_free::pbil::DoneWhenConvergedConfig,
//!     prelude::*,
//! };
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter = DoneWhenConvergedConfig::default(Count)
//!         .initialize()
//!         .into_streaming_iter();
//!     let xs = iter
//!         .find(|o| o.is_done())
//!         .expect("should converge")
//!         .best_point();
//!     println!("f({}) = {}", xs, Count.evaluate(xs.view()));
//! }
//!
//! struct Count;
//!
//! impl Problem<bool, u64> for Count {
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

#![deny(missing_docs)]

mod derive;
pub mod optimizer;
pub mod prelude;
mod problem;
