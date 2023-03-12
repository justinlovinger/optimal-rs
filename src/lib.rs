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
//!     optimizer::derivative_free::pbil::{PbilDoneWhenConverged, NumBits},
//!     prelude::*,
//! };
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter = PbilDoneWhenConverged::default(NumBits(16), |xs| f(xs)).into_streaming_iter();
//!     let xs = iter
//!         .find(|o| o.is_done())
//!         .expect("should converge")
//!         .best_point();
//!     println!("f({}) = {}", xs, f(xs.view()));
//! }
//!
//! fn f<S, D>(bs: ArrayBase<S, D>) -> Array<u64, D::Smaller>
//! where
//!     S: Data<Elem = bool>,
//!     D: Dimension + RemoveAxis,
//! {
//!     bs.fold_axis(Axis(bs.ndim() - 1), 0, |acc, b| acc + *b as u64)
//! }
//! ```

#![deny(missing_docs)]

mod derive;
pub mod optimizer;
pub mod prelude;
