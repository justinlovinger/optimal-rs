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
//! use optimal::{optimizer::derivative_free::pbil::DoneWhenConvergedConfig, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let config = DoneWhenConvergedConfig::default(16.into());
//!     let mut iter = config.iterate(|xs| f(xs), config.initial_state());
//!     // `unwrap` is safe
//!     // because the optimizer is guaranteed to converge.
//!     let bs = config.best_point(iter.find(|s| config.is_done(s)).unwrap());
//!     println!("f({}) = {}", bs, f(bs.view()));
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

pub mod optimizer;
pub mod prelude;
