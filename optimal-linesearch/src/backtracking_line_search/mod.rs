//! Line-search by backtracking step-size.
//!
//! # Examples
//!
//! ```
//! use optimal_compute_core::{argvals, run::Value, Run};
//! use optimal_linesearch::backtracking_line_search::BacktrackingLineSearchBuilder;
//!
//! fn main() {
//!     let line_search = BacktrackingLineSearchBuilder::default()
//!         .for_combined(
//!             2,
//!             |point| Value(obj_func(&point)),
//!             |point| (Value(obj_func(&point)), Value(obj_func_d(&point))),
//!         )
//!         .with_point(vec![10.0, 10.0])
//!         .computation();
//!     println!("{}", line_search);
//!     println!("{:?}", line_search.run(argvals![]));
//! }
//!
//! fn obj_func(point: &[f64]) -> f64 {
//!     point.iter().map(|x| x.powi(2)).sum()
//! }
//!
//! fn obj_func_d(point: &[f64]) -> Vec<f64> {
//!     point.iter().map(|x| 2.0 * x).collect()
//! }
//! ```

mod high_level;
pub mod low_level;
pub mod types;

pub use high_level::*;
