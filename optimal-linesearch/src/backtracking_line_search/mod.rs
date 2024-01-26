//! Line-search by backtracking step-size.
//!
//! # Examples
//!
//! ```
//! use optimal_linesearch::backtracking_line_search::BacktrackingLineSearchBuilder;
//!
//! println!(
//!     "{:?}",
//!     BacktrackingLineSearchBuilder::default()
//!         .for_(
//!             2,
//!             |point: &[f64]| point.iter().map(|x| x.powi(2)).sum(),
//!             |point| point.iter().map(|x| 2.0 * x).collect()
//!         )
//!         .point(vec![10.0, 10.0])
//!         .argmin()
//! );
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
