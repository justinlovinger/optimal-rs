//! Line-search by backtracking step-size.
//!
//! # Examples
//!
//! ```
//! use optimal_compute_core::{arg1, argvals, val, Computation, Run};
//! use optimal_linesearch::backtracking_line_search::BacktrackingLineSearchBuilder;
//!
//! let line_search = BacktrackingLineSearchBuilder::default()
//!     .for_(
//!         2,
//!         arg1!("point", f64).pow(val!(2.0)).sum(),
//!         val!(2.0) * arg1!("point", f64),
//!     )
//!     .with_point(vec![10.0, 10.0])
//!     .computation();
//! println!("{}", line_search);
//! println!("{:?}", line_search.run(argvals![]));
//! ```

mod high_level;
pub mod low_level;
pub mod types;

pub use high_level::*;
