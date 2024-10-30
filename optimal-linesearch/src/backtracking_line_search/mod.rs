//! Line-search by backtracking step-size.
//!
//! # Examples
//!
//! ```
//! use computation_types::{arg1, named_args, val, Computation, Run};
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
//! println!("{:?}", line_search.run(named_args![]));
//! ```

mod high_level;
pub mod low_level;
pub mod types;

pub use high_level::*;
