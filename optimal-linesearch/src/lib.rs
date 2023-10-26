#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Line-search optimizers.
//!
//! Fixed step-size is included
//! for being line-search-like.

pub mod backtracking_line_search;
pub mod fixed_step_size;
mod initial_step_size;
mod step_direction;
mod traits;

use std::ops::Mul;

use derive_more::Display;
use derive_num_bounded::{derive_into_inner, derive_new_from_lower_bounded_partial_ord};
use num_traits::{bounds::LowerBounded, real::Real};

pub use self::{initial_step_size::*, step_direction::*, traits::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Error returned when
/// problem length does not match state length.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
#[error("problem length does not match state length")]
pub struct MismatchedLengthError;

/// Multiplier for each component of a step direction
/// in derivative optimization.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StepSize<A>(A);

derive_new_from_lower_bounded_partial_ord!(StepSize<A: Real>);
derive_into_inner!(StepSize<A>);

impl<A> LowerBounded for StepSize<A>
where
    A: Real,
{
    fn min_value() -> Self {
        Self(A::zero() + A::epsilon())
    }
}

impl<A> Mul<A> for StepSize<A>
where
    A: Mul<Output = A>,
{
    type Output = A;

    fn mul(self, rhs: A) -> Self::Output {
        self.0 * rhs
    }
}

/// Useful traits,
/// types,
/// and functions
/// unlikely to conflict with existing definitions.
pub mod prelude {
    pub use crate::{initial_step_size::*, step_direction::*, traits::*, StepSize};
    pub use optimal_core::prelude::*;
    pub use streaming_iterator::StreamingIterator;
}
