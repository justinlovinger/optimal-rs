#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Steepest descent optimizers.

pub mod backtracking_steepest;
pub mod fixed_step_steepest;

use std::ops::Mul;

use derive_more::Display;
use derive_num_bounded::{derive_into_inner, derive_new_from_lower_bounded_partial_ord};
use num_traits::{bounds::LowerBounded, real::Real};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Error returned when
/// problem length does not match state length.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
#[error("problem length does not match state length")]
pub struct MismatchedLengthError;

/// Multiplier for each component of a step direction
/// in derivative optimization.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
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
