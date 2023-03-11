//! Derivative optimization,
//! gradient descent.

pub mod backtracking_steepest;
pub mod fixed_step_steepest;

use std::ops::Mul;

use derive_more::Display;
use num_traits::{bounds::LowerBounded, real::Real};

use crate::derive::{derive_into_inner, derive_new_from_lower_bounded_partial_ord};

/// Multiplier for each component of a step direction
/// in derivative optimization.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
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
