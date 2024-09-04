//! Types for backtracking line-search.

use std::{fmt::Debug, ops::Mul};

use derive_more::Display;
use derive_num_bounded::{derive_into_inner, derive_new_from_bounded_partial_ord};
use num_traits::{
    bounds::{LowerBounded, UpperBounded},
    real::Real,
    AsPrimitive,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::StepSize;

// `#[serde(into = "A")]` and `#[serde(try_from = "A")]` makes more sense
// than `#[serde(transparent)]`,
// but as of 2023-09-24,
// but we cannot `impl<A> From<Foo<A>> for A`
// and manually implementing `Serialize` and `Deserialize`
// is not worth the effort.

/// The sufficient decrease parameter,
/// `c_1`.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct SufficientDecreaseParameter<A>(A);

derive_new_from_bounded_partial_ord!(SufficientDecreaseParameter<A: Real>);
derive_into_inner!(SufficientDecreaseParameter<A>);

impl<A> Default for SufficientDecreaseParameter<A>
where
    A: 'static + Copy,
    f64: AsPrimitive<A>,
{
    fn default() -> Self {
        Self(0.5.as_())
    }
}

impl<A> LowerBounded for SufficientDecreaseParameter<A>
where
    A: Real,
{
    fn min_value() -> Self {
        Self(A::epsilon())
    }
}

impl<A> UpperBounded for SufficientDecreaseParameter<A>
where
    A: Real,
{
    fn max_value() -> Self {
        Self(A::one() - A::epsilon())
    }
}

impl<A> Mul<A> for SufficientDecreaseParameter<A>
where
    A: Mul<Output = A>,
{
    type Output = A;

    fn mul(self, rhs: A) -> Self::Output {
        self.0 * rhs
    }
}

/// Rate to decrease step-size while line-searching.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct BacktrackingRate<A>(A);

derive_new_from_bounded_partial_ord!(BacktrackingRate<A: Real>);
derive_into_inner!(BacktrackingRate<A>);

impl<A> Default for BacktrackingRate<A>
where
    A: 'static + Copy,
    f64: AsPrimitive<A>,
{
    fn default() -> Self {
        Self(0.5.as_())
    }
}

impl<A> LowerBounded for BacktrackingRate<A>
where
    A: Real,
{
    fn min_value() -> Self {
        Self(A::epsilon())
    }
}

impl<A> UpperBounded for BacktrackingRate<A>
where
    A: Real,
{
    fn max_value() -> Self {
        Self(A::one() - A::epsilon())
    }
}

impl<A> Mul<StepSize<A>> for BacktrackingRate<A>
where
    A: Mul<Output = A>,
{
    type Output = StepSize<A>;

    fn mul(self, rhs: StepSize<A>) -> Self::Output {
        StepSize(self.0 * rhs.into_inner())
    }
}
