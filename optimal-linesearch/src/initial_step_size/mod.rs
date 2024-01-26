//! Methods to get initial step-size for line-search.

pub use incr_prev::IncrRate;

mod incr_prev {
    use std::ops::{Div, Mul, Sub};

    use derive_more::Display;
    use derive_num_bounded::{derive_into_inner, derive_new_from_lower_bounded_partial_ord};
    use num_traits::{bounds::LowerBounded, real::Real, AsPrimitive, One};

    use crate::{backtracking_line_search::types::BacktrackingRate, StepSize};

    /// Rate to increase step-size before starting each line-search.
    #[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    pub struct IncrRate<A>(A);

    derive_new_from_lower_bounded_partial_ord!(IncrRate<A: Real>);
    derive_into_inner!(IncrRate<A>);

    impl<A> IncrRate<A>
    where
        A: 'static + Copy + One + Sub<Output = A> + Div<Output = A>,
        f64: AsPrimitive<A>,
    {
        /// Return increase rate slightly more than one step up from backtracking rate.
        pub fn from_backtracking_rate(x: BacktrackingRate<A>) -> IncrRate<A> {
            Self(2.0.as_() / x.into_inner() - A::one())
        }
    }

    impl<A> LowerBounded for IncrRate<A>
    where
        A: Real,
    {
        fn min_value() -> Self {
            Self(A::one() + A::epsilon())
        }
    }

    impl<A> Mul<StepSize<A>> for IncrRate<A>
    where
        A: Mul<Output = A>,
    {
        type Output = StepSize<A>;

        fn mul(self, rhs: StepSize<A>) -> Self::Output {
            StepSize(self.0 * rhs.into_inner())
        }
    }
}
