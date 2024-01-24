//! Line-search by backtracking step-size.
//!
//! # Examples
//!
//! ```
//! use optimal_linesearch::{
//!     backtracking_line_search::{
//!         BacktrackingLineSearch, BacktrackingRate, SufficientDecreaseParameter,
//!     },
//!     initial_step_size::IncrRate,
//!     step_direction::steepest_descent,
//!     StepSize,
//! };
//!
//! fn main() {
//!     let c_1 = SufficientDecreaseParameter::default();
//!     let backtracking_rate = BacktrackingRate::default();
//!     let incr_rate = IncrRate::from_backtracking_rate(backtracking_rate);
//!
//!     let mut step_size = StepSize::new(1.0).unwrap();
//!     let mut point = vec![10.0, 10.0];
//!     for _ in 0..100 {
//!         let value = obj_func(&point);
//!         let derivatives = obj_func_d(&point);
//!         let line_search = BacktrackingLineSearch::new(
//!             c_1,
//!             point,
//!             value,
//!             &derivatives,
//!             steepest_descent(&derivatives),
//!         );
//!         (step_size, point) = line_search.search(backtracking_rate, &obj_func, step_size);
//!         step_size = incr_rate * step_size;
//!     }
//!     println!("{:?}", point);
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

pub use self::types::*;

use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul, Neg},
};

use derive_getters::{Dissolve, Getters};

use crate::StepSize;

/// Values required for backtracking line-search.
#[derive(Clone, Debug, Dissolve, Getters)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct BacktrackingLineSearch<A> {
    c_1_times_derivatives_dot_direction: C1TimesDerivativesDotDirection<A>,
    point: Vec<A>,
    value: A,
    direction: Vec<A>,
}

impl<A> BacktrackingLineSearch<A> {
    /// Prepare for backtracking line-search.
    pub fn new(
        c_1: SufficientDecreaseParameter<A>,
        point: Vec<A>,
        value: A,
        derivatives: &[A],
        direction: Vec<A>,
    ) -> Self
    where
        A: Clone + Neg<Output = A> + Mul<Output = A> + Sum,
    {
        debug_assert_eq!(point.len(), derivatives.len());
        debug_assert_eq!(point.len(), direction.len());
        Self {
            c_1_times_derivatives_dot_direction: C1TimesDerivativesDotDirection::new(
                c_1,
                derivatives,
                &direction,
            ),
            point,
            value,
            direction,
        }
    }

    /// Return the step-size
    /// and corresponding point
    /// that minimizes the objective function.
    pub fn search<F>(
        &self,
        backtracking_rate: BacktrackingRate<A>,
        obj_func: F,
        initial_step_size: StepSize<A>,
    ) -> (StepSize<A>, Vec<A>)
    where
        A: Clone + PartialOrd + Add<Output = A> + Mul<Output = A>,
        F: Fn(&[A]) -> A,
    {
        let mut step_size = initial_step_size;
        let point = loop {
            let point_at_step = self.point_at_step(step_size.clone());
            if self.is_sufficient_decrease(step_size.clone(), obj_func(&point_at_step)) {
                break point_at_step;
            } else {
                step_size = backtracking_rate.clone() * step_size.clone();
            }
        };
        (step_size, point)
    }

    /// Return point at `step_size` in line-search direction.
    pub fn point_at_step(&self, step_size: StepSize<A>) -> Vec<A>
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        crate::descend(step_size, self.direction(), self.point())
    }

    /// Return whether `point_at_step` is sufficient.
    ///
    /// See [`is_sufficient_decrease`].
    pub fn is_sufficient_decrease(&self, step_size: StepSize<A>, value_at_step: A) -> bool
    where
        A: Clone + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        is_sufficient_decrease(
            self.c_1_times_derivatives_dot_direction.clone(),
            self.value.clone(),
            step_size,
            value_at_step,
        )
    }
}

/// The sufficient decrease condition,
/// also known as the Armijo rule,
/// mathematically written as:
/// $f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)$.
pub fn is_sufficient_decrease<A>(
    c_1_times_derivatives_dot_direction: C1TimesDerivativesDotDirection<A>,
    value: A,
    step_size: StepSize<A>,
    value_at_step: A,
) -> bool
where
    A: PartialOrd + Add<Output = A> + Mul<Output = A>,
{
    value_at_step <= value + c_1_times_derivatives_dot_direction * step_size
}

mod types {
    use std::{
        fmt::Debug,
        iter::Sum,
        ops::{Mul, Neg},
    };

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

    /// Value prepared to check for sufficient decrease.
    #[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    pub struct C1TimesDerivativesDotDirection<A>(pub A);

    impl<A> Mul<A> for C1TimesDerivativesDotDirection<A>
    where
        A: Mul<Output = A>,
    {
        type Output = A;

        fn mul(self, rhs: A) -> Self::Output {
            self.0.mul(rhs)
        }
    }

    impl<A> Mul<StepSize<A>> for C1TimesDerivativesDotDirection<A>
    where
        A: Mul<Output = A>,
    {
        type Output = A;

        fn mul(self, rhs: StepSize<A>) -> Self::Output {
            self.0.mul(rhs.0)
        }
    }

    impl<A> Mul<C1TimesDerivativesDotDirection<A>> for StepSize<A>
    where
        A: Mul<Output = A>,
    {
        type Output = A;

        fn mul(self, rhs: C1TimesDerivativesDotDirection<A>) -> Self::Output {
            self.0.mul(rhs.0)
        }
    }

    impl<A> C1TimesDerivativesDotDirection<A> {
        #[allow(missing_docs)]
        pub fn new(c_1: SufficientDecreaseParameter<A>, derivatives: &[A], direction: &[A]) -> Self
        where
            A: Clone + Neg<Output = A> + Mul<Output = A> + Sum,
        {
            Self(
                c_1.into_inner()
                    * derivatives
                        .iter()
                        .cloned()
                        .zip(direction.iter().cloned())
                        .map(|(x, y)| x * y)
                        .sum(),
            )
        }
    }

    derive_into_inner!(C1TimesDerivativesDotDirection<A>);

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
}
