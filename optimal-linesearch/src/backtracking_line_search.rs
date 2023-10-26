//! Line-search by backtracking step-size.
//!
//! # Examples
//!
//! ```
//! use optimal_linesearch::{
//!     backtracking_line_search, incr_prev_initial_step, prelude::*, steepest_descent,
//! };
//!
//! fn main() {
//!     let config = backtracking_line_search::Config::default();
//!     let initial_step_config =
//!         incr_prev_initial_step::Config::from_backtracking_rate(config.backtracking_rate);
//!     println!(
//!         "{:?}",
//!         config
//!             .build(
//!                 IndependentStepDirectionInitialStepSize::new(
//!                     steepest_descent::SteepestDescent::new(|xs| (obj_func(xs), obj_func_d(xs))),
//!                     initial_step_config.build(),
//!                 ),
//!                 obj_func,
//!                 std::iter::repeat(-10.0..=10.0).take(2),
//!             )
//!             .nth(100)
//!             .unwrap()
//!             .best_point()
//!     );
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

use std::{iter::Sum, ops::RangeInclusive, sync::Arc};

use derive_getters::{Dissolve, Getters};
use num_traits::{real::Real, AsPrimitive};
use optimal_core::Optimizer;
use rand::{
    distributions::{uniform::SampleUniform, Uniform},
    prelude::*,
};
use streaming_iterator::StreamingIterator;

use crate::{StepDirectionInitialStepSize, StepSize, ValueDerivatives};

pub use self::{
    state_machine::{
        BacktrackingLineSearchBundle, SearchStepWithEvaluatedStep, SearchWithStep, State, Valued,
    },
    types::{BacktrackingRate, SufficientDecreaseParameter},
};

/// Line-search by backtracking step-size.
#[derive(Clone, Debug, Getters, Dissolve)]
#[dissolve(rename = "into_parts")]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BacktrackingLineSearch<A, I, F>
where
    I: StepDirectionInitialStepSize,
{
    /// Optimizer configuration.
    config: Config<A>,

    /// State of optimizer.
    state: State<A>,

    /// Objective function to minimize.
    obj_func: F,

    /// Component for getting step-direction and initial step-size.
    step_direction_and_initial_size: I,
}

/// Backtracking line-search config.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Config<A> {
    /// The sufficient decrease parameter,
    /// `c_1`.
    pub c_1: SufficientDecreaseParameter<A>,

    /// Rate to decrease step size while line searching.
    pub backtracking_rate: BacktrackingRate<A>,
}

impl<A> Default for Config<A>
where
    A: 'static + Copy,
    f64: AsPrimitive<A>,
{
    fn default() -> Self {
        Self {
            c_1: Default::default(),
            backtracking_rate: Default::default(),
        }
    }
}

impl<A> Config<A> {
    /// Return a new 'BacktrackingLineSearch'.
    ///
    /// This is nondeterministic.
    ///
    /// - `step_direction_and_initial_size`: component for getting step-direction and initial step-size
    /// - `obj_func`: objective function to minimize
    /// - `initial_bounds`: bounds for generating the initial random point
    ///
    /// Note,
    /// `step_direction_and_initial_size` should use the same objective function
    /// for sound results.
    pub fn build<I, F>(
        self,
        step_direction_and_initial_size: I,
        obj_func: F,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
    ) -> BacktrackingLineSearch<A, I, F>
    where
        A: SampleUniform,
        I: StepDirectionInitialStepSize,
        F: Fn(&[A]) -> A,
    {
        BacktrackingLineSearch {
            step_direction_and_initial_size,
            state: self.initial_state_using(initial_bounds, &mut thread_rng()),
            config: self,
            obj_func,
        }
    }

    /// Return a new 'BacktrackingLineSearch'
    /// initialized using `rng`.
    ///
    /// - `step_direction_and_initial_size`: component for getting step-direction and initial step-size
    /// - `obj_func`: objective function to minimize
    /// - `initial_bounds`: bounds for generating the initial random point
    /// - `rng`: source of randomness
    ///
    /// Note,
    /// `step_direction_and_initial_size` should use the same objective function
    /// for sound results.
    pub fn build_using<I, F, R>(
        self,
        step_direction_and_initial_size: I,
        obj_func: F,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        rng: &mut R,
    ) -> BacktrackingLineSearch<A, I, F>
    where
        A: SampleUniform,
        I: StepDirectionInitialStepSize,
        F: Fn(&[A]) -> A,
        R: Rng,
    {
        BacktrackingLineSearch {
            step_direction_and_initial_size,
            state: self.initial_state_using(initial_bounds, rng),
            config: self,
            obj_func,
        }
    }

    /// Return a new 'BacktrackingLineSearch'
    /// initialized from `point`.
    ///
    /// - `step_direction_and_initial_size`: component for getting step-direction and initial step-size
    /// - `obj_func`: objective function to minimize
    /// - `point`: initial point
    ///
    /// Note,
    /// `step_direction_and_initial_size` should use the same objective function
    /// for sound results.
    pub fn build_from<I, F>(
        self,
        step_direction_and_initial_size: I,
        obj_func: F,
        point: Vec<A>,
    ) -> BacktrackingLineSearch<A, I, F>
    where
        I: StepDirectionInitialStepSize,
        F: Fn(&[A]) -> A,
    {
        BacktrackingLineSearch {
            step_direction_and_initial_size,
            config: self,
            state: State::new(point),
            obj_func,
        }
    }

    fn initial_state_using<R>(
        &self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        rng: &mut R,
    ) -> State<A>
    where
        A: SampleUniform,
        R: Rng,
    {
        State::new(
            initial_bounds
                .into_iter()
                .map(|range| {
                    let (start, end) = range.into_inner();
                    Uniform::new_inclusive(start, end).sample(rng)
                })
                .collect(),
        )
    }
}

impl<A, I, F> StreamingIterator for BacktrackingLineSearch<A, I, F>
where
    A: Real + Sum + 'static,
    I: StepDirectionInitialStepSize<Elem = A, Point = Arc<Vec<A>>>,
    I::Point: Clone + From<Vec<A>> + AsRef<Vec<A>>,
    I::Value: ValueDerivatives<A>,
    F: Fn(&[A]) -> A,
{
    type Item = Self;

    fn advance(&mut self) {
        replace_with::replace_with_or_abort(&mut self.state, |state| match state {
            State::Started { point } => {
                #[allow(clippy::arc_with_non_send_sync)] // False positive
                let point: I::Point = point.into();
                self.step_direction_and_initial_size
                    .start_iteration(point.clone(), None);
                State::PreparingLineSearch { point }
            }
            State::PreparingLineSearch { point } => {
                match self.step_direction_and_initial_size.step() {
                    Some((step_direction, step_size)) => State::InitializedSearching {
                        line_search: BacktrackingLineSearchBundle::new(
                            self.config.c_1,
                            point,
                            step_direction,
                        ),
                        step_size,
                    },
                    None => State::PreparingLineSearch { point },
                }
            }
            State::InitializedSearching {
                line_search,
                step_size,
            } => State::SearchStepped(SearchWithStep::new(line_search, step_size)),
            State::SearchStepped(x) => {
                State::SearchStepEvaluated(x.into_fake_step_evaluated(&self.obj_func))
            }
            State::SearchStepEvaluated(x) => {
                if x.is_sufficient_decrease() {
                    let (_, step_size, point_at_step) = x.into_parts();
                    State::Stepped {
                        point: point_at_step.into_parts().0,
                        last_step_size: step_size,
                    }
                } else {
                    let (line_search, step_size, _) = x.into_parts();
                    State::StepSizeDecremented {
                        line_search,
                        step_size: StepSize(
                            self.config.backtracking_rate.into_inner() * step_size.into_inner(),
                        ),
                    }
                }
            }
            State::StepSizeDecremented {
                line_search,
                step_size,
            } => State::SearchStepped(SearchWithStep::new(line_search, step_size)),
            State::Stepped {
                point,
                last_step_size,
            } => {
                #[allow(clippy::arc_with_non_send_sync)] // False positive
                let point: I::Point = point.into();
                self.step_direction_and_initial_size
                    .start_iteration(point.clone(), Some(last_step_size));
                State::PreparingLineSearch { point }
            }
        });
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

impl<A, I, F> Optimizer for BacktrackingLineSearch<A, I, F>
where
    A: Clone,
    I: StepDirectionInitialStepSize,
    I::Point: AsRef<Vec<A>>,
{
    type Point = Vec<A>;

    fn best_point(&self) -> Self::Point {
        self.point().into()
    }
}

impl<A, I, F> BacktrackingLineSearch<A, I, F>
where
    I: StepDirectionInitialStepSize,
    I::Point: AsRef<Vec<A>>,
{
    /// Return point being line-searched from.
    pub fn point(&self) -> &[A] {
        match &self.state {
            State::Started { point } => point,
            State::PreparingLineSearch { point } => point.as_ref(),
            State::InitializedSearching { line_search, .. } => line_search.point().as_ref(),
            State::SearchStepped(x) => x.line_search().point().as_ref(),
            State::SearchStepEvaluated(x) => x.line_search().point().as_ref(),
            State::StepSizeDecremented { line_search, .. } => line_search.point().as_ref(),
            State::Stepped { point, .. } => point,
        }
    }
}

mod state_machine {
    use std::{
        fmt::Debug,
        ops::{Add, Mul},
        sync::Arc,
    };

    use derive_getters::{Dissolve, Getters};

    use crate::StepSize;

    pub use self::{line_search::BacktrackingLineSearchBundle, valued::Valued};

    /// Line-search state.
    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum State<A> {
        /// Optimizer started.
        Started {
            /// Point to line-search from.
            point: Vec<A>,
        },
        /// Preparing to line-search.
        PreparingLineSearch {
            /// Point to line-search from.
            point: Arc<Vec<A>>,
        },
        /// Prepared to line-search.
        InitializedSearching {
            /// Values required for line-search.
            line_search: BacktrackingLineSearchBundle<A, Arc<Vec<A>>>,
            /// Initial step-size.
            step_size: StepSize<A>,
        },
        /// Took a line-search step.
        SearchStepped(SearchWithStep<A, Arc<Vec<A>>>),
        /// Evaluated a line-search step.
        SearchStepEvaluated(SearchStepWithEvaluatedStep<A, Arc<Vec<A>>>),
        /// Decremented step-size for next line-search iteration.
        StepSizeDecremented {
            /// Values required for line-search.
            line_search: BacktrackingLineSearchBundle<A, Arc<Vec<A>>>,
            /// Current line-search step-size.
            step_size: StepSize<A>,
        },
        /// Finished line-search and took a real step.
        Stepped {
            /// Point found by line-search.
            point: Vec<A>,
            /// Step-size used by line-search.
            last_step_size: StepSize<A>,
        },
    }

    /// Prepared line-search with point at step-size.
    #[derive(Clone, Debug, Getters, Dissolve)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[dissolve(rename = "into_parts")]
    pub struct SearchWithStep<A, P> {
        line_search: BacktrackingLineSearchBundle<A, P>,
        step_size: StepSize<A>,
        point_at_step: Vec<A>,
    }

    /// Prepared line-search with evaluated point at step-size.
    #[derive(Clone, Debug, Getters, Dissolve)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[dissolve(rename = "into_parts")]
    pub struct SearchStepWithEvaluatedStep<A, P> {
        line_search: BacktrackingLineSearchBundle<A, P>,
        step_size: StepSize<A>,
        point_at_step: Valued<Vec<A>, A>,
    }

    impl<A> State<A> {
        /// Return an initial state.
        pub(crate) fn new(point: Vec<A>) -> Self {
            State::Started { point }
        }
    }

    impl<A, P> SearchWithStep<A, P> {
        /// Prepare step at step-size in line-search direction.
        pub fn new(line_search: BacktrackingLineSearchBundle<A, P>, step_size: StepSize<A>) -> Self
        where
            A: Clone + Add<Output = A> + Mul<Output = A>,
            P: AsRef<Vec<A>>,
        {
            Self {
                point_at_step: descend(
                    line_search.point().as_ref(),
                    step_size.clone(),
                    line_search.step_direction(),
                ),
                line_search,
                step_size,
            }
        }

        /// Evaluate search-step.
        pub fn into_fake_step_evaluated(
            self,
            f: impl FnOnce(&[A]) -> A,
        ) -> SearchStepWithEvaluatedStep<A, P> {
            SearchStepWithEvaluatedStep::new(
                self.line_search,
                self.step_size,
                self.point_at_step,
                f,
            )
        }
    }

    impl<A, P> SearchStepWithEvaluatedStep<A, P> {
        // `step_size` and `point_at_step` cannot be safely passed
        // independent of `line_search`.
        fn new(
            line_search: BacktrackingLineSearchBundle<A, P>,
            step_size: StepSize<A>,
            point_at_step: Vec<A>,
            f: impl FnOnce(&[A]) -> A,
        ) -> Self {
            Self {
                line_search,
                step_size,
                point_at_step: Valued::new(point_at_step, f),
            }
        }

        /// Return whether search-step is sufficient.
        pub fn is_sufficient_decrease(&self) -> bool
        where
            A: Copy + PartialOrd + Add<Output = A> + Mul<Output = A>,
        {
            is_sufficient_decrease(
                *self.line_search.value(),
                self.step_size,
                *self
                    .line_search
                    .c_1_times_point_derivatives_dot_step_direction(),
                *self.point_at_step.value(),
            )
        }
    }

    fn descend<A>(point: &[A], step_size: StepSize<A>, step_direction: &[A]) -> Vec<A>
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        point
            .iter()
            .zip(step_direction)
            .map(|(x, d)| x.clone() + step_size.clone() * d.clone())
            .collect()
    }

    /// The sufficient decrease condition,
    /// also known as the Armijo rule,
    /// mathematically written as:
    /// $f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)$.
    fn is_sufficient_decrease<A>(
        value: A,
        step_size: StepSize<A>,
        c_1_times_point_derivatives_dot_step_direction: A,
        value_at_step: A,
    ) -> bool
    where
        A: PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        value_at_step <= value + step_size * c_1_times_point_derivatives_dot_step_direction
    }

    mod valued {
        use std::borrow::Borrow;

        use derive_getters::{Dissolve, Getters};

        /// Value `x` safely bundled with `value`.
        #[derive(Clone, Debug, PartialEq, Eq, Dissolve, Getters)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[dissolve(rename = "into_parts")]
        pub struct Valued<T, B> {
            x: T,
            value: B,
        }

        impl<T, B> Valued<T, B> {
            /// Return valued `x` evaluated by `f`.
            pub fn new<Borrowed, F>(x: T, f: F) -> Self
            where
                Borrowed: ?Sized,
                T: Borrow<Borrowed>,
                F: FnOnce(&Borrowed) -> B,
            {
                Self {
                    value: f(x.borrow()),
                    x,
                }
            }
        }
    }

    mod line_search {
        use std::{
            fmt::Debug,
            iter::Sum,
            ops::{Mul, Neg},
        };

        use derive_getters::{Dissolve, Getters};

        use crate::{
            backtracking_line_search::{SufficientDecreaseParameter, ValueDerivatives},
            DoneStepDirection,
        };

        /// Values required for line-search.
        #[derive(Clone, Debug, Dissolve, Getters)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[dissolve(rename = "into_parts")]
        pub struct BacktrackingLineSearchBundle<A, P> {
            point: P,
            value: A,
            step_direction: Vec<A>,
            c_1_times_point_derivatives_dot_step_direction: A,
        }

        impl<A, P> BacktrackingLineSearchBundle<A, P> {
            /// Bundle values required for line-search.
            pub fn new<V>(
                c_1: SufficientDecreaseParameter<A>,
                point: P,
                step_direction: DoneStepDirection<A, V>,
            ) -> Self
            where
                A: 'static + Copy + Neg<Output = A> + Mul<Output = A> + Sum,
                P: AsRef<Vec<A>>,
                V: ValueDerivatives<A>,
            {
                let (value, derivatives) = step_direction.value.into_value_derivatives();
                debug_assert_eq!(point.as_ref().len(), derivatives.len());
                Self {
                    point,
                    value,
                    c_1_times_point_derivatives_dot_step_direction: c_1.into_inner()
                        * derivatives
                            .into_iter()
                            .zip(step_direction.step_direction.iter().copied())
                            .map(|(x, y)| x * y)
                            .sum(),
                    step_direction: step_direction.step_direction,
                }
            }
        }
    }
}

mod types {
    use std::fmt::Debug;

    use derive_more::Display;
    use derive_num_bounded::{derive_into_inner, derive_new_from_bounded_partial_ord};
    use num_traits::{
        bounds::{LowerBounded, UpperBounded},
        real::Real,
        AsPrimitive,
    };
    pub use optimal_core::prelude::*;

    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

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

    /// Rate to decrease step size while line searching.
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
}
