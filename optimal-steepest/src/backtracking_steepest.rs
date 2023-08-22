#![allow(clippy::needless_doctest_main)]

//! Backtracking line search steepest descent.
//!
//! Initial line search step size is chosen by incrementing the previous step size.
//!
//! # Examples
//!
//! ```
//! use ndarray::prelude::*;
//! use optimal_steepest::backtracking_steepest::*;
//!
//! fn main() {
//!     let backtracking_rate = BacktrackingRate::default();
//!     println!(
//!         "{}",
//!         Config::new(
//!             SufficientDecreaseParameter::default(),
//!             backtracking_rate,
//!             IncrRate::from_backtracking_rate(backtracking_rate),
//!         )
//!         .start(std::iter::repeat(-10.0..=10.0).take(2), obj_func, |x| (
//!             obj_func(x),
//!             obj_func_d(x)
//!         ))
//!         .nth(100)
//!         .unwrap()
//!         .best_point()
//!     );
//! }
//!
//! fn obj_func(point: ArrayView1<f64>) -> f64 {
//!     point.map(|x| x.powi(2)).sum()
//! }
//!
//! fn obj_func_d(point: ArrayView1<f64>) -> Array1<f64> {
//!     point.map(|x| 2.0 * x)
//! }
//! ```

use std::{
    fmt::Debug,
    hint::unreachable_unchecked,
    ops::{Add, Div, Mul, Neg, RangeInclusive, Sub},
};

use derive_bounded::{
    derive_into_inner, derive_new_from_bounded_partial_ord,
    derive_new_from_lower_bounded_partial_ord,
};
use derive_getters::Getters;
use derive_more::Display;
use ndarray::{linalg::Dot, prelude::*, Data, Zip};
use num_traits::{
    bounds::{LowerBounded, UpperBounded},
    real::Real,
    AsPrimitive, One,
};
use once_cell::sync::OnceCell;
pub use optimal_core::prelude::*;
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

pub use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A running fixed step size steepest descent optimizer.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BacktrackingSteepest<A, F, FD> {
    /// Optimizer configuration.
    config: Config<A>,

    /// State of optimizer.
    state: State<A>,

    /// Objective function to minimize.
    obj_func: F,

    /// Function returning value and partial derivatives
    /// of objective function to minimize.
    obj_func_d: FD,

    #[getter(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    evaluation_cache: OnceCell<Evaluation<A>>,
}

impl<A, F, FD> BacktrackingSteepest<A, F, FD> {
    fn new(state: State<A>, config: Config<A>, obj_func: F, obj_func_d: FD) -> Self {
        Self {
            config,
            obj_func,
            obj_func_d,
            state,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Return configuration, state, and problem parameters.
    pub fn into_inner(self) -> (Config<A>, State<A>, F, FD) {
        (self.config, self.state, self.obj_func, self.obj_func_d)
    }
}

impl<A, F, FD> BacktrackingSteepest<A, F, FD>
where
    F: Fn(ArrayView1<A>) -> A,
    FD: Fn(ArrayView1<A>) -> (A, Array1<A>),
{
    /// Return value of the best point discovered,
    /// evaluating the best point if necessary.
    pub fn best_point_value(&self) -> A
    where
        A: Clone,
    {
        self.state
            .stored_best_point_value()
            .cloned()
            .unwrap_or_else(|| (self.obj_func)(self.state.best_point()))
    }

    /// Return evaluation of current state,
    /// evaluating and caching if necessary.
    pub fn evaluation(&self) -> &Evaluation<A> {
        self.evaluation_cache.get_or_init(|| self.evaluate())
    }

    fn evaluate(&self) -> Evaluation<A> {
        match &self.state {
            State::Ready(x) => Evaluation::ValueAndDerivatives((self.obj_func_d)(x.point().into())),
            State::Searching(x) => Evaluation::Value((self.obj_func)(x.point().into())),
        }
    }
}

impl<A, F, FD> StreamingIterator for BacktrackingSteepest<A, F, FD>
where
    A: Real + 'static,
    f64: AsPrimitive<A>,
    F: Fn(ArrayView1<A>) -> A,
    FD: Fn(ArrayView1<A>) -> (A, Array1<A>),
{
    type Item = Self;

    fn advance(&mut self) {
        let evaluation = self
            .evaluation_cache
            .take()
            .unwrap_or_else(|| self.evaluate());
        replace_with::replace_with_or_abort(&mut self.state, |state| {
            match state {
                State::Ready(x) => {
                    let (point_value, point_derivatives) = match evaluation {
                        Evaluation::ValueAndDerivatives(x) => x,
                        // `unreachable_unchecked` is safe if this method is safe,
                        // because `evaluate` always returns `ValueAndDerivatives`
                        // for `State::Ready`.
                        _ => unsafe { unreachable_unchecked() },
                    };
                    x.step_from_evaluated(&self.config, point_value, point_derivatives)
                }
                State::Searching(x) => {
                    let point_value = match evaluation {
                        Evaluation::Value(x) => x,
                        // `unreachable_unchecked` is safe if this method is safe,
                        // because `evaluate` always returns `Value`
                        // for `State::Searching`.
                        _ => unsafe { unreachable_unchecked() },
                    };
                    x.step_from_evaluated(&self.config, point_value)
                }
            }
        });
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

impl<A, F, FD> Optimizer for BacktrackingSteepest<A, F, FD>
where
    A: Clone,
{
    type Point = Array1<A>;

    fn best_point(&self) -> Self::Point {
        self.state.best_point().into_owned()
    }
}

/// Backtracking steepest descent configuration parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A> {
    /// The sufficient decrease parameter,
    /// `c_1`.
    pub c_1: SufficientDecreaseParameter<A>,
    /// Rate to decrease step size while line searching.
    pub backtracking_rate: BacktrackingRate<A>,
    /// Rate to increase step size before starting each line search.
    pub initial_step_size_incr_rate: IncrRate<A>,
}

/// Backtracking steepest descent state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State<A> {
    /// Ready to begin line search.
    Ready(Ready<A>),
    /// Line searching.
    Searching(Searching<A>),
}

/// Ready to begin line search.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ready<A> {
    point: Point<A>,
    last_step_size: A,
}

/// Line searching.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Searching<A> {
    point: Point<A>,
    point_value: A,
    step_direction: Array1<A>,
    c_1_times_point_derivatives_dot_step_direction: A,
    step_size: A,
    point_at_step: Point<A>,
}

/// A backtracking steepest descent evaluation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Evaluation<A> {
    /// An objective value.
    Value(A),
    /// An objective value and point derivatives.
    ValueAndDerivatives((A, Array1<A>)),
}

impl<A> Config<A> {
    /// Return a new 'Config'.
    pub fn new(
        c_1: SufficientDecreaseParameter<A>,
        backtracking_rate: BacktrackingRate<A>,
        initial_step_size_incr_rate: IncrRate<A>,
    ) -> Self {
        Self {
            c_1,
            backtracking_rate,
            initial_step_size_incr_rate,
        }
    }
}

impl<A> Config<A> {
    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    pub fn start<F, FD>(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        obj_func: F,
        obj_func_d: FD,
    ) -> BacktrackingSteepest<A, F, FD>
    where
        A: Debug + SampleUniform + Real,
        F: Fn(ArrayView1<A>) -> A,
        FD: Fn(ArrayView1<A>) -> (A, Array1<A>),
    {
        BacktrackingSteepest::new(
            self.initial_state_using(initial_bounds, &mut thread_rng()),
            self,
            obj_func,
            obj_func_d,
        )
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    pub fn start_using<F, FD, R>(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        obj_func: F,
        obj_func_d: FD,
        rng: &mut R,
    ) -> BacktrackingSteepest<A, F, FD>
    where
        A: Debug + SampleUniform + Real,
        F: Fn(ArrayView1<A>) -> A,
        FD: Fn(ArrayView1<A>) -> (A, Array1<A>),
        R: Rng,
    {
        BacktrackingSteepest::new(
            self.initial_state_using(initial_bounds, rng),
            self,
            obj_func,
            obj_func_d,
        )
    }

    /// Return this optimizer
    /// running on the given problem.
    /// if the given `state` is valid.
    pub fn start_from<F, FD>(
        self,
        obj_func: F,
        obj_func_d: FD,
        state: State<A>,
    ) -> BacktrackingSteepest<A, F, FD>
    where
        F: Fn(ArrayView1<A>) -> A,
        FD: Fn(ArrayView1<A>) -> (A, Array1<A>),
    {
        // Note,
        // this assumes states cannot be modified
        // outside of `initial_state`
        // and `step`.
        // As of the writing of this method,
        // all states are derived from an initial state.
        // Otherwise,
        // state would need validation,
        // and this method would need to return a `Result`.
        BacktrackingSteepest::new(state, self, obj_func, obj_func_d)
    }

    fn initial_state_using<R>(
        &self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        rng: &mut R,
    ) -> State<A>
    where
        A: Debug + SampleUniform + Real,
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
            StepSize::new(A::one()).unwrap(),
        )
    }
}

impl<A> State<A> {
    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> ArrayView1<A> {
        (match self {
            State::Ready(x) => x.point(),
            State::Searching(x) => x.point(),
        })
        .into()
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> ArrayView1<A> {
        (match self {
            State::Ready(x) => x.best_point(),
            State::Searching(x) => x.best_point(),
        })
        .view()
    }

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    pub fn stored_best_point_value(&self) -> Option<&A> {
        match self {
            State::Ready(_) => None,
            State::Searching(x) => Some(&x.point_value),
        }
    }
}

impl<A> State<A> {
    /// Return an initial state.
    pub fn new(point: Point<A>, initial_step_size: StepSize<A>) -> Self {
        Self::Ready(Ready {
            point,
            last_step_size: initial_step_size.0,
        })
    }

    /// Return length of point in this state.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Ready(x) => x.point.len(),
            Self::Searching(x) => x.point.len(),
        }
    }
}

impl<A> Ready<A> {
    fn point(&self) -> &Point<A> {
        self.best_point()
    }

    fn best_point(&self) -> &Point<A> {
        &self.point
    }

    fn step_from_evaluated<S>(
        self,
        config: &Config<A>,
        point_value: A,
        point_derivatives: ArrayBase<S, Ix1>,
    ) -> State<A>
    where
        A: 'static
            + Clone
            + Copy
            + Neg<Output = A>
            + Add<Output = A>
            + Sub<Output = A>
            + Div<Output = A>
            + One,
        S: Data<Elem = A>,
        f64: AsPrimitive<A>,
        ArrayBase<S, Ix1>: Dot<Array1<A>, Output = A>,
    {
        let step_direction = point_derivatives.mapv(|x| -x);
        let step_size = config.initial_step_size_incr_rate * self.last_step_size;
        State::Searching(Searching {
            point_at_step: descend(&self.point, step_size, &step_direction),
            point: self.point,
            point_value,
            c_1_times_point_derivatives_dot_step_direction: config.c_1.0
                * point_derivatives.dot(&step_direction),
            step_direction,
            step_size,
        })
    }
}

impl<A> Searching<A> {
    fn best_point(&self) -> &Point<A> {
        &self.point
    }

    #[allow(clippy::misnamed_getters)]
    fn point(&self) -> &Point<A> {
        &self.point_at_step
    }

    fn step_from_evaluated(mut self, config: &Config<A>, point_value: A) -> State<A>
    where
        A: Clone + Copy + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        if is_sufficient_decrease(
            self.point_value,
            self.step_size,
            self.c_1_times_point_derivatives_dot_step_direction,
            point_value,
        ) {
            State::Ready(Ready {
                point: self.point_at_step,
                last_step_size: self.step_size,
            })
        } else {
            self.step_size = config.backtracking_rate.0 * self.step_size;
            self.point_at_step = descend(&self.point, self.step_size, &self.step_direction);
            State::Searching(self)
        }
    }
}

fn descend<A>(point: &Point<A>, step_size: A, step_direction: &Array1<A>) -> Point<A>
where
    A: Clone + Add<Output = A> + Mul<Output = A>,
{
    Zip::from(point)
        .and(step_direction)
        .map_collect(|x, d| x.clone() + step_size.clone() * d.clone())
}

/// The sufficient decrease condition,
/// also known as the Armijo rule,
/// mathematically written as:
/// $f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)$.
fn is_sufficient_decrease<A>(
    point_value: A,
    step_size: A,
    c_1_times_point_derivatives_dot_step_direction: A,
    new_point_value: A,
) -> bool
where
    A: PartialOrd + Add<Output = A> + Mul<Output = A>,
{
    new_point_value <= point_value + step_size * c_1_times_point_derivatives_dot_step_direction
}

/// The sufficient decrease parameter,
/// `c_1`.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// Rate to increase step size before starting each line search.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

impl<A> Mul<A> for IncrRate<A>
where
    A: Mul<Output = A>,
{
    type Output = A;

    fn mul(self, rhs: A) -> Self::Output {
        self.0 * rhs
    }
}

type Point<A> = Array1<A>;