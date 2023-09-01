#![allow(clippy::needless_doctest_main)]

//! Backtracking line search steepest descent.
//!
//! Initial line search step size is chosen by incrementing the previous step size.
//!
//! # Examples
//!
//! ```
//! use optimal_steepest::backtracking_steepest::*;
//!
//! fn main() {
//!     println!(
//!         "{:?}",
//!         Config::default()
//!             .start(std::iter::repeat(-10.0..=10.0).take(2), obj_func, |x| (
//!                 obj_func(x),
//!                 obj_func_d(x)
//!             ))
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

mod state_machine;
mod types;

use std::{
    fmt::Debug,
    hint::unreachable_unchecked,
    iter::Sum,
    ops::{Div, RangeInclusive, Sub},
};

use derive_getters::Getters;
use num_traits::{real::Real, AsPrimitive, One};
use once_cell::sync::OnceCell;
pub use optimal_core::prelude::*;
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use self::state_machine::DynState;
pub use self::types::*;

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
pub struct State<A> {
    inner: DynState<A>,
}

/// A backtracking steepest descent evaluation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Evaluation<A> {
    /// An objective value.
    Value(A),
    /// An objective value and point derivatives.
    ValueAndDerivatives((A, Vec<A>)),
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
    F: Fn(&[A]) -> A,
    FD: Fn(&[A]) -> (A, Vec<A>),
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
        match &self.state.inner {
            DynState::Ready(x) => Evaluation::ValueAndDerivatives((self.obj_func_d)(x.point())),
            DynState::Searching(x) => Evaluation::Value((self.obj_func)(x.point())),
        }
    }
}

impl<A, F, FD> StreamingIterator for BacktrackingSteepest<A, F, FD>
where
    A: Real + Sum + 'static,
    f64: AsPrimitive<A>,
    F: Fn(&[A]) -> A,
    FD: Fn(&[A]) -> (A, Vec<A>),
{
    type Item = Self;

    fn advance(&mut self) {
        let evaluation = self
            .evaluation_cache
            .take()
            .unwrap_or_else(|| self.evaluate());
        replace_with::replace_with_or_abort(&mut self.state.inner, |state| {
            match state {
                DynState::Ready(x) => {
                    let (point_value, point_derivatives) = match evaluation {
                        Evaluation::ValueAndDerivatives(x) => x,
                        // `unreachable_unchecked` is safe if this method is safe,
                        // because `evaluate` always returns `ValueAndDerivatives`
                        // for `State::Ready`.
                        _ => unsafe { unreachable_unchecked() },
                    };
                    DynState::Searching(x.to_searching(
                        self.config.initial_step_size_incr_rate,
                        self.config.c_1,
                        point_value,
                        point_derivatives,
                    ))
                }
                DynState::Searching(x) => {
                    let point_value = match evaluation {
                        Evaluation::Value(x) => x,
                        // `unreachable_unchecked` is safe if this method is safe,
                        // because `evaluate` always returns `Value`
                        // for `State::Searching`.
                        _ => unsafe { unreachable_unchecked() },
                    };
                    if x.is_sufficient_decrease(point_value) {
                        DynState::Ready(x.to_ready())
                    } else {
                        DynState::Searching(x.to_searching(self.config.backtracking_rate))
                    }
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
    type Point = Vec<A>;

    fn best_point(&self) -> Self::Point {
        self.state.best_point().into()
    }
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

impl<A> Default for Config<A>
where
    A: 'static + Copy + One + Sub<Output = A> + Div<Output = A>,
    f64: AsPrimitive<A>,
{
    fn default() -> Self {
        let backtracking_rate = BacktrackingRate::default();
        Self {
            c_1: SufficientDecreaseParameter::default(),
            backtracking_rate: BacktrackingRate::default(),
            initial_step_size_incr_rate: IncrRate::from_backtracking_rate(backtracking_rate),
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
        F: Fn(&[A]) -> A,
        FD: Fn(&[A]) -> (A, Vec<A>),
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
        F: Fn(&[A]) -> A,
        FD: Fn(&[A]) -> (A, Vec<A>),
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
        F: Fn(&[A]) -> A,
        FD: Fn(&[A]) -> (A, Vec<A>),
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
    /// Return an initial state.
    pub fn new(point: Vec<A>, initial_step_size: StepSize<A>) -> Self {
        Self {
            inner: DynState::new(point, initial_step_size),
        }
    }

    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> &[A] {
        match &self.inner {
            DynState::Ready(x) => x.point(),
            DynState::Searching(x) => x.point(),
        }
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> &[A] {
        match &self.inner {
            DynState::Ready(x) => x.best_point(),
            DynState::Searching(x) => x.best_point(),
        }
    }

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    pub fn stored_best_point_value(&self) -> Option<&A> {
        match &self.inner {
            DynState::Ready(_) => None,
            DynState::Searching(x) => Some(x.point_value()),
        }
    }

    /// Return length of point in this state.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match &self.inner {
            DynState::Ready(x) => x.point().len(),
            DynState::Searching(x) => x.point().len(),
        }
    }
}
