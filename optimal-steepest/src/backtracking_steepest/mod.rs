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
    iter::Sum,
    ops::{Div, RangeInclusive, Sub},
};

use derive_getters::{Dissolve, Getters};
use num_traits::{real::Real, AsPrimitive, One};
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
#[derive(Clone, Debug, Getters, Dissolve)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[dissolve(rename = "into_parts")]
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
pub struct State<A>(DynState<A>);

/// Backtracking steepest descent state kind.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StateKind {
    /// Iteration started.
    Started,
    /// Point evaluated and differentiated.
    Evaluated,
    /// Prepared to line search.
    InitializedSearching,
    /// Took a line-search step.
    FakeStepped,
    /// Evaluated a line-search step.
    FakeStepEvaluated,
    /// Decremented step size for next line-search iteration.
    StepSizeDecremented,
    /// Finished line-search and took a real step.
    Stepped,
    /// Iteration finished.
    Finished,
    /// Incremented step size for next iteration.
    StepSizeIncremented,
}

/// A backtracking steepest descent evaluation.
#[derive(Clone, Debug)]
pub enum Evaluation<'a, A> {
    /// An objective value.
    Value(&'a A),
    /// An objective value and point derivatives.
    ValueAndDerivatives((&'a A, &'a [A])),
}

impl<A, F, FD> BacktrackingSteepest<A, F, FD> {
    fn new(state: State<A>, config: Config<A>, obj_func: F, obj_func_d: FD) -> Self {
        Self {
            config,
            obj_func,
            obj_func_d,
            state,
        }
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
        replace_with::replace_with_or_abort(&mut self.state.0, |state| match state {
            DynState::Started(x) => DynState::Evaluated(x.into_evaluated(&self.obj_func_d)),
            DynState::Evaluated(x) => {
                DynState::InitializedSearching(x.into_initialized_searching(self.config.c_1))
            }
            DynState::InitializedSearching(x) => DynState::FakeStepped(x.into_fake_stepped()),
            DynState::FakeStepped(x) => {
                DynState::FakeStepEvaluated(x.into_fake_step_evaluated(&self.obj_func))
            }
            DynState::FakeStepEvaluated(x) => {
                if x.is_sufficient_decrease() {
                    DynState::Stepped(x.into_stepped())
                } else {
                    DynState::StepSizeDecremented(
                        x.into_step_size_decremented(self.config.backtracking_rate),
                    )
                }
            }
            DynState::StepSizeDecremented(x) => DynState::FakeStepped(x.into_fake_stepped()),
            DynState::Stepped(x) => DynState::Finished(x.into_finished()),
            DynState::Finished(x) => DynState::StepSizeIncremented(
                x.into_step_size_incremented(self.config.initial_step_size_incr_rate),
            ),
            DynState::StepSizeIncremented(x) => DynState::Started(x.into_started()),
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
        Self(DynState::new(point, initial_step_size))
    }

    /// Return length of point in this state.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match &self.0 {
            DynState::Started(x) => x.point.len(),
            DynState::Evaluated(x) => x.point.x().len(),
            DynState::InitializedSearching(x) => x.line_search.point().len(),
            DynState::FakeStepped(x) => x.line_search().point().len(),
            DynState::FakeStepEvaluated(x) => x.line_search().point().len(),
            DynState::StepSizeDecremented(x) => x.line_search.point().len(),
            DynState::Stepped(x) => x.point.len(),
            DynState::Finished(x) => x.point.len(),
            DynState::StepSizeIncremented(x) => x.point.len(),
        }
    }

    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> Option<&[A]> {
        match &self.0 {
            DynState::Started(x) => Some(&x.point),
            DynState::Evaluated(_) => None,
            DynState::InitializedSearching(_) => None,
            DynState::FakeStepped(x) => Some(x.point_at_step()),
            DynState::FakeStepEvaluated(_) => None,
            DynState::StepSizeDecremented(_) => None,
            DynState::Stepped(_) => None,
            DynState::Finished(_) => None,
            DynState::StepSizeIncremented(_) => None,
        }
    }

    /// Return result of evaluation.
    pub fn evaluation(&self) -> Option<Evaluation<A>> {
        match &self.0 {
            DynState::Started(_) => None,
            DynState::Evaluated(x) => {
                let (value, derivatives) = x.point.value();
                Some(Evaluation::ValueAndDerivatives((value, derivatives)))
            }
            DynState::InitializedSearching(_) => None,
            DynState::FakeStepped(_) => None,
            DynState::FakeStepEvaluated(x) => Some(Evaluation::Value(x.point_at_step().value())),
            DynState::StepSizeDecremented(_) => None,
            DynState::Stepped(_) => None,
            DynState::Finished(_) => None,
            DynState::StepSizeIncremented(_) => None,
        }
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> &[A] {
        match &self.0 {
            DynState::Started(x) => &x.point,
            DynState::Evaluated(x) => x.point.x(),
            DynState::InitializedSearching(x) => x.line_search.point(),
            DynState::FakeStepped(x) => x.line_search().point(),
            DynState::FakeStepEvaluated(x) => x.line_search().point(),
            DynState::StepSizeDecremented(x) => x.line_search.point(),
            DynState::Stepped(x) => &x.point,
            DynState::Finished(x) => &x.point,
            DynState::StepSizeIncremented(x) => &x.point,
        }
    }

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    pub fn stored_best_point_value(&self) -> Option<&A> {
        match &self.0 {
            DynState::Started(_) => None,
            DynState::Evaluated(x) => {
                let (value, _) = x.point.value();
                Some(value)
            }
            DynState::InitializedSearching(x) => Some(x.line_search.value()),
            DynState::FakeStepped(x) => Some(x.line_search().value()),
            DynState::FakeStepEvaluated(x) => Some(x.line_search().value()),
            DynState::StepSizeDecremented(x) => Some(x.line_search.value()),
            DynState::Stepped(_) => None,
            DynState::Finished(_) => None,
            DynState::StepSizeIncremented(_) => None,
        }
    }

    /// Return kind of state of inner state-machine.
    pub fn kind(&self) -> StateKind {
        match &self.0 {
            DynState::Started(_) => StateKind::Started,
            DynState::Evaluated(_) => StateKind::Evaluated,
            DynState::InitializedSearching(_) => StateKind::InitializedSearching,
            DynState::FakeStepped(_) => StateKind::FakeStepped,
            DynState::FakeStepEvaluated(_) => StateKind::FakeStepEvaluated,
            DynState::StepSizeDecremented(_) => StateKind::StepSizeDecremented,
            DynState::Stepped(_) => StateKind::Stepped,
            DynState::Finished(_) => StateKind::Finished,
            DynState::StepSizeIncremented(_) => StateKind::StepSizeIncremented,
        }
    }
}
