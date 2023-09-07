#![allow(clippy::needless_doctest_main)]

//! Fixed-step-size steepest descent,
//! a very simple derivative optimizer.
//!
//! # Examples
//!
//! ```
//! use optimal_steepest::fixed_step_steepest::*;
//!
//! println!(
//!     "{:?}",
//!     Config::new(StepSize::new(0.5).unwrap())
//!         .start(std::iter::repeat(-10.0..=10.0).take(2), |point| point
//!             .iter()
//!             .map(|x| 2.0 * x)
//!             .collect())
//!         .nth(100)
//!         .unwrap()
//!         .best_point()
//! );
//! ```

mod state_machine;

use std::ops::{Mul, RangeInclusive, SubAssign};

use derive_getters::{Dissolve, Getters};
pub use optimal_core::prelude::*;
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use self::state_machine::*;

pub use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A running fixed-step-size steepest descent optimizer.
#[derive(Clone, Debug, Getters, Dissolve)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct FixedStepSteepest<A, FD> {
    /// Optimizer configuration.
    config: Config<A>,

    /// State of optimizer.
    state: State<A>,

    /// Derivative of objective function to minimize.
    obj_func_d: FD,
}

/// Fixed-step-size steepest descent configuration parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A> {
    /// Length of each step.
    pub step_size: StepSize<A>,
}

/// Fixed-step-size steepest descent state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct State<A>(DynState<A>);

/// Fixed-step-size steepest descent state kind.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StateKind {
    /// Iteration started.
    Started,
    /// Point differentiated.
    Evaluated,
    /// Step taken from differentiation.
    Stepped,
    /// Iteration finished.
    Finished,
}

impl<A, FD> FixedStepSteepest<A, FD> {
    fn new(state: State<A>, config: Config<A>, obj_func_d: FD) -> Self {
        Self {
            config,
            obj_func_d,
            state,
        }
    }
}

impl<A, FD> StreamingIterator for FixedStepSteepest<A, FD>
where
    A: Clone + SubAssign + Mul<Output = A>,
    FD: Fn(&[A]) -> Vec<A>,
{
    type Item = Self;

    fn advance(&mut self) {
        replace_with::replace_with_or_abort(&mut self.state.0, |state| match state {
            DynState::Started(x) => DynState::Evaluated(x.into_evaluated(&self.obj_func_d)),
            DynState::Evaluated(x) => {
                DynState::Stepped(x.into_stepped(self.config.step_size.clone()))
            }
            DynState::Stepped(x) => DynState::Finished(x.into_finished()),
            DynState::Finished(x) => DynState::Started(x.into_started()),
        });
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

impl<A, FD> Optimizer for FixedStepSteepest<A, FD>
where
    A: Clone,
{
    type Point = Vec<A>;

    fn best_point(&self) -> Self::Point {
        match &self.state.0 {
            DynState::Started(x) => x.point.clone(),
            DynState::Evaluated(x) => x.point.x().clone(),
            DynState::Stepped(x) => x.point.clone(),
            DynState::Finished(x) => x.point.clone(),
        }
    }
}

impl<A> Config<A> {
    /// Return a new 'Config'.
    pub fn new(step_size: StepSize<A>) -> Self {
        Self { step_size }
    }
}

impl<A> Config<A> {
    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    ///
    /// - `initial_bounds`: bounds for generating the initial random point
    /// - `obj_func_d`: derivative of objective function to minimize
    pub fn start<FD>(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        obj_func_d: FD,
    ) -> FixedStepSteepest<A, FD>
    where
        A: SampleUniform,
        FD: Fn(&[A]) -> Vec<A>,
    {
        FixedStepSteepest::new(
            self.initial_state_using(initial_bounds, &mut thread_rng()),
            self,
            obj_func_d,
        )
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    ///
    /// - `initial_bounds`: bounds for generating the initial random point
    /// - `obj_func_d`: derivative of objective function to minimize
    /// - `rng`: source of randomness
    pub fn start_using<FD, R>(
        self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        obj_func_d: FD,
        rng: &mut R,
    ) -> FixedStepSteepest<A, FD>
    where
        A: SampleUniform,
        FD: Fn(&[A]) -> Vec<A>,
        R: Rng,
    {
        FixedStepSteepest::new(
            self.initial_state_using(initial_bounds, rng),
            self,
            obj_func_d,
        )
    }

    /// Return this optimizer
    /// running on the given problem.
    /// if the given `state` is valid.
    ///
    /// - `obj_func_d`: derivative of objective function to minimize
    /// - `state`: initial point to start from
    pub fn start_from<FD>(self, obj_func_d: FD, state: State<A>) -> FixedStepSteepest<A, FD>
    where
        FD: Fn(&[A]) -> Vec<A>,
    {
        FixedStepSteepest::new(state, self, obj_func_d)
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
        State(DynState::new(
            initial_bounds
                .into_iter()
                .map(|range| {
                    let (start, end) = range.into_inner();
                    Uniform::new_inclusive(start, end).sample(rng)
                })
                .collect(),
        ))
    }
}

impl<A> State<A> {
    /// Return an initial state.
    pub fn new(point: Vec<A>) -> Self {
        Self(DynState::new(point))
    }

    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> Option<&[A]> {
        match &self.0 {
            DynState::Started(x) => Some(&x.point),
            DynState::Evaluated(_) => None,
            DynState::Stepped(_) => None,
            DynState::Finished(_) => None,
        }
    }

    /// Return result of evaluation.
    pub fn evaluation(&self) -> Option<&[A]> {
        match &self.0 {
            DynState::Started(_) => None,
            DynState::Evaluated(x) => Some(x.point.value()),
            DynState::Stepped(_) => None,
            DynState::Finished(_) => None,
        }
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> &[A] {
        match &self.0 {
            DynState::Started(x) => &x.point,
            DynState::Evaluated(x) => x.point.x(),
            DynState::Stepped(x) => &x.point,
            DynState::Finished(x) => &x.point,
        }
    }

    /// Return kind of state of inner state-machine.
    pub fn kind(&self) -> StateKind {
        match &self.0 {
            DynState::Started(_) => StateKind::Started,
            DynState::Evaluated(_) => StateKind::Evaluated,
            DynState::Stepped(_) => StateKind::Stepped,
            DynState::Finished(_) => StateKind::Finished,
        }
    }
}
