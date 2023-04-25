#![allow(clippy::needless_doctest_main)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::prelude::*;
//! use optimal::{optimizer::derivative_free::pbil, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let point = pbil::DoneWhenConvergedConfig::default(Count).argmin();
//!     let point_value = Count.evaluate(point.view().into());
//!     println!("f({}) = {}", point, point_value);
//! }
//!
//! #[derive(Clone, Debug)]
//! struct Count;
//!
//! impl Problem for Count {
//!     type PointElem = bool;
//!     type PointValue = u64;
//!
//!     fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
//!         point.fold(0, |acc, b| acc + *b as u64)
//!     }
//! }
//!
//! impl FixedLength for Count {
//!     fn len(&self) -> usize {
//!         16
//!     }
//! }
//! ```

mod states;
mod types;

use lazy_static::lazy_static;
use ndarray::{prelude::*, Data};
use rand::prelude::*;
use rand_xoshiro::SplitMix64;
use replace_with::replace_with_or_abort;
use std::{borrow::Borrow, fmt::Debug, marker::PhantomData};

use crate::prelude::*;

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// PBIL configuration parameters
/// with check for converged probabilities.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DoneWhenConvergedConfig<P> {
    /// Probability convergence parameter.
    pub converged_threshold: ConvergedThreshold,
    /// Regular PBIL configuration.
    pub inner: Config<P>,
}

/// Running PBIL optimizer with check for converged probabilities.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RunningDoneWhenConverged<P, C> {
    problem: PhantomData<P>,
    /// PBIL configuration parameters
    /// with check for converged probabilities.
    pub config: C,
    /// PBIL state.
    pub state: State,
}

impl<P> DoneWhenConvergedConfig<P> {
    /// Convenience function
    /// to populate every field
    /// with their default.
    pub fn default(problem: P) -> Self
    where
        P: FixedLength,
    {
        Self {
            converged_threshold: ConvergedThreshold::default(),
            inner: Config::default(problem),
        }
    }
}

impl<P> OptimizerConfig for DoneWhenConvergedConfig<P>
where
    P: Problem<PointElem = bool> + FixedLength,
{
    type Problem = P;
    type Optimizer = RunningDoneWhenConverged<P, Self>;

    fn start(self) -> Self::Optimizer {
        let state = State::initial(self.inner.problem.len());
        RunningDoneWhenConverged {
            problem: PhantomData,
            config: self,
            state,
        }
    }

    fn problem(&self) -> &P {
        &self.inner.problem
    }
}

impl<P> StochasticOptimizerConfig<SplitMix64> for DoneWhenConvergedConfig<P>
where
    P: Problem<PointElem = bool> + FixedLength,
{
    fn start_using(
        self,
        rng: &mut SplitMix64,
    ) -> RunningDoneWhenConverged<P, DoneWhenConvergedConfig<P>> {
        let state = State::initial_using(self.inner.problem.len(), rng);
        RunningDoneWhenConverged {
            problem: PhantomData,
            config: self,
            state,
        }
    }
}

impl<P, C> RunningDoneWhenConverged<P, C> {
    /// Return optimizer configuration.
    pub fn config(&self) -> &C {
        &self.config
    }

    /// Return state of optimizer.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Stop optimization run,
    /// returning configuration and state.
    pub fn into_inner(self) -> (C, State) {
        (self.config, self.state)
    }
}

impl<P, C> RunningOptimizerBase for RunningDoneWhenConverged<P, C>
where
    P: Problem<PointElem = bool>,
    C: Borrow<DoneWhenConvergedConfig<P>>,
{
    type Problem = P;

    fn problem(&self) -> &Self::Problem {
        &self.config.borrow().inner.problem
    }

    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }

    fn stored_best_point_value(&self) -> Option<&<Self::Problem as Problem>::PointValue> {
        None
    }
}

impl<P, C> RunningOptimizerStep for RunningDoneWhenConverged<P, C>
where
    P: Problem<PointElem = bool> + FixedLength,
    C: Borrow<DoneWhenConvergedConfig<P>>,
    <Self::Problem as Problem>::PointValue: Debug + PartialOrd,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config.borrow().inner.step_from_evaluated(
                self.config
                    .borrow()
                    .inner
                    .problem
                    .evaluate_population(state.points().into()),
                state,
            )
        })
    }
}

impl<P, C> Convergent for RunningDoneWhenConverged<P, C>
where
    P: Problem<PointElem = bool>,
    C: Borrow<DoneWhenConvergedConfig<P>>,
{
    fn is_done(&self) -> bool {
        converged(
            &self.config.borrow().converged_threshold,
            self.state.probabilities(),
        )
    }
}

impl<P, C> PopulationBased for RunningDoneWhenConverged<P, C>
where
    P: Problem<PointElem = bool>,
    C: Borrow<DoneWhenConvergedConfig<P>>,
{
    fn points(&self) -> ArrayView2<bool> {
        self.state.points()
    }
}

/// PBIL configuration parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<P> {
    /// An optimization problem.
    pub problem: P,
    /// Number of samples generated
    /// during steps.
    pub num_samples: NumSamples,
    /// Degree to adjust probabilities towards best point
    /// during steps.
    pub adjust_rate: AdjustRate,
    /// Probability for each probability to mutate,
    /// independently.
    pub mutation_chance: MutationChance,
    /// Degree to adjust probability towards random value
    /// when mutating.
    pub mutation_adjust_rate: MutationAdjustRate,
}

/// Running PBIL optimizer.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Running<P, C> {
    problem: PhantomData<P>,
    /// PBIL configuration parameters.
    pub config: C,
    /// PBIL state.
    pub state: State,
}

/// PBIL state.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State {
    /// Ready to start a new iteration.
    Ready(Ready),
    /// For sampling
    /// and adjusting probabilities
    /// based on samples.
    Sampling(Sampling),
    /// For mutating probabilities.
    Mutating(Mutating),
}

impl<P> Config<P> {
    /// Return a new PBIL configuration.
    pub fn new(
        problem: P,
        num_samples: NumSamples,
        adjust_rate: AdjustRate,
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
    ) -> Self {
        Self {
            problem,
            num_samples,
            adjust_rate,
            mutation_chance,
            mutation_adjust_rate,
        }
    }

    /// Convenience function
    /// to populate every field
    /// with their default.
    pub fn default(problem: P) -> Self
    where
        P: FixedLength,
    {
        Self {
            num_samples: Default::default(),
            adjust_rate: Default::default(),
            mutation_chance: MutationChance::default(problem.len()),
            mutation_adjust_rate: Default::default(),
            problem,
        }
    }
}

impl<P> Config<P> {
    /// Return the next state,
    /// given point values.
    fn step_from_evaluated<B, S>(&self, point_values: ArrayBase<S, Ix1>, state: State) -> State
    where
        B: Debug + PartialOrd,
        S: Data<Elem = B>,
    {
        match state {
            State::Ready(s) => State::Sampling(s.to_sampling(self.num_samples)),
            State::Sampling(s) => State::Mutating(s.to_mutating(self.adjust_rate, point_values)),
            State::Mutating(s) => {
                State::Ready(s.to_ready(self.mutation_chance, self.mutation_adjust_rate))
            }
        }
    }
}

impl<P> OptimizerConfig for Config<P>
where
    P: Problem<PointElem = bool> + FixedLength,
{
    type Problem = P;
    type Optimizer = Running<P, Self>;

    fn start(self) -> Running<P, Config<P>> {
        let state = State::initial(self.problem.len());
        Running {
            problem: PhantomData,
            config: self,
            state,
        }
    }

    fn problem(&self) -> &P {
        &self.problem
    }
}

impl<P> StochasticOptimizerConfig<SplitMix64> for Config<P>
where
    P: Problem<PointElem = bool> + FixedLength,
{
    fn start_using(self, rng: &mut SplitMix64) -> Running<P, Config<P>> {
        let state = State::initial_using(self.problem.len(), rng);
        Running {
            problem: PhantomData,
            config: self,
            state,
        }
    }
}

impl<P, C> Running<P, C> {
    /// Return optimizer configuration.
    pub fn config(&self) -> &C {
        &self.config
    }

    /// Return state of optimizer.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Stop optimization run,
    /// returning configuration and state.
    pub fn into_inner(self) -> (C, State) {
        (self.config, self.state)
    }
}

impl<P, C> RunningOptimizerBase for Running<P, C>
where
    P: Problem<PointElem = bool>,
    C: Borrow<Config<P>>,
{
    type Problem = P;

    fn problem(&self) -> &Self::Problem {
        &self.config.borrow().problem
    }

    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }

    fn stored_best_point_value(&self) -> Option<&<Self::Problem as Problem>::PointValue> {
        None
    }
}

impl<P, C> RunningOptimizerStep for Running<P, C>
where
    P: Problem<PointElem = bool> + FixedLength,
    C: Borrow<Config<P>>,
    <Self::Problem as Problem>::PointValue: Debug + PartialOrd,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config.borrow().step_from_evaluated(
                self.config
                    .borrow()
                    .problem
                    .evaluate_population(state.points().into()),
                state,
            )
        })
    }
}

impl<P, C> PopulationBased for Running<P, C>
where
    P: Problem<PointElem = bool>,
    C: Borrow<Config<P>>,
{
    fn points(&self) -> ArrayView2<bool> {
        self.state.points()
    }
}

impl State {
    fn initial(num_bits: usize) -> Self {
        Self::Ready(Ready::initial(num_bits))
    }

    fn initial_using<R>(num_bits: usize, rng: &mut R) -> Self
    where
        R: Rng,
    {
        Self::Ready(Ready::initial_using(num_bits, rng))
    }

    /// Return PBIL probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        match &self {
            State::Ready(s) => s.probabilities(),
            State::Sampling(s) => s.probabilities(),
            State::Mutating(s) => s.probabilities(),
        }
    }

    fn points(&self) -> ArrayView2<bool> {
        lazy_static! {
            static ref EMPTY: Array2<bool> = Array::from_elem((0, 0), false);
        }
        match self {
            State::Ready(_) => EMPTY.view(),
            State::Sampling(s) => s.samples().view(),
            State::Mutating(_) => EMPTY.view(),
        }
    }

    fn best_point(&self) -> CowArray<bool, Ix1> {
        finalize(self.probabilities()).into()
    }
}
