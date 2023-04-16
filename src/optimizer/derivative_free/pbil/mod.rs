#![allow(clippy::needless_doctest_main)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::{Data, RemoveAxis, prelude::*};
//! use optimal::{optimizer::derivative_free::pbil, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter =
//!         pbil::RunningDoneWhenConverged::new(pbil::DoneWhenConvergedConfig::default(Count))
//!             .into_streaming_iter();
//!     let o = iter.find(|o| o.is_done()).expect("should converge");
//!     println!("f({}) = {}", o.best_point(), o.best_point_value());
//! }
//!
//! struct Count;
//!
//! impl Problem for Count {
//!     type PointElem = bool;
//!     type PointValue = u64;
//!
//!     fn evaluate<S>(&self, point: ArrayBase<S, Ix1>) -> u64
//!     where
//!         S: ndarray::RawData<Elem = bool> + Data,
//!     {
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
use replace_with::replace_with_or_abort;
use std::{borrow::Borrow, fmt::Debug, marker::PhantomData};

use crate::prelude::*;

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Running PBIL optimizer with check for converged probabilities.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RunningDoneWhenConverged<B, P, C> {
    point_value: PhantomData<B>,
    problem: PhantomData<P>,
    /// PBIL configuration parameters
    /// with check for converged probabilities.
    pub config: C,
    /// PBIL state.
    pub state: State,
}

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

impl<B, P, C> RunningOptimizer for RunningDoneWhenConverged<B, P, C>
where
    B: Debug + PartialOrd,
    P: Problem<PointElem = bool, PointValue = B> + FixedLength,
    C: Borrow<DoneWhenConvergedConfig<P>>,
{
    type PointElem = bool;
    type PointValue = B;
    type Config = C;
    type State = State;

    fn new(config: Self::Config) -> Self {
        let state = State::initial(config.borrow().inner.problem.len());
        Self {
            point_value: PhantomData,
            problem: PhantomData,
            config,
            state,
        }
    }

    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config.borrow().inner.step_from_evaluated(
                self.config
                    .borrow()
                    .inner
                    .problem
                    .evaluate_all(state.points().view()),
                state,
            )
        })
    }

    fn config(&self) -> &C {
        &self.config
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn stop(self) -> (C, State) {
        (self.config, self.state)
    }

    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }

    fn stored_best_point_value(&self) -> Option<&B> {
        None
    }
}

impl<B, P, C> StochasticRunningOptimizer for RunningDoneWhenConverged<B, P, C>
where
    B: Debug + PartialOrd,
    P: Problem<PointElem = bool, PointValue = B> + FixedLength,
    C: Borrow<DoneWhenConvergedConfig<P>>,
{
    fn new_using<R>(config: C, rng: &mut R) -> RunningDoneWhenConverged<B, P, C>
    where
        R: Rng,
    {
        let state = State::initial_using(config.borrow().inner.problem.len(), rng);
        Self {
            point_value: PhantomData,
            problem: PhantomData,
            config,
            state,
        }
    }
}

impl<B, P, C> Convergent for RunningDoneWhenConverged<B, P, C>
where
    B: Debug + PartialOrd,
    P: Problem<PointElem = bool, PointValue = B> + FixedLength,
    C: Borrow<DoneWhenConvergedConfig<P>>,
{
    fn is_done(&self) -> bool {
        converged(
            &self.config.borrow().converged_threshold,
            self.state.probabilities(),
        )
    }
}

impl<B, P, C> PopulationBased for RunningDoneWhenConverged<B, P, C>
where
    B: Debug + PartialOrd,
    P: Problem<PointElem = bool, PointValue = B> + FixedLength,
    C: Borrow<DoneWhenConvergedConfig<P>>,
{
    fn points(&self) -> ArrayView2<Self::PointElem> {
        self.state.points()
    }
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

impl<P> OptimizerConfig for DoneWhenConvergedConfig<P> {
    type Problem = P;

    fn problem(&self) -> &P {
        &self.borrow().inner.problem
    }
}

/// Running PBIL optimizer.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Running<B, P, C> {
    point_value: PhantomData<B>,
    problem: PhantomData<P>,
    /// PBIL configuration parameters.
    pub config: C,
    /// PBIL state.
    pub state: State,
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

impl<B, P, C> RunningOptimizer for Running<B, P, C>
where
    B: Debug + PartialOrd,
    P: Problem<PointElem = bool, PointValue = B> + FixedLength,
    C: Borrow<Config<P>>,
{
    type PointElem = bool;
    type PointValue = B;
    type Config = C;
    type State = State;

    fn new(config: Self::Config) -> Running<B, P, C> {
        let state = State::initial(config.borrow().problem.len());
        Self {
            point_value: PhantomData,
            problem: PhantomData,
            config,
            state,
        }
    }

    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config.borrow().step_from_evaluated(
                self.config.borrow().problem.evaluate_all(state.points()),
                state,
            )
        })
    }

    fn config(&self) -> &C {
        &self.config
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn stop(self) -> (C, State) {
        (self.config, self.state)
    }

    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }

    fn stored_best_point_value(&self) -> Option<&B> {
        None
    }
}

impl<B, P, C> StochasticRunningOptimizer for Running<B, P, C>
where
    B: Debug + PartialOrd,
    P: Problem<PointElem = bool, PointValue = B> + FixedLength,
    C: Borrow<Config<P>>,
{
    fn new_using<R>(config: C, rng: &mut R) -> Running<B, P, C>
    where
        R: Rng,
    {
        let state = State::initial_using(config.borrow().problem.len(), rng);
        Self {
            point_value: PhantomData,
            problem: PhantomData,
            config,
            state,
        }
    }
}

impl<B, P, C> PopulationBased for Running<B, P, C>
where
    B: Debug + PartialOrd,
    P: Problem<PointElem = bool, PointValue = B> + FixedLength,
    C: Borrow<Config<P>>,
{
    fn points(&self) -> ArrayView2<Self::PointElem> {
        self.state.points()
    }
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

impl<P> OptimizerConfig for Config<P> {
    type Problem = P;

    fn problem(&self) -> &P {
        &self.borrow().problem
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
