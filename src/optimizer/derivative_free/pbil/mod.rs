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
//!     let point = pbil::PbilDoneWhenConverged::default_for(Count).argmin();
//!     let point_value = Count.evaluate(point.view().into());
//!     println!("f({}) = {}", point, point_value);
//! }
//!
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

use std::fmt::Debug;

use lazy_static::lazy_static;
use ndarray::{prelude::*, Data};
use rand::prelude::*;
use rand_xoshiro::SplitMix64;

use crate::{optimizer::MismatchedLengthError, prelude::*};

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// PBIL with check for converged probabilities.
pub type PbilDoneWhenConverged<P> = Optimizer<P, DoneWhenConvergedConfig>;

/// PBIL configuration parameters
/// with check for converged probabilities.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DoneWhenConvergedConfig {
    /// Probability convergence parameter.
    pub converged_threshold: ConvergedThreshold,
    /// Regular PBIL configuration.
    pub inner: Config,
}

impl<P> OptimizerConfig<P> for DoneWhenConvergedConfig
where
    P: Problem<PointElem = bool> + FixedLength,
    P::PointValue: Debug + PartialOrd,
{
    type Err = <Config as OptimizerConfig<P>>::Err;

    type State = <Config as OptimizerConfig<P>>::State;

    type StateErr = <Config as OptimizerConfig<P>>::StateErr;

    fn validate(&self, problem: &P) -> Result<(), Self::Err> {
        self.inner.validate(problem)
    }

    fn validate_state(&self, problem: &P, state: &Self::State) -> Result<(), Self::StateErr> {
        self.inner.validate_state(problem, state)
    }

    unsafe fn initial_state(&self, problem: &P) -> Self::State {
        self.inner.initial_state(problem)
    }

    unsafe fn step(&self, problem: &P, state: Self::State) -> Self::State {
        self.inner.step(problem, state)
    }
}

impl<P> DefaultFor<&P> for DoneWhenConvergedConfig
where
    P: Problem<PointElem = bool> + FixedLength,
    P::PointValue: Debug + PartialOrd,
{
    fn default_for(problem: &P) -> Self
    where
        P: FixedLength,
    {
        Self {
            converged_threshold: ConvergedThreshold::default(),
            inner: Config::default_for(problem),
        }
    }
}

impl<P> StochasticOptimizerConfig<P, SplitMix64> for DoneWhenConvergedConfig
where
    P: Problem<PointElem = bool> + FixedLength,
    P::PointValue: Debug + PartialOrd,
{
    unsafe fn initial_state_using(&self, problem: &P, rng: &mut SplitMix64) -> Self::State {
        self.inner.initial_state_using(problem, rng)
    }
}

impl<P> Convergent<P> for DoneWhenConvergedConfig
where
    P: Problem<PointElem = bool> + FixedLength,
    P::PointValue: Debug + PartialOrd,
{
    fn is_done(&self, state: &Self::State) -> bool {
        converged(&self.converged_threshold, state.probabilities())
    }
}

/// PBIL optimizer.
pub type Pbil<P> = Optimizer<P, Config>;

/// PBIL configuration parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config {
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

impl Config {
    /// Return a new PBIL configuration.
    pub fn new(
        num_samples: NumSamples,
        adjust_rate: AdjustRate,
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
    ) -> Self {
        Self {
            num_samples,
            adjust_rate,
            mutation_chance,
            mutation_adjust_rate,
        }
    }
}

impl Config {
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

impl<P> OptimizerConfig<P> for Config
where
    P: Problem<PointElem = bool> + FixedLength,
    P::PointValue: Debug + PartialOrd,
{
    type Err = ();

    type State = State;

    type StateErr = MismatchedLengthError;

    fn validate(&self, _problem: &P) -> Result<(), Self::Err> {
        Ok(())
    }

    fn validate_state(&self, problem: &P, state: &Self::State) -> Result<(), Self::StateErr> {
        // If `Sampling::samples` could be changed independent of `probabilities`,
        // it would need to be validated.
        if state.probabilities().len() == problem.len() {
            Ok(())
        } else {
            Err(MismatchedLengthError)
        }
    }

    unsafe fn initial_state(&self, problem: &P) -> Self::State {
        State::initial(problem.len())
    }

    unsafe fn step(&self, problem: &P, state: Self::State) -> Self::State {
        self.step_from_evaluated(problem.evaluate_population(state.points().into()), state)
    }
}

impl<P> DefaultFor<&P> for Config
where
    P: Problem<PointElem = bool> + FixedLength,
    P::PointValue: Debug + PartialOrd,
{
    fn default_for(problem: &P) -> Self
    where
        P: FixedLength,
    {
        Self {
            num_samples: NumSamples::default(),
            adjust_rate: AdjustRate::default(),
            mutation_chance: MutationChance::default_for(problem),
            mutation_adjust_rate: MutationAdjustRate::default(),
        }
    }
}

impl<P> StochasticOptimizerConfig<P, SplitMix64> for Config
where
    P: Problem<PointElem = bool> + FixedLength,
    P::PointValue: Debug + PartialOrd,
{
    unsafe fn initial_state_using(&self, problem: &P, rng: &mut SplitMix64) -> Self::State {
        State::initial_using(problem.len(), rng)
    }
}

impl<P> OptimizerState<P> for State
where
    P: Problem<PointElem = bool>,
{
    fn best_point(&self) -> CowArray<<P as Problem>::PointElem, Ix1> {
        self.best_point()
    }
}

impl<P> PopulationBased<P> for State
where
    P: Problem<PointElem = bool>,
{
    fn points(&self) -> ArrayView2<P::PointElem> {
        self.points()
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
