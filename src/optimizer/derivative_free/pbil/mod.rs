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
//!     let mut o = pbil::Config::start_default_for(Count);
//!     let point = pbil::UntilConvergedConfig::default().argmin(&mut o);
//!     let point_value = Count.evaluate(point.view().into());
//!     println!("f({point}) = {point_value}");
//! }
//!
//! struct Count;
//!
//! impl Problem for Count {
//!     type Point<'a> = CowArray<'a, bool, Ix1>;
//!     type Value = u64;
//!
//!     fn evaluate(&self, point: Self::Point<'_>) -> Self::Value {
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

use ndarray::prelude::*;
use rand_xoshiro::SplitMix64;

use crate::{optimizer::MismatchedLengthError, prelude::*};

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

trait Probabilities {
    fn probabilities(&self) -> &Array1<Probability>;
}

impl<P, C> Probabilities for RunningOptimizer<P, C>
where
    C: OptimizerConfig<P>,
    C::State: Probabilities,
{
    fn probabilities(&self) -> &Array1<Probability> {
        self.state().probabilities()
    }
}

impl Probabilities for State {
    fn probabilities(&self) -> &Array1<Probability> {
        self.probabilities()
    }
}

/// PBIL runner
/// to check for converged probabilities.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UntilConvergedConfig {
    /// Probability convergence parameter.
    pub converged_threshold: ConvergedThreshold,
}

impl<I> RunnerConfig<I> for UntilConvergedConfig
where
    I: Probabilities,
{
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn is_done(&self, it: &I, _: &Self::State) -> bool {
        converged(&self.converged_threshold, it.probabilities())
    }
}

/// PBIL optimizer.
pub type Pbil<P> = Optimizer<P, Config>;

/// PBIL configuration parameters.
#[derive(Clone, Debug, PartialEq)]
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

impl<P> DefaultFor<P> for Config
where
    P: FixedLength,
{
    fn default_for(problem: P) -> Self {
        Self {
            num_samples: NumSamples::default(),
            adjust_rate: AdjustRate::default(),
            mutation_chance: MutationChance::default_for(problem),
            mutation_adjust_rate: MutationAdjustRate::default(),
        }
    }
}

impl<P> OptimizerConfig<P> for Config
where
    for<'a> P: Problem<Point<'a> = CowArray<'a, bool, Ix1>> + FixedLength + 'a,
    P::Value: Debug + PartialOrd,
{
    type State = State;

    type StateErr = MismatchedLengthError;

    type Evaluation = Option<Array1<P::Value>>;

    fn validate_state(&self, problem: &P, state: &Self::State) -> Result<(), Self::StateErr> {
        // If `Sampling::samples` could be changed independent of `probabilities`,
        // it would need to be validated.
        if state.probabilities().len() == problem.len() {
            Ok(())
        } else {
            Err(MismatchedLengthError)
        }
    }

    fn initial_state(&self, problem: &P) -> Self::State {
        State::Ready(Ready::initial(problem.len()))
    }

    unsafe fn evaluate(&self, problem: &P, state: &Self::State) -> Self::Evaluation {
        match state {
            State::Ready(_) => None,
            State::Sampling(s) => Some(problem.evaluate_population(s.samples().into())),
            State::Mutating(_) => None,
        }
    }

    unsafe fn step_from_evaluated(
        &self,
        evaluation: Self::Evaluation,
        state: Self::State,
    ) -> Self::State {
        match state {
            State::Ready(s) => State::Sampling(s.to_sampling(self.num_samples)),
            State::Sampling(s) => {
                // `unwrap_unchecked` is safe if this method is safe,
                // because `evaluate` always returns `Some`
                // for `State::Sampling`.
                State::Mutating(
                    s.to_mutating(self.adjust_rate, unsafe { evaluation.unwrap_unchecked() }),
                )
            }
            State::Mutating(s) => {
                State::Ready(s.to_ready(self.mutation_chance, self.mutation_adjust_rate))
            }
        }
    }
}

impl<P> StochasticOptimizerConfig<P, SplitMix64> for Config
where
    for<'a> P: Problem<Point<'a> = CowArray<'a, bool, Ix1>> + FixedLength + 'a,
    P::Value: Debug + PartialOrd,
{
    fn initial_state_using(&self, problem: &P, rng: &mut SplitMix64) -> Self::State {
        State::Ready(Ready::initial_using(problem.len(), rng))
    }
}

impl<P> OptimizerState<P> for State
where
    for<'a> P: Problem<Point<'a> = CowArray<'a, bool, Ix1>> + 'a,
{
    type Evaluatee<'a> = Option<ArrayView2<'a, bool>>;

    fn evaluatee(&self) -> Self::Evaluatee<'_> {
        match self {
            State::Ready(_) => None,
            State::Sampling(s) => Some(s.samples().into()),
            State::Mutating(_) => None,
        }
    }

    fn best_point(&self) -> P::Point<'_> {
        finalize(self.probabilities()).into()
    }
}

impl State {
    /// Return PBIL probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        match &self {
            State::Ready(s) => s.probabilities(),
            State::Sampling(s) => s.probabilities(),
            State::Mutating(s) => s.probabilities(),
        }
    }
}
