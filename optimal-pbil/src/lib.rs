#![allow(clippy::needless_doctest_main)]
#![warn(missing_debug_implementations)]
// `missing_docs` does not work with `IsVariant`,
// see <https://github.com/JelteF/derive_more/issues/215>.
// #![warn(missing_docs)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use optimal_pbil::*;
//!
//! println!(
//!     "{:?}",
//!     UntilProbabilitiesConvergedConfig::default()
//!         .start(Config::start_default_for(16, |point| point.iter().filter(|x| **x).count()))
//!         .argmin()
//! );
//! ```

mod states;
mod types;

use std::fmt::Debug;

use default_for::DefaultFor;
use derive_getters::Getters;
use derive_more::IsVariant;
use once_cell::sync::OnceCell;
pub use optimal_core::prelude::*;
use rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus};

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Error returned when
/// problem length does not match state length.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
#[error("problem length does not match state length")]
pub struct MismatchedLengthError;

/// A type containing an array of probabilities.
pub trait Probabilities {
    /// Return probabilities.
    fn probabilities(&self) -> &[Probability];
}

impl<B, F> Probabilities for Pbil<B, F> {
    fn probabilities(&self) -> &[Probability] {
        self.state().probabilities()
    }
}

/// PBIL runner
/// to check for converged probabilities.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UntilProbabilitiesConverged<I> {
    config: UntilProbabilitiesConvergedConfig,
    it: I,
}

/// Config for PBIL runner
/// to check for converged probabilities.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UntilProbabilitiesConvergedConfig {
    /// Probability convergence parameter.
    pub threshold: ProbabilityThreshold,
}

impl UntilProbabilitiesConvergedConfig {
    /// Return this runner
    /// wrapping the given iterator.
    pub fn start<I>(self, it: I) -> UntilProbabilitiesConverged<I> {
        UntilProbabilitiesConverged { config: self, it }
    }
}

impl<I> UntilProbabilitiesConverged<I> {
    /// Return configuration and iterator.
    pub fn into_inner(self) -> (UntilProbabilitiesConvergedConfig, I) {
        (self.config, self.it)
    }
}

impl<I> StreamingIterator for UntilProbabilitiesConverged<I>
where
    I: StreamingIterator + Probabilities,
{
    type Item = I::Item;

    fn advance(&mut self) {
        self.it.advance()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.it.get()
    }

    fn is_done(&self) -> bool {
        self.it.is_done()
            || self.it.probabilities().iter().all(|p| {
                p > &self.config.threshold.upper_bound() || p < &self.config.threshold.lower_bound()
            })
    }
}

/// A running PBIL optimizer.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pbil<B, F> {
    /// Optimizer configuration.
    config: Config,

    /// State of optimizer.
    state: State<B>,

    /// Objective function to minimize.
    obj_func: F,

    #[getter(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    evaluation_cache: OnceCell<Evaluation<B>>,
}

impl<B, F> Pbil<B, F> {
    fn new(state: State<B>, config: Config, obj_func: F) -> Self {
        Self {
            config,
            obj_func,
            state,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Return configuration, state, and problem parameters.
    pub fn into_inner(self) -> (Config, State<B>, F) {
        (self.config, self.state, self.obj_func)
    }
}

impl<B, F> Pbil<B, F>
where
    F: Fn(&[bool]) -> B,
{
    /// Return value of the best point discovered.
    pub fn best_point_value(&self) -> B {
        (self.obj_func)(&self.best_point())
    }

    /// Return evaluation of current state,
    /// evaluating and caching if necessary.
    pub fn evaluation(&self) -> &Evaluation<B> {
        self.evaluation_cache.get_or_init(|| self.evaluate())
    }

    fn evaluate(&self) -> Evaluation<B> {
        self.state.evaluatee().map(|xs| (self.obj_func)(xs))
    }
}

impl<B, F> StreamingIterator for Pbil<B, F>
where
    B: PartialOrd,
    F: Fn(&[bool]) -> B,
{
    type Item = Self;

    fn advance(&mut self) {
        let evaluation = self
            .evaluation_cache
            .take()
            .unwrap_or_else(|| self.evaluate());
        replace_with::replace_with_or_abort(&mut self.state, |state| match state {
            // `evaluation.unwrap_unchecked()` is safe
            // because we always have a sample to evaluate
            // when it is called.
            State::Ready(s) => {
                State::Sampling(s.to_sampling(unsafe { evaluation.unwrap_unchecked() }))
            }
            State::Sampling(s) => {
                let value = unsafe { evaluation.unwrap_unchecked() };
                if s.samples_generated() < self.config.num_samples.into_inner() {
                    State::Sampling(s.to_sampling(value))
                } else if self.config.mutation_chance.is_zero() {
                    State::Ready(s.to_ready(self.config.adjust_rate, value))
                } else {
                    State::Mutating(s.to_mutating(self.config.adjust_rate, value))
                }
            }
            State::Mutating(s) => State::Ready(s.to_ready(
                self.config.mutation_chance,
                self.config.mutation_adjust_rate,
            )),
        });
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

impl<B, F> Optimizer for Pbil<B, F> {
    type Point = Vec<bool>;

    fn best_point(&self) -> Self::Point {
        self.state.best_point()
    }
}

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
#[derive(Clone, Debug, PartialEq, IsVariant)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State<B> {
    /// Ready to start sampling.
    Ready(Ready),
    /// For sampling
    /// and adjusting probabilities
    /// based on samples.
    Sampling(Sampling<B>),
    /// For mutating probabilities.
    Mutating(Mutating),
}

type Evaluation<B> = Option<B>;

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

impl DefaultFor<usize> for Config {
    fn default_for(num_bits: usize) -> Self {
        Self {
            num_samples: NumSamples::default(),
            adjust_rate: AdjustRate::default(),
            mutation_chance: MutationChance::default_for(num_bits),
            mutation_adjust_rate: MutationAdjustRate::default(),
        }
    }
}

impl Config {
    /// Return this optimizer default
    /// running on the given problem.
    ///
    /// # Arguments
    ///
    /// - `num_bits`: number of bits in each point
    /// - `obj_func`: objective function to minimize
    pub fn start_default_for<B, F>(num_bits: usize, obj_func: F) -> Pbil<B, F>
    where
        F: Fn(&[bool]) -> B,
    {
        Self::default_for(num_bits).start(num_bits, obj_func)
    }

    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    ///
    /// # Arguments
    ///
    /// - `num_bits`: number of bits in each point
    /// - `obj_func`: objective function to minimize
    pub fn start<B, F>(self, num_bits: usize, obj_func: F) -> Pbil<B, F>
    where
        F: Fn(&[bool]) -> B,
    {
        Pbil::new(State::Ready(Ready::initial(num_bits)), self, obj_func)
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    ///
    /// # Arguments
    ///
    /// - `num_bits`: number of bits in each point
    /// - `obj_func`: objective function to minimize
    /// - `rng`: source of randomness
    pub fn start_using<B, F>(self, num_bits: usize, obj_func: F, rng: &mut SplitMix64) -> Pbil<B, F>
    where
        F: Fn(&[bool]) -> B,
    {
        Pbil::new(
            State::Ready(Ready::initial_using(num_bits, rng)),
            self,
            obj_func,
        )
    }

    /// Return this optimizer
    /// running on the given problem.
    /// if the given `state` is valid.
    ///
    /// # Arguments
    ///
    /// - `obj_func`: objective function to minimize
    /// - `state`: PBIL state to start from
    pub fn start_from<B, F>(self, obj_func: F, state: State<B>) -> Pbil<B, F>
    where
        F: Fn(&[bool]) -> B,
    {
        Pbil::new(state, self, obj_func)
    }
}

impl<B> State<B> {
    /// Return custom initial state.
    pub fn new(probabilities: Vec<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self::Ready(Ready::new(probabilities, rng))
    }

    /// Return number of bits being optimized.
    pub fn num_bits(&self) -> usize {
        self.probabilities().len()
    }

    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> Option<&[bool]> {
        match self {
            State::Ready(s) => Some(s.sample()),
            State::Sampling(s) => Some(s.sample()),
            State::Mutating(_) => None,
        }
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> Vec<bool> {
        self.probabilities()
            .iter()
            .map(|p| f64::from(*p) >= 0.5)
            .collect()
    }
}

impl<B> Probabilities for State<B> {
    fn probabilities(&self) -> &[Probability] {
        match &self {
            State::Ready(s) => s.probabilities(),
            State::Sampling(s) => s.probabilities(),
            State::Mutating(s) => s.probabilities(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbil_should_not_mutate_if_chance_is_zero() {
        Config {
            num_samples: NumSamples::default(),
            adjust_rate: AdjustRate::default(),
            mutation_chance: MutationChance::new(0.0).unwrap(),
            mutation_adjust_rate: MutationAdjustRate::default(),
        }
        .start(16, |point| point.iter().filter(|x| **x).count())
        .inspect(|x| assert!(!x.state().is_mutating()))
        .nth(100);
    }
}
