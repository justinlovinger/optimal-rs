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

mod state_machine;
mod types;

use std::fmt::Debug;

use default_for::DefaultFor;
use derive_getters::Getters;
use derive_more::IsVariant;
use once_cell::sync::OnceCell;
pub use optimal_core::prelude::*;
use rand::prelude::*;
use rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus};

use self::state_machine::DynState;
pub use self::types::*;

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
pub struct State<B> {
    inner: DynState<B>,
}

/// PBIL state kind.
#[derive(Clone, Debug, PartialEq, IsVariant)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StateKind {
    /// Ready to start sampling.
    Ready,
    /// For sampling
    /// and adjusting probabilities
    /// based on samples.
    Sampling,
    /// For mutating probabilities.
    Mutating,
}

type Evaluation<B> = Option<B>;

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
        replace_with::replace_with_or_abort(&mut self.state.inner, |state| match state {
            // `evaluation.unwrap_unchecked()` is safe
            // because we always have a sample to evaluate
            // when it is called.
            DynState::Ready(s) => {
                DynState::Sampling(s.to_sampling(unsafe { evaluation.unwrap_unchecked() }))
            }
            DynState::Sampling(s) => {
                let value = unsafe { evaluation.unwrap_unchecked() };
                if s.samples_generated() < self.config.num_samples.into_inner() {
                    DynState::Sampling(s.to_sampling(value))
                } else if self.config.mutation_chance.is_zero() {
                    DynState::Ready(s.to_ready(self.config.adjust_rate, value))
                } else {
                    DynState::Mutating(s.to_mutating(self.config.adjust_rate, value))
                }
            }
            DynState::Mutating(s) => DynState::Ready(s.to_ready(
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
    /// - `len`: number of bits in each point
    /// - `obj_func`: objective function to minimize
    pub fn start_default_for<B, F>(len: usize, obj_func: F) -> Pbil<B, F>
    where
        F: Fn(&[bool]) -> B,
    {
        Self::default_for(len).start(len, obj_func)
    }

    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    ///
    /// # Arguments
    ///
    /// - `len`: number of bits in each point
    /// - `obj_func`: objective function to minimize
    pub fn start<B, F>(self, len: usize, obj_func: F) -> Pbil<B, F>
    where
        F: Fn(&[bool]) -> B,
    {
        Pbil::new(State::initial(len), self, obj_func)
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    ///
    /// # Arguments
    ///
    /// - `len`: number of bits in each point
    /// - `obj_func`: objective function to minimize
    /// - `rng`: source of randomness
    pub fn start_using<B, F>(self, len: usize, obj_func: F, rng: &mut SplitMix64) -> Pbil<B, F>
    where
        F: Fn(&[bool]) -> B,
    {
        Pbil::new(State::initial_using(len, rng), self, obj_func)
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
    /// Return recommended initial state.
    ///
    /// # Arguments
    ///
    /// - `len`: number of bits in each sample
    fn initial(len: usize) -> Self {
        Self::new(
            [Probability::default()].repeat(len),
            Xoshiro256PlusPlus::from_entropy(),
        )
    }

    /// Return recommended initial state.
    ///
    /// # Arguments
    ///
    /// - `len`: number of bits in each sample
    /// - `rng`: source of randomness
    fn initial_using<R>(len: usize, rng: R) -> Self
    where
        R: Rng,
    {
        Self::new(
            [Probability::default()].repeat(len),
            Xoshiro256PlusPlus::from_rng(rng).expect("RNG should initialize"),
        )
    }

    /// Return custom initial state.
    pub fn new(probabilities: Vec<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self {
            inner: DynState::new(probabilities, rng),
        }
    }

    /// Return number of bits being optimized.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.probabilities().len()
    }

    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> Option<&[bool]> {
        match &self.inner {
            DynState::Ready(s) => Some(s.sample()),
            DynState::Sampling(s) => Some(s.sample()),
            DynState::Mutating(_) => None,
        }
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> Vec<bool> {
        self.probabilities()
            .iter()
            .map(|p| f64::from(*p) >= 0.5)
            .collect()
    }

    /// Return kind of state of inner state-machine.
    pub fn kind(&self) -> StateKind {
        match self.inner {
            DynState::Ready(_) => StateKind::Ready,
            DynState::Sampling(_) => StateKind::Sampling,
            DynState::Mutating(_) => StateKind::Mutating,
        }
    }
}

impl<B> Probabilities for State<B> {
    fn probabilities(&self) -> &[Probability] {
        match &self.inner {
            DynState::Ready(s) => s.probabilities(),
            DynState::Sampling(s) => s.probabilities(),
            DynState::Mutating(s) => s.probabilities(),
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
        .inspect(|x| assert!(!x.state().kind().is_mutating()))
        .nth(100);
    }
}
