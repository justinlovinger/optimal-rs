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
mod until_probabilities_converged;

use derive_getters::{Dissolve, Getters};
use derive_more::IsVariant;
pub use optimal_core::prelude::*;
use rand::prelude::*;
use rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus};

use self::state_machine::DynState;
pub use self::{types::*, until_probabilities_converged::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Error returned when
/// problem length does not match state length.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
#[error("problem length does not match state length")]
pub struct MismatchedLengthError;

/// A running PBIL optimizer.
#[derive(Clone, Debug, Getters, Dissolve)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct Pbil<B, F> {
    /// Optimizer configuration.
    config: Config,

    /// State of optimizer.
    state: State<B>,

    /// Objective function to minimize.
    obj_func: F,
}

/// PBIL configuration parameters.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct State<B>(DynState<B>);

/// PBIL state kind.
#[derive(Clone, Debug, PartialEq, IsVariant)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StateKind {
    /// Iteration started.
    Started,
    /// Sample generated.
    Sampled,
    /// Sample evaluated.
    Evaluated,
    /// Samples compared.
    Compared,
    /// Probabilities adjusted.
    Adjusted,
    /// Probabilities mutated.
    Mutated,
    /// Iteration finished.
    Finished,
}

impl<B, F> Pbil<B, F> {
    fn new(state: State<B>, config: Config, obj_func: F) -> Self {
        Self {
            config,
            obj_func,
            state,
        }
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
}

impl<B, F> StreamingIterator for Pbil<B, F>
where
    B: PartialOrd,
    F: Fn(&[bool]) -> B,
{
    type Item = Self;

    fn advance(&mut self) {
        replace_with::replace_with_or_abort(&mut self.state.0, |state| match state {
            DynState::Started(x) => {
                DynState::SampledFirst(x.into_initialized_sampling().into_sampled_first())
            }
            DynState::SampledFirst(x) => {
                DynState::EvaluatedFirst(x.into_evaluated_first(&self.obj_func))
            }
            DynState::EvaluatedFirst(x) => DynState::Sampled(x.into_sampled()),
            DynState::Sampled(x) => DynState::Evaluated(x.into_evaluated(&self.obj_func)),
            DynState::Evaluated(x) => DynState::Compared(x.into_compared()),
            DynState::Compared(x) => {
                if x.samples_generated < self.config.num_samples.into_inner() {
                    DynState::Sampled(x.into_sampled())
                } else {
                    DynState::Adjusted(x.into_adjusted(self.config.adjust_rate))
                }
            }
            DynState::Adjusted(x) => {
                if self.config.mutation_chance.into_inner() > 0.0 {
                    DynState::Mutated(x.into_mutated(
                        self.config.mutation_chance,
                        self.config.mutation_adjust_rate,
                    ))
                } else {
                    DynState::Finished(x.into_finished())
                }
            }
            DynState::Mutated(x) => DynState::Finished(x.into_finished()),
            DynState::Finished(x) => DynState::Started(x.into_started()),
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

impl Config {
    /// Return default 'Config'.
    pub fn default_for(num_bits: usize) -> Self {
        Self {
            num_samples: NumSamples::default(),
            adjust_rate: AdjustRate::default(),
            mutation_chance: MutationChance::default_for(num_bits),
            mutation_adjust_rate: MutationAdjustRate::default(),
        }
    }

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
        Self(DynState::new(probabilities, rng))
    }

    /// Return number of bits being optimized.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.probabilities().len()
    }

    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> Option<&[bool]> {
        match &self.0 {
            DynState::Started(_) => None,
            DynState::SampledFirst(x) => Some(&x.sample),
            DynState::EvaluatedFirst(_) => None,
            DynState::Sampled(x) => Some(&x.sample),
            DynState::Evaluated(_) => None,
            DynState::Compared(_) => None,
            DynState::Adjusted(_) => None,
            DynState::Mutated(_) => None,
            DynState::Finished(_) => None,
        }
    }

    /// Return result of evaluation.
    pub fn evaluation(&self) -> Option<&B> {
        match &self.0 {
            DynState::Started(_) => None,
            DynState::SampledFirst(_) => None,
            DynState::EvaluatedFirst(x) => Some(x.sample.value()),
            DynState::Sampled(_) => None,
            DynState::Evaluated(x) => Some(x.sample.value()),
            DynState::Compared(_) => None,
            DynState::Adjusted(_) => None,
            DynState::Mutated(_) => None,
            DynState::Finished(_) => None,
        }
    }

    /// Return sample if stored.
    pub fn sample(&self) -> Option<&[bool]> {
        match &self.0 {
            DynState::Started(_) => None,
            DynState::SampledFirst(x) => Some(&x.sample),
            DynState::EvaluatedFirst(x) => Some(x.sample.x()),
            DynState::Sampled(x) => Some(&x.sample),
            DynState::Evaluated(x) => Some(x.sample.x()),
            DynState::Compared(_) => None,
            DynState::Adjusted(_) => None,
            DynState::Mutated(_) => None,
            DynState::Finished(_) => None,
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
        match self.0 {
            DynState::Started(_) => StateKind::Started,
            DynState::SampledFirst(_) => StateKind::Sampled,
            DynState::EvaluatedFirst(_) => StateKind::Evaluated,
            DynState::Sampled(_) => StateKind::Sampled,
            DynState::Evaluated(_) => StateKind::Evaluated,
            DynState::Compared(_) => StateKind::Compared,
            DynState::Adjusted(_) => StateKind::Adjusted,
            DynState::Mutated(_) => StateKind::Mutated,
            DynState::Finished(_) => StateKind::Finished,
        }
    }
}

impl<B> Probabilities for State<B> {
    fn probabilities(&self) -> &[Probability] {
        match &self.0 {
            DynState::Started(x) => &x.probabilities,
            DynState::SampledFirst(x) => x.probabilities.probabilities(),
            DynState::EvaluatedFirst(x) => x.probabilities.probabilities(),
            DynState::Sampled(x) => x.probabilities.probabilities(),
            DynState::Evaluated(x) => x.probabilities.probabilities(),
            DynState::Compared(x) => x.probabilities.probabilities(),
            DynState::Adjusted(x) => &x.probabilities,
            DynState::Mutated(x) => &x.probabilities,
            DynState::Finished(x) => &x.probabilities,
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
        .inspect(|x| assert!(!x.state().kind().is_mutated()))
        .nth(100);
    }
}
