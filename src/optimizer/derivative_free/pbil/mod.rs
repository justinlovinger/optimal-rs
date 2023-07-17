#![allow(clippy::needless_doctest_main)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::prelude::*;
//! use optimal::{optimizer::derivative_free::pbil::*, prelude::*};
//!
//! println!(
//!     "{}",
//!     UntilConvergedConfig::default().argmin(&mut Config::start_default_for(16, |points| {
//!         points.map_axis(Axis(1), |bits| bits.iter().filter(|x| **x).count())
//!     }))
//! );
//! ```

mod states;
mod types;

use std::fmt::Debug;

use derive_getters::Getters;
use ndarray::prelude::*;
use once_cell::sync::OnceCell;
use rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus};

use crate::prelude::*;

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A type containing an array of probabilities.
pub trait Probabilities {
    /// Return probabilities.
    fn probabilities(&self) -> &Array1<Probability>;
}

impl<B, F> Probabilities for Pbil<B, F> {
    fn probabilities(&self) -> &Array1<Probability> {
        self.state().probabilities()
    }
}

/// PBIL runner
/// to check for converged probabilities.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UntilConvergedConfig {
    /// Probability convergence parameter.
    pub threshold: ProbabilityThreshold,
}

impl<I> RunnerConfig<I> for UntilConvergedConfig
where
    I: Probabilities,
{
    type State = ();

    fn initial_state(&self) -> Self::State {}

    fn is_done(&self, it: &I, _: &Self::State) -> bool {
        it.probabilities()
            .iter()
            .all(|p| p > &self.threshold.upper_bound() || p < &self.threshold.lower_bound())
    }
}

/// A running PBIL optimizer.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pbil<B, F> {
    /// Optimizer configuration.
    config: Config,

    /// Objective function to minimize.
    obj_func: F,

    /// State of optimizer.
    state: State,

    #[getter(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    evaluation_cache: OnceCell<Evaluation<B>>,
}

impl<B, F> Pbil<B, F> {
    fn new(state: State, config: Config, obj_func: F) -> Self {
        Self {
            config,
            obj_func,
            state,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Return configuration, state, and problem parameters.
    pub fn into_inner(self) -> (Config, State, F) {
        (self.config, self.state, self.obj_func)
    }
}

impl<B, F> Pbil<B, F>
where
    F: Fn(ArrayView2<bool>) -> Array1<B>,
{
    /// Return value of the best point discovered.
    pub fn best_point_value(&self) -> B {
        (self.obj_func)(
            self.best_point()
                .view()
                .into_shape((1, self.state().num_bits()))
                .unwrap(),
        )
        .into_iter()
        .next()
        .unwrap()
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
    B: Debug + PartialOrd,
    F: Fn(ArrayView2<bool>) -> Array1<B>,
{
    type Item = Self;

    fn advance(&mut self) {
        let evaluation = self
            .evaluation_cache
            .take()
            .unwrap_or_else(|| self.evaluate());
        replace_with::replace_with_or_abort(&mut self.state, |state| {
            match state {
                State::Ready(s) => State::Sampling(s.to_sampling(self.config.num_samples)),
                State::Sampling(s) => {
                    // `unwrap_unchecked` is safe
                    // because `evaluate` always returns `Some`
                    // for `State::Sampling`.
                    State::Mutating(s.to_mutating(self.config.adjust_rate, unsafe {
                        evaluation.unwrap_unchecked()
                    }))
                }
                State::Mutating(s) => State::Ready(s.to_ready(
                    self.config.mutation_chance,
                    self.config.mutation_adjust_rate,
                )),
            }
        });
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

impl<B, F> Optimizer for Pbil<B, F> {
    type Point = Array1<bool>;

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

type Evaluation<B> = Option<Array1<B>>;

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
        F: Fn(ArrayView2<bool>) -> Array1<B>,
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
        F: Fn(ArrayView2<bool>) -> Array1<B>,
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
        F: Fn(ArrayView2<bool>) -> Array1<B>,
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
    pub fn start_from<B, F>(self, obj_func: F, state: State) -> Pbil<B, F>
    where
        F: Fn(ArrayView2<bool>) -> Array1<B>,
    {
        Pbil::new(state, self, obj_func)
    }
}

impl State {
    /// Return custom initial state.
    pub fn new(probabilities: Array1<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self::Ready(Ready::new(probabilities, rng))
    }

    /// Return number of bits being optimized.
    pub fn num_bits(&self) -> usize {
        self.probabilities().len()
    }

    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> Option<ArrayView2<bool>> {
        match self {
            State::Ready(_) => None,
            State::Sampling(s) => Some(s.samples().into()),
            State::Mutating(_) => None,
        }
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> Array1<bool> {
        self.probabilities().map(|p| f64::from(*p) >= 0.5)
    }
}

impl Probabilities for State {
    fn probabilities(&self) -> &Array1<Probability> {
        match &self {
            State::Ready(s) => s.probabilities(),
            State::Sampling(s) => s.probabilities(),
            State::Mutating(s) => s.probabilities(),
        }
    }
}
