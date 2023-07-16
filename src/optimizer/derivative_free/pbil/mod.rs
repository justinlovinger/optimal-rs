#![allow(clippy::needless_doctest_main)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::prelude::*;
//! use optimal::{optimizer::derivative_free::pbil::*, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut o = Config::start_default_for(Count);
//!     let point = UntilConvergedConfig::default().argmin(&mut o);
//!     let point_value = Count.evaluate(point.view().into_shape((1, Count.len())).unwrap())[0];
//!     println!("f({point}) = {point_value}");
//! }
//!
//! struct Count;
//!
//! impl Problem for Count {
//!     type Value = u64;
//!
//!     fn evaluate(&self, points: ArrayView2<bool>) -> Array1<Self::Value> {
//!         points.fold_axis(Axis(1), 0, |acc, b| acc + *b as u64)
//!     }
//!
//!     fn len(&self) -> usize {
//!         16
//!     }
//! }
//! ```

mod states;
mod types;

use std::fmt::Debug;

use blanket::blanket;
use derive_getters::Getters;
use ndarray::prelude::*;
use once_cell::sync::OnceCell;
use rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus};

use crate::{optimizer::MismatchedLengthError, prelude::*};

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A type containing an array of probabilities.
pub trait Probabilities {
    /// Return probabilities.
    fn probabilities(&self) -> &Array1<Probability>;
}

impl<P> Probabilities for Pbil<P>
where
    P: Problem,
{
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

/// A PBIL optimization problem.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
#[allow(clippy::len_without_is_empty)] // Problems should never have no length.
pub trait Problem {
    /// Value of a point in this problem space.
    type Value;

    /// Return the objective values of points in this problem space.
    fn evaluate(&self, points: ArrayView2<bool>) -> Array1<Self::Value>;

    /// Return the length of each point in this problem space.
    fn len(&self) -> usize;
}

/// A running PBIL optimizer.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pbil<P>
where
    P: Problem,
{
    /// Optimizer configuration.
    config: Config,

    /// Problem being optimized.
    problem: P,

    /// State of optimizer.
    state: State,

    #[getter(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    evaluation_cache: OnceCell<Evaluation<P::Value>>,
}

impl<P> Pbil<P>
where
    P: Problem,
{
    fn new(state: State, config: Config, problem: P) -> Self {
        Self {
            problem,
            config,
            state,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Return configuration, problem, and state.
    pub fn into_inner(self) -> (Config, P, State) {
        (self.config, self.problem, self.state)
    }
}

impl<P> Pbil<P>
where
    P: Problem,
{
    /// Return value of the best point discovered.
    pub fn best_point_value(&self) -> P::Value {
        self.problem()
            .evaluate(
                self.best_point()
                    .view()
                    .into_shape((1, self.problem().len()))
                    .unwrap(),
            )
            .into_iter()
            .next()
            .unwrap()
    }

    /// Return evaluation of current state,
    /// evaluating and caching if necessary.
    pub fn evaluation(&self) -> &Evaluation<P::Value> {
        self.evaluation_cache.get_or_init(|| self.evaluate())
    }

    fn evaluate(&self) -> Evaluation<P::Value> {
        self.state.evaluatee().map(|xs| self.problem.evaluate(xs))
    }
}

impl<P> StreamingIterator for Pbil<P>
where
    P: Problem,
    P::Value: Debug + PartialOrd,
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

impl<P> Optimizer for Pbil<P>
where
    P: Problem,
{
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

impl<P> DefaultFor<P> for Config
where
    P: Problem,
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

impl Config {
    /// Return this optimizer default
    /// running on the given problem.
    pub fn start_default_for<P>(problem: P) -> Pbil<P>
    where
        P: Problem,
    {
        Self::default_for(&problem).start(problem)
    }

    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    pub fn start<P>(self, problem: P) -> Pbil<P>
    where
        P: Problem,
    {
        Pbil::new(State::Ready(Ready::initial(problem.len())), self, problem)
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    pub fn start_using<P>(self, problem: P, rng: &mut SplitMix64) -> Pbil<P>
    where
        P: Problem,
    {
        Pbil::new(
            State::Ready(Ready::initial_using(problem.len(), rng)),
            self,
            problem,
        )
    }

    /// Return this optimizer
    /// running on the given problem.
    /// if the given `state` is valid.
    #[allow(clippy::result_large_err)]
    pub fn start_from<P>(
        self,
        problem: P,
        state: State,
    ) -> Result<Pbil<P>, (MismatchedLengthError, Self, P, State)>
    where
        P: Problem,
    {
        // If `Sampling::samples` could be changed independent of `probabilities`,
        // it would need to be validated.
        if state.probabilities().len() == problem.len() {
            Ok(Pbil::new(state, self, problem))
        } else {
            Err((MismatchedLengthError, self, problem, state))
        }
    }
}

impl State {
    /// Return custom initial state.
    pub fn new(probabilities: Array1<Probability>, rng: Xoshiro256PlusPlus) -> Self {
        Self::Ready(Ready::new(probabilities, rng))
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
