#![allow(clippy::needless_doctest_main)]

//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::{Data, RemoveAxis, prelude::*};
//! use optimal::{
//!     optimizer::derivative_free::pbil::DoneWhenConvergedConfig,
//!     prelude::*,
//! };
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter = DoneWhenConvergedConfig::default(Count)
//!         .start()
//!         .into_streaming_iter();
//!     let o = iter.find(|o| o.is_done()).expect("should converge");
//!     println!("f({}) = {}", o.best_point(), o.best_point_value());
//! }
//!
//! struct Count;
//!
//! impl Problem<bool, u64> for Count {
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
pub struct RunningDoneWhenConverged<B, BorrowedP, P, C> {
    point_value: PhantomData<B>,
    borrowed_problem: PhantomData<BorrowedP>,
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
pub struct DoneWhenConvergedConfig<BorrowedP, P> {
    /// Probability convergence parameter.
    pub converged_threshold: ConvergedThreshold,
    /// Regular PBIL configuration.
    pub inner: Config<BorrowedP, P>,
}

impl<B, BorrowedP, P, C> RunningDoneWhenConverged<B, BorrowedP, P, C> {
    /// Convenience function to return a 'PbilDoneWhenConverged'
    /// without setting 'PhantomData'.
    pub fn new(config: C, state: State) -> Self {
        Self {
            point_value: PhantomData,
            borrowed_problem: PhantomData,
            problem: PhantomData,
            config,
            state,
        }
    }
}

impl<B, BorrowedP, P, C> RunningOptimizer<bool, B, C, State>
    for RunningDoneWhenConverged<B, BorrowedP, P, C>
where
    B: Debug + PartialOrd,
    BorrowedP: Problem<bool, B>,
    P: Borrow<BorrowedP>,
    C: Borrow<DoneWhenConvergedConfig<BorrowedP, P>>,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config.borrow().inner.step_from_evaluated(
                self.config
                    .borrow()
                    .inner
                    .problem
                    .borrow()
                    .evaluate_all(state.points().view()),
                state,
            )
        })
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

    fn config(&self) -> &C {
        &self.config
    }

    fn stored_best_point_value(&self) -> Option<B> {
        None
    }
}

impl<B, BorrowedP, P, C> Convergent for RunningDoneWhenConverged<B, BorrowedP, P, C>
where
    C: Borrow<DoneWhenConvergedConfig<BorrowedP, P>>,
{
    fn is_done(&self) -> bool {
        converged(
            &self.config.borrow().converged_threshold,
            self.state.probabilities(),
        )
    }
}

impl<B, BorrowedP, P, C> PopulationBased<bool> for RunningDoneWhenConverged<B, BorrowedP, P, C> {
    fn points(&self) -> ArrayView2<bool> {
        self.state.points()
    }
}

impl<BorrowedP, P> DoneWhenConvergedConfig<BorrowedP, P> {
    /// Convenience function
    /// to populate every field
    /// with their default.
    pub fn default(problem: P) -> Self
    where
        BorrowedP: FixedLength,
        P: Borrow<BorrowedP>,
    {
        Self {
            converged_threshold: ConvergedThreshold::default(),
            inner: Config::default(problem),
        }
    }
}

impl<B, BorrowedP, P, C> StochasticOptimizerConfig<RunningDoneWhenConverged<B, BorrowedP, P, C>>
    for C
where
    BorrowedP: FixedLength,
    P: Borrow<BorrowedP>,
    C: Borrow<DoneWhenConvergedConfig<BorrowedP, P>>,
{
    fn start_using<R>(self, rng: &mut R) -> RunningDoneWhenConverged<B, BorrowedP, P, C>
    where
        R: Rng,
    {
        let state = State::initial_using(self.borrow().inner.problem.borrow().len(), rng);
        RunningDoneWhenConverged::new(self, state)
    }
}

impl<'a, B, BorrowedP, P, C>
    OptimizerConfig<'a, RunningDoneWhenConverged<B, BorrowedP, P, C>, BorrowedP> for C
where
    BorrowedP: FixedLength,
    P: Borrow<BorrowedP> + 'a,
    C: Borrow<DoneWhenConvergedConfig<BorrowedP, P>>,
{
    fn start(self) -> RunningDoneWhenConverged<B, BorrowedP, P, C> {
        let state = State::initial(self.borrow().inner.problem.borrow().len());
        RunningDoneWhenConverged::new(self, state)
    }

    fn problem(&'a self) -> &'a BorrowedP {
        self.borrow().inner.problem.borrow()
    }
}

/// Running PBIL optimizer.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Running<B, BorrowedP, P, C> {
    point_value: PhantomData<B>,
    borrowed_problem: PhantomData<BorrowedP>,
    problem: PhantomData<P>,
    /// PBIL configuration parameters.
    pub config: C,
    /// PBIL state.
    pub state: State,
}

/// PBIL configuration parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<BorrowedP, P> {
    borrowed_problem: PhantomData<BorrowedP>,
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State {
    /// Initial and post-evaluation state.
    Init(Init),
    /// State with samples ready for evaluation.
    PreEval(PreEval),
}

impl<B, BorrowedP, P, C> Running<B, BorrowedP, P, C> {
    /// Return a new 'Pbil'.
    pub fn new(config: C, state: State) -> Self {
        Self {
            point_value: PhantomData,
            borrowed_problem: PhantomData,
            problem: PhantomData,
            config,
            state,
        }
    }
}

impl<B, BorrowedP, P, C> RunningOptimizer<bool, B, C, State> for Running<B, BorrowedP, P, C>
where
    B: Debug + PartialOrd,
    BorrowedP: Problem<bool, B>,
    P: Borrow<BorrowedP>,
    C: Borrow<Config<BorrowedP, P>>,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config.borrow().step_from_evaluated(
                self.config
                    .borrow()
                    .problem
                    .borrow()
                    .evaluate_all(state.points()),
                state,
            )
        })
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

    fn config(&self) -> &C {
        &self.config
    }

    fn stored_best_point_value(&self) -> Option<B> {
        None
    }
}

impl<B, BorrowedP, P, C> PopulationBased<bool> for Running<B, BorrowedP, P, C> {
    fn points(&self) -> ArrayView2<bool> {
        self.state.points()
    }
}

impl<BorrowedP, P> Config<BorrowedP, P> {
    /// Return a new PBIL configuration.
    pub fn new(
        problem: P,
        num_samples: NumSamples,
        adjust_rate: AdjustRate,
        mutation_chance: MutationChance,
        mutation_adjust_rate: MutationAdjustRate,
    ) -> Self {
        Self {
            borrowed_problem: PhantomData,
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
        BorrowedP: FixedLength,
        P: Borrow<BorrowedP>,
    {
        Self {
            borrowed_problem: PhantomData,
            num_samples: Default::default(),
            adjust_rate: Default::default(),
            mutation_chance: MutationChance::default(problem.borrow().len()),
            mutation_adjust_rate: Default::default(),
            problem,
        }
    }
}

impl<BorrowedP, P> Config<BorrowedP, P> {
    /// Return the next state,
    /// given point values.
    fn step_from_evaluated<B, S>(&self, point_values: ArrayBase<S, Ix1>, state: State) -> State
    where
        B: Debug + PartialOrd,
        S: Data<Elem = B>,
    {
        match state {
            State::Init(s) => State::PreEval(s.to_pre_eval(self.num_samples)),
            State::PreEval(s) => {
                let mut s = s.to_init(self.adjust_rate, point_values);
                s.mutate(self.mutation_chance, self.mutation_adjust_rate);
                State::Init(s)
            }
        }
    }
}

impl<B, BorrowedP, P, C> StochasticOptimizerConfig<Running<B, BorrowedP, P, C>> for C
where
    BorrowedP: FixedLength,
    P: Borrow<BorrowedP>,
    C: Borrow<Config<BorrowedP, P>>,
{
    fn start_using<R>(self, rng: &mut R) -> Running<B, BorrowedP, P, C>
    where
        R: Rng,
    {
        let state = State::initial_using(self.borrow().problem.borrow().len(), rng);
        Running::new(self, state)
    }
}

impl<'a, B, BorrowedP, P, C> OptimizerConfig<'a, Running<B, BorrowedP, P, C>, BorrowedP> for C
where
    BorrowedP: FixedLength,
    P: Borrow<BorrowedP> + 'a,
    C: Borrow<Config<BorrowedP, P>>,
{
    fn start(self) -> Running<B, BorrowedP, P, C> {
        let state = State::initial(self.borrow().problem.borrow().len());
        Running::new(self, state)
    }

    fn problem(&'a self) -> &'a BorrowedP {
        self.borrow().problem.borrow()
    }
}

impl State {
    fn initial(num_bits: usize) -> Self {
        Self::Init(Init::initial(num_bits))
    }

    fn initial_using<R>(num_bits: usize, rng: &mut R) -> Self
    where
        R: Rng,
    {
        Self::Init(Init::initial_using(num_bits, rng))
    }

    /// Return PBIL probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        match &self {
            State::Init(s) => s.probabilities(),
            State::PreEval(s) => s.probabilities(),
        }
    }

    fn points(&self) -> ArrayView2<bool> {
        lazy_static! {
            static ref EMPTY: Array2<bool> = Array::from_elem((0, 0), false);
        }
        match self {
            State::Init(_) => EMPTY.view(),
            State::PreEval(s) => s.samples().view(),
        }
    }

    fn best_point(&self) -> CowArray<bool, Ix1> {
        finalize(self.probabilities()).into()
    }
}
