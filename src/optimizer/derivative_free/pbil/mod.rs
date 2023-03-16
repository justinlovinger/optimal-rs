//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::{Data, RemoveAxis, prelude::*};
//! use optimal::{
//!     optimizer::derivative_free::pbil::{DoneWhenConvergedConfig, NumBits},
//!     prelude::*,
//! };
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let config = DoneWhenConvergedConfig::default(NumBits(16));
//!     let mut iter = config.initialize(&f).into_streaming_iter();
//!     let xs = iter
//!         .find(|o| o.is_done())
//!         .expect("should converge")
//!         .best_point();
//!     println!("f({}) = {}", xs, f(xs.view()));
//! }
//!
//! fn f(bs: ArrayView1<bool>) -> u64 {
//!     bs.fold(0, |acc, b| acc + *b as u64)
//! }
//! ```

mod states;
mod types;

use lazy_static::lazy_static;
use ndarray::{prelude::*, Data};
use rand::prelude::*;
use replace_with::replace_with_or_abort;
use std::{fmt::Debug, marker::PhantomData};

use crate::prelude::*;

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// PBIL optimizer with check for converged probabilities.
#[derive(Clone, Debug)]
pub struct PbilDoneWhenConverged<'a, B, F> {
    point_value: PhantomData<B>,
    /// PBIL configuration parameters
    /// with check for converged probabilities.
    pub config: &'a DoneWhenConvergedConfig,
    /// An objective function.
    pub objective: &'a F,
    /// PBIL state.
    pub state: State,
}

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

impl<'a, B, F> PbilDoneWhenConverged<'a, B, F>
where
    F: Problem<bool, B>,
{
    /// Convenience function to return a 'PbilDoneWhenConverged'
    /// without setting 'PhantomData'.
    pub fn new(config: &'a DoneWhenConvergedConfig, objective: &'a F, state: State) -> Self {
        Self {
            point_value: PhantomData,
            config,
            state,
            objective,
        }
    }
}

impl<B, F> Step for PbilDoneWhenConverged<'_, B, F>
where
    B: Debug + PartialOrd,
    F: Problem<bool, B>,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config
                .inner
                .step_from_evaluated(self.objective.evaluate_all(state.points().view()), state)
        })
    }
}

impl<B, F> IsDone for PbilDoneWhenConverged<'_, B, F> {
    fn is_done(&self) -> bool {
        converged(&self.config.converged_threshold, self.state.probabilities())
    }
}

impl<B, F> Points<bool> for PbilDoneWhenConverged<'_, B, F> {
    fn points(&self) -> ArrayView2<bool> {
        self.state.points()
    }
}

impl<B, F> BestPoint<bool> for PbilDoneWhenConverged<'_, B, F> {
    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }
}

impl DoneWhenConvergedConfig {
    /// Convenience function
    /// to populate every field
    /// with their default.
    pub fn default(num_bits: NumBits) -> Self {
        Self {
            converged_threshold: ConvergedThreshold::default(),
            inner: Config::default(num_bits),
        }
    }
}

impl<'a, B, F> InitializeUsing<'a, F, PbilDoneWhenConverged<'a, B, F>> for DoneWhenConvergedConfig
where
    F: Problem<bool, B>,
{
    fn initialize_using<R>(
        &'a self,
        objective: &'a F,
        rng: &mut R,
    ) -> PbilDoneWhenConverged<'a, B, F>
    where
        R: Rng,
    {
        PbilDoneWhenConverged::new(
            self,
            objective,
            State::initial_using(self.inner.num_bits, rng),
        )
    }
}

impl<'a, B, F> Initialize<'a, F, PbilDoneWhenConverged<'a, B, F>> for DoneWhenConvergedConfig
where
    F: Problem<bool, B>,
{
    fn initialize(&'a self, objective: &'a F) -> PbilDoneWhenConverged<'a, B, F> {
        PbilDoneWhenConverged::new(self, objective, State::initial(self.inner.num_bits))
    }
}

/// PBIL optimizer.
#[derive(Clone, Debug)]
pub struct Pbil<'a, B, F> {
    point_value: PhantomData<B>,
    /// PBIL configuration parameters.
    pub config: &'a Config,
    /// An objective function.
    pub objective: &'a F,
    /// PBIL state.
    pub state: State,
}

/// PBIL configuration parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config {
    /// Number of bits in generated points
    /// and probabilities in PBIL.
    pub num_bits: NumBits,
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

impl<'a, B, F> Pbil<'a, B, F>
where
    F: Problem<bool, B>,
{
    /// Convenience function to return a 'Pbil'
    /// without setting 'PhantomData'.
    pub fn new(config: &'a Config, objective: &'a F, state: State) -> Self {
        Self {
            point_value: PhantomData,
            config,
            state,
            objective,
        }
    }
}

impl<B, F> Step for Pbil<'_, B, F>
where
    B: Debug + PartialOrd,
    F: Problem<bool, B>,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| {
            self.config
                .step_from_evaluated(self.objective.evaluate_all(state.points()), state)
        })
    }
}

impl<B, F> Points<bool> for Pbil<'_, B, F> {
    fn points(&self) -> ArrayView2<bool> {
        self.state.points()
    }
}

impl<B, F> BestPoint<bool> for Pbil<'_, B, F> {
    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }
}

impl Config {
    /// Convenience function
    /// to populate every field
    /// with their default.
    pub fn default(num_bits: NumBits) -> Self {
        Self {
            num_bits,
            num_samples: Default::default(),
            adjust_rate: Default::default(),
            mutation_chance: MutationChance::default(num_bits),
            mutation_adjust_rate: Default::default(),
        }
    }

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

impl<'a, B, F> InitializeUsing<'a, F, Pbil<'a, B, F>> for Config
where
    F: Problem<bool, B>,
{
    fn initialize_using<R>(&'a self, objective: &'a F, rng: &mut R) -> Pbil<'a, B, F>
    where
        R: Rng,
    {
        Pbil::new(self, objective, State::initial_using(self.num_bits, rng))
    }
}

impl<'a, B, F> Initialize<'a, F, Pbil<'a, B, F>> for Config
where
    F: Problem<bool, B>,
{
    fn initialize(&'a self, objective: &'a F) -> Pbil<'a, B, F> {
        Pbil::new(self, objective, State::initial(self.num_bits))
    }
}

impl State {
    fn initial(num_bits: NumBits) -> Self {
        Self::Init(Init::initial(usize::from(num_bits)))
    }

    fn initial_using<R>(num_bits: NumBits, rng: &mut R) -> Self
    where
        R: Rng,
    {
        Self::Init(Init::initial_using(usize::from(num_bits), rng))
    }

    /// Return PBIL probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        match &self {
            State::Init(s) => s.probabilities(),
            State::PreEval(s) => s.probabilities(),
        }
    }
}

impl Points<bool> for State {
    fn points(&self) -> ArrayView2<bool> {
        lazy_static! {
            static ref EMPTY: Array2<bool> = Array::from_elem((0, 0), false);
        }
        match self {
            State::Init(_) => EMPTY.view(),
            State::PreEval(s) => s.samples().view(),
        }
    }
}

impl BestPoint<bool> for State {
    fn best_point(&self) -> CowArray<bool, Ix1> {
        finalize(self.probabilities()).into()
    }
}
