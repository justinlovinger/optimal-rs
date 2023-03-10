//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::{Data, RemoveAxis, prelude::*};
//! use optimal::{
//!     optimizer::derivative_free::pbil::{PbilDoneWhenConverged, NumBits},
//!     prelude::*,
//! };
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter = PbilDoneWhenConverged::default(NumBits(16), |xs| f(xs)).iterate();
//!     let xs = iter
//!         .find(|o| o.is_done())
//!         .expect("should converge")
//!         .best_point();
//!     println!("f({}) = {}", xs, f(xs.view()));
//! }
//!
//! fn f<S, D>(bs: ArrayBase<S, D>) -> Array<u64, D::Smaller>
//! where
//!     S: Data<Elem = bool>,
//!     D: Dimension + RemoveAxis,
//! {
//!     bs.fold_axis(Axis(bs.ndim() - 1), 0, |acc, b| acc + *b as u64)
//! }
//! ```

mod states;
mod types;

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
pub struct PbilDoneWhenConverged<R, B, F> {
    point_value: PhantomData<B>,
    /// PBIL configuration parameters
    /// with check for converged probabilities.
    pub config: DoneWhenConvergedConfig,
    /// PBIL state.
    pub state: State<R>,
    /// Derivative-free objective function.
    pub objective_function: F,
}

impl<R, B, F> PbilDoneWhenConverged<R, B, F>
where
    F: Fn(CowArray<bool, Ix2>) -> Array1<B>,
{
    /// Convenience function to return a 'PbilDoneWhenConverged'
    /// without setting 'PhantomData'.
    pub fn new(config: DoneWhenConvergedConfig, state: State<R>, objective_function: F) -> Self {
        Self {
            point_value: PhantomData,
            config,
            state,
            objective_function,
        }
    }
}

impl<B, F> PbilDoneWhenConverged<SmallRng, B, F>
where
    F: Fn(CowArray<bool, Ix2>) -> Array1<B>,
{
    /// Convenience function
    /// to populate every field
    /// with their default.
    pub fn default(num_bits: NumBits, objective_function: F) -> Self {
        let config = DoneWhenConvergedConfig::default(num_bits);
        let state = config.initial_state();
        Self::new(config, state, objective_function)
    }
}

impl<R, B, F> Step for PbilDoneWhenConverged<R, B, F>
where
    R: Rng,
    B: Debug + PartialOrd,
    F: Fn(CowArray<bool, Ix2>) -> Array1<B>,
{
    fn step(&mut self) {
        self.step_from_evaluated((self.objective_function)(self.points()))
    }
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

impl<R, B, F> IsDone for PbilDoneWhenConverged<R, B, F> {
    fn is_done(&self) -> bool {
        converged(&self.config.converged_threshold, self.state.probabilities())
    }
}

impl InitialState<State<SmallRng>> for DoneWhenConvergedConfig {
    fn initial_state(&self) -> State<SmallRng> {
        self.inner.initial_state()
    }
}

impl<R, B, F> StepFromEvaluated<B> for PbilDoneWhenConverged<R, B, F>
where
    B: Debug + PartialOrd,
    R: Rng,
{
    fn step_from_evaluated<S>(&mut self, point_values: ArrayBase<S, Ix1>)
    where
        S: Data<Elem = B>,
    {
        replace_with_or_abort(self, |o| {
            let mut pbil = Pbil {
                point_value: o.point_value,
                config: o.config.inner,
                state: o.state,
                objective_function: o.objective_function,
            };
            pbil.step_from_evaluated(point_values);
            PbilDoneWhenConverged {
                point_value: pbil.point_value,
                config: DoneWhenConvergedConfig {
                    converged_threshold: o.config.converged_threshold,
                    inner: pbil.config,
                },
                state: pbil.state,
                objective_function: pbil.objective_function,
            }
        });
    }
}

impl<R, B, F> Points<bool> for PbilDoneWhenConverged<R, B, F> {
    fn points(&self) -> CowArray<bool, Ix2> {
        self.state.points()
    }
}

impl<R, B, F> BestPoint<bool> for PbilDoneWhenConverged<R, B, F> {
    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }
}

/// PBIL optimizer.
#[derive(Clone, Debug)]
pub struct Pbil<R, B, F> {
    point_value: PhantomData<B>,
    /// PBIL configuration parameters.
    pub config: Config,
    /// PBIL state.
    pub state: State<R>,
    /// Derivative-free objective function.
    pub objective_function: F,
}

impl<R, B, F> Pbil<R, B, F> {
    /// Convenience function to return a 'Pbil'
    /// without setting 'PhantomData'.
    pub fn new(config: Config, state: State<R>, objective_function: F) -> Self {
        Self {
            point_value: PhantomData,
            config,
            state,
            objective_function,
        }
    }
}

impl<B, F> Pbil<SmallRng, B, F> {
    /// Convenience function
    /// to populate every field
    /// with their default.
    pub fn default(num_bits: NumBits, objective_function: F) -> Self {
        let config = Config::default(num_bits);
        Self {
            point_value: PhantomData,
            state: config.initial_state(),
            config,
            objective_function,
        }
    }
}

impl<R, B, F> Step for Pbil<R, B, F>
where
    R: Rng,
    B: Debug + PartialOrd,
    F: Fn(CowArray<bool, Ix2>) -> Array1<B>,
{
    fn step(&mut self) {
        self.step_from_evaluated((self.objective_function)(self.points()))
    }
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
}

/// PBIL state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State<R> {
    /// Initial and post-evaluation state.
    Init(Init<R>),
    /// State with samples ready for evaluation.
    PreEval(PreEval<R>),
}

impl<R> State<R> {
    /// Return PBIL probabilities.
    pub fn probabilities(&self) -> &Array1<Probability> {
        match &self {
            State::Init(s) => s.probabilities(),
            State::PreEval(s) => s.probabilities(),
        }
    }
}

impl InitialState<State<SmallRng>> for Config {
    fn initial_state(&self) -> State<SmallRng> {
        State::Init(Init::new(
            Array::from_elem(usize::from(self.num_bits), Probability::default()),
            SmallRng::from_entropy(),
        ))
    }
}

impl<R, B, F> StepFromEvaluated<B> for Pbil<R, B, F>
where
    R: Rng,
    B: Debug + PartialOrd,
{
    fn step_from_evaluated<S>(&mut self, point_values: ArrayBase<S, Ix1>)
    where
        S: Data<Elem = B>,
    {
        replace_with_or_abort(&mut self.state, |state| match state {
            State::Init(s) => State::PreEval(s.to_pre_eval(self.config.num_samples)),
            State::PreEval(s) => {
                let mut s = s.to_init(self.config.adjust_rate, point_values);
                s.mutate(
                    self.config.mutation_chance,
                    self.config.mutation_adjust_rate,
                );
                State::Init(s)
            }
        });
    }
}

impl<R, B, F> Points<bool> for Pbil<R, B, F> {
    fn points(&self) -> CowArray<bool, Ix2> {
        self.state.points()
    }
}

impl IsDone for Config {}

impl<R, B, F> BestPoint<bool> for Pbil<R, B, F> {
    fn best_point(&self) -> CowArray<bool, Ix1> {
        self.state.best_point()
    }
}

impl<R> Points<bool> for State<R> {
    fn points(&self) -> CowArray<bool, Ix2> {
        match self {
            State::Init(_) => Array::from_elem((0, 0), false).into(),
            State::PreEval(s) => s.samples().view().into(),
        }
    }
}

impl<R> BestPoint<bool> for State<R> {
    fn best_point(&self) -> CowArray<bool, Ix1> {
        finalize(self.probabilities()).into()
    }
}
