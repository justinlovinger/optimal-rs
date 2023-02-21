//! Population-based incremental learning (PBIL).
//!
//! # Examples
//!
//! ```
//! use ndarray::{Data, RemoveAxis, prelude::*};
//! use optimal::{optimizer::derivative_free::pbil::DoneWhenConvergedConfig, prelude::*};
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let config = DoneWhenConvergedConfig::default(16.into());
//!     let mut iter = config.iterate(|xs| f(xs), config.initial_state());
//!     // `unwrap` is safe
//!     // because the optimizer is guaranteed to converge.
//!     let bs = config.best_point(iter.find(|s| config.is_done(s)).unwrap());
//!     println!("f({}) = {}", bs, f(bs.view()));
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
use std::fmt::Debug;

use crate::prelude::*;

pub use self::{states::*, types::*};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// PBIL with check for converged probabilities.
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

impl InitialState<State<SmallRng>> for DoneWhenConvergedConfig {
    fn initial_state(&self) -> State<SmallRng> {
        self.inner.initial_state()
    }
}

impl<A, R> StepFromEvaluated<A, State<R>, State<R>> for DoneWhenConvergedConfig
where
    A: Debug + PartialOrd,
    R: Rng,
{
    fn step_from_evaluated<S: Data<Elem = A>>(
        &self,
        point_values: ArrayBase<S, Ix1>,
        state: State<R>,
    ) -> State<R> {
        self.inner.step_from_evaluated(point_values, state)
    }
}

impl<R> Points<bool, State<R>> for DoneWhenConvergedConfig {
    fn points<'a>(&'a self, state: &'a State<R>) -> CowArray<bool, Ix2> {
        self.inner.points(state)
    }
}

impl<R> IsDone<State<R>> for DoneWhenConvergedConfig {
    fn is_done(&self, state: &State<R>) -> bool {
        converged(&self.converged_threshold, state.probabilities())
    }
}

impl<R> BestPoint<bool, State<R>> for DoneWhenConvergedConfig {
    fn best_point<'a>(&'a self, state: &'a State<R>) -> CowArray<bool, Ix1> {
        self.inner.best_point(state)
    }
}

/// PBIL configuration.
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

impl<A, R> StepFromEvaluated<A, State<R>, State<R>> for Config
where
    A: Debug + PartialOrd,
    R: Rng,
{
    fn step_from_evaluated<S: Data<Elem = A>>(
        &self,
        point_values: ArrayBase<S, Ix1>,
        state: State<R>,
    ) -> State<R> {
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

impl<R> Points<bool, State<R>> for Config {
    fn points<'a>(&'a self, state: &'a State<R>) -> CowArray<bool, Ix2> {
        match state {
            State::Init(_) => Array::from_elem((0, 0), false).into(),
            State::PreEval(s) => s.samples().view().into(),
        }
    }
}

impl<S> IsDone<S> for Config {}

impl<R> BestPoint<bool, State<R>> for Config {
    fn best_point<'a>(&'a self, state: &'a State<R>) -> CowArray<bool, Ix1> {
        finalize(state.probabilities()).into()
    }
}
