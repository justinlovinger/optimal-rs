//! A reference optimizer of random points.
//!
//! # Examples
//!
//! ```
//! use optimal::{optimizer::derivative_free::random::Config, prelude::*};
//! use ndarray::{Data, RemoveAxis, prelude::*};
//! use rand::distributions::Uniform;
//! use std::iter::repeat;
//!
//! fn main() {
//!     let config = Config::new(100, repeat(Uniform::new(-5., 5.)).take(2).collect());
//!     // The random optimizer finishes in one step.
//!     let state = config.step(|xs| f(xs), config.initial_state());
//!     println!(
//!         "f({}) = {}",
//!         config.best_point(&state),
//!         config.best_point_value(&state).unwrap()
//!     );
//! }
//!
//! fn f<S, D>(xs: ArrayBase<S, D>) -> Array<f64, D::Smaller>
//! where
//!     S: Data<Elem = f64>,
//!     D: Dimension + RemoveAxis,
//! {
//!     xs.sum_axis(Axis(xs.ndim() - 1))
//! }
//! ```

mod states;

use ndarray::{prelude::*, Data};
use rand::prelude::*;
use std::marker::PhantomData;

use crate::prelude::*;

pub use self::states::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Random optimizer configuration.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A, D> {
    point_type: PhantomData<A>,
    /// Number of random points to generate.
    pub num_points: usize, // TODO: should use type wrapper and disallow zero points.
    /// Distribution for each element in points.
    pub distributions: Array1<D>,
}

impl<A, D> Config<A, D>
where
    D: Distribution<A>,
{
    /// Convenience function to return a 'Config'
    /// without setting 'PhantomData'.
    pub fn new(num_points: usize, distributions: Array1<D>) -> Self {
        Self {
            point_type: PhantomData,
            num_points,
            distributions,
        }
    }
}

/// Random optimizer state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State<A, B> {
    /// Initial state,
    /// ready to evaluate points.
    Init(Init<A>),
    /// Final state,
    /// with points evaluated.
    Done(Done<A, B>),
}

impl<A, B, D> InitialState<State<A, B>> for Config<A, D>
where
    D: Distribution<A>,
{
    fn initial_state(&self) -> State<A, B> {
        State::Init(Init::new(
            self.num_points,
            &self.distributions,
            SmallRng::from_entropy(),
        ))
    }
}

impl<A, B, D> StepFromEvaluated<B, State<A, B>, State<A, B>> for Config<A, D>
where
    A: Clone,
    B: Clone + PartialOrd,
    D: Distribution<A>,
{
    fn step_from_evaluated<S>(
        &self,
        point_values: ArrayBase<S, Ix1>,
        state: State<A, B>,
    ) -> State<A, B>
    where
        S: Data<Elem = B>,
    {
        match state {
            State::Init(s) => State::Done(s.to_done(point_values)),
            State::Done(s) => State::Done(s),
        }
    }
}

impl<A, B, D> Points<A, State<A, B>> for Config<A, D>
where
    A: Clone + Default,
{
    fn points<'a>(&'a self, state: &'a State<A, B>) -> CowArray<A, Ix2> {
        match state {
            State::Init(s) => s.points().into(),
            State::Done(_) => Array::from_elem((0, 0), A::default()).into(),
        }
    }
}

impl<A, B, D> IsDone<State<A, B>> for Config<A, D> {
    fn is_done(&self, state: &State<A, B>) -> bool {
        match state {
            State::Init(_) => false,
            State::Done(_) => true,
        }
    }
}

impl<A, B, D> BestPoint<A, State<A, B>> for Config<A, D> {
    fn best_point<'a>(&'a self, state: &'a State<A, B>) -> CowArray<A, Ix1> {
        match state {
            // What should the best point be
            // for an initial state?
            // For now,
            // the first point is returned.
            State::Init(s) => s.points().row(0).into(),
            State::Done(s) => s.best_point().into(),
        }
    }
}

impl<A, B, D> BestPointValue<B, State<A, B>> for Config<A, D>
where
    B: Clone,
{
    fn best_point_value(&self, state: &State<A, B>) -> Option<B> {
        match state {
            State::Init(_) => None,
            State::Done(s) => Some(s.best_point_value().clone()),
        }
    }
}
