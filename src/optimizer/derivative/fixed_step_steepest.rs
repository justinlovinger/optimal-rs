#![allow(clippy::needless_doctest_main)]

//! Fixed step size steepest descent,
//! a very simple derivative optimizer.
//!
//! # Examples
//!
//! ```
//! use ndarray::prelude::*;
//! use optimal::{optimizer::derivative::fixed_step_steepest::*, prelude::*};
//!
//! fn main() {
//!     println!(
//!         "{}",
//!         Config::new(StepSize::new(0.5).unwrap()).start(Sphere)
//!             .nth(100)
//!             .unwrap()
//!             .best_point()
//!     );
//! }
//!
//! #[derive(Clone, Debug)]
//! struct Sphere;
//!
//! impl Problem for Sphere {
//!     type Elem = f64;
//!
//!     type Bounds = std::iter::Take<std::iter::Repeat<std::ops::RangeInclusive<f64>>>;
//!
//!     fn differentiate(&self, point: ArrayView1<Self::Elem>) -> Array1<Self::Elem> {
//!         point.map(|x| 2.0 * x)
//!     }
//!
//!     fn bounds(&self) -> Self::Bounds {
//!         std::iter::repeat(-10.0..=10.0).take(2)
//!     }
//! }
//! ```

use std::ops::{Mul, RangeInclusive, SubAssign};

use blanket::blanket;
use derive_getters::Getters;
use ndarray::prelude::*;
use once_cell::sync::OnceCell;
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use crate::{optimizer::MismatchedLengthError, prelude::*};

use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A fixed step size steepest descent optimization problem.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Problem {
    /// Element of a point in this problem space.
    type Elem;

    /// Bounds for points in this problem space.
    type Bounds: Iterator<Item = RangeInclusive<Self::Elem>>;

    /// Return partial derivatives of a point.
    fn differentiate(&self, point: ArrayView1<Self::Elem>) -> Array1<Self::Elem>;

    /// Return bounds for this problem.
    fn bounds(&self) -> Self::Bounds;
}

/// A running fixed step size steepest descent optimizer.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "P: Serialize, P::Elem: Serialize",
        deserialize = "P: Deserialize<'de>, P::Elem: Deserialize<'de>"
    ))
)]
pub struct FixedStepSteepest<P>
where
    P: Problem,
{
    /// Optimizer configuration.
    config: Config<P::Elem>,

    /// Problem being optimized.
    problem: P,

    /// State of optimizer.
    state: Point<P::Elem>,

    #[getter(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    evaluation_cache: OnceCell<Point<P::Elem>>,
}

impl<P> FixedStepSteepest<P>
where
    P: Problem,
{
    fn new(state: Point<P::Elem>, config: Config<P::Elem>, problem: P) -> Self {
        Self {
            problem,
            config,
            state,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Return configuration, problem, and state.
    pub fn into_inner(self) -> (Config<P::Elem>, P, Point<P::Elem>) {
        (self.config, self.problem, self.state)
    }
}

impl<P> FixedStepSteepest<P>
where
    P: Problem,
{
    /// Return evaluation of current state,
    /// evaluating and caching if necessary.
    pub fn evaluation(&self) -> &Array1<P::Elem> {
        self.evaluation_cache.get_or_init(|| self.evaluate())
    }

    fn evaluate(&self) -> Array1<P::Elem> {
        self.problem.differentiate(self.state.view())
    }
}

impl<P> StreamingIterator for FixedStepSteepest<P>
where
    P: Problem,
    P::Elem: Clone + SubAssign + Mul<Output = P::Elem>,
{
    type Item = Self;

    fn advance(&mut self) {
        let evaluation = self
            .evaluation_cache
            .take()
            .unwrap_or_else(|| self.evaluate());
        self.state.zip_mut_with(&evaluation, |x, d| {
            *x -= self.config.step_size.clone() * d.clone()
        });
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

impl<P> Optimizer for FixedStepSteepest<P>
where
    P: Problem,
    P::Elem: Clone,
{
    type Point = Array1<P::Elem>;

    fn best_point(&self) -> Self::Point {
        self.state.clone()
    }
}

/// Fixed step size steepest descent configuration parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A> {
    /// Length of each step.
    pub step_size: StepSize<A>,
}

type Point<A> = Array1<A>;

impl<A> Config<A> {
    /// Return a new 'Config'.
    pub fn new(step_size: StepSize<A>) -> Self {
        Self { step_size }
    }
}

impl<A> Config<A> {
    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    pub fn start<P>(self, problem: P) -> FixedStepSteepest<P>
    where
        A: SampleUniform,
        P: Problem<Elem = A>,
    {
        FixedStepSteepest::new(
            self.initial_state_using(problem.bounds(), &mut thread_rng()),
            self,
            problem,
        )
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    pub fn start_using<P, R>(self, problem: P, rng: &mut R) -> FixedStepSteepest<P>
    where
        A: SampleUniform,
        P: Problem<Elem = A>,
        R: Rng,
    {
        FixedStepSteepest::new(
            self.initial_state_using(problem.bounds(), rng),
            self,
            problem,
        )
    }

    /// Return this optimizer
    /// running on the given problem.
    /// if the given `state` is valid.
    #[allow(clippy::type_complexity)]
    pub fn start_from<P>(
        self,
        problem: P,
        state: Point<P::Elem>,
    ) -> Result<FixedStepSteepest<P>, (MismatchedLengthError, Self, P, Point<P::Elem>)>
    where
        P: Problem<Elem = A>,
    {
        if state.len() == problem.bounds().count() {
            Ok(FixedStepSteepest::new(state, self, problem))
        } else {
            Err((MismatchedLengthError, self, problem, state))
        }
    }

    fn initial_state_using<R>(
        &self,
        initial_bounds: impl Iterator<Item = RangeInclusive<A>>,
        rng: &mut R,
    ) -> Point<A>
    where
        A: SampleUniform,
        R: Rng,
    {
        initial_bounds
            .map(|range| {
                let (start, end) = range.into_inner();
                Uniform::new_inclusive(start, end).sample(rng)
            })
            .collect()
    }
}
