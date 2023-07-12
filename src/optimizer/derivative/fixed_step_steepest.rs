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
//!     type Point<'a> = CowArray<'a, f64, Ix1>;
//!     type Value = f64;
//!
//!     fn evaluate(&self, point: Self::Point<'_>) -> Self::Value {
//!         point.map(|x| x.powi(2)).sum()
//!     }
//! }
//!
//! impl Differentiable for Sphere {
//!     type Derivative = Array1<f64>;
//!
//!     fn differentiate(&self, point: Self::Point<'_>) -> Self::Derivative {
//!         point.map(|x| 2.0 * x)
//!     }
//! }
//!
//! impl FixedLength for Sphere {
//!     fn len(&self) -> usize {
//!         2
//!     }
//! }
//!
//! impl Bounded for Sphere {
//!     type Bounds = std::iter::Take<std::iter::Repeat<std::ops::RangeInclusive<f64>>>;
//!
//!     fn bounds(&self) -> Self::Bounds {
//!         std::iter::repeat(-10.0..=10.0).take(self.len())
//!     }
//! }
//! ```

use std::ops::{Mul, RangeInclusive, SubAssign};

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

/// A running fixed step size steepest descent optimizer.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FixedStepSteepest<A, P> {
    /// Optimizer configuration.
    config: Config<A>,

    /// Problem being optimized.
    problem: P,

    /// State of optimizer.
    state: Point<A>,

    #[getter(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    evaluation_cache: OnceCell<Point<A>>,
}

impl<A, P> FixedStepSteepest<A, P> {
    fn new(state: Point<A>, config: Config<A>, problem: P) -> Self {
        Self {
            problem,
            config,
            state,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Return configuration, problem, and state.
    pub fn into_inner(self) -> (Config<A>, P, Point<A>) {
        (self.config, self.problem, self.state)
    }
}

impl<A, P> FixedStepSteepest<A, P>
where
    for<'a> P: Differentiable<Point<'a> = CowArray<'a, A, Ix1>, Derivative = Array1<A>> + 'a,
{
    /// Return evaluation of current state,
    /// evaluating and caching if necessary.
    pub fn evaluation(&self) -> &P::Derivative {
        self.evaluation_cache.get_or_init(|| self.evaluate())
    }

    fn evaluate(&self) -> P::Derivative {
        self.problem.differentiate(self.state.view().into())
    }
}

impl<A, P> StreamingIterator for FixedStepSteepest<A, P>
where
    for<'a> P: Differentiable<Point<'a> = CowArray<'a, A, Ix1>, Derivative = Array1<A>> + 'a,
    A: Clone + SubAssign + Mul<Output = A>,
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

impl<A, P> Optimizer<P> for FixedStepSteepest<A, P>
where
    for<'a> P: Differentiable<Point<'a> = CowArray<'a, A, Ix1>> + 'a,
{
    fn best_point(&self) -> P::Point<'_> {
        self.state.view().into()
    }

    fn best_point_value(&self) -> P::Value {
        self.problem().evaluate(self.best_point())
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
    pub fn start<P>(self, problem: P) -> FixedStepSteepest<A, P>
    where
        A: SampleUniform,
        P: Bounded + FixedLength,
        P::Bounds: Iterator<Item = RangeInclusive<A>>,
    {
        FixedStepSteepest::new(
            self.initial_state_using(problem.bounds().take(problem.len()), &mut thread_rng()),
            self,
            problem,
        )
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    pub fn start_using<P, R>(self, problem: P, rng: &mut R) -> FixedStepSteepest<A, P>
    where
        A: SampleUniform,
        P: Bounded + FixedLength,
        P::Bounds: Iterator<Item = RangeInclusive<A>>,
        R: Rng,
    {
        FixedStepSteepest::new(
            self.initial_state_using(problem.bounds().take(problem.len()), rng),
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
        state: Point<A>,
    ) -> Result<FixedStepSteepest<A, P>, (MismatchedLengthError, Self, P, Point<A>)>
    where
        P: FixedLength,
    {
        if state.len() == problem.len() {
            Ok(FixedStepSteepest::new(state, self, problem))
        } else {
            Err((MismatchedLengthError, self, problem, state))
        }
    }

    fn initial_state_using<R>(
        &self,
        bounds: impl Iterator<Item = RangeInclusive<A>>,
        rng: &mut R,
    ) -> Point<A>
    where
        A: SampleUniform,
        R: Rng,
    {
        bounds
            .map(|range| {
                let (start, end) = range.into_inner();
                Uniform::new_inclusive(start, end).sample(rng)
            })
            .collect()
    }
}
