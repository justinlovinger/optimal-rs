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

use ndarray::prelude::*;
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use crate::{optimizer::MismatchedLengthError, prelude::*};

use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Fixed step size steepest descent optimizer.
pub type FixedStepSteepest<A, P> = RunningOptimizer<P, Config<A>>;

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

impl<A, P> OptimizerConfig<P> for Config<A>
where
    A: Clone + SubAssign + Mul<Output = A> + SampleUniform,
    for<'a> P: Differentiable<Point<'a> = CowArray<'a, A, Ix1>, Derivative = Array1<A>>
        + FixedLength
        + Bounded
        + 'a,
    P::Bounds: Iterator<Item = RangeInclusive<A>>,
{
    type State = Point<A>;

    type StateErr = MismatchedLengthError;

    type Evaluation = P::Derivative;

    fn validate_state(&self, problem: &P, state: &Self::State) -> Result<(), Self::StateErr> {
        if state.len() == problem.len() {
            Ok(())
        } else {
            Err(MismatchedLengthError)
        }
    }

    fn initial_state(&self, problem: &P) -> Self::State {
        let mut rng = thread_rng();
        problem
            .bounds()
            .take(problem.len())
            .map(|range| {
                let (start, end) = range.into_inner();
                Uniform::new_inclusive(start, end).sample(&mut rng)
            })
            .collect()
    }

    unsafe fn evaluate(&self, problem: &P, state: &Self::State) -> Self::Evaluation {
        problem.differentiate(state.view().into())
    }

    unsafe fn step_from_evaluated(
        &self,
        evaluation: Self::Evaluation,
        mut state: Self::State,
    ) -> Self::State {
        state.zip_mut_with(&evaluation, |x, d| *x -= self.step_size.clone() * d.clone());
        state
    }
}

impl<A, P> OptimizerState<P> for Point<A>
where
    for<'a> P: Problem<Point<'a> = CowArray<'a, A, Ix1>> + 'a,
{
    type Evaluatee<'a> = ArrayView1<'a, A> where A: 'a;

    fn evaluatee(&self) -> Self::Evaluatee<'_> {
        self.into()
    }

    fn best_point(&self) -> P::Point<'_> {
        self.into()
    }
}
