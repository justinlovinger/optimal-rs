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
//!         FixedStepSteepest::new(Sphere, Config::new(StepSize::new(0.5).unwrap()))
//!             .expect("should never fail")
//!             .start()
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
//!     type PointElem = f64;
//!     type PointValue = f64;
//!
//!     fn evaluate(&self, point: CowArray<Self::PointElem, Ix1>) -> Self::PointValue {
//!         point.map(|x| x.powi(2)).sum()
//!     }
//! }
//!
//! impl Differentiable for Sphere {
//!     fn differentiate(&self, point: CowArray<Self::PointElem, Ix1>) -> Array1<Self::PointElem> {
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
//!     fn bounds(&self) -> Box<dyn Iterator<Item = std::ops::RangeInclusive<Self::PointElem>>> {
//!         Box::new(std::iter::repeat(-10.0..=10.0).take(self.len()))
//!     }
//! }
//! ```

use std::ops::{Mul, SubAssign};

use ndarray::{prelude::*, Data};
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use crate::{optimizer::MismatchedLengthError, prelude::*};

use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Fixed step size steepest descent optimizer.
pub type FixedStepSteepest<A, P> = Optimizer<P, Config<A>>;

/// Fixed step size steepest descent configuration parameters.
#[derive(Clone, Debug)]
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

impl<A> Config<A>
where
    A: Clone + SubAssign + Mul<Output = A>,
{
    /// step from one state to another
    /// given point derivatives.
    fn step_from_evaluated<S>(
        &self,
        point_derivatives: ArrayBase<S, Ix1>,
        mut state: Point<A>,
    ) -> Point<A>
    where
        S: Data<Elem = A>,
    {
        state.zip_mut_with(&point_derivatives, |x, d| {
            *x -= self.step_size.clone() * d.clone()
        });
        state
    }
}

impl<A, P> OptimizerConfig<P> for Config<A>
where
    A: Clone + SubAssign + Mul<Output = A> + SampleUniform,
    P: Differentiable<PointElem = A> + FixedLength + Bounded,
{
    type Err = ();

    type State = Point<A>;

    type StateErr = MismatchedLengthError;

    fn validate(&self, _problem: &P) -> Result<(), Self::Err> {
        Ok(())
    }

    fn validate_state(&self, problem: &P, state: &Self::State) -> Result<(), Self::StateErr> {
        if state.len() == problem.len() {
            Ok(())
        } else {
            Err(MismatchedLengthError)
        }
    }

    unsafe fn initial_state(&self, problem: &P) -> Self::State {
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

    unsafe fn step(&self, problem: &P, state: Self::State) -> Self::State {
        self.step_from_evaluated(problem.differentiate(state.view().into()), state)
    }
}

impl<A, P> OptimizerState<P> for Point<A>
where
    P: Problem<PointElem = A>,
{
    type Evaluatee = Array1<P::PointElem>;

    fn evaluatee(&self) -> &Self::Evaluatee {
        self
    }

    fn best_point(&self) -> CowArray<P::PointElem, Ix1> {
        self.into()
    }
}
