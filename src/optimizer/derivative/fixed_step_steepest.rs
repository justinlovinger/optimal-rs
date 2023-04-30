#![allow(clippy::needless_doctest_main)]

//! Fixed step size steepest descent,
//! a very simple derivative optimizer.
//!
//! # Examples
//!
//! ```
//! use ndarray::prelude::*;
//! use ndarray_rand::RandomExt;
//! use optimal::{optimizer::derivative::fixed_step_steepest::*, prelude::*};
//! use rand::distributions::Uniform;
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter = Config::new(Sphere, StepSize::new(0.5).unwrap())
//!         .start()
//!         .into_streaming_iter();
//!     println!("{}", iter.nth(100).unwrap().best_point());
//! }
//!
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

use std::{
    borrow::Borrow,
    marker::PhantomData,
    ops::{Mul, SubAssign},
};

use ndarray::{prelude::*, Data};
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};
use replace_with::replace_with_or_abort;

use crate::prelude::*;

use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Fixed step size steepest descent configuration parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A, P> {
    /// A differentiable objective function.
    pub problem: P,
    /// Length of each step.
    pub step_size: StepSize<A>,
}

/// Running fixed step size steepest descent optimizer.
#[derive(Clone, Debug)]
pub struct Running<A, P, C> {
    problem: PhantomData<P>,
    /// Fixed step size steepest descent configuration parameters.
    pub config: C,
    /// Fixed step size steepest descent state,
    /// a point.
    pub state: Point<A>,
}

type Point<A> = Array1<A>;

impl<A, P> Config<A, P> {
    /// Return a new 'Config'.
    pub fn new(problem: P, step_size: StepSize<A>) -> Self {
        Self { problem, step_size }
    }
}

impl<A, P> OptimizerConfig for Config<A, P>
where
    P: Differentiable<PointElem = A, PointValue = A> + FixedLength + Bounded,
    P::PointElem: Clone + SubAssign + Mul<Output = A> + SampleUniform,
{
    type Problem = P;

    type Optimizer = Running<A, P, Self>;

    fn start(self) -> Self::Optimizer {
        let mut rng = thread_rng();
        let state = self
            .problem
            .bounds()
            .take(self.problem.len())
            .map(|range| {
                let (start, end) = range.into_inner();
                Uniform::new_inclusive(start, end).sample(&mut rng)
            })
            .collect();
        Running::new(self, state)
    }

    fn problem(&self) -> &Self::Problem {
        &self.problem
    }
}

impl<A, P, C> Running<A, P, C> {
    fn new(config: C, state: Point<A>) -> Self {
        Self {
            problem: PhantomData,
            config,
            state,
        }
    }

    /// Return optimizer configuration.
    pub fn config(&self) -> &C {
        &self.config
    }

    /// Return state of optimizer.
    pub fn state(&self) -> &Point<A> {
        &self.state
    }

    /// Stop optimization run,
    /// returning configuration and state.
    pub fn into_inner(self) -> (C, Point<A>) {
        (self.config, self.state)
    }
}

impl<A, P, C> RunningOptimizerBase for Running<A, P, C>
where
    P: Problem<PointElem = A, PointValue = A>,
    C: Borrow<Config<A, P>>,
{
    type Problem = P;

    fn problem(&self) -> &Self::Problem {
        &self.config.borrow().problem
    }

    fn best_point(&self) -> CowArray<A, Ix1> {
        (&self.state).into()
    }

    fn stored_best_point_value(&self) -> Option<&A> {
        None
    }
}

impl<A, P, C> RunningOptimizerStep for Running<A, P, C>
where
    A: Clone + SubAssign + Mul<Output = A>,
    P: Differentiable<PointElem = A, PointValue = A>,
    C: Borrow<Config<A, P>>,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |point| {
            self.config.borrow().step_from_evaluated(
                self.config
                    .borrow()
                    .problem
                    .differentiate(point.view().into()),
                point,
            )
        });
    }
}

impl<A, P, C> PointBased for Running<A, P, C>
where
    A: Clone + SubAssign + Mul<Output = A>,
    P: Differentiable<PointElem = A, PointValue = A>,
    C: Borrow<Config<A, P>>,
{
    fn point(&self) -> Option<ArrayView1<A>> {
        Some(self.state.view())
    }
}

impl<A, P> Config<A, P>
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
