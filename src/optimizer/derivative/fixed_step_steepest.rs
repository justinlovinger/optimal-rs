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
//! println!(
//!     "{}",
//!     Config::new(StepSize::new(0.5).unwrap())
//!         .start(std::iter::repeat(-10.0..=10.0).take(2), |point| point
//!             .map(|x| 2.0 * x))
//!         .nth(100)
//!         .unwrap()
//!         .best_point()
//! );
//! ```

use std::ops::{Mul, RangeInclusive, SubAssign};

use derive_getters::Getters;
use ndarray::prelude::*;
use once_cell::sync::OnceCell;
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use crate::prelude::*;

use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A running fixed step size steepest descent optimizer.
#[derive(Clone, Debug, Getters)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FixedStepSteepest<A, FD> {
    /// Optimizer configuration.
    config: Config<A>,

    /// Derivative of objective function to minimize.
    obj_func_d: FD,

    /// State of optimizer.
    state: Point<A>,

    #[getter(skip)]
    #[cfg_attr(feature = "serde", serde(skip))]
    evaluation_cache: OnceCell<Point<A>>,
}

impl<A, FD> FixedStepSteepest<A, FD> {
    fn new(state: Point<A>, config: Config<A>, obj_func_d: FD) -> Self {
        Self {
            config,
            obj_func_d,
            state,
            evaluation_cache: OnceCell::new(),
        }
    }

    /// Return configuration, state, and problem parameters.
    pub fn into_inner(self) -> (Config<A>, Point<A>, FD) {
        (self.config, self.state, self.obj_func_d)
    }
}

impl<A, FD> FixedStepSteepest<A, FD>
where
    FD: Fn(ArrayView1<A>) -> Array1<A>,
{
    /// Return evaluation of current state,
    /// evaluating and caching if necessary.
    pub fn evaluation(&self) -> &Array1<A> {
        self.evaluation_cache.get_or_init(|| self.evaluate())
    }

    fn evaluate(&self) -> Array1<A> {
        (self.obj_func_d)(self.state.view())
    }
}

impl<A, FD> StreamingIterator for FixedStepSteepest<A, FD>
where
    A: Clone + SubAssign + Mul<Output = A>,
    FD: Fn(ArrayView1<A>) -> Array1<A>,
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

impl<A, FD> Optimizer for FixedStepSteepest<A, FD>
where
    A: Clone,
{
    type Point = Array1<A>;

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
    ///
    /// - `initial_bounds`: bounds for generating the initial random point
    /// - `obj_func_d`: derivative of objective function to minimize
    pub fn start<FD>(
        self,
        initial_bounds: impl Iterator<Item = RangeInclusive<A>>,
        obj_func_d: FD,
    ) -> FixedStepSteepest<A, FD>
    where
        A: SampleUniform,
        FD: Fn(ArrayView1<A>) -> Array1<A>,
    {
        FixedStepSteepest::new(
            self.initial_state_using(initial_bounds, &mut thread_rng()),
            self,
            obj_func_d,
        )
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    ///
    /// - `initial_bounds`: bounds for generating the initial random point
    /// - `obj_func_d`: derivative of objective function to minimize
    /// - `rng`: source of randomness
    pub fn start_using<FD, R>(
        self,
        initial_bounds: impl Iterator<Item = RangeInclusive<A>>,
        obj_func_d: FD,
        rng: &mut R,
    ) -> FixedStepSteepest<A, FD>
    where
        A: SampleUniform,
        FD: Fn(ArrayView1<A>) -> Array1<A>,
        R: Rng,
    {
        FixedStepSteepest::new(
            self.initial_state_using(initial_bounds, rng),
            self,
            obj_func_d,
        )
    }

    /// Return this optimizer
    /// running on the given problem.
    /// if the given `state` is valid.
    ///
    /// - `obj_func_d`: derivative of objective function to minimize
    /// - `state`: initial point to start from
    pub fn start_from<FD>(self, obj_func_d: FD, state: Point<A>) -> FixedStepSteepest<A, FD>
    where
        FD: Fn(ArrayView1<A>) -> Array1<A>,
    {
        FixedStepSteepest::new(state, self, obj_func_d)
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
