#![allow(clippy::needless_doctest_main)]

//! Fixed step-size descent optimization.
//!
//! # Examples
//!
//! ```
//! use optimal_linesearch::{fixed_step_size, prelude::*, steepest_descent};
//!
//! println!(
//!     "{:?}",
//!     fixed_step_size::Config {
//!         step_size: StepSize::new(0.5).unwrap()
//!     }
//!     .build(
//!         steepest_descent::SteepestDescent::new(|point| point
//!             .iter()
//!             .map(|x| 2.0 * x)
//!             .collect::<Vec<_>>()),
//!         std::iter::repeat(-10.0..=10.0).take(2),
//!     )
//!     .nth(100)
//!     .unwrap()
//!     .best_point()
//! );
//! ```

use std::{
    ops::{Mul, RangeInclusive, SubAssign},
    sync::Arc,
};

use derive_getters::{Dissolve, Getters};
pub use optimal_core::prelude::*;
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use crate::StepDirection;

pub use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A running fixed step-size optimizer.
#[derive(Clone, Debug, Getters, Dissolve)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct FixedStepSize<A, D> {
    /// Optimizer configuration.
    config: Config<A>,

    /// State of optimizer.
    state: State<A>,

    /// Component for getting step-direction.
    step_direction: D,
}

/// Fixed step-size configuration parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A> {
    /// Length of each step.
    pub step_size: StepSize<A>,
}

/// Fixed step-size state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum State<A> {
    /// Optimizer started.
    Started {
        /// Point to step from.
        point: Vec<A>,
    },
    /// Getting step-direction from component.
    GettingStepDirection {
        /// Point to step from.
        point: Arc<Vec<A>>,
    },
    /// Took a step from previous `point` in step-direction.
    Stepped {
        /// New point.
        point: Vec<A>,
    },
}

impl<A> State<A> {
    fn new(point: Vec<A>) -> Self {
        Self::Started { point }
    }
}

impl<A> Config<A> {
    /// Return a new 'FixedStepSize'.
    ///
    /// This is nondeterministic.
    ///
    /// - `step_direction`: component for getting step-direction
    /// - `initial_bounds`: bounds for generating the initial random point
    pub fn build<D>(
        self,
        step_direction: D,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
    ) -> FixedStepSize<A, D>
    where
        A: SampleUniform,
    {
        FixedStepSize {
            step_direction,
            state: self.initial_state_using(initial_bounds, &mut thread_rng()),
            config: self,
        }
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    ///
    /// - `step_direction`: component for getting step-direction
    /// - `initial_bounds`: bounds for generating the initial random point
    /// - `rng`: source of randomness
    pub fn build_using<D, R>(
        self,
        step_direction: D,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        rng: &mut R,
    ) -> FixedStepSize<A, D>
    where
        A: SampleUniform,
        R: Rng,
    {
        FixedStepSize {
            step_direction,
            state: self.initial_state_using(initial_bounds, rng),
            config: self,
        }
    }

    /// Return this optimizer
    /// running on the given problem.
    /// if the given `state` is valid.
    ///
    /// - `step_direction`: component for getting step-direction
    /// - `point`: initial point
    pub fn build_from<D>(self, step_direction: D, point: Vec<A>) -> FixedStepSize<A, D> {
        FixedStepSize {
            config: self,
            step_direction,
            state: State::new(point),
        }
    }

    fn initial_state_using<R>(
        &self,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<A>>,
        rng: &mut R,
    ) -> State<A>
    where
        A: SampleUniform,
        R: Rng,
    {
        State::new(
            initial_bounds
                .into_iter()
                .map(|range| {
                    let (start, end) = range.into_inner();
                    Uniform::new_inclusive(start, end).sample(rng)
                })
                .collect(),
        )
    }
}

impl<A, D> StreamingIterator for FixedStepSize<A, D>
where
    A: Clone + SubAssign + Mul<Output = A>,
    D: StepDirection<Elem = A, Point = Arc<Vec<A>>>,
{
    type Item = Self;

    fn advance(&mut self) {
        replace_with::replace_with_or_abort(&mut self.state, |state| match state {
            State::Started { point } => {
                let point: D::Point = point.into();
                self.step_direction.start_iteration(point.clone());
                State::GettingStepDirection { point }
            }
            State::GettingStepDirection { point } => match self.step_direction.step() {
                Some(step_direction) => {
                    let mut point = Arc::try_unwrap(point).unwrap_or_else(|arc| (*arc).clone());
                    point
                        .iter_mut()
                        .zip(step_direction.step_direction)
                        .for_each(|(x, d)| *x -= self.config.step_size.clone() * d);
                    State::Stepped { point }
                }
                None => State::GettingStepDirection { point },
            },
            State::Stepped { point } => {
                let point: D::Point = point.into();
                self.step_direction.start_iteration(point.clone());
                State::GettingStepDirection { point }
            }
        });
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

impl<A, FD> Optimizer for FixedStepSize<A, FD>
where
    A: Clone,
{
    type Point = Vec<A>;

    fn best_point(&self) -> Self::Point {
        self.point().into()
    }
}

impl<A, D> FixedStepSize<A, D> {
    /// Return the point being stepped from.
    pub fn point(&self) -> &[A] {
        match &self.state {
            State::Started { point } => point,
            State::GettingStepDirection { point } => point.as_ref(),
            State::Stepped { point } => point,
        }
    }
}
