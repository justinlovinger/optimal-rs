use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

use num_traits::{AsPrimitive, One};
pub use optimal_core::prelude::*;

use crate::StepSize;

use super::Config;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Backtracking steepest descent state.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State<A> {
    /// Ready to begin line search.
    Ready(Ready<A>),
    /// Line searching.
    Searching(Searching<A>),
}

/// Ready to begin line search.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ready<A> {
    point: Vec<A>,
    last_step_size: A,
}

/// Line searching.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Searching<A> {
    point: Vec<A>,
    point_value: A,
    step_direction: Vec<A>,
    c_1_times_point_derivatives_dot_step_direction: A,
    step_size: A,
    point_at_step: Vec<A>,
}

impl<A> State<A> {
    /// Return data to be evaluated.
    pub fn evaluatee(&self) -> &[A] {
        match self {
            State::Ready(x) => x.point(),
            State::Searching(x) => x.point(),
        }
    }

    /// Return the best point discovered.
    pub fn best_point(&self) -> &[A] {
        match self {
            State::Ready(x) => x.best_point(),
            State::Searching(x) => x.best_point(),
        }
    }

    /// Return the value of the best point discovered,
    /// if possible
    /// without evaluating.
    pub fn stored_best_point_value(&self) -> Option<&A> {
        match self {
            State::Ready(_) => None,
            State::Searching(x) => Some(&x.point_value),
        }
    }

    /// Return an initial state.
    pub fn new(point: Vec<A>, initial_step_size: StepSize<A>) -> Self {
        Self::Ready(Ready {
            point,
            last_step_size: initial_step_size.0,
        })
    }

    /// Return length of point in this state.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Ready(x) => x.point.len(),
            Self::Searching(x) => x.point.len(),
        }
    }
}

impl<A> Ready<A> {
    pub(super) fn point(&self) -> &[A] {
        self.best_point()
    }

    pub(super) fn best_point(&self) -> &[A] {
        &self.point
    }

    pub(super) fn step_from_evaluated(
        self,
        config: &Config<A>,
        point_value: A,
        point_derivatives: Vec<A>,
    ) -> State<A>
    where
        A: 'static
            + Clone
            + Copy
            + Neg<Output = A>
            + Add<Output = A>
            + Sub<Output = A>
            + Div<Output = A>
            + One
            + Sum,
        f64: AsPrimitive<A>,
    {
        let step_direction = point_derivatives.iter().map(|x| -*x).collect::<Vec<_>>();
        let step_size = config.initial_step_size_incr_rate * self.last_step_size;
        State::Searching(Searching {
            point_at_step: descend(&self.point, step_size, &step_direction),
            point: self.point,
            point_value,
            c_1_times_point_derivatives_dot_step_direction: config.c_1.into_inner()
                * point_derivatives
                    .into_iter()
                    .zip(step_direction.iter().copied())
                    .map(|(x, y)| x * y)
                    .sum(),
            step_direction,
            step_size,
        })
    }
}

impl<A> Searching<A> {
    pub(super) fn best_point(&self) -> &[A] {
        &self.point
    }

    #[allow(clippy::misnamed_getters)]
    pub(super) fn point(&self) -> &[A] {
        &self.point_at_step
    }

    pub(super) fn step_from_evaluated(mut self, config: &Config<A>, point_value: A) -> State<A>
    where
        A: Clone + Copy + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        if is_sufficient_decrease(
            self.point_value,
            self.step_size,
            self.c_1_times_point_derivatives_dot_step_direction,
            point_value,
        ) {
            State::Ready(Ready {
                point: self.point_at_step,
                last_step_size: self.step_size,
            })
        } else {
            self.step_size = config.backtracking_rate.into_inner() * self.step_size;
            self.point_at_step = descend(&self.point, self.step_size, &self.step_direction);
            State::Searching(self)
        }
    }
}

fn descend<A>(point: &[A], step_size: A, step_direction: &[A]) -> Vec<A>
where
    A: Clone + Add<Output = A> + Mul<Output = A>,
{
    point
        .iter()
        .zip(step_direction)
        .map(|(x, d)| x.clone() + step_size.clone() * d.clone())
        .collect()
}

/// The sufficient decrease condition,
/// also known as the Armijo rule,
/// mathematically written as:
/// $f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)$.
fn is_sufficient_decrease<A>(
    point_value: A,
    step_size: A,
    c_1_times_point_derivatives_dot_step_direction: A,
    new_point_value: A,
) -> bool
where
    A: PartialOrd + Add<Output = A> + Mul<Output = A>,
{
    new_point_value <= point_value + step_size * c_1_times_point_derivatives_dot_step_direction
}
