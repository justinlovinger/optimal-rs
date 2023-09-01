use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

use num_traits::{AsPrimitive, One};
pub use optimal_core::prelude::*;

use crate::StepSize;

use super::types::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DynState<A> {
    Ready(State<A, Ready<A>>),
    Searching(State<A, Searching<A>>),
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct State<A, T> {
    point: Vec<A>,
    inner: T,
}

/// Ready to begin line search.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ready<A> {
    last_step_size: A,
}

/// Line searching.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Searching<A> {
    point_value: A,
    step_direction: Vec<A>,
    c_1_times_point_derivatives_dot_step_direction: A,
    step_size: A,
    point_at_step: Vec<A>,
}

impl<A> DynState<A> {
    /// Return an initial state.
    pub fn new(point: Vec<A>, initial_step_size: StepSize<A>) -> Self {
        DynState::Ready(State {
            point,
            inner: Ready {
                last_step_size: initial_step_size.0,
            },
        })
    }
}

impl<A, T> State<A, T> {
    pub fn best_point(&self) -> &[A] {
        &self.point
    }
}

impl<A> State<A, Ready<A>> {
    pub fn point(&self) -> &[A] {
        &self.point
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_searching(
        self,
        initial_step_size_incr_rate: IncrRate<A>,
        c_1: SufficientDecreaseParameter<A>,
        point_value: A,
        point_derivatives: Vec<A>,
    ) -> State<A, Searching<A>>
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
        let step_size = initial_step_size_incr_rate * self.inner.last_step_size;
        State {
            inner: Searching {
                point_at_step: descend(&self.point, step_size, &step_direction),
                point_value,
                c_1_times_point_derivatives_dot_step_direction: c_1.into_inner()
                    * point_derivatives
                        .into_iter()
                        .zip(step_direction.iter().copied())
                        .map(|(x, y)| x * y)
                        .sum(),
                step_direction,
                step_size,
            },
            point: self.point,
        }
    }
}

impl<A> State<A, Searching<A>> {
    #[allow(clippy::misnamed_getters)]
    pub fn point(&self) -> &[A] {
        &self.inner.point_at_step
    }

    pub fn point_value(&self) -> &A {
        &self.inner.point_value
    }

    pub fn is_sufficient_decrease(&self, point_value: A) -> bool
    where
        A: Copy + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        is_sufficient_decrease(
            self.inner.point_value,
            self.inner.step_size,
            self.inner.c_1_times_point_derivatives_dot_step_direction,
            point_value,
        )
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_ready(self) -> State<A, Ready<A>>
    where
        A: Clone + Copy + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        State {
            point: self.inner.point_at_step,
            inner: Ready {
                last_step_size: self.inner.step_size,
            },
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_searching(mut self, backtracking_rate: BacktrackingRate<A>) -> Self
    where
        A: Clone + Copy + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        self.inner.step_size = backtracking_rate.into_inner() * self.inner.step_size;
        self.inner.point_at_step = descend(
            &self.point,
            self.inner.step_size,
            &self.inner.step_direction,
        );
        self
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
