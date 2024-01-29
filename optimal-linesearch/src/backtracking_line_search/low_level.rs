//! Low-level functions offering greater flexibility.
//!
//! # Examples
//!
//! ```
//! use optimal_linesearch::{
//!     backtracking_line_search::{low_level::*, types::*},
//!     initial_step_size::IncrRate,
//!     step_direction::steepest_descent,
//!     StepSize,
//! };
//!
//! fn main() {
//!     let c_1 = SufficientDecreaseParameter::default();
//!     let backtracking_rate = BacktrackingRate::default();
//!     let incr_rate = IncrRate::from_backtracking_rate(backtracking_rate);
//!     
//!     let mut step_size = StepSize::new(1.0).unwrap();
//!     let mut point = vec![10.0, 10.0];
//!     for _ in 0..100 {
//!         let value = obj_func(&point);
//!         let derivatives = obj_func_d(&point);
//!         let direction = steepest_descent(derivatives.iter().copied()).collect();
//!         (step_size, point) = BacktrackingSearcher::new(
//!             c_1,
//!             point,
//!             value,
//!             derivatives,
//!             direction,
//!         )
//!         .search(backtracking_rate, &obj_func, step_size);
//!         step_size = incr_rate * step_size;
//!     }
//!     println!("{:?}", point);
//! }
//!
//! fn obj_func(point: &[f64]) -> f64 {
//!     point.iter().map(|x| x.powi(2)).sum()
//! }
//!
//! fn obj_func_d(point: &[f64]) -> Vec<f64> {
//!     point.iter().map(|x| 2.0 * x).collect()
//! }
//! ```

use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul, Neg},
};

use derive_getters::{Dissolve, Getters};

use crate::{descend, StepSize};

use super::types::*;

/// Values required for backtracking line-search.
#[derive(Clone, Debug, Dissolve, Getters)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[dissolve(rename = "into_parts")]
pub struct BacktrackingSearcher<A> {
    c_1_times_derivatives_dot_direction: C1TimesDerivativesDotDirection<A>,
    point: Vec<A>,
    value: A,
    direction: Vec<A>,
}

impl<A> BacktrackingSearcher<A> {
    /// Prepare for backtracking line-search.
    pub fn new(
        c_1: SufficientDecreaseParameter<A>,
        point: Vec<A>,
        value: A,
        derivatives: impl IntoIterator<Item = A>,
        direction: Vec<A>,
    ) -> Self
    where
        A: Clone + Neg<Output = A> + Mul<Output = A> + Sum,
    {
        debug_assert_eq!(point.len(), direction.len());
        Self {
            c_1_times_derivatives_dot_direction: C1TimesDerivativesDotDirection::new(
                c_1,
                derivatives,
                direction.iter().cloned(),
            ),
            point,
            value,
            direction,
        }
    }

    /// Return the step-size
    /// and corresponding point
    /// that minimizes the objective function.
    pub fn search<F>(
        &self,
        backtracking_rate: BacktrackingRate<A>,
        obj_func: F,
        initial_step_size: StepSize<A>,
    ) -> (StepSize<A>, Vec<A>)
    where
        A: Clone + PartialOrd + Add<Output = A> + Mul<Output = A>,
        F: Fn(&[A]) -> A,
    {
        let mut step_size = initial_step_size;
        let point = loop {
            let point_at_step = self.point_at_step(step_size.clone()).collect::<Vec<_>>();
            if self.is_sufficient_decrease(step_size.clone(), obj_func(&point_at_step)) {
                break point_at_step;
            } else {
                step_size = backtracking_rate.clone() * step_size.clone();
            }
        };
        (step_size, point)
    }

    /// Return point at `step_size` in line-search direction.
    pub fn point_at_step(&self, step_size: StepSize<A>) -> impl Iterator<Item = A> + '_
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        descend(
            step_size,
            self.direction.iter().cloned(),
            self.point.iter().cloned(),
        )
    }

    /// Return whether `point_at_step` is sufficient.
    ///
    /// See [`is_sufficient_decrease`].
    pub fn is_sufficient_decrease(&self, step_size: StepSize<A>, value_at_step: A) -> bool
    where
        A: Clone + PartialOrd + Add<Output = A> + Mul<Output = A>,
    {
        is_sufficient_decrease(
            self.c_1_times_derivatives_dot_direction.clone(),
            self.value.clone(),
            step_size,
            value_at_step,
        )
    }
}

/// The sufficient decrease condition,
/// also known as the Armijo rule,
/// mathematically written as:
/// $f(x_k + a_k p_k) <= f(x_k) + c_1 a_k p_k^T grad_f(x_k)$.
pub fn is_sufficient_decrease<A>(
    c_1_times_derivatives_dot_direction: C1TimesDerivativesDotDirection<A>,
    value: A,
    step_size: StepSize<A>,
    value_at_step: A,
) -> bool
where
    A: PartialOrd + Add<Output = A> + Mul<Output = A>,
{
    value_at_step <= value + c_1_times_derivatives_dot_direction * step_size
}
