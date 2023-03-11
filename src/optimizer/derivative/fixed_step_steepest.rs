//! Fixed step size steepest descent,
//! a very simple derivative optimizer.
//!
//! # Examples
//!
//! ```
//! use ndarray::{prelude::*, Data};
//! use ndarray_rand::RandomExt;
//! use optimal::{optimizer::derivative::fixed_step_steepest::*, prelude::*};
//! use rand::distributions::Uniform;
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let mut iter = (FixedStepSteepestDescent {
//!         config: Config { step_size: StepSize::new(0.5).unwrap() },
//!         state: Array::random(2, Uniform::new(-1.0, 1.0)),
//!         objective_derivatives_function: |xs: ArrayView1<f64>| f_prime(xs),
//!     })
//!     .iterate();
//!     println!("{}", iter.nth(100).unwrap().state);
//! }
//!
//! fn f<S>(point: ArrayBase<S, Ix1>) -> f64
//! where
//!     S: Data<Elem = f64>,
//! {
//!     point.map(|x| x.powi(2)).sum()
//! }
//!
//! fn f_prime<S>(point: ArrayBase<S, Ix1>) -> Array1<f64>
//! where
//!     S: Data<Elem = f64>,
//! {
//!     point.map(|x| 2.0 * x)
//! }
//! ```

use std::ops::{Mul, SubAssign};

use ndarray::{prelude::*, Data};
use replace_with::replace_with_or_abort;

use crate::prelude::*;

use super::StepSize;

/// Fixed step size steepest descent optimizer.
pub struct FixedStepSteepestDescent<A, F> {
    /// Fixed step size steepest descent configuration parameters.
    pub config: Config<A>,
    /// Fixed step size steepest descent state,
    /// a point.
    pub state: Point<A>,
    /// Function returning partial derivatives of objective function.
    pub objective_derivatives_function: F,
}

impl<A, F> Step for FixedStepSteepestDescent<A, F>
where
    A: Clone + SubAssign + Mul<Output = A>,
    F: Fn(ArrayView1<A>) -> Array1<A>,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |point| {
            self.config
                .step_from_evaluated((self.objective_derivatives_function)(point.view()), point)
        });
    }
}

impl<A, F> BestPoint<A> for FixedStepSteepestDescent<A, F> {
    fn best_point(&self) -> CowArray<A, Ix1> {
        (&self.state).into()
    }
}

/// Fixed step size steepest descent configuration parameters.
pub struct Config<A> {
    /// Length of each step.
    pub step_size: StepSize<A>,
}

impl<A> Config<A>
where
    A: Clone + SubAssign + Mul<Output = A>,
{
    /// step from one state to another
    /// given point derivatives.
    pub fn step_from_evaluated<S>(
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

type Point<A> = Array1<A>;
