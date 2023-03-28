#![allow(clippy::needless_doctest_main)]

//! Backtracking line search steepest descent.
//!
//! Initial line search step size is chosen by incrementing the previous step size.
//!
//! # Examples
//!
//! ```
//! use ndarray::{prelude::*, Data};
//! use ndarray_rand::RandomExt;
//! use optimal::{optimizer::derivative::backtracking_steepest::*, prelude::*};
//! use rand::distributions::Uniform;
//! use streaming_iterator::StreamingIterator;
//!
//! fn main() {
//!     let backtracking_rate = BacktrackingRate::default();
//!     let mut iter = Running::new(
//!         Config::new(
//!             Sphere,
//!             SufficientDecreaseParameter::default(),
//!             backtracking_rate,
//!             IncrRate::from_backtracking_rate(backtracking_rate),
//!         ),
//!         State::new(
//!             Array::random(2, Uniform::new(-1.0, 1.0)),
//!             StepSize::new(1.0).unwrap(),
//!         ),
//!     )
//!     .into_streaming_iter();
//!     println!("{}", iter.nth(100).unwrap().best_point());
//! }
//!
//! struct Sphere;
//!
//! impl Problem<f64, f64> for Sphere {
//!     fn evaluate<S>(&self, point: ArrayBase<S, Ix1>) -> f64
//!     where
//!         S: ndarray::RawData<Elem = f64> + Data,
//!     {
//!         point.map(|x| x.powi(2)).sum()
//!     }
//! }
//!
//! impl Differentiable<f64, f64> for Sphere {
//!     fn differentiate<S>(&self, point: ArrayBase<S, Ix1>) -> Array1<f64>
//!     where
//!         S: ndarray::RawData<Elem = f64> + Data,
//!     {
//!         point.map(|x| 2.0 * x)
//!     }
//! }
//! ```

use std::{
    borrow::Borrow,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use derive_more::Display;
use ndarray::{linalg::Dot, prelude::*, Data, Zip};
use num_traits::{
    bounds::{LowerBounded, UpperBounded},
    real::Real,
    AsPrimitive, One, Zero,
};
use replace_with::replace_with_or_abort;

use crate::{
    derive::{
        derive_into_inner, derive_new_from_bounded_partial_ord,
        derive_new_from_lower_bounded_partial_ord,
    },
    prelude::*,
};

use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Running backtracking line search steepest descent optimizer
/// with initial line search step size chosen by incrementing previous step size.
#[derive(Clone, Debug)]
pub struct Running<A, BorrowedP, P, C> {
    borrowed_problem: PhantomData<BorrowedP>,
    problem: PhantomData<P>,
    /// Backtracking steepest descent configuration parameters.
    pub config: C,
    /// Backtracking steepest descent state.
    pub state: State<A>,
}

/// Backtracking steepest descent configuration parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A, BorrowedP, P> {
    borrowed_problem: PhantomData<BorrowedP>,
    /// A differentiable objective function.
    pub problem: P,
    /// The sufficient decrease parameter,
    /// `c_1`.
    pub c_1: SufficientDecreaseParameter<A>,
    /// Rate to decrease step size while line searching.
    pub backtracking_rate: BacktrackingRate<A>,
    /// Rate to increase step size before starting each line search.
    pub initial_step_size_incr_rate: IncrRate<A>,
}

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
    point: Point<A>,
    last_step_size: A,
}

/// Line searching.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Searching<A> {
    point: Point<A>,
    point_value: A,
    step_direction: Array1<A>,
    c_1_times_point_derivatives_dot_step_direction: A,
    step_size: A,
    point_at_step: Point<A>,
}

impl<A, BorrowedP, P, C> Running<A, BorrowedP, P, C> {
    /// Return a new 'BacktrackingSteepestDescent'.
    pub fn new(config: C, state: State<A>) -> Self {
        Self {
            borrowed_problem: PhantomData,
            problem: PhantomData,
            config,
            state,
        }
    }
}

impl<A, BorrowedP, P> Config<A, BorrowedP, P> {
    /// Return a new 'Config'.
    pub fn new(
        problem: P,
        c_1: SufficientDecreaseParameter<A>,
        backtracking_rate: BacktrackingRate<A>,
        initial_step_size_incr_rate: IncrRate<A>,
    ) -> Self {
        Self {
            borrowed_problem: PhantomData,
            problem,
            c_1,
            backtracking_rate,
            initial_step_size_incr_rate,
        }
    }
}

impl<A, BorrowedP, P, C> RunningOptimizer for Running<A, BorrowedP, P, C>
where
    A: 'static
        + Clone
        + Copy
        + PartialOrd
        + Neg<Output = A>
        + Add<Output = A>
        + Sub<Output = A>
        + Div<Output = A>
        + Zero
        + One,
    BorrowedP: Differentiable<A, A>,
    P: Borrow<BorrowedP>,
    C: Borrow<Config<A, BorrowedP, P>>,
    f64: AsPrimitive<A>,
{
    fn step(&mut self) {
        replace_with_or_abort(&mut self.state, |state| match state {
            State::Ready(x) => {
                let (point_value, point_derivatives) = self
                    .config
                    .borrow()
                    .problem
                    .borrow()
                    .evaluate_differentiate(x.point_to_evaluate().view());
                x.step_from_evaluated(self.config.borrow(), point_value, point_derivatives)
            }
            State::Searching(x) => {
                let point_value = self
                    .config
                    .borrow()
                    .problem
                    .borrow()
                    .evaluate(x.point_to_evaluate().view());
                x.step_from_evaluated(self.config.borrow(), point_value)
            }
        })
    }
}

impl<A, BorrowedP, P, C> crate::prelude::PointBased<A> for Running<A, BorrowedP, P, C> {
    fn point(&self) -> Option<ArrayView1<A>> {
        self.state.point()
    }
}

impl<A, BorrowedP, P, C> BestPoint<A> for Running<A, BorrowedP, P, C> {
    fn best_point(&self) -> CowArray<A, Ix1> {
        self.state.best_point()
    }
}

impl<A> State<A> {
    /// Return an initial state.
    pub fn new(point: Point<A>, initial_step_size: StepSize<A>) -> Self {
        Self::Ready(Ready {
            point,
            last_step_size: initial_step_size.0,
        })
    }
}

impl<A> crate::prelude::PointBased<A> for State<A> {
    fn point(&self) -> Option<ArrayView1<A>> {
        Some(
            (match self {
                State::Ready(x) => x.point_to_evaluate(),
                State::Searching(x) => x.point_to_evaluate(),
            })
            .into(),
        )
    }
}

impl<A> BestPoint<A> for State<A> {
    fn best_point(&self) -> CowArray<A, Ix1> {
        (match self {
            State::Ready(x) => x.point(),
            State::Searching(x) => x.point(),
        })
        .into()
    }
}

impl<A> Ready<A> {
    fn point(&self) -> &Point<A> {
        &self.point
    }

    fn point_to_evaluate(&self) -> &Point<A> {
        self.point()
    }

    fn step_from_evaluated<BorrowedP, P, S>(
        self,
        config: &Config<A, BorrowedP, P>,
        point_value: A,
        point_derivatives: ArrayBase<S, Ix1>,
    ) -> State<A>
    where
        A: 'static
            + Clone
            + Copy
            + Neg<Output = A>
            + Add<Output = A>
            + Sub<Output = A>
            + Div<Output = A>
            + One,
        S: Data<Elem = A>,
        f64: AsPrimitive<A>,
        ArrayBase<S, Ix1>: Dot<Array1<A>, Output = A>,
    {
        let step_direction = point_derivatives.mapv(|x| -x);
        let step_size = config.initial_step_size_incr_rate * self.last_step_size;
        State::Searching(Searching {
            point_at_step: descend(&self.point, step_size, &step_direction),
            point: self.point,
            point_value,
            c_1_times_point_derivatives_dot_step_direction: config.c_1.0
                * point_derivatives.dot(&step_direction),
            step_direction,
            step_size,
        })
    }
}

impl<A> Searching<A> {
    fn point(&self) -> &Point<A> {
        &self.point
    }

    fn point_to_evaluate(&self) -> &Point<A> {
        &self.point_at_step
    }

    fn step_from_evaluated<BorrowedP, P>(
        mut self,
        config: &Config<A, BorrowedP, P>,
        point_value: A,
    ) -> State<A>
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
            self.step_size = config.backtracking_rate.0 * self.step_size;
            self.point_at_step = descend(&self.point, self.step_size, &self.step_direction);
            State::Searching(self)
        }
    }
}

fn descend<A>(point: &Point<A>, step_size: A, step_direction: &Array1<A>) -> Point<A>
where
    A: Clone + Add<Output = A> + Mul<Output = A>,
{
    Zip::from(point)
        .and(step_direction)
        .map_collect(|x, d| x.clone() + step_size.clone() * d.clone())
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

/// The sufficient decrease parameter,
/// `c_1`.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SufficientDecreaseParameter<A>(A);

derive_new_from_bounded_partial_ord!(SufficientDecreaseParameter<A: Real>);
derive_into_inner!(SufficientDecreaseParameter<A>);

impl<A> Default for SufficientDecreaseParameter<A>
where
    A: 'static + Copy,
    f64: AsPrimitive<A>,
{
    fn default() -> Self {
        Self(0.5.as_())
    }
}

impl<A> LowerBounded for SufficientDecreaseParameter<A>
where
    A: Real,
{
    fn min_value() -> Self {
        Self(A::epsilon())
    }
}

impl<A> UpperBounded for SufficientDecreaseParameter<A>
where
    A: Real,
{
    fn max_value() -> Self {
        Self(A::one() - A::epsilon())
    }
}

/// Rate to decrease step size while line searching.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BacktrackingRate<A>(A);

derive_new_from_bounded_partial_ord!(BacktrackingRate<A: Real>);
derive_into_inner!(BacktrackingRate<A>);

impl<A> Default for BacktrackingRate<A>
where
    A: 'static + Copy,
    f64: AsPrimitive<A>,
{
    fn default() -> Self {
        Self(0.5.as_())
    }
}

impl<A> LowerBounded for BacktrackingRate<A>
where
    A: Real,
{
    fn min_value() -> Self {
        Self(A::epsilon())
    }
}

impl<A> UpperBounded for BacktrackingRate<A>
where
    A: Real,
{
    fn max_value() -> Self {
        Self(A::one() - A::epsilon())
    }
}

/// Rate to increase step size before starting each line search.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IncrRate<A>(A);

derive_new_from_lower_bounded_partial_ord!(IncrRate<A: Real>);
derive_into_inner!(IncrRate<A>);

impl<A> IncrRate<A>
where
    A: 'static + Copy + One + Sub<Output = A> + Div<Output = A>,
    f64: AsPrimitive<A>,
{
    /// Return increase rate slightly more than one step up from backtracking rate.
    pub fn from_backtracking_rate(x: BacktrackingRate<A>) -> IncrRate<A> {
        Self(2.0.as_() / x.into_inner() - A::one())
    }
}

impl<A> LowerBounded for IncrRate<A>
where
    A: Real,
{
    fn min_value() -> Self {
        Self(A::one() + A::epsilon())
    }
}

impl<A> Mul<A> for IncrRate<A>
where
    A: Mul<Output = A>,
{
    type Output = A;

    fn mul(self, rhs: A) -> Self::Output {
        self.0 * rhs
    }
}

type Point<A> = Array1<A>;
