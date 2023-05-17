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
//!     println!(
//!         "{}",
//!         BacktrackingSteepest::new(
//!             Sphere,
//!             Config::new(
//!                 SufficientDecreaseParameter::default(),
//!                 backtracking_rate,
//!                 IncrRate::from_backtracking_rate(backtracking_rate),
//!             ),
//!         )
//!         .expect("should never fail")
//!         .start()
//!         .nth(100)
//!         .unwrap()
//!         .best_point()
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

use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use derive_more::Display;
use ndarray::{linalg::Dot, prelude::*, Data, Zip};
use num_traits::{
    bounds::{LowerBounded, UpperBounded},
    real::Real,
    AsPrimitive, One,
};
use rand::{
    distributions::uniform::{SampleUniform, Uniform},
    prelude::*,
};

use crate::{
    derive::{
        derive_into_inner, derive_new_from_bounded_partial_ord,
        derive_new_from_lower_bounded_partial_ord,
    },
    optimizer::MismatchedLengthError,
    prelude::*,
};

use super::StepSize;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Running backtracking line search steepest descent optimizer
/// with initial line search step size chosen by incrementing previous step size.
pub type BacktrackingSteepest<A, P> = Optimizer<P, Config<A>>;

/// Backtracking steepest descent configuration parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A> {
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

impl<A> Config<A> {
    /// Return a new 'Config'.
    pub fn new(
        c_1: SufficientDecreaseParameter<A>,
        backtracking_rate: BacktrackingRate<A>,
        initial_step_size_incr_rate: IncrRate<A>,
    ) -> Self {
        Self {
            c_1,
            backtracking_rate,
            initial_step_size_incr_rate,
        }
    }
}

impl<A, P> OptimizerConfig<P> for Config<A>
where
    A: Debug + SampleUniform + Real + 'static,
    P: Differentiable<PointElem = A, PointValue = A> + FixedLength + Bounded,
    f64: AsPrimitive<A>,
{
    type Err = ();

    type State = State<A>;

    type StateErr = MismatchedLengthError;

    fn validate(&self, _problem: &P) -> Result<(), Self::Err> {
        Ok(())
    }

    fn validate_state(&self, problem: &P, state: &Self::State) -> Result<(), Self::StateErr> {
        // Note,
        // this assumes states cannot be modified
        // outside of `initial_state`
        // and `step`.
        // As of the writing of this method,
        // all states are derived from an initial state
        // and the only way for a state to be invalid
        // is if it was from a different problem.
        match state {
            State::Ready(x) => {
                if x.point.len() == problem.len() {
                    Ok(())
                } else {
                    Err(MismatchedLengthError)
                }
            }
            State::Searching(x) => {
                if x.point.len() == problem.len() {
                    Ok(())
                } else {
                    Err(MismatchedLengthError)
                }
            }
        }
    }

    unsafe fn initial_state(&self, problem: &P) -> Self::State {
        let mut rng = thread_rng();
        State::new(
            problem
                .bounds()
                .take(problem.len())
                .map(|range| {
                    let (start, end) = range.into_inner();
                    Uniform::new_inclusive(start, end).sample(&mut rng)
                })
                .collect(),
            StepSize::new(A::one()).unwrap(),
        )
    }

    unsafe fn step(&self, problem: &P, state: Self::State) -> Self::State {
        match state {
            State::Ready(x) => {
                let (point_value, point_derivatives) =
                    problem.evaluate_differentiate(x.point().into());
                x.step_from_evaluated(self, point_value, point_derivatives)
            }
            State::Searching(x) => {
                let point_value = problem.evaluate(x.point().into());
                x.step_from_evaluated(self, point_value)
            }
        }
    }
}

impl<A, P> OptimizerState<P> for State<A>
where
    P: Problem<PointElem = A, PointValue = A>,
{
    type Evaluatee<'a> = ArrayView1<'a, P::PointElem> where A: 'a;

    fn evaluatee(&self) -> Self::Evaluatee<'_> {
        (match self {
            State::Ready(x) => x.point(),
            State::Searching(x) => x.point(),
        })
        .into()
    }

    fn best_point(&self) -> CowArray<P::PointElem, Ix1> {
        (match self {
            State::Ready(x) => x.best_point(),
            State::Searching(x) => x.best_point(),
        })
        .into()
    }

    fn stored_best_point_value(&self) -> Option<&P::PointValue> {
        match self {
            State::Ready(_) => None,
            State::Searching(x) => Some(&x.point_value),
        }
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

impl<A> Ready<A> {
    fn point(&self) -> &Point<A> {
        self.best_point()
    }

    fn best_point(&self) -> &Point<A> {
        &self.point
    }

    fn step_from_evaluated<S>(
        self,
        config: &Config<A>,
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
    fn best_point(&self) -> &Point<A> {
        &self.point
    }

    #[allow(clippy::misnamed_getters)]
    fn point(&self) -> &Point<A> {
        &self.point_at_step
    }

    fn step_from_evaluated(mut self, config: &Config<A>, point_value: A) -> State<A>
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
