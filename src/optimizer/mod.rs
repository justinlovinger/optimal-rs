//! Mathematical optimization framework.
//!
//! An optimizer is split between a static configuration
//! and a dynamic state.
//! Traits are defined on the configuration
//! and may take a state.

pub mod derivative_free;
mod iterate;

use ndarray::{prelude::*, Data};

pub use self::iterate::*;

impl<S, T> InitialState<S> for Box<T>
where
    T: ?Sized + InitialState<S>,
{
    fn initial_state(&self) -> S {
        self.as_ref().initial_state()
    }
}

/// An optimizer with a recommended initial state.
pub trait InitialState<S> {
    /// Return the recommended initial state.
    fn initial_state(&self) -> S;
}

impl<A, B, S1, S2, C> Step<A, B, S1, S2> for C
where
    C: Points<A, S1> + StepFromEvaluated<B, S1, S2>,
{
    fn step<F>(&self, f: F, state: S1) -> S2
    where
        F: Fn(CowArray<A, Ix2>) -> Array1<B>,
    {
        self.step_from_evaluated(f(self.points(&state)), state)
    }
}

/// An automatically implemented extension to [`StepFromEvaluated`]
/// and [`Points`]
/// providing a higher-order function API.
pub trait Step<A, B, S1, S2> {
    /// Return the next state.
    fn step<F>(&self, f: F, state: S1) -> S2
    where
        F: Fn(CowArray<A, Ix2>) -> Array1<B>;
}

impl<A, S1, S2, T> StepFromEvaluated<A, S1, S2> for Box<T>
where
    T: ?Sized + StepFromEvaluated<A, S1, S2>,
{
    fn step_from_evaluated<S: Data<Elem = A>>(
        &self,
        point_values: ArrayBase<S, Ix1>,
        state: S1,
    ) -> S2 {
        self.as_ref().step_from_evaluated(point_values, state)
    }
}

/// The core of an optimizer,
/// step from one state to another
/// given point values.
pub trait StepFromEvaluated<A, S1, S2> {
    /// Return the next state,
    /// given point values.
    fn step_from_evaluated<S: Data<Elem = A>>(
        &self,
        point_values: ArrayBase<S, Ix1>,
        state: S1,
    ) -> S2;
}

impl<A, S, T> Points<A, S> for Box<T>
where
    T: ?Sized + Points<A, S>,
{
    fn points<'a>(&'a self, state: &'a S) -> CowArray<A, Ix2> {
        self.as_ref().points(state)
    }
}

/// The secondary core of an optimizer,
/// providing points needing evaluation
/// to guide the optimizer.
pub trait Points<A, S> {
    /// Return points to be evaluated.
    fn points<'a>(&'a self, state: &'a S) -> CowArray<A, Ix2>;
}

impl<S, T> IsDone<S> for Box<T>
where
    T: ?Sized + IsDone<S>,
{
    fn is_done(&self, state: &S) -> bool {
        self.as_ref().is_done(state)
    }
}

/// Indicate whether or not an optimizer is done.
pub trait IsDone<S> {
    /// Return if optimizer is done.
    #[allow(unused_variables)]
    fn is_done(&self, state: &S) -> bool {
        false
    }
}

impl<A, S, T> BestPoint<A, S> for Box<T>
where
    T: ?Sized + BestPoint<A, S>,
{
    fn best_point<'a>(&'a self, state: &'a S) -> CowArray<A, Ix1> {
        self.as_ref().best_point(state)
    }
}

/// An optimizer able to return the best point discovered.
pub trait BestPoint<A, S> {
    /// Return the best point discovered.
    fn best_point<'a>(&'a self, state: &'a S) -> CowArray<A, Ix1>;
}

impl<A, S, T> BestPointValue<A, S> for Box<T>
where
    T: ?Sized + BestPointValue<A, S>,
{
    fn best_point_value(&self, state: &S) -> Option<A> {
        self.as_ref().best_point_value(state)
    }
}

/// An optimizer able to return the value of the best point discovered.
pub trait BestPointValue<A, S> {
    /// Return the value of the best point discovered.
    fn best_point_value(&self, state: &S) -> Option<A>;
}
