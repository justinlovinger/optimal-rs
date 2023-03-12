//! Mathematical optimization framework.
//!
//! An optimizer is split between a static configuration
//! and a dynamic state.
//! Traits are defined on the configuration
//! and may take a state.

pub mod derivative;
pub mod derivative_free;
mod iterate;

use ndarray::prelude::*;

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

// impl<A, B, S1, S2, C> Step<A, B, S1, S2> for C
// where
//     C: Points<A, S1> + StepFromEvaluated<B, S1, S2>,
// {
//     fn step<F>(&self, f: F, state: S1) -> S2
//     where
//         F: Fn(CowArray<A, Ix2>) -> Array1<B>,
//     {
//         self.step_from_evaluated(f(self.points(&state)), state)
//     }
// }

/// The core of an optimizer,
/// step from one state to another,
/// improving objective value.
pub trait Step {
    /// Perform an optimization step.
    fn step(&mut self);
}

impl<A, T> Points<A> for Box<T>
where
    T: ?Sized + Points<A>,
{
    fn points(&self) -> CowArray<A, Ix2> {
        self.as_ref().points()
    }
}

/// The secondary core of an optimizer,
/// providing points needing evaluation
/// to guide the optimizer.
pub trait Points<A> {
    /// Return points to be evaluated.
    fn points(&self) -> CowArray<A, Ix2>;
}

impl<T> IsDone for Box<T>
where
    T: ?Sized + IsDone,
{
    fn is_done(&self) -> bool {
        self.as_ref().is_done()
    }
}

/// Indicate whether or not an optimizer is done.
pub trait IsDone {
    /// Return if optimizer is done.
    fn is_done(&self) -> bool;
}

impl<A, T> BestPoint<A> for Box<T>
where
    T: ?Sized + BestPoint<A>,
{
    fn best_point(&self) -> CowArray<A, Ix1> {
        self.as_ref().best_point()
    }
}

/// An optimizer able to return the best point discovered.
pub trait BestPoint<A> {
    /// Return the best point discovered.
    fn best_point(&self) -> CowArray<A, Ix1>;
}

impl<A, T> BestPointValue<A> for Box<T>
where
    T: ?Sized + BestPointValue<A>,
{
    fn best_point_value(&self) -> Option<A> {
        self.as_ref().best_point_value()
    }
}

/// An optimizer able to return the value of the best point discovered.
pub trait BestPointValue<A> {
    /// Return the value of the best point discovered.
    fn best_point_value(&self) -> Option<A>;
}
